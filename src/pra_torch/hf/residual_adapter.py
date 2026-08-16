"""Tiny PRA-conditioned residual adapters for frozen Hugging Face decoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRAHFResidualAdapter(nn.Module):
    """Correct one layer's PRA residual through a small bottleneck MLP."""

    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.down = nn.Linear(self.hidden_size, self.bottleneck)
        self.up = nn.Linear(self.bottleneck, self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start exactly at frozen PRA and learn only a residual correction."""
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Return frozen PRA's residual plus a learned FP32 correction."""
        # The adapter calibrates the counterfactual effect of memory itself.
        # ``hidden_states`` remains in the stable call signature but is not an
        # input feature: delta = y_mem - y_local is the complete intervention.
        del hidden_states
        correction = self.up(F.silu(self.down(memory_residual.float())))
        return memory_residual.float() + correction


class PRAHFResidualAdapterBank(nn.Module):
    """Lazily own one residual adapter per PRA-active decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        layer_ids: tuple[int, ...],
        *,
        bottleneck: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_ids = tuple(sorted({int(layer_id) for layer_id in layer_ids}))
        self.adapters = nn.ModuleDict()
        self.register_buffer(
            "_device_anchor",
            torch.empty(0),
            persistent=False,
        )
        self.bottleneck = 0
        if bottleneck:
            self.configure(bottleneck)

    @property
    def enabled(self) -> bool:
        """Whether the memory-active path should apply learned corrections."""
        return self.bottleneck > 0

    def configure(self, bottleneck: int, *, reset: bool = True) -> None:
        """Enable one width or disable the residual adapter with width zero."""
        bottleneck = int(bottleneck)
        if bottleneck < 0:
            raise ValueError("Residual-adapter bottleneck must be non-negative.")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.bottleneck = bottleneck
        if bottleneck == 0:
            return
        key = str(bottleneck)
        if key not in self.adapters:
            created = nn.ModuleDict(
                {
                    str(layer_id): PRAHFResidualAdapter(
                        self.hidden_size,
                        bottleneck,
                    )
                    for layer_id in self.layer_ids
                }
            )
            self.adapters[key] = created.to(self._device_anchor.device)
        elif reset:
            for adapter in self.adapters[key].values():
                adapter.reset_parameters()
        for parameter in self.adapters[key].parameters():
            parameter.requires_grad_(True)

    def transform(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        memory_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the active layer-local correction, or return PRA unchanged."""
        if not self.enabled:
            return memory_residual
        adapter = self.adapters[str(self.bottleneck)][str(int(layer_id))]
        return adapter(hidden_states, memory_residual)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters belonging to the active bottleneck."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @property
    def trainable_parameter_count(self) -> int:
        """Count active residual-adapter scalars."""
        return sum(parameter.numel() for parameter in self.trainable_parameters())
