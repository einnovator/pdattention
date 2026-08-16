"""PRA-conditioned LoRA deltas for late decoder attention projections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRAHFConditionalOutputLoRA(nn.Module):
    """Add a low-rank delta to one layer's native attention output projection.

    The input is the pre-``o_proj`` attention result ``[B,T,D]``. The module
    returns only the learned delta; its zero-initialized up projection makes
    rank activation initially identical to frozen PRA.
    """

    def __init__(
        self,
        input_width: int,
        output_width: int,
        rank: int,
        *,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(self.input_width, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_width, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the low-rank branch as an exact zero function."""
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, attention_features: torch.Tensor) -> torch.Tensor:
        """Return the FP32 LoRA delta for ``[B,T,D]`` attention features."""
        features = F.dropout(
            attention_features.float(),
            p=self.dropout,
            training=self.training,
        )
        return self.scaling * self.up(self.down(features))


class PRAHFConditionalOutputLoRABank(nn.Module):
    """Lazily own one conditional output-projection LoRA per PRA layer."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        layer_ids: tuple[int, ...],
        *,
        rank: int = 0,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.layer_ids = tuple(sorted({int(layer_id) for layer_id in layer_ids}))
        self.adapters = nn.ModuleDict()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.rank = 0
        self.alpha = 0.0
        self.dropout = 0.0
        if rank:
            self.configure(rank, alpha=alpha, dropout=dropout)

    @property
    def enabled(self) -> bool:
        """Whether selected-memory attention should receive a LoRA delta."""
        return self.rank > 0

    def configure(
        self,
        rank: int,
        *,
        alpha: float | None = None,
        dropout: float = 0.0,
        reset: bool = True,
    ) -> None:
        """Select a rank or disable LoRA with rank zero."""
        rank = int(rank)
        dropout = float(dropout)
        if rank < 0:
            raise ValueError("Conditional LoRA rank must be non-negative.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Conditional LoRA dropout must be in [0, 1).")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha) if rank else 0.0
        self.dropout = dropout
        if rank == 0:
            return
        if self.alpha <= 0.0:
            raise ValueError("Conditional LoRA alpha must be positive.")
        key = self._key(rank, self.alpha, dropout)
        if key not in self.adapters:
            created = nn.ModuleDict(
                {
                    str(layer_id): PRAHFConditionalOutputLoRA(
                        self.input_width,
                        self.output_width,
                        rank,
                        alpha=self.alpha,
                        dropout=dropout,
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
        attention_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return this layer's conditional delta, or an exact zero tensor."""
        if not self.enabled:
            return attention_features.new_zeros(
                *attention_features.shape[:-1],
                self.output_width,
            )
        key = self._key(self.rank, self.alpha, self.dropout)
        return self.adapters[key][str(int(layer_id))](attention_features)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only factors for the active rank and hyperparameters."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @property
    def trainable_parameter_count(self) -> int:
        """Count active LoRA scalars across the selected late-layer band."""
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    @staticmethod
    def _key(rank: int, alpha: float, dropout: float) -> str:
        """Create a stable ModuleDict key for one lazy configuration."""
        return f"r{rank}_a{alpha:g}_d{dropout:g}".replace(".", "p")
