"""Trainable calibration gates for the Hugging Face PRA memory residual."""

from __future__ import annotations

import torch
import torch.nn as nn


MEMORY_GATE_FIXED = "fixed"
MEMORY_GATE_SINGLE = "single"
MEMORY_GATE_PER_LAYER = "per_layer"
MEMORY_GATE_MODES = {
    MEMORY_GATE_FIXED,
    MEMORY_GATE_SINGLE,
    MEMORY_GATE_PER_LAYER,
}


class PRAHFMemoryGate(nn.Module):
    """Own fixed, shared-scalar, and per-layer PRA memory scales.

    The module is registered once on the wrapped HF model. Attention adapters
    keep non-owning references so a shared scalar is not registered repeatedly.
    Only parameters selected by the active mode require gradients.
    """

    def __init__(
        self,
        layer_ids: tuple[int, ...],
        *,
        mode: str = MEMORY_GATE_FIXED,
        initial_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.layer_ids = tuple(sorted({int(layer_id) for layer_id in layer_ids}))
        if not self.layer_ids:
            raise ValueError("A PRA memory gate requires at least one layer.")
        self.shared = nn.Parameter(torch.tensor(float(initial_value)))
        self.per_layer = nn.ParameterDict(
            {
                str(layer_id): nn.Parameter(torch.tensor(float(initial_value)))
                for layer_id in self.layer_ids
            }
        )
        self.fixed_value = float(initial_value)
        self.mode = MEMORY_GATE_FIXED
        self.configure(mode, initial_value=initial_value)

    def configure(self, mode: str, *, initial_value: float | None = None) -> None:
        """Select gate ownership and optionally reset all candidate values."""
        if mode not in MEMORY_GATE_MODES:
            raise ValueError(f"Unsupported HF PRA memory gate mode: {mode}")
        if initial_value is not None:
            value = float(initial_value)
            self.fixed_value = value
            with torch.no_grad():
                self.shared.fill_(value)
                for parameter in self.per_layer.values():
                    parameter.fill_(value)
        self.mode = mode
        self.shared.requires_grad_(mode == MEMORY_GATE_SINGLE)
        for parameter in self.per_layer.values():
            parameter.requires_grad_(mode == MEMORY_GATE_PER_LAYER)

    def value(self, layer_id: int, reference: torch.Tensor) -> torch.Tensor:
        """Return this layer's scalar on the residual tensor device."""
        if self.mode == MEMORY_GATE_FIXED:
            return torch.tensor(
                self.fixed_value,
                device=reference.device,
                dtype=torch.float32,
            )
        if self.mode == MEMORY_GATE_SINGLE:
            parameter = self.shared
        else:
            try:
                parameter = self.per_layer[str(int(layer_id))]
            except KeyError as error:
                raise ValueError(f"No PRA memory gate exists for layer {layer_id}.") from error
        return parameter.to(device=reference.device)

    @property
    def requires_delta_path(self) -> bool:
        """Whether attention must compute local and memory outputs separately."""
        return self.mode != MEMORY_GATE_FIXED or self.fixed_value != 1.0

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters owned by the active gate variant."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def values(self) -> dict[int, float]:
        """Serialize the effective scalar at every injected layer."""
        if self.mode == MEMORY_GATE_FIXED:
            return {layer_id: self.fixed_value for layer_id in self.layer_ids}
        if self.mode == MEMORY_GATE_SINGLE:
            value = float(self.shared.detach().cpu())
            return {layer_id: value for layer_id in self.layer_ids}
        return {
            layer_id: float(self.per_layer[str(layer_id)].detach().cpu())
            for layer_id in self.layer_ids
        }
