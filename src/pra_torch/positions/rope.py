"""Rotary position embedding (RoPE) for per-head attention queries and keys."""

from __future__ import annotations

import torch

from .base import PositionEncoding, batched_position_ids


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs ``(x0,x1)`` to ``(-x1,x0)``."""
    paired = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.flatten(-2)


class RotaryPositionEncoding(PositionEncoding):
    """Apply complex-plane rotations whose phase depends on logical token position."""

    mode = "rope"
    has_bounded_positions = False

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        if theta <= 0:
            raise ValueError("rope_theta must be positive.")
        frequencies = 1.0 / (
            float(theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def _cos_sin(
        self,
        position_ids: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = batched_position_ids(position_ids, reference.shape[0]).to(
            device=reference.device,
            dtype=torch.float32,
        )
        phase = positions[..., None] * self.inverse_frequencies.to(reference.device)
        phase = torch.repeat_interleave(phase, 2, dim=-1).unsqueeze(1)
        return phase.cos().to(reference.dtype), phase.sin().to(reference.dtype)

    def apply_rotary(self, tensor: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Rotate one ``[B,H,T,Dh]`` tensor at the supplied logical positions."""
        cosine, sine = self._cos_sin(position_ids, tensor)
        return tensor * cosine + rotate_half(tensor) * sine

    def transform_qk(self, query, key, position_ids):
        return self.apply_rotary(query, position_ids), self.apply_rotary(key, position_ids)
