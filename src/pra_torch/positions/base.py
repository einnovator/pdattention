"""Position-encoding interfaces shared by decoder and retrieved native-KV paths."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PositionEncoding(nn.Module, ABC):
    """Define how token embeddings and per-head Q/K receive position information."""

    mode: str
    has_bounded_positions: bool = False

    def apply_embeddings(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        absolute_embedding: nn.Embedding,
    ) -> torch.Tensor:
        """Return positioned hidden states shaped ``[B,T,D]``."""
        _ = position_ids, absolute_embedding
        return token_embeddings

    @abstractmethod
    def transform_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Position query/key tensors shaped ``[B,H,T,Dh]``."""


def batched_position_ids(position_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize ``[T]`` or ``[B,T]`` positions to ``[B,T]``."""
    if position_ids.ndim == 1:
        return position_ids.unsqueeze(0).expand(batch_size, -1)
    if position_ids.ndim != 2 or position_ids.shape[0] != batch_size:
        raise ValueError("position_ids must have shape [tokens] or [batch,tokens].")
    return position_ids
