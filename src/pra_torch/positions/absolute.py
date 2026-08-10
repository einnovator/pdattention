"""Learned absolute position encoding used by the original PRA models."""

import torch
import torch.nn as nn

from .base import PositionEncoding


class AbsolutePositionEncoding(PositionEncoding):
    """Add the model's learned table to embeddings and leave projected Q/K unchanged."""

    mode = "absolute"
    has_bounded_positions = True

    def apply_embeddings(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        absolute_embedding: nn.Embedding,
    ) -> torch.Tensor:
        return token_embeddings + absolute_embedding(position_ids)

    def transform_qk(self, query, key, position_ids):
        _ = position_ids
        return query, key
