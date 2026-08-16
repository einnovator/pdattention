"""Parameter-free absolute sinusoidal positions for controlled comparisons."""

import math

import torch
import torch.nn as nn

from .base import PositionEncoding, batched_position_ids


class SinusoidalPositionEncoding(PositionEncoding):
    """Add the original transformer sinusoidal basis to token embeddings."""

    mode = "sinusoidal"
    has_bounded_positions = False

    def apply_embeddings(self, token_embeddings, position_ids, absolute_embedding: nn.Embedding):
        _ = absolute_embedding
        positions = batched_position_ids(position_ids, token_embeddings.shape[0]).to(
            device=token_embeddings.device,
            dtype=torch.float32,
        )
        width = token_embeddings.shape[-1]
        frequency = torch.exp(
            torch.arange(0, width, 2, device=token_embeddings.device, dtype=torch.float32)
            * (-math.log(10_000.0) / width)
        )
        phase = positions[..., None] * frequency
        encoding = torch.zeros_like(token_embeddings)
        encoding[..., 0::2] = phase.sin().to(encoding.dtype)
        encoding[..., 1::2] = phase.cos().to(encoding.dtype)
        return token_embeddings + encoding

    def transform_qk(self, query, key, position_ids):
        _ = position_ids
        return query, key
