"""Existing one-gist pooling modes expressed with the uniform ``[G,D]`` contract."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .base import ComputedGists, GistContext
from .common import empty_gists, mean_cosine_separation, validate_points


class GRUGistPooler(nn.Module):
    """Learned recurrent reduction from ``[B,T,D]`` to ``[B,D]``."""

    def __init__(
        self,
        d_model: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_size = int(hidden_size or d_model)
        self.bidirectional = bool(bidirectional)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        directions = 2 if self.bidirectional else 1
        self.output = nn.Linear(hidden_size * directions, d_model)

    def forward(self, token_keys: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Pool padded sequences, optionally using their true lengths."""
        if token_keys.ndim != 3:
            raise ValueError(f"Expected token keys [batch,tokens,model], got {token_keys.shape}.")
        if lengths is not None:
            packed = pack_padded_sequence(
                token_keys,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(token_keys)
        pooled = (
            torch.cat((hidden[-2], hidden[-1]), dim=-1)
            if self.bidirectional
            else hidden[-1]
        )
        return self.output(pooled)


class SingleGistStrategy:
    """Apply mean, last, ref-end, or registered GRU pooling exactly once."""

    def __init__(self, mode: str):
        self.mode = mode

    def _index(self, context: GistContext) -> int:
        if context.tokenizer is None:
            raise ValueError("ref_end gist mode requires a tokenizer.")
        marker_ids = context.tokenizer.encode(context.ref_end_token)
        if len(marker_ids) != 1:
            raise ValueError(
                f"Reference terminator {context.ref_end_token!r} is not one atomic token."
            )
        indices = [
            index for index, token_id in enumerate(context.token_ids) if token_id == marker_ids[0]
        ]
        if len(indices) != 1:
            raise ValueError(
                f"Expected exactly one {context.ref_end_token!r} token, found {len(indices)}."
            )
        return indices[0]

    def _pool(self, points: torch.Tensor, context: GistContext) -> torch.Tensor:
        if self.mode == "mean":
            return points.mean(dim=0)
        if self.mode == "last":
            return points[-1]
        if self.mode == "ref_end":
            return points[self._index(context)]
        if self.mode == "gru":
            if context.gru_pooler is None:
                raise ValueError("gist_mode='gru' requires the model's registered GRU pooler.")
            lengths = torch.tensor([points.shape[0]], device=points.device)
            return context.gru_pooler(points.unsqueeze(0), lengths=lengths).squeeze(0)
        raise ValueError(f"Unsupported single-gist mode: {self.mode}")

    def compute(self, *, keys, values, num_gists, config, context) -> ComputedGists:
        validate_points(keys, values)
        if keys.shape[0] == 0:
            return empty_gists(
                keys,
                values,
                mode=self.mode,
                requested_gists=int(num_gists),
                actual_gists=0,
            )
        key_gist = self._pool(keys, context).unsqueeze(0)
        value_gist = self._pool(values, context).unsqueeze(0) if values is not None else None
        return ComputedGists(
            k=key_gist,
            v=value_gist,
            metadata={
                "mode": self.mode,
                "requested_gists": int(num_gists),
                "actual_gists": 1,
                "occupancy": [int(keys.shape[0])],
                "mean_cosine_separation": mean_cosine_separation(key_gist),
            },
        )
