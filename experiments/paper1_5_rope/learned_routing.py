"""Minimal learned semantic-routing geometry for the controlled RoPE study.

The router transforms attention-input hidden states only. It never owns or
modifies the post-RoPE K/V tensors that native attention materializes after
selection.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLinearRouter(nn.Module):
    """Project query and chunk hidden states into one compact cosine space.

    Queries have shape ``[B, D]`` and candidate chunks have shape
    ``[B, C, D]``. The returned scores are ``[B, C]``. Separate projections
    permit the two roles to learn different linear metrics without changing
    the frozen transformer or its native attention projections.
    """

    def __init__(self, d_model: int, routing_dim: int):
        super().__init__()
        if d_model <= 0 or routing_dim <= 0:
            raise ValueError("d_model and routing_dim must be positive")
        self.d_model = int(d_model)
        self.routing_dim = int(routing_dim)
        self.query_projection = nn.Linear(d_model, routing_dim, bias=False)
        self.chunk_projection = nn.Linear(d_model, routing_dim, bias=False)

    def forward(
        self,
        query_hidden: torch.Tensor,
        chunk_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if query_hidden.ndim != 2 or query_hidden.shape[-1] != self.d_model:
            raise ValueError(f"Expected query hidden states [B,{self.d_model}]")
        if (
            chunk_hidden.ndim != 3
            or chunk_hidden.shape[0] != query_hidden.shape[0]
            or chunk_hidden.shape[-1] != self.d_model
        ):
            raise ValueError(f"Expected chunk hidden states [B,C,{self.d_model}]")
        query = F.normalize(self.query_projection(query_hidden), dim=-1)
        chunks = F.normalize(self.chunk_projection(chunk_hidden), dim=-1)
        return torch.einsum("bd,bcd->bc", query, chunks)


def cosine_scores(query: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """Return cosine scores for ``[B,D]`` queries and ``[B,C,D]`` chunks."""

    if query.ndim != 2 or chunks.ndim != 3:
        raise ValueError("Expected query [B,D] and chunks [B,C,D]")
    if query.shape[0] != chunks.shape[0] or query.shape[-1] != chunks.shape[-1]:
        raise ValueError("Query and chunk batch/feature dimensions must match")
    return torch.einsum(
        "bd,bcd->bc",
        F.normalize(query, dim=-1),
        F.normalize(chunks, dim=-1),
    )


def contrastive_margin_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Penalize every positive candidate that trails an in-example negative."""

    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must share shape [B,C]")
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    pair_mask = positive_mask[:, :, None] & ~positive_mask[:, None, :]
    if not bool(pair_mask.any()):
        raise ValueError("Every training batch needs at least one positive-negative pair")
    pair_losses = F.relu(
        float(margin) - scores[:, :, None] + scores[:, None, :]
    )
    return pair_losses[pair_mask].mean()


def shuffled_positive_mask(positive_mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Replace each evidence set with equally many non-evidence labels.

    This within-example control preserves candidate count and class balance. It
    intentionally fails when an example has too few negatives, because silently
    retaining true labels would weaken the control.
    """

    if positive_mask.ndim != 2:
        raise ValueError("positive_mask must have shape [B,C]")
    rng = random.Random(int(seed))
    output = torch.zeros_like(positive_mask, dtype=torch.bool)
    for row_index, row in enumerate(positive_mask.to(dtype=torch.bool).cpu()):
        positives = [index for index, value in enumerate(row.tolist()) if value]
        negatives = [index for index, value in enumerate(row.tolist()) if not value]
        if not positives or len(negatives) < len(positives):
            raise ValueError("Shuffled labels require at least as many negatives as positives")
        for index in rng.sample(negatives, len(positives)):
            output[row_index, index] = True
    return output.to(positive_mask.device)


def train_router(
    router: AsymmetricLinearRouter,
    query_hidden: torch.Tensor,
    chunk_hidden: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    steps: int = 300,
    learning_rate: float = 1e-2,
    margin: float = 0.2,
) -> list[float]:
    """Fit only ``A_q`` and ``A_c`` on frozen, precomputed representations."""

    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    history = []
    router.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_margin_loss(
            router(query_hidden, chunk_hidden), positive_mask, margin=margin
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    router.eval()
    return history


def rank_candidate_ids(
    scores: torch.Tensor,
    candidate_ids: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Sort candidates by descending score with identity-stable tie breaking."""

    if scores.ndim != 2 or scores.shape[0] != len(candidate_ids):
        raise ValueError("scores and candidate_ids must describe the same batch")
    rankings = []
    for row_scores, row_ids in zip(scores.detach().cpu(), candidate_ids, strict=True):
        if len(row_ids) != row_scores.numel() or len(set(row_ids)) != len(row_ids):
            raise ValueError("Candidate IDs must be unique and align with score columns")
        scored = zip(row_ids, row_scores.tolist(), strict=True)
        rankings.append([identity for identity, _ in sorted(scored, key=lambda item: (-item[1], item[0]))])
    return rankings


def materialize_native_payload(
    payload_by_id: dict[str, tuple[torch.Tensor, torch.Tensor]],
    selected_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate selected native post-RoPE K/V without transforming either."""

    if not selected_ids:
        raise ValueError("At least one selected chunk is required")
    selected = [payload_by_id[identity] for identity in selected_ids]
    return (
        torch.cat([key for key, _ in selected], dim=2),
        torch.cat([value for _, value in selected], dim=2),
    )


def trainable_parameter_count(module: nn.Module) -> int:
    """Count parameters optimized by the learned routing metric."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
