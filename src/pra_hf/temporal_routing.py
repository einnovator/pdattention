"""Temporal query aggregation and routing diagnostics for Paper 2.9.

The functions in this module operate only on routing representations. They do
not modify causal backbone states, native keys/values, or the materialization
budget. Query rows are ordered token states and memory rows are routing-only
representatives grouped by chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch


TemporalReducer = Literal["current", "mean", "recency", "late_max", "late_top_mean"]


@dataclass(frozen=True)
class TemporalWindow:
    """Token provenance for one routing decision.

    ``anchor`` is the token where routing would have been requested. ``start``
    and ``stop`` delimit the states made visible to the router. A positive
    ``look_ahead`` is analysis-only unless every state belongs to known prompt
    prefill; causal delayed commitment moves the anchor forward instead.
    """

    anchor: int
    start: int
    stop: int
    look_behind: int
    look_ahead: int

    @property
    def length(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


@dataclass(frozen=True)
class RoutingDiagnostics:
    """Uncertainty and participation statistics for one chunk-score vector."""

    entropy: float
    normalized_entropy: float
    top1_margin: float
    effective_candidates: float
    contributing_query_states: float
    contributing_memory_modes: float
    score_concentration: float


def temporal_window(
    token_count: int,
    *,
    anchor: int,
    look_behind: int,
    look_ahead: int = 0,
) -> TemporalWindow:
    """Return a clipped temporal routing window with exact token provenance."""
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if anchor < 0 or anchor >= token_count:
        raise IndexError("anchor is outside the token sequence.")
    if look_behind <= 0 or look_ahead < 0:
        raise ValueError("look_behind must be positive and look_ahead non-negative.")
    start = max(0, anchor - int(look_behind) + 1)
    stop = min(token_count, anchor + int(look_ahead) + 1)
    return TemporalWindow(anchor, start, stop, int(look_behind), int(look_ahead))


def recency_weights(length: int, *, decay: float = 0.75, device=None) -> torch.Tensor:
    """Return validation-free geometric weights, newest query state first."""
    if length <= 0 or not 0 < decay <= 1:
        raise ValueError("length must be positive and decay in (0, 1].")
    ages = torch.arange(length - 1, -1, -1, dtype=torch.float32, device=device)
    weights = decay**ages
    return weights / weights.sum()


def _validate_inputs(
    projected_queries: torch.Tensor,
    projected_memory: torch.Tensor,
    memory_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if projected_queries.ndim == 1:
        projected_queries = projected_queries.unsqueeze(0)
    if projected_queries.ndim != 2 or projected_memory.ndim != 3:
        raise ValueError("Queries and memory must be [Q,r] and [C,M,r].")
    if projected_queries.shape[-1] != projected_memory.shape[-1]:
        raise ValueError("Query and memory ranks must match.")
    if memory_mask.shape != projected_memory.shape[:2]:
        raise ValueError("memory_mask must have shape [C,M].")
    if not bool(memory_mask.any(dim=1).all()):
        raise ValueError("Every chunk needs at least one routing representative.")
    return projected_queries, projected_memory, memory_mask.to(torch.bool)


def temporal_chunk_scores(
    projected_queries: torch.Tensor,
    projected_memory: torch.Tensor,
    memory_mask: torch.Tensor,
    *,
    reducer: TemporalReducer = "current",
    top_r: int = 4,
    recency_decay: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score temporal queries against compact memory representatives.

    Args:
        projected_queries: Temporal query states ``[Q, r]``, oldest to newest.
        projected_memory: Routing representatives ``[chunks, modes, r]``.
        memory_mask: Valid representative mask ``[chunks, modes]``.
        reducer: Early query pooling (``current``, ``mean``, ``recency``) or
            token-by-mode late interaction (``late_max``, ``late_top_mean``).
        top_r: Number of strongest memory modes retained by top-mean reduction.

    Returns:
        Chunk scores ``[chunks]`` and the unreduced interaction tensor
        ``[query_states, chunks, memory_modes]`` for diagnostics.
    """
    queries, memory, mask = _validate_inputs(
        projected_queries, projected_memory, memory_mask
    )
    if top_r <= 0:
        raise ValueError("top_r must be positive.")
    interactions = torch.einsum("qr,cmr->qcm", queries, memory)
    interactions = interactions / sqrt(float(memory.shape[-1]))
    interactions = interactions.masked_fill(~mask.unsqueeze(0), float("-inf"))

    if reducer in {"current", "mean", "recency"}:
        if reducer == "current":
            pooled = queries[-1]
        elif reducer == "mean":
            pooled = queries.mean(dim=0)
        else:
            weights = recency_weights(
                len(queries), decay=recency_decay, device=queries.device
            ).to(queries.dtype)
            pooled = torch.einsum("q,qr->r", weights, queries)
        pooled_dots = torch.einsum("r,cmr->cm", pooled, memory)
        pooled_dots = pooled_dots / sqrt(float(memory.shape[-1]))
        pooled_dots = pooled_dots.masked_fill(~mask, float("-inf"))
        count = min(int(top_r), memory.shape[1])
        values = pooled_dots.topk(count, dim=-1).values
        finite = torch.isfinite(values)
        scores = values.masked_fill(~finite, 0).sum(dim=-1)
        scores = scores / finite.sum(dim=-1).clamp_min(1)
        return scores, interactions

    if reducer == "late_max":
        per_query = interactions.amax(dim=-1)
    elif reducer == "late_top_mean":
        count = min(int(top_r), memory.shape[1])
        values = interactions.topk(count, dim=-1).values
        finite = torch.isfinite(values)
        per_query = values.masked_fill(~finite, 0).sum(dim=-1)
        per_query = per_query / finite.sum(dim=-1).clamp_min(1)
    else:
        raise ValueError(f"Unsupported temporal reducer: {reducer}")
    weights = recency_weights(
        len(queries), decay=recency_decay, device=queries.device
    ).to(queries.dtype)
    return torch.einsum("q,qc->c", weights, per_query), interactions


def score_diagnostics(
    scores: torch.Tensor,
    interactions: torch.Tensor,
    *,
    selected_chunk: int | None = None,
) -> RoutingDiagnostics:
    """Summarize score uncertainty and late-interaction participation."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector.")
    probabilities = torch.softmax(scores.float(), dim=0)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    entropy = -(probabilities * log_probabilities).sum()
    normalizer = torch.tensor(float(len(scores)), device=scores.device).log().clamp_min(1)
    top = scores.float().topk(min(2, len(scores))).values
    margin = top[0] - top[1] if len(top) > 1 else torch.tensor(float("inf"))
    chunk = int(torch.argmax(scores)) if selected_chunk is None else int(selected_chunk)
    if chunk < 0 or chunk >= len(scores):
        raise IndexError("selected_chunk is outside scores.")
    local = interactions[:, chunk]
    finite = torch.isfinite(local)
    local = local.masked_fill(~finite, float("-inf"))
    winners = local.argmax(dim=-1)
    contributing_queries = int(torch.isfinite(local).any(dim=-1).sum())
    contributing_modes = len(set(winners[torch.isfinite(local).any(dim=-1)].tolist()))
    flat = torch.softmax(local[finite].float(), dim=0) if bool(finite.any()) else torch.ones(1)
    concentration = float(flat.max().item())
    return RoutingDiagnostics(
        entropy=float(entropy.item()),
        normalized_entropy=float((entropy / normalizer).item()),
        top1_margin=float(margin.item()),
        effective_candidates=float(torch.exp(entropy).item()),
        contributing_query_states=float(contributing_queries),
        contributing_memory_modes=float(contributing_modes),
        score_concentration=concentration,
    )


def selection_churn(previous: torch.Tensor, current: torch.Tensor) -> float:
    """Return one minus Jaccard similarity between two selected chunk sets."""
    left = set(int(value) for value in previous.flatten().tolist())
    right = set(int(value) for value in current.flatten().tolist())
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def delayed_commit_index(
    margins: torch.Tensor,
    entropies: torch.Tensor,
    *,
    start: int,
    maximum_delay: int,
    margin_threshold: float | None = None,
    entropy_threshold: float | None = None,
) -> int:
    """Choose the first confidence-qualified causal decision before a deadline."""
    if margins.ndim != 1 or entropies.shape != margins.shape:
        raise ValueError("margins and entropies must be aligned vectors.")
    if start < 0 or start >= len(margins) or maximum_delay < 0:
        raise ValueError("Invalid start or maximum_delay.")
    stop = min(start + int(maximum_delay), len(margins) - 1)
    for index in range(start, stop + 1):
        margin_ok = margin_threshold is None or float(margins[index]) >= margin_threshold
        entropy_ok = entropy_threshold is None or float(entropies[index]) <= entropy_threshold
        if margin_ok and entropy_ok:
            return index
    return stop


def stride_update_mask(token_count: int, stride: int, *, force_last: bool = True) -> torch.Tensor:
    """Mark token positions where a slower-clock router recomputes selection."""
    if token_count <= 0 or stride <= 0:
        raise ValueError("token_count and stride must be positive.")
    mask = torch.zeros(token_count, dtype=torch.bool)
    mask[:: int(stride)] = True
    if force_last:
        mask[-1] = True
    return mask


def interaction_contrast(
    baseline: float,
    temporal_only: float,
    memory_only: float,
    combined: float,
) -> float:
    """Return the 2x2 query-by-memory interaction beyond additive effects."""
    return float(combined - temporal_only - memory_only + baseline)
