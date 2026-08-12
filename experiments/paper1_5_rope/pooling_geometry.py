"""Pooling constructions and metrics for the controlled RoPE geometry study."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from pra_torch.positions import RotaryPositionEncoding


def contiguous_subspans(token_count: int, gist_count: int) -> tuple[tuple[int, int], ...]:
    """Partition ``token_count`` tokens into balanced non-empty contiguous spans."""

    if token_count <= 0 or gist_count <= 0:
        raise ValueError("token_count and gist_count must be positive")
    if gist_count > token_count:
        raise ValueError("gist_count cannot exceed token_count")
    width, remainder = divmod(int(token_count), int(gist_count))
    spans = []
    start = 0
    for index in range(gist_count):
        end = start + width + int(index < remainder)
        spans.append((start, end))
        start = end
    return tuple(spans)


def subspan_centers(
    source_positions: torch.Tensor,
    gist_count: int,
) -> torch.Tensor:
    """Return exact inclusive centers for contiguous source-position subspans."""

    if source_positions.ndim != 1:
        raise ValueError("source_positions must have shape [T]")
    spans = contiguous_subspans(source_positions.numel(), gist_count)
    return torch.stack(
        [
            (source_positions[start].float() + source_positions[end - 1].float()) / 2.0
            for start, end in spans
        ]
    )


def pre_rope_mean(raw_key: torch.Tensor) -> torch.Tensor:
    """Average token keys ``[B,H,T,Dh]`` into one position-neutral gist."""

    if raw_key.ndim != 4 or raw_key.shape[2] == 0:
        raise ValueError("raw_key must have shape [B,H,T,Dh] with T > 0")
    return raw_key.mean(dim=2, keepdim=True)


def post_rope_mean(positioned_key: torch.Tensor) -> torch.Tensor:
    """Average exact token-level post-RoPE keys into one mixed-phase gist."""

    if positioned_key.ndim != 4 or positioned_key.shape[2] == 0:
        raise ValueError("positioned_key must have shape [B,H,T,Dh] with T > 0")
    return positioned_key.mean(dim=2, keepdim=True)


def centered_rope_subgists(
    raw_key: torch.Tensor,
    source_positions: torch.Tensor,
    gist_count: int,
    rope: RotaryPositionEncoding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean pre-RoPE keys per subspan and rotate once at each exact center.

    Returns centered gists ``[B,H,G,Dh]`` and fractional centers ``[G]``.
    Token-level native keys are not modified or reused as gist storage.
    """

    if raw_key.ndim != 4 or raw_key.shape[2] != source_positions.numel():
        raise ValueError("raw_key [B,H,T,Dh] must align with source_positions [T]")
    effective_count = min(int(gist_count), raw_key.shape[2])
    spans = contiguous_subspans(raw_key.shape[2], effective_count)
    centers = subspan_centers(source_positions.to(raw_key.device), effective_count)
    means = torch.cat(
        [raw_key[:, :, start:end, :].mean(dim=2, keepdim=True) for start, end in spans],
        dim=2,
    )
    return rope.apply_rotary(means, centers), centers


def qk_gist_score(query: torch.Tensor, gists: torch.Tensor) -> torch.Tensor:
    """Reduce native-style QK logits to one score per batch item.

    ``query`` is ``[B,H,1,Dh]`` and ``gists`` is ``[B,H,G,Dh]``. Heads are
    averaged first; the best separately addressable gist supplies the chunk score.
    """

    if query.ndim != 4 or gists.ndim != 4 or query.shape[2] != 1:
        raise ValueError("Expected query [B,H,1,Dh] and gists [B,H,G,Dh]")
    if query.shape[:2] != gists.shape[:2] or query.shape[-1] != gists.shape[-1]:
        raise ValueError("query and gists must share batch, head, and head-width dimensions")
    logits = torch.einsum("bhqd,bhgd->bhqg", query, gists) / math.sqrt(query.shape[-1])
    return logits[:, :, 0, :].mean(dim=1).max(dim=-1).values


def native_token_chunk_score(query: torch.Tensor, positioned_key: torch.Tensor) -> torch.Tensor:
    """Return the maximum exact token-QK score after averaging attention heads."""

    return qk_gist_score(query, positioned_key)


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute deterministic population Pearson correlation, returning zero if constant."""

    if len(left) != len(right) or not left:
        raise ValueError("Correlation inputs must be non-empty and equally sized")
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(right, dtype=torch.float64)
    left_centered = left_tensor - left_tensor.mean()
    right_centered = right_tensor - right_tensor.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denominator) == 0.0:
        return 0.0
    return float((left_centered @ right_centered) / denominator)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(float(value) for value in values), key=lambda row: (row[1], row[0]))
    ranks = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = 0.5 * ((cursor + 1) + end)
        for original_index, _ in ordered[cursor:end]:
            ranks[original_index] = average
        cursor = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman correlation using average ranks for exact ties."""

    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def topk_overlap(left: Sequence[float], right: Sequence[float], k: int) -> float:
    """Return overlap divided by ``k`` for deterministic descending top-k sets."""

    if len(left) != len(right) or not left or k <= 0:
        raise ValueError("top-k inputs must be non-empty, equally sized, and use k > 0")
    count = min(int(k), len(left))
    left_ids = set(sorted(range(len(left)), key=lambda i: (-float(left[i]), i))[:count])
    right_ids = set(sorted(range(len(right)), key=lambda i: (-float(right[i]), i))[:count])
    return len(left_ids & right_ids) / count
