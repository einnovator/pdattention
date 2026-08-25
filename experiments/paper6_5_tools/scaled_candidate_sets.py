"""Tensorized candidate ordering for the Paper 6.5 scaling experiment."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def stable_descending_order(scores: torch.Tensor) -> tuple[int, ...]:
    """Rank one score vector, preserving input/URI order for exact ties."""

    if scores.ndim != 1:
        raise ValueError("stable_descending_order expects a one-dimensional tensor.")
    return tuple(int(value) for value in torch.argsort(scores, descending=True, stable=True).tolist())


def candidate_orders(
    channel_scores: Mapping[str, torch.Tensor],
    *,
    max_candidates: int,
) -> dict[str, tuple[int, ...]]:
    """Return exact fused/raw/diversity prefixes without sorting full URI maps repeatedly.

    Input rows must already follow stable URI order. The result matches
    ``discover_candidate_set`` for custom channels and ``allow_unsafe=True``.
    """

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")
    if not channel_scores:
        return {"fused_score": (), "raw_union": (), "diversity_union": ()}
    if any(scores.ndim != 1 for scores in channel_scores.values()):
        raise ValueError("All channel score tensors must be one-dimensional and aligned.")
    lengths = {int(scores.numel()) for scores in channel_scores.values()}
    if len(lengths) != 1:
        raise ValueError("All channel score tensors must be one-dimensional and aligned.")
    count = next(iter(lengths))
    limit = min(max_candidates, count)
    rankings = {name: stable_descending_order(scores) for name, scores in channel_scores.items()}

    # Runtime fusion uses Python floats. Float64 reproduces its tie behavior
    # while retaining a vectorized implementation for large catalogs.
    fused = torch.zeros(count, dtype=torch.float64)
    for scores in channel_scores.values():
        values = scores.double()
        low = float(values.min())
        high = float(values.max())
        fused += (values - low) / (high - low) if high > low else (values > 0).float()
    fused_order = stable_descending_order(fused)[:limit]

    pool: dict[int, tuple[int, float, str]] = {}
    for channel, ranking in rankings.items():
        scores = channel_scores[channel]
        for rank, index in enumerate(ranking[:limit], start=1):
            candidate = (rank, -float(scores[index]), channel)
            previous = pool.get(index)
            if previous is None or candidate < previous:
                pool[index] = candidate
    raw_order = tuple(
        index for index, _ in sorted(pool.items(), key=lambda row: (row[1], row[0]))[:limit]
    )

    diversity = []
    seen = set()
    depth = 0
    while len(diversity) < limit and depth < count:
        for ranking in rankings.values():
            index = ranking[depth]
            if index not in seen:
                diversity.append(index)
                seen.add(index)
                if len(diversity) == limit:
                    break
        depth += 1
    return {
        "fused_score": fused_order,
        "raw_union": raw_order,
        "diversity_union": tuple(diversity),
    }
