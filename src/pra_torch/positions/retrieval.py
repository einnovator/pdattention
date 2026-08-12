"""Position assignment policies for already-selected retrieved memory chunks."""

from __future__ import annotations

import math

import torch


RETRIEVAL_POSITION_POLICIES = (
    "exact",
    "local",
    "fixed",
    "clipped",
    "log_compressed",
    "bucketed",
)


def _translate_to_nearest_distance(
    source_positions: torch.Tensor,
    *,
    query_position: int,
    nearest_distance: int,
) -> torch.Tensor:
    """Translate a chunk so its last token is ``nearest_distance`` behind Q."""
    target_last = int(query_position) - max(int(nearest_distance), 1)
    return source_positions + (target_last - int(source_positions[-1]))


def assign_retrieval_positions(
    source_positions: torch.Tensor,
    query_position: int,
    policy: str,
    *,
    distance: int | None = None,
) -> torch.Tensor:
    """Place one historical chunk while retaining all of its internal spacing.

    ``source_positions`` is an ordered integer tensor ``[M]``. Distances use a
    nearest-token convention: ``D = query_position - assigned_positions[-1]``.
    This convention keeps every assigned memory token strictly before the query,
    including when the requested distance is smaller than the chunk length.
    """
    if source_positions.ndim != 1 or not source_positions.numel():
        raise ValueError("source_positions must be a non-empty [memory_tokens] tensor.")
    if source_positions.dtype == torch.bool or source_positions.is_floating_point():
        raise TypeError("source_positions must contain integer logical positions.")
    if policy not in RETRIEVAL_POSITION_POLICIES:
        raise ValueError(f"Unsupported retrieval position policy: {policy}")
    if source_positions.numel() > 1 and not bool(
        (source_positions[1:] > source_positions[:-1]).all()
    ):
        raise ValueError("source_positions must be strictly increasing.")

    source_positions = source_positions.to(dtype=torch.long)
    query_position = int(query_position)
    exact_distance = query_position - int(source_positions[-1])
    if policy == "exact":
        if exact_distance <= 0:
            raise ValueError("Exact retrieved memory must be historical to the query.")
        return source_positions.clone()
    if policy == "local":
        effective_distance = 1
    else:
        if distance is None or int(distance) <= 0:
            raise ValueError(f"{policy} retrieval positioning requires positive distance.")
        limit = int(distance)
        if policy == "fixed":
            effective_distance = limit
        else:
            if exact_distance <= 0:
                raise ValueError("Compressed retrieved memory must be historical to the query.")
            if policy == "clipped":
                effective_distance = min(exact_distance, limit)
            elif policy == "log_compressed":
                if exact_distance <= limit:
                    effective_distance = exact_distance
                else:
                    ratio = (exact_distance - limit) / limit
                    effective_distance = limit + round(limit * math.log2(1.0 + ratio))
            else:  # bucketed
                boundaries = (limit, 4 * limit, 16 * limit, 64 * limit)
                effective_distance = next(
                    (boundary for boundary in boundaries if exact_distance <= boundary),
                    boundaries[-1],
                )
    return _translate_to_nearest_distance(
        source_positions,
        query_position=query_position,
        nearest_distance=effective_distance,
    )
