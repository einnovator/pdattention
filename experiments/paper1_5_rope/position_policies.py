"""Experimental position policies for materializing raw retrieved RoPE keys.

These policies deliberately live outside ``src/pra_torch``. Canonical PRA stores
post-position native keys; the alternatives below are research probes for a
possible raw-key cache whose phase is assigned only after routing.
"""

from __future__ import annotations

import math

import torch


POLICIES = (
    "exact_logical",
    "local_chunk",
    "clipped",
    "log_compressed",
    "bucketed",
    "remote_past",
)


def _translated_chunk(
    source_positions: torch.Tensor,
    *,
    query_position: int,
    nearest_distance: int,
) -> torch.Tensor:
    """Move a complete chunk while retaining every intra-chunk displacement."""
    target_last = int(query_position) - max(int(nearest_distance), 1)
    return source_positions + (target_last - int(source_positions[-1]))


def materialization_positions(
    source_positions: torch.Tensor,
    query_position: int,
    policy: str,
    *,
    distance_limit: int,
) -> torch.Tensor:
    """Assign positions to one historical chunk selected for a future query.

    ``source_positions`` is an ordered ``[M]`` integer tensor and
    ``query_position`` is the first direct-query position. Approximate policies
    shift the chunk as a unit, so token order and exact local spacing survive.
    ``distance_limit`` defines the near-history horizon used by compression.
    """
    if source_positions.ndim != 1 or not source_positions.numel():
        raise ValueError("source_positions must be a non-empty [memory_tokens] tensor.")
    if source_positions.dtype == torch.bool or source_positions.is_floating_point():
        raise TypeError("source_positions must contain integer logical positions.")
    if int(distance_limit) <= 0:
        raise ValueError("distance_limit must be positive.")
    if policy not in POLICIES:
        raise ValueError(f"Unsupported materialization position policy: {policy}")
    ordered = source_positions[1:] > source_positions[:-1]
    if source_positions.numel() > 1 and not bool(ordered.all()):
        raise ValueError("source_positions must be strictly increasing.")

    source_positions = source_positions.to(dtype=torch.long)
    query_position = int(query_position)
    nearest_distance = query_position - int(source_positions[-1])
    if nearest_distance <= 0:
        raise ValueError("Retrieved memory must be strictly historical to the query.")
    if policy == "exact_logical":
        return source_positions.clone()
    if policy == "local_chunk":
        return _translated_chunk(
            source_positions,
            query_position=query_position,
            nearest_distance=1,
        )
    if policy == "clipped":
        effective = min(nearest_distance, int(distance_limit))
    elif policy == "log_compressed":
        if nearest_distance <= distance_limit:
            effective = nearest_distance
        else:
            excess_ratio = (nearest_distance - distance_limit) / distance_limit
            effective = distance_limit + round(
                distance_limit * math.log2(1.0 + excess_ratio)
            )
    elif policy == "bucketed":
        boundaries = (
            int(distance_limit),
            4 * int(distance_limit),
            16 * int(distance_limit),
            64 * int(distance_limit),
        )
        effective = next(
            (boundary for boundary in boundaries if nearest_distance <= boundary),
            boundaries[-1],
        )
    else:  # remote_past
        effective = (
            nearest_distance if nearest_distance <= distance_limit else 4 * distance_limit
        )
    return _translated_chunk(
        source_positions,
        query_position=query_position,
        nearest_distance=effective,
    )
