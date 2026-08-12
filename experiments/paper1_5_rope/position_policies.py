"""Experimental position policies for materializing raw retrieved RoPE keys.

These policies deliberately live outside ``src/pra_torch``. Canonical PRA stores
post-position native keys; the alternatives below are research probes for a
possible raw-key cache whose phase is assigned only after routing.
"""

from __future__ import annotations

import torch

from pra_torch.positions.retrieval import assign_retrieval_positions


POLICIES = (
    "exact_logical",
    "local_chunk",
    "clipped",
    "log_compressed",
    "bucketed",
    "remote_past",
)


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
    if int(distance_limit) <= 0:
        raise ValueError("distance_limit must be positive.")
    if policy not in POLICIES:
        raise ValueError(f"Unsupported materialization position policy: {policy}")
    mapped_policy = {
        "exact_logical": "exact",
        "local_chunk": "local",
        "remote_past": "fixed",
    }.get(policy, policy)
    mapped_distance = (
        4 * int(distance_limit) if policy == "remote_past" else int(distance_limit)
    )
    return assign_retrieval_positions(
        source_positions,
        query_position,
        mapped_policy,
        distance=mapped_distance,
    )
