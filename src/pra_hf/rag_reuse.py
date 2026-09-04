"""Deterministic accounting helpers for changing-selection RAG reuse."""

from __future__ import annotations

from typing import Mapping, Sequence


def longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the exact token-prefix overlap used by ordinary APC controls."""

    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def reusable_resource_tokens(
    resource_ids: Sequence[str],
    token_counts: Mapping[str, int],
    resident_resource_ids: set[str],
) -> int:
    """Count selected K/V tokens whose immutable resource is already resident."""

    missing = set(resource_ids) - set(token_counts)
    if missing:
        raise ValueError(f"missing resource token counts: {sorted(missing)}")
    return sum(
        token_counts[resource_id]
        for resource_id in resource_ids
        if resource_id in resident_resource_ids
    )


def greedy_overlap_sequences(
    resource_sets: Sequence[Sequence[str]],
    *,
    sequence_length: int,
    sequence_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Build disjoint query sequences with high consecutive resource overlap.

    This is a workload-construction policy, not a learned selector. Ties use
    original cohort order so the resulting sequences are reproducible.
    """

    if sequence_length < 2 or sequence_count <= 0:
        raise ValueError("reuse sequences require length >= 2 and positive count")
    normalized = tuple(frozenset(values) for values in resource_sets)
    if any(not values for values in normalized):
        raise ValueError("every query must select at least one resource")
    available = set(range(len(normalized)))
    sequences: list[tuple[int, ...]] = []
    while available and len(sequences) < sequence_count:
        start = max(
            available,
            key=lambda index: (
                sum(len(normalized[index] & normalized[other]) for other in available if other != index),
                -index,
            ),
        )
        sequence = [start]
        available.remove(start)
        union = set(normalized[start])
        while available and len(sequence) < sequence_length:
            previous = normalized[sequence[-1]]
            next_index = max(
                available,
                key=lambda index: (
                    len(previous & normalized[index]),
                    len(union & normalized[index]),
                    -index,
                ),
            )
            sequence.append(next_index)
            available.remove(next_index)
            union.update(normalized[next_index])
        if len(sequence) >= 2:
            sequences.append(tuple(sequence))
    return tuple(sequences)
