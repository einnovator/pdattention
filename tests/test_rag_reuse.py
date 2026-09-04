from __future__ import annotations

import pytest

from pra_hf.rag_reuse import (
    greedy_overlap_sequences,
    longest_common_prefix_length,
    reusable_resource_tokens,
)


def test_longest_common_prefix_length_stops_at_first_change() -> None:
    assert longest_common_prefix_length((1, 2, 3, 4), (1, 2, 8, 4)) == 2
    assert longest_common_prefix_length((1, 2), (1, 2, 3)) == 2


def test_reusable_resource_tokens_counts_only_resident_resources() -> None:
    assert reusable_resource_tokens(
        ("a", "b", "c"), {"a": 10, "b": 20, "c": 30}, {"b", "x"}
    ) == 20


def test_greedy_overlap_sequences_prefers_consecutive_overlap() -> None:
    sequences = greedy_overlap_sequences(
        (("a", "b"), ("b", "c"), ("x",), ("c", "d")),
        sequence_length=3,
        sequence_count=1,
    )
    assert len(sequences) == 1
    selected = sequences[0]
    assert len(selected) == 3
    sets = [set(values) for values in (("a", "b"), ("b", "c"), ("x",), ("c", "d"))]
    assert sets[selected[0]] & sets[selected[1]]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"sequence_length": 1, "sequence_count": 1},
        {"sequence_length": 2, "sequence_count": 0},
    ),
)
def test_greedy_overlap_sequences_rejects_invalid_shape(kwargs) -> None:
    with pytest.raises(ValueError):
        greedy_overlap_sequences((("a",), ("b",)), **kwargs)
