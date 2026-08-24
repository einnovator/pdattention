from types import SimpleNamespace

import torch

from experiments.paper2_5_iterative_pra.precompute_native_qk_features import (
    _default_offset,
)
from experiments.paper2_8_qk_compression.run_confirmation import (
    _oracle_indices,
    _semantic_unit,
    _stable_random_indices,
)
from experiments.paper2_8_qk_compression.run_confirmation_generation import (
    _fixed_selection,
    _selected_lookup,
)


def test_confirmation_split_starts_after_frozen_validation_and_test() -> None:
    assert _default_offset("validation") == 0
    assert _default_offset("test") == 8
    assert _default_offset("confirmation") == 24


def test_oracle_selection_fills_exact_budget_without_duplicates() -> None:
    positives = torch.tensor([False, True, False, True, False])
    teacher = torch.tensor([0.1, 0.2, 0.5, 0.4, 0.3])
    selected = _oracle_indices(positives, teacher, 4)
    assert selected == [1, 2, 3, 4]
    assert len(selected) == len(set(selected)) == 4


def test_random_control_is_stable_and_query_independent() -> None:
    first = _stable_random_indices("example-a", 20, 4)
    assert first == _stable_random_indices("example-a", 20, 4)
    assert len(first) == len(set(first)) == 4


def test_semantic_normalization_is_bounded() -> None:
    values = _semantic_unit(torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(values, torch.tensor([-1.0, 0.0, 1.0]))


def test_selection_lookup_uses_only_frozen_ensemble_rows() -> None:
    rows = [
        {
            "dataset": "qasper",
            "example_id": "x",
            "condition": "lowrank_r16_ensemble",
            "seed": "-1",
            "selected_chunks": "4 2 7 1",
        },
        {
            "dataset": "qasper",
            "example_id": "x",
            "condition": "lowrank_r16",
            "seed": "11",
            "selected_chunks": "0 1 2 3",
        },
    ]
    assert _selected_lookup(rows) == {
        ("qasper", "x", "lowrank_r16_ensemble"): [4, 2, 7, 1]
    }


def test_native_kv_replay_preserves_chunk_identity_and_order() -> None:
    chunks = [
        SimpleNamespace(chunk_id=f"c{i}", logical_start=32 * i, logical_end=32 * (i + 1))
        for i in range(5)
    ]
    entry = SimpleNamespace(
        uri="benchmark://x",
        layer_memory={27: SimpleNamespace(chunks=chunks)},
    )
    fixed = _fixed_selection(
        entry,
        27,
        [3, 1, 4, 0],
        [(32 * i, 32 * (i + 1)) for i in range(5)],
    )
    assert [hit.chunk_id for hit in fixed[0]] == ["c3", "c1", "c4", "c0"]
    assert [hit.rank_within_reference for hit in fixed[0]] == [1, 2, 3, 4]
