import torch

from experiments.paper2_8_qk_compression.run_multidataset_extension import (
    _balanced,
    _mean_seed_selection_jaccard,
    _secondary_budget,
    _select_validation_policy,
)


def _feature(example_id: str, positive_chunks: int):
    mask = torch.zeros(12, dtype=torch.bool)
    mask[:positive_chunks] = True
    return {"example_id": example_id, "local_positive_mask": mask}


def test_secondary_budget_uses_validation_ninetieth_percentile_and_cap():
    mostly_six = [_feature(f"e{index}", 6) for index in range(9)] + [_feature("e9", 12)]
    assert _secondary_budget(mostly_six) == 6
    mostly_twelve = [_feature(f"x{index}", 12) for index in range(10)]
    assert _secondary_budget(mostly_twelve) == 8


def test_balanced_mixed_projection_cycles_smaller_dataset_cohorts():
    groups = {
        "a": [_feature("a0", 1), _feature("a1", 1)],
        "b": [_feature(f"b{index}", 1) for index in range(5)],
    }
    balanced = _balanced(groups)
    assert len(balanced) == 10
    assert sum(row["example_id"].startswith("a") for row in balanced) == 5
    assert sum(row["example_id"].startswith("b") for row in balanced) == 5


def test_seed_selection_stability_averages_identity_paired_jaccard():
    rows = [
        {"example_id": "a", "selected_chunks": "0 1"},
        {"example_id": "a", "selected_chunks": "0 2"},
        {"example_id": "b", "selected_chunks": "3 4"},
        {"example_id": "b", "selected_chunks": "3 4"},
    ]
    assert _mean_seed_selection_jaccard(rows) == (1 / 3 + 1) / 2


def test_validation_policy_compares_python_chunk_identities():
    feature = {
        "example_id": "example",
        "local_positive_mask": torch.tensor([False, True]),
    }
    evidence = torch.tensor([0.0, 1.0])
    distractor = torch.tensor([1.0, 0.0])
    scores = {
        ("exact", -1): (evidence, 0, 0.0, "control"),
        ("bm25", -1): (distractor, 0, 0.0, "control"),
        ("approximate", -1): (distractor, 0, 0.0, "control"),
        ("inherited_hybrid", -1): (distractor, 0, 0.0, "control"),
    }
    for regime in ("zero_shot", "retrained", "mixed"):
        scores[(f"rank16_{regime}", -1)] = (evidence, 0, 0.0, "ensemble")
    policies = _select_validation_policy(
        {"example": (scores, distractor)},
        {"dataset": [feature]},
        {"dataset": 1},
    )
    assert policies["dataset"]["best_lexical"] == "exact"
    assert policies["dataset"]["fusion"]["retrained"]["validation_recall"] == 1.0
