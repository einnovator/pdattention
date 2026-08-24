from experiments.paper2_9_look_ahead_back.run_temporal_study import select_policies
from experiments.paper2_9_look_ahead_back.precompute_temporal_queries import (
    identity_shard_name,
)


def _row(split, condition, reducer, window, recall):
    return {
        "dataset": "qasper",
        "split": split,
        "condition": condition,
        "memory": "rank16",
        "layer": 27,
        "reducer": reducer,
        "look_behind": window,
        "evidence_recall": recall,
        "evidence_precision": recall,
        "any_evidence": recall,
        "chain_completion": recall,
        "mrr": recall,
        "normalized_entropy": 0.5,
        "top1_margin": 0.5,
        "score_concentration": 0.5,
    }


def test_temporal_policy_is_selected_from_validation_only():
    rows = [
        _row("validation", "rank16_l27_mean_b2", "mean", 2, 0.8),
        _row("validation", "rank16_l27_late_max_b4", "late_max", 4, 0.7),
        _row("test", "rank16_l27_mean_b2", "mean", 2, 0.0),
        _row("test", "rank16_l27_late_max_b4", "late_max", 4, 1.0),
    ]
    selected = select_policies(rows)["qasper"]
    assert selected["condition"] == "rank16_l27_mean_b2"
    assert selected["validation_evidence_recall"] == 0.8


def test_identity_shard_names_are_portable_and_stable():
    name = identity_shard_name("1603.01514:111afb77")
    assert name == identity_shard_name("1603.01514:111afb77")
    assert name.endswith(".pt")
    assert ":" not in name
