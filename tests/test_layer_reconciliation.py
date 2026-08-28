"""Offline checks for corrected Paper 3 layer-profile reconciliation."""

from experiments.paper3_kv_materialization.run_layer_reconciliation import _schedules
from experiments.paper3_kv_materialization.summarize_layer_reconciliation import (
    _pareto,
    _recommendations,
)


def _row(dataset: str, profile: str, quality: float, cost: float) -> dict:
    return {
        "dataset": dataset,
        "profile": profile,
        "gold_mean_logprob_delta_vs_none_mean": quality,
        "native_kv_token_states_mean": cost,
    }


def test_reconciliation_schedules_separate_late_and_sparse_placement():
    schedules = _schedules(28)

    assert schedules["last_20"] == tuple(range(8, 28))
    assert schedules["last_8"] == tuple(range(20, 28))
    assert schedules["early_4"] == (0, 1, 2, 3)
    assert schedules["middle_4"] == (12, 13, 14, 15)
    assert schedules["even_4"] == (0, 9, 18, 27)
    assert schedules["even_8"] == (0, 4, 8, 12, 15, 19, 23, 27)


def test_recommendations_keep_quality_balanced_and_economy_objectives_distinct():
    rows = [
        _row("fixture", "all", 0.8, 280),
        _row("fixture", "last_20", 2.0, 200),
        _row("fixture", "last_8", 1.5, 80),
        _row("fixture", "last_4", 0.5, 40),
        _row("fixture", "last_1", 0.01, 10),
    ]

    result = _recommendations(rows)["per_dataset"]["fixture"]

    assert result["quality_max"] == "last_20"
    assert result["balanced"] == "last_8"
    assert result["economy"] == "last_1"
    assert result["reference_correctness"].startswith("all")


def test_pareto_marks_only_quality_cost_nondominated_profiles():
    rows = [
        _row("fixture", "a", 1.0, 10),
        _row("fixture", "b", 0.5, 20),
        _row("fixture", "c", 2.0, 30),
    ]

    result = {row["profile"]: row["pareto"] for row in _pareto(rows)}

    assert result == {"a": True, "b": False, "c": True}
