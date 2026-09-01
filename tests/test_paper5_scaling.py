from experiments.paper5_scaling_laws.scaling_core import fit_candidate_laws, pareto_frontier, percentile
from experiments.paper5_scaling_laws.run_scaling_study import (
    ScalingConfig,
    build_memory_pool,
    exact_search,
    retrieval_metrics,
)
import pytest
import torch


def test_fit_candidate_laws_recovers_constant_curve():
    fits = fit_candidate_laws([1, 2, 4, 8, 16], [0.8, 0.8, 0.8, 0.8, 0.8])

    constant = next(fit for fit in fits if fit.family == "constant")
    assert constant.rmse == 0.0
    assert constant.parameters["c"] == 0.8


def test_fit_candidate_laws_identifies_power_growth():
    fits = fit_candidate_laws([1, 2, 4, 8, 16], [2, 4, 8, 16, 32])

    power = next(fit for fit in fits if fit.family == "power")
    assert power.r_squared > 0.999999
    assert abs(power.parameters["exponent"] - 1.0) < 1e-8


def test_pareto_frontier_respects_mixed_objective_directions():
    rows = [
        {"name": "slow-good", "quality": 0.9, "latency": 3.0},
        {"name": "fast-good", "quality": 0.9, "latency": 2.0},
        {"name": "fast-poor", "quality": 0.7, "latency": 1.0},
        {"name": "dominated", "quality": 0.6, "latency": 4.0},
    ]

    names = {
        row["name"]
        for row in pareto_frontier(rows, maximize=["quality"], minimize=["latency"])
    }
    assert names == {"fast-good", "fast-poor"}


def test_percentile_validates_and_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_scaling_config_rejects_misaligned_active_budget():
    config = ScalingConfig(active_kv_budgets=(31,))

    with pytest.raises(ValueError, match="active budgets"):
        config.validate()


def test_controlled_exact_search_recovers_planted_working_set():
    pool = build_memory_pool(
        1024,
        dimension=32,
        queries=4,
        evidence_regions=4,
        seed=17,
        device=torch.device("cpu"),
    )

    indices, _, timings = exact_search(pool, 4, repeats=2)
    metrics = retrieval_metrics(indices, pool.evidence)

    assert timings and all(value > 0 for value in timings)
    assert metrics["evidence_recall"] == 1.0
    assert metrics["task_accuracy"] == 1.0


def test_hard_negatives_create_a_nontrivial_retrieval_budget():
    pool = build_memory_pool(
        1024,
        dimension=32,
        queries=4,
        evidence_regions=4,
        seed=17,
        device=torch.device("cpu"),
        hard_negatives=16,
        hard_negative_noise=0.01,
    )

    narrow, _, _ = exact_search(pool, 4, repeats=1)
    wide, _, _ = exact_search(pool, 32, repeats=1)
    assert retrieval_metrics(narrow, pool.evidence)["evidence_recall"] < 1.0
    assert retrieval_metrics(wide, pool.evidence)["evidence_recall"] == 1.0
