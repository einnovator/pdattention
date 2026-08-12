import pytest

from experiments.paper2_hf.qa.run_overnight_lora_sweep import (
    SweepConfig,
    choose_pareto_winner,
    select_stage_c_configs,
    stage_a_configs,
    stage_b_configs,
    top_stage_a_ranks,
    validation_metrics,
)


def _record(config_id, rank, score, parameters=None):
    return {
        "config_id": config_id,
        "rank": rank,
        "steps": 64,
        "step_multiplier": 2.0,
        "learning_rate": 1e-3,
        "learning_rate_multiplier": 1.0,
        "memory_use_parameters": parameters or rank * 100,
        "combined_oracle_delta_logp": score,
    }


def test_stages_cover_only_predeclared_bounded_grid():
    stage_a = stage_a_configs()
    assert len(stage_a) == 8
    assert {config.rank for config in stage_a} == {4, 8, 16, 32}
    assert {config.steps for config in stage_a} == {32, 64}
    assert {config.learning_rate_multiplier for config in stage_a} == {1.0}

    stage_b = stage_b_configs((8, 16, 32))
    assert len(stage_b) == 18
    assert {config.steps for config in stage_b} == {64, 128}
    assert {config.learning_rate_multiplier for config in stage_b} == {0.5, 1.0, 2.0}
    assert all(config.rank <= 32 for config in (*stage_a, *stage_b))


def test_validation_metric_equal_weights_datasets_and_keeps_routed_secondary():
    rows = [
        {
            "dataset": dataset,
            "condition": condition,
            "gold_sequence_logprob_delta_vs_none_mean": value,
        }
        for dataset, condition, value in (
            ("hotpotqa", "oracle", 2.0),
            ("qasper", "oracle", 4.0),
            ("hotpotqa", "routed", 1.0),
            ("qasper", "routed", -1.0),
        )
    ]
    metrics = validation_metrics(rows)
    assert metrics["combined_oracle_delta_logp"] == 3.0
    assert metrics["combined_routed_delta_logp"] == 0.0


def test_stage_selection_is_validation_only_rank_diverse_and_pareto_bounded():
    records = [
        _record("r4", 4, 4.70, 400),
        _record("r8", 8, 4.95, 800),
        _record("r16", 16, 5.00, 1600),
        _record("r32", 32, 5.03, 3200),
    ]
    assert top_stage_a_ranks(records) == [32, 16, 8]
    assert select_stage_c_configs(records) == ["r32", "r16", "r8"]
    assert choose_pareto_winner(records, tolerance=0.10)["config_id"] == "r8"
    assert choose_pareto_winner(records, tolerance=0.02)["config_id"] == "r32"


def test_sweep_config_ids_separate_checkpoint_contracts():
    baseline = SweepConfig(8, 32, 1e-3, "A")
    longer = SweepConfig(8, 128, 1e-3, "B")
    lower_lr = SweepConfig(8, 128, 5e-4, "B")
    assert len({baseline.config_id, longer.config_id, lower_lr.config_id}) == 3
    assert baseline.variant.lora_rank == 8
    assert baseline.variant.residual_width == 0


def test_validation_metric_requires_both_datasets():
    with pytest.raises(ValueError, match="Both validation datasets"):
        validation_metrics(
            [
                {
                    "dataset": "hotpotqa",
                    "condition": "oracle",
                    "gold_sequence_logprob_delta_vs_none_mean": 1.0,
                }
            ]
        )
