import pytest

from experiments.paper2_hf.summarize_overnight_lora_sweep import (
    paired_vs_frozen,
    validation_grid_tex,
)


def test_paired_summary_uses_fixed_test_control_and_all_seeds():
    seeds = [11, 23, 37, 53, 71]
    rows = []
    for dataset in ("hotpotqa", "qasper"):
        for condition in ("oracle", "routed"):
            rows.append(
                {
                    "seed": 0,
                    "variant": "fixed",
                    "dataset": dataset,
                    "condition": condition,
                    "gold_sequence_logprob_delta_vs_none": 1.0,
                }
            )
            for seed in seeds:
                rows.append(
                    {
                        "seed": seed,
                        "variant": "lora",
                        "dataset": dataset,
                        "condition": condition,
                        "gold_sequence_logprob_delta_vs_none": 1.0 + seed / 100.0,
                    }
                )
                rows.append(
                    {
                        "seed": seed,
                        "variant": "combo_lora",
                        "dataset": dataset,
                        "condition": condition,
                        "gold_sequence_logprob_delta_vs_none": 0.5,
                    }
                )
    artifact = {
        "manifest": {"seeds": seeds},
        "stage_c_finalists": ["lora"],
        "test_seed_aggregates": rows,
    }
    paired = paired_vs_frozen(artifact)

    assert len(paired) == 8
    lora = next(
        row
        for row in paired
        if row["variant"] == "lora"
        and row["dataset"] == "hotpotqa"
        and row["condition"] == "oracle"
    )
    assert lora["paired_differences"] == pytest.approx([0.11, 0.23, 0.37, 0.53, 0.71])
    assert lora["same_direction"] is True
    combo = next(row for row in paired if row["variant"] == "combo_lora")
    assert combo["mean_delta_logp_difference"] == -0.5


def test_validation_grid_tex_contains_every_screened_configuration():
    artifact = {
        "validation_ranking": [
            {
                "stage": "A",
                "rank": 4,
                "steps": 32,
                "learning_rate": 1e-3,
                "hotpotqa_oracle_delta_logp": 2.0,
                "qasper_oracle_delta_logp": 1.0,
                "combined_oracle_delta_logp": 1.5,
            },
            {
                "stage": "B",
                "rank": 8,
                "steps": 128,
                "learning_rate": 2e-3,
                "hotpotqa_oracle_delta_logp": 3.0,
                "qasper_oracle_delta_logp": 2.0,
                "combined_oracle_delta_logp": 2.5,
            },
        ]
    }

    rendered = validation_grid_tex(artifact)

    assert "A & 4 & 32 & 1.0e-03" in rendered
    assert "B & 8 & 128 & 2.0e-03" in rendered
    assert rendered.count(" \\\\") == 3
