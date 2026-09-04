import pytest

from experiments.paper3_2_rag.aggregate_multiseed import (
    _aggregate_composition,
    _aggregate_nonprefix,
)


def test_composition_aggregate_weights_metrics_and_sums_matches() -> None:
    manifests = [
        {
            "seed": 11,
            "summary": {
                "conditions": [
                    {
                        "condition": "NATIVE_GLOBAL_REBOUND",
                        "resource_order_name": "canonical",
                        "examples": 2,
                        "token_f1": 0.25,
                    }
                ],
                "fresh_packed_comparisons": {
                    "NATIVE_GLOBAL_REBOUND": {
                        "pairs": 2,
                        "output_matches": 1,
                        "first_step_logit_hash_matches": 0,
                        "first_step_js_divergence_mean": 0.2,
                        "gold_nll_mean_abs_delta": 0.3,
                    }
                },
                "fresh_packed_order_sensitivity": {
                    "example": {"orders": 2, "unique_outputs": 2}
                },
            },
        },
        {
            "seed": 23,
            "summary": {
                "conditions": [
                    {
                        "condition": "NATIVE_GLOBAL_REBOUND",
                        "resource_order_name": "canonical",
                        "examples": 6,
                        "token_f1": 0.75,
                    }
                ],
                "fresh_packed_comparisons": {
                    "NATIVE_GLOBAL_REBOUND": {
                        "pairs": 6,
                        "output_matches": 3,
                        "first_step_logit_hash_matches": 1,
                        "first_step_js_divergence_mean": 0.4,
                        "gold_nll_mean_abs_delta": 0.5,
                    }
                },
                "fresh_packed_order_sensitivity": {
                    "example": {"orders": 2, "unique_outputs": 1}
                },
            },
        },
    ]

    result = _aggregate_composition(manifests)

    assert result["conditions"] == [
        {
            "condition": "NATIVE_GLOBAL_REBOUND",
            "resource_order_name": "canonical",
            "examples": 8,
            "token_f1": 0.625,
        }
    ]
    comparison = result["fresh_packed_comparisons"]["NATIVE_GLOBAL_REBOUND"]
    assert comparison["pairs"] == 8
    assert comparison["output_matches"] == 4
    assert comparison["first_step_logit_hash_matches"] == 1
    assert comparison["first_step_js_divergence_mean"] == pytest.approx(0.35)
    assert set(result["fresh_packed_order_sensitivity"]) == {
        "seed11:example",
        "seed23:example",
    }


def test_nonprefix_aggregate_keeps_seed_on_sequence_rows() -> None:
    manifests = [
        {
            "seed": 11,
            "summary": {
                "conditions": [
                    {
                        "condition": "PRA_GLOBAL_REBOUND",
                        "turns": 2,
                        "exact_output_parity_with_fresh": 0.5,
                        "newly_encoded_tokens": 20.0,
                        "reused_tokens": 10.0,
                        "token_f1": 0.1,
                        "total_with_materialization_ms": 100.0,
                    }
                ],
                "sequence_cumulative": [{"sequence_id": 1, "turns": 2}],
            },
        },
        {
            "seed": 23,
            "summary": {
                "conditions": [
                    {
                        "condition": "PRA_GLOBAL_REBOUND",
                        "turns": 6,
                        "exact_output_parity_with_fresh": 0.25,
                        "newly_encoded_tokens": 40.0,
                        "reused_tokens": 30.0,
                        "token_f1": 0.2,
                        "total_with_materialization_ms": 200.0,
                    }
                ],
                "sequence_cumulative": [{"sequence_id": 1, "turns": 6}],
            },
        },
    ]

    result = _aggregate_nonprefix(manifests)

    assert result["conditions"][0]["turns"] == 8
    assert result["conditions"][0]["reused_tokens"] == 25.0
    assert [row["seed"] for row in result["sequence_cumulative"]] == [11, 23]
