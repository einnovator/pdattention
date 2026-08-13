import copy

import pytest

from experiments.paper2_hf.score_behavioral_judge_results import score_response


def _truth():
    return {
        "schema_version": "1.0",
        "items": [
            {
                "item_id": "judge_1_ab",
                "pair_group_id": "pair_1",
                "source_example_id": "example_1",
                "dataset": "fixture",
                "comparison_group": "native_no_context_vs_pra",
                "condition_a": "native_no_context",
                "condition_b": "pra_routed_frozen",
            },
            {
                "item_id": "judge_1_ba",
                "pair_group_id": "pair_1",
                "source_example_id": "example_1",
                "dataset": "fixture",
                "comparison_group": "native_no_context_vs_pra",
                "condition_a": "pra_routed_frozen",
                "condition_b": "native_no_context",
            },
        ],
    }


def _response():
    common = {"semantic_equivalence": 80, "confidence": 90, "reason": "Materially similar."}
    return {
        "schema_version": "1.0",
        "judge_name": "fixture-judge",
        "items": [
            {
                "item_id": "judge_1_ab",
                "relative_quality": 20,
                "validity_a": 60,
                "validity_b": 80,
                **common,
            },
            {
                "item_id": "judge_1_ba",
                "relative_quality": -20,
                "validity_a": 80,
                "validity_b": 60,
                **common,
            },
        ],
    }


def test_scoring_orients_conditions_and_collapses_order_reversal():
    result = score_response(_response(), _truth())
    aggregate = result["aggregates"][0]

    assert result["presentation_count"] == 2
    assert result["underlying_pair_count"] == 1
    assert aggregate["semantic_equivalence_mean"] == 80
    assert aggregate["relative_quality_target_mean"] == 20
    assert aggregate["target_validity_mean"] == 80
    assert aggregate["comparator_validity_mean"] == 60
    assert result["order_reversal"]["relative_quality_target_absolute_difference"] == 0


def test_scoring_rejects_missing_or_invalid_responses():
    missing = _response()
    missing["items"].pop()
    with pytest.raises(ValueError, match="IDs do not match"):
        score_response(missing, _truth())

    invalid = copy.deepcopy(_response())
    invalid["items"][0]["confidence"] = 101
    with pytest.raises(ValueError, match="out-of-range confidence"):
        score_response(invalid, _truth())
