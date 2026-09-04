from __future__ import annotations

from experiments.paper3_2_rag.calibrate_composition_policy import (
    evaluate_policy,
    fit_policy,
)


def _row(condition: str, js: float, fraction: float | None, prediction: str) -> dict:
    return {
        "example_id": "example-1",
        "resource_order_name": "canonical",
        "condition": condition,
        "question_type": "comparison",
        "physical_native_tokens": 100,
        "resource_order": ["a", "b"],
        "repair_fraction": fraction,
        "first_step_js_divergence": js,
        "prediction": prediction,
        "token_f1": 1.0 if prediction == "gold" else 0.0,
        "gold_answer_mean_nll": js,
        "repaired_token_count": int(100 * (fraction or 0.0)),
    }


def test_policy_is_fit_on_calibration_outcomes_and_frozen_for_evaluation() -> None:
    calibration = [
        _row("NATIVE_GLOBAL_REBOUND", 0.2, None, "wrong"),
        _row("REPAIR_PREFIX_0.25", 0.01, 0.25, "gold"),
    ]
    policy = fit_policy(calibration, cost_weight=0.05, max_repair_fraction=0.5)
    assert policy["global_action"] == "REPAIR_PREFIX_0.25"

    evaluation = [
        _row("FRESH_PACKED", 0.0, None, "gold"),
        _row("NATIVE_GLOBAL_REBOUND", 0.1, None, "wrong"),
        _row("REPAIR_PREFIX_0.25", 0.02, 0.25, "gold"),
    ]
    result = evaluate_policy(evaluation, policy)
    assert result["fresh_packed_reference"]["mean_first_step_js"] == 0.0
    assert result["query_conditioned_policy"]["exact_output_recovery"] == 1.0
    assert result["query_conditioned_policy"]["mean_repair_fraction"] == 0.25
