from __future__ import annotations

import pytest
import torch

from pra_hf import AdaptiveRetryAgent as PublicAdaptiveRetryAgent
from pra_hf import EffortProfile as PublicEffortProfile

from pra_hf.adaptive_runtime import (
    AdaptiveRetryAgent,
    AttemptResult,
    ControllerFeatures,
    HandRuleController,
    LinearEffortController,
    StopPolicy,
    calibration_metrics,
    default_effort_profiles,
    semantic_consistency,
    token_entropy,
    validate_effort_ladder,
)


def test_default_effort_profiles_are_deterministic_and_monotonic() -> None:
    first = default_effort_profiles()
    second = default_effort_profiles()
    assert first == second
    assert [profile.name for profile in first] == ["E0_low", "E1_medium", "E2_high"]
    assert first[1].dominates(first[0])
    assert first[2].dominates(first[1])
    assert first[0].control_vector["B"]["native_kv_tokens"] == 256
    assert PublicAdaptiveRetryAgent is AdaptiveRetryAgent
    assert PublicEffortProfile is type(first[0])


def test_non_monotonic_effort_ladder_is_rejected() -> None:
    profiles = list(default_effort_profiles())
    broken = profiles[1].__class__(
        **{**profiles[1].__dict__, "native_kv_budget": 128}
    )
    with pytest.raises(ValueError, match="not monotonic"):
        validate_effort_ladder((profiles[0], broken, profiles[2]))


def test_controller_features_reject_oracle_leakage() -> None:
    with pytest.raises(ValueError, match="Evaluator-only"):
        ControllerFeatures.from_runtime_mapping(
            {"routing_entropy": 0.5, "oracle_evidence_recall": 1.0}
        )


def test_entropy_consistency_and_calibration_metrics() -> None:
    entropy = token_entropy(torch.tensor([[0.0, 0.0]]))
    assert float(entropy[0]) == pytest.approx(0.693147, rel=1e-5)
    assert semantic_consistency("Paris, France", "France: Paris") == 1.0
    assert semantic_consistency("alpha", "beta") == 0.0
    metrics = calibration_metrics([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["brier"] < 0.05


def test_linear_controller_learns_discrete_minimum_effort() -> None:
    features = [
        ControllerFeatures(routing_entropy=value, root_score_gap=1.0 - value)
        for value in (0.05, 0.1, 0.45, 0.55, 0.85, 0.95)
    ]
    targets = ["E0_low", "E0_low", "E1_medium", "E1_medium", "E2_high", "E2_high"]
    controller = LinearEffortController.fit(
        features,
        targets,
        ("E0_low", "E1_medium", "E2_high"),
        feature_names=("routing_entropy", "root_score_gap"),
    )
    assert controller.choose(ControllerFeatures(routing_entropy=0.03, root_score_gap=0.97)) == "E0_low"
    assert controller.choose(ControllerFeatures(routing_entropy=0.97, root_score_gap=0.03)) == "E2_high"


def test_retry_escalates_monotonically_and_reuses_state() -> None:
    calls: list[tuple[str, str | None]] = []

    def execute(profile, previous):
        calls.append((profile.name, previous.answer if previous else None))
        hard = profile.level == 0
        return AttemptResult(
            answer="uncertain" if hard else "stable answer",
            features=ControllerFeatures(
                routing_entropy=0.9 if hard else 0.2,
                root_score_gap=0.01 if hard else 0.5,
                answer_margin=-0.2 if hard else 1.0,
                path_convergence=0.2 if hard else 0.9,
            ),
            incorrect_probability=0.8 if hard else 0.1,
            search_seconds=0.01,
            materialization_seconds=0.01,
            generation_seconds=0.02,
            active_native_kv=profile.native_kv_budget,
            selected_parents=tuple(range(profile.retained_roots)),
            reusable_state={"profile": profile.name},
            reused_search_items=0 if previous is None else 1,
            reused_kv_tokens=0 if previous is None else previous.active_native_kv,
            metadata={
                "query_spans": ((3, 7),),
                "query_region_confidence": 0.8,
                "query_region_method": "structural",
                "query_region_expansion": int(previous is not None),
            },
        )

    agent = AdaptiveRetryAgent(
        default_effort_profiles(),
        StopPolicy(
            max_routing_entropy=0.7,
            min_answer_margin=0.0,
            min_retry_consistency=0.0,
        ),
        max_retries=2,
    )
    controller = HandRuleController(0.4, 0.8, 0.2, 0.05)
    result = agent.run(
        execute,
        ControllerFeatures(routing_entropy=0.1, root_score_gap=0.8),
        controller=controller,
    )
    assert result.answer == "stable answer"
    assert [trace.effort for trace in result.traces] == ["E0_low", "E1_medium"]
    assert calls == [("E0_low", None), ("E1_medium", "uncertain")]
    assert result.traces[1].reused_kv_tokens == 256
    assert result.traces[0].escalation_reasons
    assert result.traces[1].stop_reason == "confidence"
    assert result.traces[0].query_start == 3
    assert result.traces[0].query_end == 7
    assert result.traces[0].query_region_count == 1
    assert result.traces[0].query_region_confidence == pytest.approx(0.8)
    assert result.traces[0].query_region_method == "structural"
    assert result.traces[1].query_region_expansion == 1


def test_retry_enforces_maximum_active_kv_budget() -> None:
    agent = AdaptiveRetryAgent(
        default_effort_profiles(),
        StopPolicy(max_incorrect_probability=0.0),
        max_active_kv=300,
    )

    def execute(profile, previous):
        assert previous is None
        return AttemptResult(
            "answer",
            ControllerFeatures(),
            1.0,
            0.0,
            0.0,
            0.0,
            profile.native_kv_budget,
        )

    result = agent.run(execute, ControllerFeatures())
    assert result.attempts == 1
    assert result.final_effort == "E0_low"
    assert result.traces[0].stop_reason == "max_effort"
