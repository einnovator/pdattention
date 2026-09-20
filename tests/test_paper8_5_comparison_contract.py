from dataclasses import replace

import pytest

from experiments.paper8_5_agent_memory import (
    ComparisonAxis,
    ComparisonContractError,
    ExperimentIdentity,
    FrozenPlanObservation,
    validate_controlled_comparison,
    validate_frozen_plan_across_engines,
)


def _identity() -> ExperimentIdentity:
    return ExperimentIdentity(
        agent_id="mini-swe-agent",
        agent_revision="2.4.6",
        engine_id="reference-openai-endpoint",
        engine_revision="ollama-locked",
        model_id="qwen3-coder:30b",
        model_revision="06c1097e",
        tokenizer_id="qwen3-coder",
        tokenizer_revision="06c1097e",
        chat_template_digest="template-sha256",
        task_ids=("django__django-15277",),
        task_snapshot_digest="workspace-sha256",
        policy_id="matched-causal-tail",
        policy_revision="v1",
        policy_parameters_digest="policy-sha256",
        materialization_mode="ordinary-text-causal",
        temperature=0,
        top_p=1,
        seed=0,
        max_completion_tokens=1024,
        context_limit=131072,
    )


def test_cross_agent_allows_only_agent_identity_to_change() -> None:
    left = _identity()
    right = replace(left, agent_id="openhands", agent_revision="1.49.2")
    validate_controlled_comparison(left, right, axis=ComparisonAxis.AGENT)


def test_cross_agent_rejects_model_drift() -> None:
    left = _identity()
    right = replace(
        left,
        agent_id="openhands",
        agent_revision="1.49.2",
        model_id="qwen2.5-coder:14b",
    )
    with pytest.raises(ComparisonContractError, match="model_id"):
        validate_controlled_comparison(left, right, axis="agent")


def test_cross_engine_allows_only_engine_identity_to_change() -> None:
    left = _identity()
    right = replace(left, engine_id="mlx", engine_revision="0.29.3")
    validate_controlled_comparison(left, right, axis=ComparisonAxis.ENGINE)


def test_cross_engine_rejects_agent_or_policy_drift() -> None:
    left = _identity()
    right = replace(
        left,
        engine_id="mlx",
        engine_revision="0.29.3",
        agent_id="openhands",
        policy_parameters_digest="different-policy",
    )
    with pytest.raises(ComparisonContractError, match="agent_id"):
        validate_controlled_comparison(left, right, axis="engine")


def _plan() -> FrozenPlanObservation:
    return FrozenPlanObservation(
        history_digest="history-sha256",
        plan_digest="plan-sha256",
        selected_record_ids=("system", "task", "turn-7-action", "turn-7-result"),
        full_logical_tokens=10000,
        selected_logical_tokens=6000,
        tokenizer_revision="06c1097e",
    )


def test_same_frozen_plan_has_engine_independent_logical_saving() -> None:
    plan = _plan()
    reference = validate_frozen_plan_across_engines({
        "mlx": plan,
        "sglang-mlx": replace(plan),
        "llama.cpp": replace(plan),
    })
    assert reference.logical_saving_fraction == pytest.approx(0.4)


def test_engine_logical_token_difference_fails_closed() -> None:
    plan = _plan()
    rounded = replace(plan, selected_logical_tokens=6144)
    with pytest.raises(ComparisonContractError, match="selected_logical_tokens"):
        validate_frozen_plan_across_engines({"mlx": plan, "vllm": rounded})


def test_agent_native_compaction_is_not_a_controlled_arm() -> None:
    with pytest.raises(ComparisonContractError, match="compaction"):
        replace(_identity(), native_agent_compaction=True)


def test_diagnostic_tokenizer_is_not_a_controlled_arm() -> None:
    with pytest.raises(ComparisonContractError, match="whitespace"):
        replace(
            _identity(),
            tokenizer_id="whitespace_v1_diagnostic",
            tokenizer_revision="whitespace_v1",
        )
