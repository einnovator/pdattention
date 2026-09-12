import copy

import pytest

from experiments.paper8_5_agent_memory.reduce_frozen_replays import (
    _digest,
    reduce_replays,
    render_markdown,
)


def _row(
    decision,
    content,
    command,
    *,
    valid=True,
    status="completed",
    full=100,
    selected=100,
    materialized=100,
    requested=100,
    retention=1.0,
    overflow=0,
):
    return {
        "decision": decision,
        "decision_status": status,
        "failure_reason": None if valid else "missing_generated_command",
        "action_valid": valid,
        "action_validation_reason": None if valid else "missing_generated_command",
        "trajectory_message_index": decision * 2,
        "generated_content": content,
        "generated_command": command,
        "full_history_tokens": full,
        "selected_whole_record_tokens": selected,
        "materialized_tokens": materialized,
        "requested_budget_tokens": requested,
        "realized_record_retention_fraction": retention,
        "mandatory_overflow_tokens": overflow,
    }


def _artifact(policy, rows, *, reference_digest=None):
    configuration = {
        "trajectory_digest": "trajectory-sha256",
        "model": "qwen3-coder:30b",
        "tokenizer": "tokenizer-sha256",
        "reference_replay_digest": reference_digest,
    }
    return {
        "schema_version": 2,
        "study": "paper8_5_agent_memory_frozen_replay",
        "instance_id": "task-1",
        "model": configuration["model"],
        "tokenizer": configuration["tokenizer"],
        "comparison_reference": (
            "contemporaneous_full_replay" if reference_digest else "historical_trajectory"
        ),
        "reference_replay_digest": reference_digest,
        "policy": policy,
        "run_configuration": configuration,
        "run_configuration_digest": _digest(configuration),
        "attempted_decisions": len(rows),
        "completed_decisions": sum(row["decision_status"] == "completed" for row in rows),
        "rows": rows,
    }


def _comparison_artifacts():
    full = _artifact("full", [
        _row(1, "thought A\n```bash\ncmd-a\n```", "cmd-a", full=100),
        _row(2, "thought B\n```bash\ncmd-b\n```", "cmd-b", full=200),
        _row(3, "thought C\n```bash\ncmd-c\n```", "cmd-c", full=300),
    ])
    full_digest = _digest(full)
    candidate = _artifact("middle_recency", [
        _row(
            1,
            full["rows"][0]["generated_content"],
            "cmd-a",
            full=100,
            selected=80,
            materialized=75,
            requested=85,
            retention=0.8,
        ),
        _row(
            2,
            "different thought\n```bash\ncmd-b\n```",
            "cmd-b",
            full=200,
            selected=150,
            materialized=140,
            requested=160,
            retention=0.75,
        ),
        _row(
            3,
            "THOUGHT: no action",
            None,
            valid=False,
            full=300,
            selected=200,
            materialized=190,
            requested=210,
            retention=0.5,
            overflow=5,
        ),
    ], reference_digest=full_digest)
    transport = _artifact("tail", [
        _row(1, full["rows"][0]["generated_content"], "cmd-a", full=100),
        _row(2, full["rows"][1]["generated_content"], "cmd-b", full=200),
        _row(
            3,
            "",
            None,
            valid=False,
            status="failed_transport_generation",
            full=300,
        ),
    ], reference_digest=full_digest)
    transport["rows"][2]["failure_reason"] = "empty_generated_content"
    transport["rows"][2]["action_validation_reason"] = "empty_generated_content"
    return full, candidate, transport


def test_reducer_reports_matched_metrics_and_markdown():
    full, candidate, transport = _comparison_artifacts()
    summary = reduce_replays(
        [full, candidate, transport], sources=["full.json", "recency.json", "tail.json"]
    )

    assert summary["arm_count"] == 3
    recency = summary["arms"][1]
    assert recency["decisions_attempted"] == recency["decisions_completed"] == 3
    assert recency["valid_action_rate"] == pytest.approx(2 / 3)
    assert recency["exact_content_rate_vs_full"] == pytest.approx(1 / 3)
    assert recency["exact_command_rate_vs_full"] == pytest.approx(2 / 3)
    assert recency["first_content_divergence"] == 2
    assert recency["first_command_divergence"] == 3
    assert recency["first_action_validity_divergence"] == 3
    assert recency["realized_retention"] == pytest.approx({
        "mean": (0.8 + 0.75 + 0.5) / 3,
        "min": 0.5,
        "max": 0.8,
    })
    assert recency["cumulative_full_history_tokens"] == 600
    assert recency["cumulative_selected_whole_record_tokens"] == 430
    assert recency["cumulative_materialized_tokens"] == 405
    assert recency["logical_token_saving_tokens"] == 170
    assert recency["logical_token_saving_fraction"] == pytest.approx(170 / 600)
    assert recency["mandatory_overflow_tokens"] == 5
    assert recency["unused_matched_budget_tokens"] == 25
    assert recency["transport_failures"] == 0
    assert recency["format_invalid_decisions"] == 1
    assert summary["arms"][2]["decisions_completed"] == 2
    assert summary["arms"][2]["transport_failure_decisions"] == [3]

    markdown = render_markdown(summary)
    assert "| Policy | Tried/done | Valid | Exact content | Exact command |" in markdown
    assert "| middle_recency@100%-ceiling | 3/3 | 66.7% | 33.3% | 66.7% | 2/3/3 |" in markdown
    assert "| tail@100%-ceiling | 3/2" in markdown


def test_reducer_distinguishes_same_policy_with_different_budget_contracts():
    full, candidate, _ = _comparison_artifacts()
    matched = copy.deepcopy(candidate)
    matched["matched_budget_source_digest"] = "a" * 64
    matched["run_configuration"]["matched_budget_source_digest"] = "a" * 64
    matched["run_configuration_digest"] = _digest(matched["run_configuration"])
    floor = copy.deepcopy(candidate)
    floor["budget_fraction"] = 0.9
    floor["run_configuration"]["budget_fraction"] = 0.9
    floor["run_configuration"]["whole_turn_budget_interpretation"] = "retention_floor_round_up"
    floor["run_configuration_digest"] = _digest(floor["run_configuration"])

    summary = reduce_replays([full, matched, floor])

    assert summary["arms"][1]["treatment"] == "middle_recency@matched-ceiling:aaaaaaaa"
    assert summary["arms"][2]["treatment"] == "middle_recency@90%-floor"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("trajectory_digest", "trajectory_digest differs"),
        ("model", "model differs"),
        ("tokenizer", "tokenizer differs"),
        ("reference_replay_digest", "reference replay digest does not match FULL"),
    ],
)
def test_reducer_fails_closed_on_mismatched_comparison_identity(field, message):
    full, candidate, _ = _comparison_artifacts()
    changed = copy.deepcopy(candidate)
    if field in {"model", "tokenizer"}:
        changed[field] = f"different-{field}"
    changed["run_configuration"][field] = f"different-{field}"
    if field == "reference_replay_digest":
        changed["reference_replay_digest"] = f"different-{field}"
    changed["run_configuration_digest"] = _digest(changed["run_configuration"])

    with pytest.raises(ValueError, match=message):
        reduce_replays([full, changed], sources=["full.json", "candidate.json"])
