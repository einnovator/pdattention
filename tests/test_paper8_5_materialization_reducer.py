from experiments.paper8_5_agent_memory.reduce_frozen_replays import _digest, reduce_replays


def _artifact(*, mode: str, reference_digest=None):
    artifact = {
        "schema_version": 2,
        "study": "paper8_5_agent_memory_frozen_replay",
        "instance_id": "owner__task-1",
        "model": "model",
        "tokenizer": "tokenizer",
        "policy": "full",
        "budget_fraction": 1.0,
        "materialization_mode": mode,
        "comparison_reference": (
            "historical_trajectory"
            if mode == "whole_record"
            else "contemporaneous_full_replay"
        ),
        "reference_replay_digest": reference_digest,
        "run_configuration": {
            "trajectory_digest": "trajectory",
            "model": "model",
            "tokenizer": "tokenizer",
            "materialization_mode": mode,
            "materialization_threshold_tokens": 512,
            "seed": 0,
            "reference_replay_digest": reference_digest,
        },
        "rows": [{
            "decision": 1,
            "decision_status": "completed",
            "trajectory_message_index": 2,
            "action_valid": True,
            "generated_content": "THOUGHT: ok\n```mswea_bash_command\necho ok\n```",
            "exact_content": True,
            "exact_command": True,
            "reference_command": "echo ok",
            "generated_command": "echo ok",
            "full_history_tokens": 100,
            "selected_logical_tokens": 100,
            "selected_whole_record_tokens": 100,
            "materialized_tokens": 100 if mode == "whole_record" else 80,
            "requested_budget_tokens": 100,
            "realized_record_retention_fraction": 1.0,
            "realized_materialized_retention_fraction": (
                1.0 if mode == "whole_record" else 0.8
            ),
            "mandatory_overflow_tokens": 0,
            "whole_turn_budget_overshoot_tokens": 0,
            "whole_turn_budget_undershoot_tokens": 0,
            "materialized_budget_unused_tokens": 0,
            "excluded_group_count": 0,
            "excluded_tokens": 0,
        }],
    }
    artifact["run_configuration_digest"] = _digest(artifact["run_configuration"])
    artifact["attempted_decisions"] = len(artifact["rows"])
    artifact["completed_decisions"] = len(artifact["rows"])
    return artifact


def test_reducer_accepts_full_selection_with_distinct_materializers():
    baseline = _artifact(mode="whole_record")
    structured = _artifact(
        mode="tool_structured_evidence",
        reference_digest=_digest(baseline),
    )
    summary = reduce_replays([baseline, structured])

    assert [row["treatment"] for row in summary["arms"]] == [
        "FULL",
        "FULL;materialization=tool_structured_evidence;threshold=512",
    ]
