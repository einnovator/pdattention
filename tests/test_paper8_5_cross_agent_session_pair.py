import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.reduce_cross_agent_session_pair import (
    reduce_pair,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign(root: Path, *, policy: str, calls: tuple[int, ...], selected: int) -> None:
    root.mkdir()
    tasks = [f"org__task-{index}" for index in range(len(calls))]
    _write_json(root / "campaign_state.json", {
        "agent": "pi",
        "session_id": "session",
        "task_count_declared": len(tasks),
        "task_order": tasks,
        "policy": policy,
        "policy_parameters": {"frontier_recent_user_prompts": 2},
        "episodes": [
            {
                "ordinal": index + 1,
                "instance_id": task,
                "patch_bytes": 10 if not (policy == "full" and index == 1) else 0,
                "event_summary": {"assistant_model_calls": count},
            }
            for index, (task, count) in enumerate(zip(tasks, calls))
        ],
    })
    rows = []
    request = 0
    for episode, count in enumerate(calls):
        for _ in range(count):
            request += 1
            rows.append({
                "request_index": request,
                "full_tokens": 100,
                "materialized_tokens": (
                    100 if policy == "full" or episode == 0 else selected
                ),
                "excluded_causal_group_count": 0 if episode == 0 else 2,
            })
    (root / "proxy_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_reducer_separates_preservation_recovery_and_savings(tmp_path: Path) -> None:
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _campaign(control, policy="full", calls=(2, 2), selected=100)
    _campaign(candidate, policy="frontier_dag_retirement", calls=(2, 1), selected=50)
    control_report = tmp_path / "control-report.json"
    candidate_report = tmp_path / "candidate-report.json"
    cohort = ["org__task-0", "org__task-1"]
    _write_json(control_report, {
        "submitted_ids": cohort, "resolved_ids": ["org__task-0"]
    })
    _write_json(candidate_report, {
        "submitted_ids": cohort, "resolved_ids": cohort
    })

    result = reduce_pair(control, candidate, control_report, candidate_report)

    assert result["conditional_preservation_fraction"] == 1.0
    assert result["paired_success_lost_count"] == 0
    assert result["selective_recovery_diagnostic_count"] == 1
    assert result["candidate_own_saving_fraction"] == pytest.approx(1 / 6)
    assert result["paired_saving_fraction"] == pytest.approx(3 / 8)
    assert result["aggregate_call_delta"] == -1
    assert result["task_results"][1]["classification"] == (
        "selective_recovery_diagnostic"
    )


def test_reducer_rejects_incomparable_task_orders(tmp_path: Path) -> None:
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _campaign(control, policy="full", calls=(1,), selected=100)
    _campaign(candidate, policy="frontier_dag_retirement", calls=(1,), selected=50)
    state = json.loads((candidate / "campaign_state.json").read_text())
    state["task_order"] = ["different__task"]
    _write_json(candidate / "campaign_state.json", state)
    control_report = tmp_path / "control-report.json"
    candidate_report = tmp_path / "candidate-report.json"
    _write_json(control_report, {
        "submitted_ids": ["org__task-0"], "resolved_ids": []
    })
    _write_json(candidate_report, {
        "submitted_ids": ["different__task"], "resolved_ids": []
    })

    with pytest.raises(ValueError, match="campaign identity mismatch"):
        reduce_pair(control, candidate, control_report, candidate_report)
