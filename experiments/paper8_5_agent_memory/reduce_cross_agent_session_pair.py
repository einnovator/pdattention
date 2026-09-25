"""Reduce one persistent FULL/selective agent-session pair.

The reducer keeps three quantities separate:

* candidate-own saving: removed context on the candidate's realized trajectory;
* paired saving: candidate materialization versus the FULL trajectory's token cost;
* official task resolution: task-clustered SWE-bench outcomes.

A selective recovery where FULL failed is retained as a diagnostic and is never
counted as paired preservation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _trace(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(row.get("request_index") or 0) for row in rows]
    if indices != list(range(1, len(rows) + 1)):
        raise ValueError(f"trace is not contiguous and one-based: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolution(report: Mapping[str, Any], task_order: list[str]) -> dict[str, bool]:
    submitted = set(str(value) for value in report.get("submitted_ids", ()))
    if submitted != set(task_order):
        raise ValueError("official report does not cover the declared task cohort")
    resolved = set(str(value) for value in report.get("resolved_ids", ()))
    return {task: task in resolved for task in task_order}


def _episode_ranges(state: Mapping[str, Any], trace_rows: list[dict[str, Any]]):
    cursor = 0
    result = []
    for episode in state.get("episodes", ()):
        summary = episode.get("event_summary") or {}
        calls = int(summary.get("assistant_model_calls") or 0)
        if calls < 1:
            raise ValueError("episode has no audited assistant call count")
        selected = trace_rows[cursor: cursor + calls]
        if len(selected) != calls:
            raise ValueError("episode call counts exceed the request trace")
        result.append((episode, selected))
        cursor += calls
    if cursor != len(trace_rows):
        raise ValueError("episode call counts do not partition the request trace")
    return result


def reduce_pair(
    control_dir: Path,
    candidate_dir: Path,
    control_report_path: Path,
    candidate_report_path: Path,
) -> dict[str, Any]:
    control_state_path = control_dir / "campaign_state.json"
    candidate_state_path = candidate_dir / "campaign_state.json"
    control_trace_path = control_dir / "proxy_trace.jsonl"
    candidate_trace_path = candidate_dir / "proxy_trace.jsonl"
    control_state = _load(control_state_path)
    candidate_state = _load(candidate_state_path)
    control_report = _load(control_report_path)
    candidate_report = _load(candidate_report_path)
    control_trace = _trace(control_trace_path)
    candidate_trace = _trace(candidate_trace_path)

    task_order = [str(value) for value in control_state.get("task_order", ())]
    identity_fields = ("agent", "session_id", "task_order", "task_count_declared")
    mismatches = {
        field: {
            "control": control_state.get(field),
            "candidate": candidate_state.get(field),
        }
        for field in identity_fields
        if control_state.get(field) != candidate_state.get(field)
    }
    if mismatches:
        raise ValueError(f"campaign identity mismatch: {mismatches}")
    if control_state.get("policy") != "full":
        raise ValueError("control campaign is not FULL")
    if not task_order or len(task_order) != int(control_state["task_count_declared"]):
        raise ValueError("control task cohort is empty or inconsistent")

    control_resolution = _resolution(control_report, task_order)
    candidate_resolution = _resolution(candidate_report, task_order)
    control_episodes = _episode_ranges(control_state, control_trace)
    candidate_episodes = _episode_ranges(candidate_state, candidate_trace)
    if [row[0].get("instance_id") for row in control_episodes] != task_order:
        raise ValueError("control episode order differs from task order")
    if [row[0].get("instance_id") for row in candidate_episodes] != task_order:
        raise ValueError("candidate episode order differs from task order")

    task_rows = []
    for task, control, candidate in zip(
        task_order, control_episodes, candidate_episodes
    ):
        control_episode, control_requests = control
        candidate_episode, candidate_requests = candidate
        control_full = sum(int(row.get("full_tokens") or 0) for row in control_requests)
        candidate_full = sum(int(row.get("full_tokens") or 0) for row in candidate_requests)
        candidate_materialized = sum(
            int(row.get("materialized_tokens") or 0) for row in candidate_requests
        )
        control_ok = control_resolution[task]
        candidate_ok = candidate_resolution[task]
        if control_ok and candidate_ok:
            classification = "paired_full_success_preserved"
        elif control_ok:
            classification = "paired_full_success_lost"
        elif candidate_ok:
            classification = "selective_recovery_diagnostic"
        else:
            classification = "both_unresolved"
        task_rows.append({
            "instance_id": task,
            "classification": classification,
            "control_resolved": control_ok,
            "candidate_resolved": candidate_ok,
            "control_calls": len(control_requests),
            "candidate_calls": len(candidate_requests),
            "call_delta": len(candidate_requests) - len(control_requests),
            "control_patch_bytes": int(control_episode.get("patch_bytes") or 0),
            "candidate_patch_bytes": int(candidate_episode.get("patch_bytes") or 0),
            "control_full_tokens": control_full,
            "candidate_full_tokens": candidate_full,
            "candidate_materialized_tokens": candidate_materialized,
            "candidate_own_saving_fraction": (
                1.0 - candidate_materialized / candidate_full
                if candidate_full else 0.0
            ),
            "paired_saving_fraction": (
                1.0 - candidate_materialized / control_full
                if control_full else 0.0
            ),
            "candidate_excluded_group_counts": sorted({
                int(row.get("excluded_causal_group_count") or 0)
                for row in candidate_requests
            }),
        })

    control_total = sum(int(row.get("full_tokens") or 0) for row in control_trace)
    candidate_full_total = sum(
        int(row.get("full_tokens") or 0) for row in candidate_trace
    )
    candidate_materialized_total = sum(
        int(row.get("materialized_tokens") or 0) for row in candidate_trace
    )
    paired_full_successes = sum(row["control_resolved"] for row in task_rows)
    preserved = sum(
        row["control_resolved"] and row["candidate_resolved"] for row in task_rows
    )
    lost = sum(
        row["control_resolved"] and not row["candidate_resolved"]
        for row in task_rows
    )
    recoveries = sum(
        not row["control_resolved"] and row["candidate_resolved"]
        for row in task_rows
    )
    return {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_persistent_session_pair",
        "claim_boundary": (
            "Task-clustered single-repeat evidence. Selective recoveries are "
            "diagnostic and are not paired-preservation wins."
        ),
        "agent": control_state["agent"],
        "session_id": control_state["session_id"],
        "task_order": task_order,
        "control_policy": control_state["policy"],
        "candidate_policy": candidate_state["policy"],
        "candidate_policy_parameters": candidate_state.get("policy_parameters"),
        "control_official_resolved_count": sum(control_resolution.values()),
        "candidate_official_resolved_count": sum(candidate_resolution.values()),
        "paired_full_success_count": paired_full_successes,
        "paired_success_preserved_count": preserved,
        "paired_success_lost_count": lost,
        "conditional_preservation_fraction": (
            preserved / paired_full_successes if paired_full_successes else None
        ),
        "selective_recovery_diagnostic_count": recoveries,
        "control_calls": len(control_trace),
        "candidate_calls": len(candidate_trace),
        "aggregate_call_delta": len(candidate_trace) - len(control_trace),
        "control_full_tokens": control_total,
        "candidate_full_tokens": candidate_full_total,
        "candidate_materialized_tokens": candidate_materialized_total,
        "candidate_own_saving_fraction": (
            1.0 - candidate_materialized_total / candidate_full_total
            if candidate_full_total else 0.0
        ),
        "paired_saving_fraction": (
            1.0 - candidate_materialized_total / control_total
            if control_total else 0.0
        ),
        "task_results": task_rows,
        "artifacts": {
            "control_campaign_state_sha256": _sha256(control_state_path),
            "candidate_campaign_state_sha256": _sha256(candidate_state_path),
            "control_trace_sha256": _sha256(control_trace_path),
            "candidate_trace_sha256": _sha256(candidate_trace_path),
            "control_official_report_sha256": _sha256(control_report_path),
            "candidate_official_report_sha256": _sha256(candidate_report_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce_pair(
        args.control.resolve(), args.candidate.resolve(),
        args.control_report.resolve(), args.candidate_report.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
