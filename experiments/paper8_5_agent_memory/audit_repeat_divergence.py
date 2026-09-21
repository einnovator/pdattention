"""Audit deterministic-control divergence between two mini-swe trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .recordizer import _command
from .submission_protocol import validate_unified_git_diff


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"{path}: expected a mini-swe trajectory object")
    return value


def _actions(trajectory: Mapping[str, Any]) -> list[str | None]:
    return [
        _command(str(message.get("content") or ""))
        for message in trajectory["messages"]
        if message.get("role") == "assistant"
    ]


def build_audit(path_a: Path, path_b: Path) -> dict[str, Any]:
    trajectories = (_load(path_a), _load(path_b))
    messages = [trajectory["messages"] for trajectory in trajectories]
    comparable = min(map(len, messages))
    first_message = next((
        index for index in range(comparable)
        if (
            messages[0][index].get("role"),
            messages[0][index].get("content"),
        ) != (
            messages[1][index].get("role"),
            messages[1][index].get("content"),
        )
    ), None)
    actions = [_actions(trajectory) for trajectory in trajectories]
    comparable_actions = min(map(len, actions))
    first_action = next((
        index for index in range(comparable_actions)
        if actions[0][index] != actions[1][index]
    ), None)
    runs = []
    for label, path, trajectory, run_actions in zip(
        ("A", "B"), (path_a, path_b), trajectories, actions
    ):
        info = dict(trajectory.get("info") or {})
        submission = str(info.get("submission") or "")
        validation = validate_unified_git_diff(submission)
        runs.append({
            "label": label,
            "trajectory_label": path.name,
            "trajectory_sha256": _sha256(path),
            "message_count": len(trajectory["messages"]),
            "action_count": len(run_actions),
            "exit_status": info.get("exit_status"),
            "submission_valid_git_diff": validation.valid,
            "submission_validation_reason": validation.reason,
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_repeat_control_divergence",
        "same_instance": trajectories[0].get("instance_id") == trajectories[1].get("instance_id"),
        "instance_id": trajectories[0].get("instance_id"),
        "common_visible_message_prefix": first_message if first_message is not None else comparable,
        "first_divergent_message_index_zero_based": first_message,
        "first_divergent_message_role": (
            messages[0][first_message].get("role")
            if first_message is not None else None
        ),
        "first_divergent_action_index_one_based": (
            first_action + 1 if first_action is not None else None
        ),
        "first_divergent_actions": (
            {"A": actions[0][first_action], "B": actions[1][first_action]}
            if first_action is not None else None
        ),
        "runs": runs,
        "interpretation": (
            "Visible histories were identical before the first divergent assistant "
            "action; the fork is backend/model trajectory variability, not selection."
            if first_message is not None
            and messages[0][first_message].get("role") == "assistant" else
            "The trajectories diverged after different visible histories."
        ),
    }


def _markdown(audit: Mapping[str, Any]) -> str:
    divergent = audit.get("first_divergent_actions") or {}
    rows = [
        "# Repeat-control divergence audit",
        "",
        f"- Instance: `{audit.get('instance_id')}`",
        f"- Common visible-message prefix: {audit.get('common_visible_message_prefix')}",
        f"- First divergent action: {audit.get('first_divergent_action_index_one_based')}",
        f"- Interpretation: {audit.get('interpretation')}",
        "",
        "| Run | Actions | Exit | Valid Git diff |",
        "|---|---:|---|---|",
    ]
    rows.extend(
        f"| {run['label']} | {run['action_count']} | {run['exit_status']} | "
        f"{run['submission_valid_git_diff']} |"
        for run in audit["runs"]
    )
    if divergent:
        rows.extend((
            "", "## First changed action", "",
            f"- A: `{divergent.get('A')}`",
            f"- B: `{divergent.get('B')}`",
        ))
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-a", type=Path, required=True)
    parser.add_argument("--trajectory-b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.trajectory_a, args.trajectory_b)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
