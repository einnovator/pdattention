"""Compare an autonomous agent's plain and PRA interaction histories."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .context_treatment import (
    _mandatory_indices,
    _pinned_task_indices,
    _turn_bundles,
)


ACTION = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory(arm: Path) -> dict[str, Any]:
    for name in ("trajectory.json", "django__django-15277.traj.json"):
        path = arm / name
        if path.is_file():
            return _load_json(path)
    matches = list(arm.glob("**/*.traj.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one trajectory below {arm}, found {matches}")
    return _load_json(matches[0])


def _actions(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("content", ""))
        for row in messages
        if row.get("role") == "assistant"
    ]


def _command(action: str) -> str | None:
    match = ACTION.search(action)
    return match.group(1).strip() if match else None


def _request_prefix(
    messages: list[Mapping[str, Any]], assistant_number: int,
) -> list[Mapping[str, Any]]:
    observed = 0
    for index, row in enumerate(messages):
        if row.get("role") == "assistant":
            observed += 1
            if observed == assistant_number:
                return messages[:index]
    return messages


def _excerpt(text: Any, limit: int = 280) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit]


def _selection_rows(arm: Path) -> list[dict[str, Any]]:
    path = arm / "selection_fixture.jsonl"
    if not path.is_file():
        path = arm / "selection.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _outcome(arm: Path) -> dict[str, Any]:
    path = arm / "results.jsonl"
    if not path.is_file():
        return {}
    line = next((line for line in path.read_text(encoding="utf-8").splitlines() if line), "")
    row = json.loads(line) if line else {}
    return {
        key: row.get(key) for key in (
            "benchmark_score", "resolved", "model_call_count", "termination_reason",
            "grader_outcome", "error_type", "patch_bytes", "files_inspected",
            "files_modified", "token_saving_fraction_estimate",
        )
    }


def _first_matching(commands: list[str | None], pattern: str) -> int | None:
    expression = re.compile(pattern)
    return next(
        (index for index, command in enumerate(commands, 1) if command and expression.search(command)),
        None,
    )


def analyze_arm(
    task_dir: Path, baseline_messages: list[Mapping[str, Any]], arm_name: str,
) -> dict[str, Any]:
    arm = task_dir / arm_name
    trajectory = _trajectory(arm)
    messages = trajectory["messages"]
    baseline_actions = _actions(baseline_messages)
    arm_actions = _actions(messages)
    shared = 0
    for plain, treated in zip(baseline_actions, arm_actions):
        if plain != treated:
            break
        shared += 1
    first_divergence = (
        shared + 1 if shared < min(len(baseline_actions), len(arm_actions)) else None
    )
    baseline_commands = [_command(action) for action in baseline_actions]
    arm_commands = [_command(action) for action in arm_actions]
    shared_commands = 0
    for plain, treated in zip(baseline_commands, arm_commands):
        if plain != treated:
            break
        shared_commands += 1
    first_command_divergence = (
        shared_commands + 1
        if shared_commands < min(len(baseline_commands), len(arm_commands)) else None
    )
    selection = _selection_rows(arm)
    telemetry = _jsonl_rows(arm / "request_telemetry.jsonl")
    divergence_context: dict[str, Any] | None = None
    context_request = first_command_divergence or first_divergence
    if context_request is not None:
        prefix = _request_prefix(messages, context_request)
        mandatory = _mandatory_indices(prefix)
        pinned = _pinned_task_indices(prefix, mandatory)
        candidates = [
            index for index in range(len(prefix))
            if index not in mandatory and index not in pinned
        ]
        bundles = _turn_bundles(prefix, candidates, 256)
        candidate_ids = [segment_id for bundle in bundles for segment_id, _ in bundle]
        selected_resources = (
            selection[context_request - 1].get("resources", [])
            if context_request <= len(selection) else []
        )
        selected_ids = [str(row["resource_id"]) for row in selected_resources]
        selected_id_set = set(selected_ids)
        divergence_context = {
            "request_index": context_request,
            "logical_messages": len(prefix),
            "mandatory_active_tail": [
                {
                    "message_index": index,
                    "role": prefix[index].get("role"),
                    "excerpt": _excerpt(prefix[index].get("content")),
                }
                for index in sorted(mandatory)
            ],
            "pinned_task_message_indices": sorted(pinned),
            "candidate_resource_ids": candidate_ids,
            "selected_resource_ids": selected_ids,
            "omitted_candidate_resource_ids": [
                item for item in candidate_ids if item not in selected_id_set
            ],
            "selected_resource_excerpts": [
                {
                    "resource_id": row["resource_id"],
                    "excerpt": _excerpt(row.get("text")),
                }
                for row in selected_resources
            ],
            "token_accounting": {
                key: telemetry[context_request - 1].get(key)
                for key in (
                    "logical_input_tokens_estimate", "mandatory_tokens_estimate",
                    "selected_tokens_estimate", "physical_input_tokens_estimate",
                    "token_saving_fraction_estimate",
                )
            } if context_request <= len(telemetry) else {},
            "plain_action": {
                "command": baseline_commands[context_request - 1],
                "excerpt": _excerpt(baseline_actions[context_request - 1], 500),
            },
            "treated_action": {
                "command": arm_commands[context_request - 1],
                "excerpt": _excerpt(arm_actions[context_request - 1], 500),
            },
        }
    repeated = [
        {"command": command, "count": count}
        for command, count in Counter(
            command for command in arm_commands if command
        ).most_common()
        if count > 1
    ]
    unique_commands = len(set(command for command in arm_commands if command))
    return {
        "arm": arm_name,
        "actions": len(arm_actions),
        "shared_initial_actions_exact": shared,
        "first_divergent_action": first_divergence,
        "shared_initial_commands_exact": shared_commands,
        "first_divergent_command": first_command_divergence,
        "all_shared_actions_exact": first_divergence is None,
        "divergence_context": divergence_context,
        "repeated_commands": repeated[:10],
        "command_repetition_fraction": (
            1.0 - unique_commands / len(arm_commands) if arm_commands else 0.0
        ),
        "milestones": {
            "first_value_source_inspection": _first_matching(
                arm_commands, r"expressions\.py|class Value"
            ),
            "first_source_mutation": _first_matching(
                arm_commands, r"\bsed\s+-i\b|\bapply_patch\b|\bperl\s+-[pi]"
            ),
            "first_git_diff": _first_matching(arm_commands, r"\bgit diff\b"),
            "first_patch_file_creation": _first_matching(
                arm_commands, r"git diff.*>\s*patch\.txt"
            ),
            "exact_submission_contract": _first_matching(
                arm_commands,
                r"^echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch\.txt$",
            ),
        },
        "final_submission_excerpt": _excerpt(
            trajectory.get("info", {}).get("submission"), 500
        ),
        "outcome": _outcome(arm),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Task interaction divergence audit",
        "",
        f"Baseline: `{report['baseline']}` ({report['baseline_actions']} actions).",
        "",
        "| Arm | Score | Actions | Exact responses | First response diff | Exact commands | First command diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        outcome = arm["outcome"]
        lines.append(
            f"| `{arm['arm']}` | {outcome.get('benchmark_score')} | {arm['actions']} | "
            f"{arm['shared_initial_actions_exact']} | {arm['first_divergent_action'] or 'none'} | "
            f"{arm['shared_initial_commands_exact']} | {arm['first_divergent_command'] or 'none'} |"
        )
    progress_spine = [
        arm for arm in report["arms"] if "progress_spine" in arm["arm"]
    ]
    reduced = [
        arm for arm in report["arms"]
        if arm["arm"] != "pra_100_liveprefix" and arm not in progress_spine
    ]
    common_command_break = {
        arm["first_divergent_command"] for arm in reduced
        if arm["first_divergent_command"] is not None
    }
    common_selected = {
        tuple((arm.get("divergence_context") or {}).get("selected_resource_ids", ()))
        for arm in reduced
    }
    lines.extend([
        "",
        "## Main finding",
        "",
        "The 100% PRA arm is response-for-response exact with the successful plain run, "
        "which qualifies the engine and transport path when no history is removed.",
    ])
    if len(common_command_break) == 1 and len(common_selected) == 1:
        request = next(iter(common_command_break))
        example = reduced[0]["divergence_context"]
        accounting = example.get("token_accounting", {})
        lines.extend([
            "",
            f"Every legacy v3 reduced-context arm first changes the executed plan at action {request} "
            "and exposes the same selected records at that point. The selector keeps the "
            "five structural task segments and the active tail, but drops the first two "
            "completed action/observation bundles. Nominal 75%, 50%, and 25% profiles "
            "therefore collapse to the same early prompt because the structural floor "
            "dominates the budget.",
            "",
            "At that request the whitespace accounting is "
            f"logical={accounting.get('logical_input_tokens_estimate')}, "
            f"mandatory-tail={accounting.get('mandatory_tokens_estimate')}, "
            f"selected-resource={accounting.get('selected_tokens_estimate')}, and "
            f"physical={accounting.get('physical_input_tokens_estimate')} tokens. "
            "The small early saving is purchased by deleting the agent's progress state.",
        ])
    lines.extend([
        "",
        "The legacy v3 failure mode is a progress-memory loop, not a template or K/V attachment "
        "fault: retrieval is queried only with the latest tool observation and repeatedly "
        "surfaces older, lexically similar CharField inspections. It does not preserve a "
        "stable plan, the latest successful mutation, verification state, or the patch "
        "submission lifecycle.",
    ])
    for arm in progress_spine:
        outcome = arm["outcome"]
        milestones = arm["milestones"]
        lines.extend(["", (
            "The v4 progress-spine repair pins the two latest completed causal turns "
            "plus the latest mutation and verification turns. "
            f"In `{arm['arm']}`, its first command difference is action "
            f"{arm['first_divergent_command']}. "
            + (
                "The 75% arm takes a narrower but semantically equivalent source view, "
                "later recovers from a stale-line edit, creates the patch at action "
                f"{milestones['first_patch_file_creation']}, submits exactly at action "
                f"{milestones['exact_submission_contract']}, and resolves the task."
                if outcome.get("benchmark_score") == 1.0 else
                "The 50% arm retains the exact earlier CharField path but does not use it, "
                "launches redundant whole-tree searches, corrupts the method boundary while "
                "editing, and reaches the 50-step limit without a submitted patch. This is "
                "a consumption/edit-execution failure despite the required state being selected."
            )
            + " Its estimated logical-context saving is "
            f"{outcome.get('token_saving_fraction_estimate')}."
        )])
    for arm in report["arms"]:
        context = arm.get("divergence_context")
        if not context:
            continue
        lines.extend([
            "",
            f"## {arm['arm']}: first divergence",
            "",
            f"The first differing executed command is action {context['request_index']}.",
            f"Selected resource IDs: `{context['selected_resource_ids']}`.",
            f"Omitted historical candidate IDs: `{context['omitted_candidate_resource_ids']}`.",
            "",
            f"Plain command: `{context['plain_action']['command']}`",
            "",
            f"Treated command: `{context['treated_action']['command']}`",
        ])
        if arm["repeated_commands"]:
            lines.extend([
                "",
                "Most repeated later commands: " + "; ".join(
                    f"`{row['command']}` ({row['count']}x)"
                    for row in arm["repeated_commands"][:3]
                ),
            ])
        milestones = arm["milestones"]
        lines.extend([
            "",
            "Milestones: "
            f"Value inspection={milestones['first_value_source_inspection']}; "
            f"source mutation={milestones['first_source_mutation']}; "
            f"git diff={milestones['first_git_diff']}; "
            f"patch file={milestones['first_patch_file_creation']}; "
            f"exact submission={milestones['exact_submission_contract']}.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="plain_clean_receipt")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = _trajectory(args.task_dir / args.baseline)
    report = {
        "schema_version": 1,
        "instance_id": baseline.get("instance_id"),
        "baseline": args.baseline,
        "baseline_actions": len(_actions(baseline["messages"])),
        "arms": [
            analyze_arm(args.task_dir, baseline["messages"], arm) for arm in args.arms
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
