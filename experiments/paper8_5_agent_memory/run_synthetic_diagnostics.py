"""Generate and structurally qualify Paper 8.5 synthetic diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .materialization import (
    MaterializationMode,
    ToolObservationMaterializer,
    materialize_plan,
)
from .model import AgentMemoryBudget
from .negative_selection import (
    NEGATIVE_POLICY_RULES,
    NegativeHeuristicSelector,
    NegativeSelectionConfig,
)
from .recordizer import recordize_minisweagent_messages
from .selectors import FullHistorySelector, whitespace_tokens
from .serialization import serialize_materialized_messages
from .synthetic_tasks import SyntheticAgentTask, synthetic_agent_tasks


def _prefix_for_decision(
    messages: tuple[Mapping[str, Any], ...], decision: int
) -> list[Mapping[str, Any]]:
    ordinal = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        ordinal += 1
        if ordinal == decision:
            return list(messages[:index])
    raise ValueError(f"trajectory has no assistant decision {decision}")


def _query(messages: list[Mapping[str, Any]]) -> str:
    task = next(
        (str(row.get("content", "")) for row in messages if row.get("role") == "user"),
        "",
    )
    active = str(messages[-1].get("content", "")) if messages else ""
    return f"{task}\n{active}"


def _selected_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(str(row.get("content", "")) for row in messages)


def _policy_row(
    task: SyntheticAgentTask,
    prefix: list[Mapping[str, Any]],
    policy: str,
) -> dict[str, Any]:
    history = recordize_minisweagent_messages(prefix)
    full_tokens = sum(whitespace_tokens(row.content) for row in history.records)
    selector = NegativeHeuristicSelector(NegativeSelectionConfig(
        rules=NEGATIVE_POLICY_RULES[policy],
        protected_head_turns=0,
        protected_tail_turns=0,
    ))
    plan = selector.select(
        history=history,
        query=_query(prefix),
        budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
    )
    materialized = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(),
        query=_query(prefix),
    )
    selected = serialize_materialized_messages(history, materialized)
    text = _selected_text(selected)
    actual = tuple(row.causal_group_id for row in plan.exclusions)
    expected = tuple(task.expected_excluded_groups[policy])
    evidence = {needle: needle in text for needle in task.required_evidence}
    return {
        "policy": policy,
        "expected_excluded_groups": list(expected),
        "actual_excluded_groups": list(actual),
        "activation_matches_expectation": actual == expected,
        "required_evidence_retained": evidence,
        "all_required_evidence_retained": all(evidence.values()),
        "full_tokens": plan.full_history_tokens,
        "selected_tokens": plan.selected_tokens,
        "saving_fraction": 1.0 - plan.selected_tokens / plan.full_history_tokens,
        "plan_digest": plan.digest,
    }


def _materialization_rows(
    task: SyntheticAgentTask,
    prefix: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    history = recordize_minisweagent_messages(prefix)
    full_tokens = sum(whitespace_tokens(row.content) for row in history.records)
    plan = FullHistorySelector().select(
        history=history,
        query=_query(prefix),
        budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
    )
    rows = []
    for mode in (
        MaterializationMode.WHOLE_RECORD,
        MaterializationMode.TOOL_HEAD_TAIL,
        MaterializationMode.TOOL_MATCHED_SPAN,
        MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
    ):
        materialized = materialize_plan(
            history,
            plan,
            ToolObservationMaterializer(
                mode=mode,
                threshold_tokens=1,
                head_lines=3,
                tail_lines=3,
                match_context_lines=1,
                max_matched_lines=8,
            ),
            query=_query(prefix),
        )
        serialized = serialize_materialized_messages(history, materialized)
        text = _selected_text(serialized)
        marker = task.materialization_marker
        rows.append({
            "mode": mode.value,
            "full_selected_tokens": materialized.full_selected_tokens,
            "materialized_tokens": materialized.materialized_tokens,
            "detail_reduction_fraction": materialized.detail_reduction_fraction,
            "marker_retained": marker in text if marker else None,
            "record_modes": [row.mode.value for row in materialized.records],
        })
    return rows


def run_synthetic_diagnostics(
    *, tasks: tuple[SyntheticAgentTask, ...] | None = None,
) -> dict[str, Any]:
    tasks = tasks or synthetic_agent_tasks()
    rows = []
    for task in tasks:
        prefix = _prefix_for_decision(task.messages, task.probe_decision)
        policy_rows = [
            _policy_row(task, prefix, policy)
            for policy in task.expected_excluded_groups
        ]
        materialization_rows = _materialization_rows(task, prefix)
        structured = next(
            row for row in materialization_rows
            if row["mode"] == MaterializationMode.TOOL_STRUCTURED_EVIDENCE.value
        )
        head_tail = next(
            row for row in materialization_rows
            if row["mode"] == MaterializationMode.TOOL_HEAD_TAIL.value
        )
        marker_gate = (
            True
            if task.materialization_marker is None
            else bool(structured["marker_retained"] and not head_tail["marker_retained"])
        )
        rows.append({
            "task_id": task.task_id,
            "purpose": task.purpose,
            "probe_decision": task.probe_decision,
            "reference_command": task.reference_command,
            "policy_rows": policy_rows,
            "materialization_rows": materialization_rows,
            "mechanism_gate_passed": (
                all(
                    row["activation_matches_expectation"]
                    and row["all_required_evidence_retained"]
                    for row in policy_rows
                )
                and marker_gate
            ),
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_synthetic_mechanism_diagnostics",
        "evidence_class": "synthetic_structural_only_not_model_or_task_quality",
        "tokenizer": "whitespace_v1_structural_only",
        "task_count": len(rows),
        "all_mechanism_gates_passed": all(row["mechanism_gate_passed"] for row in rows),
        "tasks": rows,
    }


def _readme(result: Mapping[str, Any]) -> str:
    lines = [
        "# Paper 8.5 synthetic mechanism diagnostics",
        "",
        "These tasks isolate rule activation and observation materialization. They are",
        "structural diagnostics, not model-quality, SWE-bench, or runtime evidence.",
        "",
        "| Task | Probe | Mechanism gate |",
        "|---|---:|---:|",
    ]
    for row in result["tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['probe_decision']} | "
            f"{'PASS' if row['mechanism_gate_passed'] else 'FAIL'} |"
        )
    lines.extend((
        "",
        "The structured-evidence materializer uses current task/query terms, command",
        "semantics, paths, tracebacks, failure lines, diff headers/hunks, and source",
        "symbols. If it finds no positive evidence, it retains the entire observation",
        "rather than falling back to arbitrary head/tail sampling.",
        "",
        "Each task directory contains a complete mini-swe-agent-shaped trajectory that",
        "can be passed to `run_frozen_replay.py` for model-facing qualification.",
        "The separately reduced model-facing smoke is documented in `MODEL_SMOKE.md`.",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    tasks = synthetic_agent_tasks()
    for task in tasks:
        task_directory = output / task.task_id
        task_directory.mkdir(parents=True, exist_ok=True)
        (task_directory / "trajectory.json").write_text(
            json.dumps(task.trajectory(), indent=2) + "\n",
            encoding="utf-8",
        )
    result = run_synthetic_diagnostics(tasks=tasks)
    (output / "structural_diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_readme(result), encoding="utf-8")
    print(json.dumps({
        "output_directory": str(output),
        "task_count": result["task_count"],
        "all_mechanism_gates_passed": result["all_mechanism_gates_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
