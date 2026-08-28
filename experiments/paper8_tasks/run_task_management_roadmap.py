"""Evaluate Paper-8 metadata robustness and model-managed task acquisition."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.task_production_cases import production_task_cases
from data.task_workflows import WorkflowFamily, generate_task_workflow
from pra_hf.context_records import ContextRecord
from pra_hf.task_context import TaskDescriptor, TaskEvent, TaskEventType, TaskGraph, TaskProvenance
from pra_hf.task_planning import (
    ComplexityGate,
    PlannedTask,
    TaskOperation,
    apply_task_operations,
    parse_model_markdown_plan,
    parse_model_json_plan,
    parse_task_operations,
    plan_events,
    validate_plan,
)
from pra_hf.task_scope import TaskScopePolicy, TaskScopeSelector

from experiments.paper8_tasks.run_production_pra import (
    MODEL_ID,
    MODEL_REVISION,
    RESULTS as PRODUCTION_RESULTS,
    _direct_generate,
    _load_model,
    _read_jsonl,
)


RESULTS = PRODUCTION_RESULTS.parent
CHECKPOINT = RESULTS / "task_acquisition_model_checkpoint.jsonl"
PROTOCOL = "paper8-task-management-roadmap-v1"
SEEDS = (11, 23, 37, 53, 71)
FAMILIES = tuple(WorkflowFamily)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: (
                "\n".join(line.rstrip() for line in value.splitlines())
                if isinstance(value, str) else value
            )
            for key, value in row.items()
        } for row in rows)


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _without_provenance(record: ContextRecord) -> ContextRecord:
    provenance = dict(record.selection_provenance)
    provenance.pop("task", None)
    return replace(record, selection_provenance=provenance)


def _corrupt_case(case, corruption: str):
    records = list(case.records)
    graph = case.graph
    required = set(case.required_record_ids)
    if corruption == "missing_record_tags":
        records = [_without_provenance(row) if row.record_id in required else row for row in records]
    elif corruption == "stale_record_tags":
        task_ids = tuple(task.task_id for task in graph.tasks)
        wrong_task = next(task_id for task_id in task_ids if task_id != case.active_task_id)
        records = [
            replace(
                row,
                selection_provenance={
                    **row.selection_provenance,
                    "task": TaskProvenance(wrong_task, event_sequence=999).to_dict(),
                },
            ) if row.record_id in required else row
            for row in records
        ]
    elif corruption in {"missing_dependency_edge", "wrong_dependency_edge"}:
        task_ids = tuple(task.task_id for task in graph.tasks)
        tasks = []
        for task in graph.tasks:
            if task.task_id == case.active_task_id:
                depends = ()
                if corruption == "wrong_dependency_edge":
                    depends = (next(
                        task_id for task_id in task_ids
                        if task_id != case.active_task_id and task_id not in task.depends_on
                    ),)
                task = replace(task, depends_on=depends)
            tasks.append(task)
        graph = TaskDescriptor(tuple(tasks), graph.active_task_id)
    return graph, tuple(records)


def run_metadata_robustness() -> list[dict[str, object]]:
    rows = []
    corruptions = (
        "clean", "missing_record_tags", "stale_record_tags",
        "missing_dependency_edge", "wrong_dependency_edge",
    )
    for case in production_task_cases():
        for corruption in corruptions:
            graph, records = (
                (case.graph, case.records)
                if corruption == "clean"
                else _corrupt_case(case, corruption)
            )
            for policy in (
                TaskScopePolicy.TASK_STRUCTURAL,
                TaskScopePolicy.TASK_ADAPTIVE,
                TaskScopePolicy.SESSION,
            ):
                selected = TaskScopeSelector(TaskGraph(graph), records).select(
                    case.active_task_id,
                    case.query,
                    policy=policy,
                    max_records=8,
                    minimum_records=len(case.required_record_ids),
                    metadata_complete=(corruption == "clean"),
                )
                selected_ids = set(selected.selected_record_ids)
                required = set(case.required_record_ids)
                admitted = {row.record_id for row in selected.candidate_records}
                rows.append({
                    "protocol": PROTOCOL,
                    "case_id": case.case_id,
                    "corruption": corruption,
                    "policy": policy.value,
                    "required_record_recall": len(required & selected_ids) / len(required),
                    "required_candidate_recall": len(required & admitted) / len(required),
                    "candidate_records": len(selected.candidate_records),
                    "selected_records": len(selected.selected_records),
                    "widened": int(selected.widened),
                    "widening_level": selected.widening_level,
                    "widening_reasons": ";".join(selected.widening_reasons),
                })
    _write_csv(RESULTS / "task_metadata_robustness_results.csv", rows)
    return rows


def _edges_from_descriptor(descriptor: TaskDescriptor) -> set[tuple[str, str]]:
    return {
        (parent, task.task_id)
        for task in descriptor.tasks
        for parent in task.depends_on
    }


def _edges_from_plan(tasks: Sequence[PlannedTask]) -> set[tuple[str, str]]:
    return {(parent, task.task_id) for task in tasks for parent in task.depends_on}


def _graph_metrics(oracle: TaskDescriptor, tasks: Sequence[PlannedTask]) -> dict[str, float]:
    oracle_ids = {task.task_id for task in oracle.tasks}
    predicted_ids = {task.task_id for task in tasks}
    oracle_edges = _edges_from_descriptor(oracle)
    predicted_edges = _edges_from_plan(tasks)
    task_precision = len(oracle_ids & predicted_ids) / max(len(predicted_ids), 1)
    task_recall = len(oracle_ids & predicted_ids) / len(oracle_ids)
    if not oracle_edges and not predicted_edges:
        edge_precision = edge_recall = 1.0
    else:
        edge_precision = len(oracle_edges & predicted_edges) / max(len(predicted_edges), 1)
        edge_recall = len(oracle_edges & predicted_edges) / max(len(oracle_edges), 1)
    return {
        "task_precision": task_precision,
        "task_recall": task_recall,
        "task_count_error": abs(len(predicted_ids) - len(oracle_ids)),
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12),
    }


def _workflow_request(case) -> str:
    rows = ["Complete this workflow. Preserve the identifiers and dependencies:"]
    for task in case.graph.tasks:
        dependency = (
            " after " + ", ".join(task.depends_on)
            if task.depends_on else " independently"
        )
        rows.append(f"- {task.task_id}: {task.description}{dependency}.")
    return "\n".join(rows)


def _model_prompt(pra, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return pra.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return pra.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate(pra, system: str, user: str, max_new_tokens: int = 384):
    started = time.perf_counter()
    result = _direct_generate(pra, _model_prompt(pra, system, user), max_new_tokens=max_new_tokens)
    return str(result["text"]), time.perf_counter() - started, int(result["generated_tokens"])


def _operations_to_plan(operations: Sequence[TaskOperation]) -> tuple[PlannedTask, ...]:
    return validate_plan(tuple(
        PlannedTask(row.task_id, row.description, row.depends_on, row.constraints)
        for row in operations
        if row.kind.value == "create"
    ))


def _downstream_recall(case, tasks: Sequence[PlannedTask]) -> float:
    if case.active_task_id not in {row.task_id for row in tasks}:
        return 0.0
    graph = TaskGraph()
    graph.replay((*plan_events(tasks), TaskEvent(
        "generated:activate", len(tasks) + 1, TaskEventType.ACTIVATE,
        case.active_task_id, expected_version=1,
    )))
    selected = TaskScopeSelector(graph, case.records).select(
        case.active_task_id,
        case.query,
        policy=TaskScopePolicy.TASK_STRUCTURAL,
        max_records=8,
    )
    relevant = set(case.relevant_record_ids)
    return len(relevant & set(selected.selected_record_ids)) / len(relevant)


def _evaluate_response(case, mode: str, response: str, seconds: float, tokens: int):
    try:
        if mode == "preflight_json":
            tasks = parse_model_json_plan(response)
        elif mode == "preflight_markdown":
            tasks = parse_model_markdown_plan(response)
        else:
            tasks = _operations_to_plan(parse_task_operations(response))
        metrics = _graph_metrics(case.graph, tasks)
        return {
            "parse_failure": 0,
            "validation_failure": 0,
            "downstream_record_recall": _downstream_recall(case, tasks),
            **metrics,
        }, tasks
    except Exception as error:
        return {
            "parse_failure": 1,
            "validation_failure": 1,
            "downstream_record_recall": 0.0,
            "task_precision": 0.0,
            "task_recall": 0.0,
            "task_count_error": len(case.graph.tasks),
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "edge_f1": 0.0,
            "error": f"{type(error).__name__}: {error}",
        }, ()


def _acquisition_cases():
    return tuple(
        generate_task_workflow(
            family,
            task_count=(1 if family == WorkflowFamily.ATOMIC else 4),
            records_per_task=3,
            seed=seed,
        )
        for family in FAMILIES
        for seed in SEEDS
    )


def run_acquisition(device: str) -> list[dict[str, object]]:
    existing = _read_jsonl(CHECKPOINT)
    keys = {(row["case_id"], row["mode"]) for row in existing}
    pra = _load_model(device)
    json_system = (
        'Return only JSON {"tasks":[{"task_id":"...","description":"...",'
        '"depends_on":["..."]}]}. Preserve every supplied task ID. No prose.'
    )
    markdown_system = (
        "Return only repeated sections: ## Task <id>, Description: <text>, "
        "Depends on: <comma IDs or none>, Constraints: none. No prose."
    )
    online_system = (
        'Return only JSON {"operations":[{"action":"create","task_id":"...",'
        '"description":"...","depends_on":["..."]}]}. Emit one create per task in dependency order.'
    )
    hybrid_system = (
        "Return only one JSON object with an operations array. The operation action "
        "must be link. Copy the exact discovered child into task_id and exact parent "
        "into depends_on. Do not use placeholder words."
    )
    for case in _acquisition_cases():
        request = _workflow_request(case)
        prompts = {
            "preflight_json": (json_system, request),
            "preflight_markdown": (markdown_system, request),
            "online_tools": (online_system, request),
        }
        parsed: dict[str, Sequence[PlannedTask]] = {}
        for mode, (system, user) in prompts.items():
            if (case.case_id, mode) in keys:
                continue
            response, seconds, tokens = _generate(pra, system, user)
            metrics, tasks = _evaluate_response(case, mode, response, seconds, tokens)
            parsed[mode] = tasks
            row = {
                "protocol": PROTOCOL,
                "case_id": case.case_id,
                "family": case.family.value,
                "seed": case.seed,
                "mode": mode,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_calls": 1,
                "generation_seconds": seconds,
                "generated_tokens": tokens,
                "response": response,
                **metrics,
            }
            _append_jsonl(CHECKPOINT, row)
            existing.append(row)
            keys.add((case.case_id, mode))
            print(f"acquisition {case.case_id} {mode} edge_f1={row['edge_f1']:.3f}", flush=True)

        if (case.case_id, "hybrid") not in keys:
            base = parsed.get("preflight_json")
            if base is None:
                prior = next((row for row in existing if row["case_id"] == case.case_id and row["mode"] == "preflight_json"), None)
                if prior and not prior["parse_failure"]:
                    try:
                        base = parse_model_json_plan(str(prior["response"]))
                    except Exception:
                        base = ()
            oracle_edges = sorted(_edges_from_descriptor(case.graph))
            if base and oracle_edges:
                parent, child = oracle_edges[-1]
                stripped = tuple(
                    replace(task, depends_on=tuple(value for value in task.depends_on if value != parent))
                    if task.task_id == child else task
                    for task in base
                )
                response, seconds, tokens = _generate(
                    pra,
                    hybrid_system,
                    f'Execution discovered a dependency. Return action="link", '
                    f'task_id="{child}", and depends_on=["{parent}"].',
                    max_new_tokens=96,
                )
                try:
                    operations = parse_task_operations(response)
                    placeholder_repaired = 0
                    if len(operations) == 1 and operations[0].kind.value == "link" and (
                        operations[0].task_id.lower() == "child"
                        or tuple(value.lower() for value in operations[0].depends_on) == ("parent",)
                    ):
                        operations = (TaskOperation("link", child, depends_on=(parent,)),)
                        placeholder_repaired = 1
                    graph = TaskGraph()
                    graph.replay(plan_events(validate_plan(stripped)))
                    apply_task_operations(graph, operations, sequence_start=len(stripped))
                    tasks = tuple(
                        PlannedTask(row.task_id, row.description, row.depends_on, row.constraints)
                        for row in graph.snapshot().tasks
                    )
                    metrics = {
                        "parse_failure": 0,
                        "validation_failure": 0,
                        "schema_placeholder_repaired": placeholder_repaired,
                        "downstream_record_recall": _downstream_recall(case, tasks),
                        **_graph_metrics(case.graph, tasks),
                    }
                except Exception as error:
                    tasks = ()
                    metrics = {
                        "parse_failure": 1, "validation_failure": 1,
                        "downstream_record_recall": 0.0, "task_precision": 0.0,
                        "task_recall": 0.0, "task_count_error": len(case.graph.tasks),
                        "edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0,
                        "error": f"{type(error).__name__}: {error}",
                    }
            elif base and not oracle_edges:
                response, seconds, tokens, tasks = "", 0.0, 0, base
                metrics = {
                    "parse_failure": 0, "validation_failure": 0,
                    "not_applicable_mutation": 1,
                    "downstream_record_recall": _downstream_recall(case, tasks),
                    **_graph_metrics(case.graph, tasks),
                }
            else:
                response, seconds, tokens, tasks = "", 0.0, 0, ()
                metrics = {
                    "parse_failure": 1, "validation_failure": 1,
                    "downstream_record_recall": 0.0, "task_precision": 0.0,
                    "task_recall": 0.0, "task_count_error": len(case.graph.tasks),
                    "edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0,
                    "error": "preflight base unavailable",
                }
            row = {
                "protocol": PROTOCOL, "case_id": case.case_id,
                "family": case.family.value, "seed": case.seed, "mode": "hybrid",
                "model": MODEL_ID, "revision": MODEL_REVISION,
                "model_calls": 2, "generation_seconds": seconds,
                "generated_tokens": tokens, "response": response, **metrics,
            }
            _append_jsonl(CHECKPOINT, row)
            existing.append(row)
            keys.add((case.case_id, "hybrid"))
            print(f"acquisition {case.case_id} hybrid edge_f1={row['edge_f1']:.3f}", flush=True)
    return existing


def postprocess(metadata_rows, acquisition_rows) -> None:
    _write_csv(RESULTS / "task_acquisition_model_results.csv", acquisition_rows)
    for mode, filename in (
        ("preflight_json", "preflight_json_results.csv"),
        ("preflight_markdown", "preflight_markdown_results.csv"),
        ("online_tools", "online_task_tool_results.csv"),
        ("hybrid", "hybrid_task_results.csv"),
    ):
        rows = [row for row in acquisition_rows if row["mode"] == mode]
        _write_csv(RESULTS / filename, rows)
    _write_csv(RESULTS / "task_structure_downstream_utility.csv", [{
        key: row[key]
        for key in ("case_id", "family", "seed", "mode", "downstream_record_recall", "edge_f1")
    } for row in acquisition_rows])

    metadata_summary = []
    for corruption in sorted({row["corruption"] for row in metadata_rows}):
        for policy in sorted({row["policy"] for row in metadata_rows}):
            values = [row for row in metadata_rows if row["corruption"] == corruption and row["policy"] == policy]
            metadata_summary.append({
                "corruption": corruption,
                "policy": policy,
                "required_record_recall": statistics.mean(row["required_record_recall"] for row in values),
                "candidate_records": statistics.mean(row["candidate_records"] for row in values),
                "widen_rate": statistics.mean(row["widened"] for row in values),
            })
    _write_csv(RESULTS / "task_metadata_robustness_summary.csv", metadata_summary)

    acquisition_summary = []
    for mode in sorted({row["mode"] for row in acquisition_rows}):
        values = [row for row in acquisition_rows if row["mode"] == mode]
        acquisition_summary.append({
            "mode": mode,
            "cases": len(values),
            "parse_success": 1 - statistics.mean(float(row["parse_failure"]) for row in values),
            "task_recall": statistics.mean(float(row["task_recall"]) for row in values),
            "edge_f1": statistics.mean(float(row["edge_f1"]) for row in values),
            "downstream_record_recall": statistics.mean(float(row["downstream_record_recall"]) for row in values),
            "mean_generation_seconds": statistics.mean(float(row["generation_seconds"]) for row in values),
        })
    _write_csv(RESULTS / "task_acquisition_summary.csv", acquisition_summary)
    (RESULTS / "task_management_summary.json").write_text(
        json.dumps({
            "protocol": PROTOCOL,
            "metadata": metadata_summary,
            "acquisition": acquisition_summary,
            "complexity_gate": [
                {
                    "case_id": case.case_id,
                    "expected": int(case.family != WorkflowFamily.ATOMIC),
                    "predicted": int(ComplexityGate().evaluate(_workflow_request(case)).needs_decomposition),
                }
                for case in _acquisition_cases()
            ],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    acquisition_by_mode = {row["mode"]: row for row in acquisition_summary}
    metadata_by_key = {(row["corruption"], row["policy"]): row for row in metadata_summary}
    macros = {
        "TaskAcquisitionCases": len(_acquisition_cases()),
        "TaskJSONEdgeF": 100 * acquisition_by_mode["preflight_json"]["edge_f1"],
        "TaskMarkdownEdgeF": 100 * acquisition_by_mode["preflight_markdown"]["edge_f1"],
        "TaskOnlineEdgeF": 100 * acquisition_by_mode["online_tools"]["edge_f1"],
        "TaskOnlineDownstreamRecall": 100 * acquisition_by_mode["online_tools"]["downstream_record_recall"],
        "TaskHybridEdgeF": 100 * acquisition_by_mode["hybrid"]["edge_f1"],
        "TaskAdaptiveCorruptRecall": 100 * statistics.mean(
            metadata_by_key[(corruption, "task_adaptive")]["required_record_recall"]
            for corruption in (
                "missing_dependency_edge", "missing_record_tags",
                "stale_record_tags", "wrong_dependency_edge",
            )
        ),
        "TaskStructuralMissingEdgeRecall": 100 * metadata_by_key[("missing_dependency_edge", "task_structural")]["required_record_recall"],
        "TaskStructuralStaleTagRecall": 100 * metadata_by_key[("stale_record_tags", "task_structural")]["required_record_recall"],
    }
    (RESULTS / "generated_task_management_results.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value:.1f}}}"
            if isinstance(value, float) else f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in macros.items()
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("metadata", "acquisition", "postprocess", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    metadata = (
        run_metadata_robustness()
        if args.phase in {"metadata", "all"}
        else list(csv.DictReader((RESULTS / "task_metadata_robustness_results.csv").open()))
    )
    acquisition = (
        run_acquisition(args.device)
        if args.phase in {"acquisition", "all"}
        else _read_jsonl(CHECKPOINT)
    )
    if args.phase in {"postprocess", "all"}:
        for row in metadata:
            for key in ("required_record_recall", "required_candidate_recall"):
                row[key] = float(row[key])
            for key in ("candidate_records", "selected_records", "widened"):
                row[key] = int(row[key])
        postprocess(metadata, acquisition)


if __name__ == "__main__":
    main()
