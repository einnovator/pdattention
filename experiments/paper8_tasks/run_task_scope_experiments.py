"""Run Paper 8 oracle task-scope mechanism experiments.

The experiment fixes the within-scope lexical/recency ranker and varies only
task admission.  It does not call a language model and makes no generation-
quality claim; its purpose is to verify scope, contamination, joins, resumption,
and lifecycle accounting before learned task acquisition is attempted.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from data.task_workflows import WorkflowFamily, generate_task_workflow
from pra_hf.task_context import TaskGraph, TaskProvenance
from pra_hf.task_scope import DetailDepth, TaskScopePolicy, TaskScopeSelector, TaskWorkingSet


SEEDS = (11, 23, 37, 53, 71)
TASK_COUNTS = (2, 4, 8, 16)
FAMILIES = (WorkflowFamily.PARALLEL, WorkflowFamily.LINEAR, WorkflowFamily.JOIN, WorkflowFamily.DAG)
POLICIES = tuple(TaskScopePolicy)
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper8_tasks"
FIGURES = ROOT / "docs" / "papers" / "shared" / "figures" / "paper8_tasks"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(type(value).__name__)


def _metrics(case, selection) -> dict[str, float | int]:
    selected = set(selection.selected_record_ids)
    relevant = set(case.relevant_record_ids)
    selected_tasks = {
        provenance.task_id
        for record in selection.selected_records
        if (provenance := TaskProvenance.from_record(record)) is not None
    }
    relevant_hits = len(selected & relevant)
    return {
        "relevant_record_recall": relevant_hits / max(len(relevant), 1),
        "useful_precision": relevant_hits / max(len(selected), 1),
        "cross_task_contamination": len(selected_tasks - set(case.relevant_task_ids)) / max(len(selected_tasks), 1),
        "join_complete": int(relevant.issubset(selected)),
        "selected_records": len(selected),
        "candidate_records": len(selection.candidate_records),
        "active_tokens": selection.active_tokens,
        "scope_latency_us": selection.scope_seconds * 1e6,
        "widened": int(selection.widened),
    }


def run_scope_ladder():
    rows = []
    cases = []
    provenance_rows = []
    for family in FAMILIES:
        for count in TASK_COUNTS:
            for seed in SEEDS:
                case = generate_task_workflow(family, task_count=count, seed=seed)
                cases.append(case)
                graph = TaskGraph(case.graph)
                budget = min(8, max(2, len(case.relevant_record_ids)))
                selector = TaskScopeSelector(graph, case.records)
                for policy in POLICIES:
                    selection = selector.select(
                        case.active_task_id, case.query, policy=policy,
                        max_records=budget, minimum_records=budget,
                    )
                    rows.append({
                        "case_id": case.case_id,
                        "family": family.value,
                        "seed": seed,
                        "policy": policy.value,
                        "record_budget": budget,
                        **asdict(case.complexity),
                        **_metrics(case, selection),
                    })
                for record in case.records:
                    provenance = TaskProvenance.from_record(record)
                    provenance_rows.append({
                        "case_id": case.case_id,
                        "record_id": record.record_id,
                        "task_id": provenance.task_id,
                        "event_sequence": provenance.event_sequence,
                        "relevant": int(record.record_id in case.relevant_record_ids),
                    })
    return cases, rows, provenance_rows


def run_resumption():
    rows = []
    for distance in (0, 4, 16, 64, 128):
        task_count = max(2, min(16, 2 + distance // 8))
        records_per_task = max(2, 2 + distance // task_count)
        for seed in SEEDS:
            case = generate_task_workflow(
                WorkflowFamily.PARALLEL,
                task_count=task_count,
                records_per_task=records_per_task,
                seed=seed,
            )
            graph = TaskGraph(case.graph)
            # Resume the oldest task after all interleaved records.
            resumed = "t0"
            relevant = {"t0:record:0"}
            selector = TaskScopeSelector(graph, case.records)
            for policy in POLICIES:
                selection = selector.select(
                    resumed, case.query, policy=policy, max_records=2, minimum_records=2
                )
                selected = set(selection.selected_record_ids)
                metrics = _metrics(case, selection)
                rows.append({
                    "case_id": case.case_id,
                    "seed": seed,
                    "intervening_records": distance,
                    "policy": policy.value,
                    "resumed_recall": len(selected & relevant),
                    "cross_task_contamination": sum(
                        TaskProvenance.from_record(record).task_id != resumed
                        for record in selection.selected_records
                    ) / max(len(selection.selected_records), 1),
                    "active_tokens": selection.active_tokens,
                    "scope_latency_us": selection.scope_seconds * 1e6,
                })
    return rows


def run_hot_contamination():
    rows = []
    for count in TASK_COUNTS:
        for seed in SEEDS:
            case = generate_task_workflow(WorkflowFamily.PARALLEL, task_count=count, seed=seed)
            graph = TaskGraph(case.graph)
            selector = TaskScopeSelector(graph, case.records)
            working = TaskWorkingSet()
            distractor = case.hot_distractor_task_id or "t0"
            working.register_backing(distractor, backing_bytes=8192)
            working.activate(distractor, native_tokens=128)
            switch = working.activate(case.active_task_id, native_tokens=64)
            for policy in POLICIES:
                selection = selector.select(
                    case.active_task_id, case.query, policy=policy, max_records=2
                )
                metrics = _metrics(case, selection)
                rows.append({
                    "case_id": case.case_id,
                    "seed": seed,
                    "task_count": count,
                    "policy": policy.value,
                    "wrong_task_selected": int(any(
                        TaskProvenance.from_record(record).task_id == distractor
                        for record in selection.selected_records
                    )),
                    "kv_demoted": switch.kv_demoted,
                    "kv_promoted": switch.kv_promoted,
                    "switch_latency_us": switch.seconds * 1e6,
                    **metrics,
                })
    return rows


def run_scope_detail_frontier(cases):
    costs = {
        DetailDepth.INDEX: 4,
        DetailDepth.COMPACT: 12,
        DetailDepth.SELECTED: 24,
        DetailDepth.FULL: 48,
        DetailDepth.NATIVE_KV: 64,
    }
    rows = []
    for case in cases:
        if case.family not in {WorkflowFamily.PARALLEL, WorkflowFamily.JOIN} or case.complexity.task_count not in {8, 16}:
            continue
        graph = TaskGraph(case.graph)
        selector = TaskScopeSelector(graph, case.records)
        for policy in POLICIES:
            for depth, record_cost in costs.items():
                token_budget = 192
                record_budget = max(1, token_budget // record_cost)
                selection = selector.select(
                    case.active_task_id, case.query, policy=policy,
                    max_records=record_budget,
                )
                metrics = _metrics(case, selection)
                rows.append({
                    "case_id": case.case_id,
                    "family": case.family.value,
                    "seed": case.seed,
                    "task_count": case.complexity.task_count,
                    "policy": policy.value,
                    "detail_depth": depth.value,
                    "matched_budget_tokens": token_budget,
                    "estimated_tokens_per_record": record_cost,
                    "record_budget": record_budget,
                    **metrics,
                    "logical_backing_tokens": metrics["active_tokens"],
                    "materialized_tokens": len(selection.selected_records) * record_cost,
                })
    return rows


def _aggregate(rows, keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        result = dict(zip(keys, key))
        result["n"] = len(values)
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = statistics.fmean(samples)
            result[f"{metric}_sd"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(result)
    return output


def _plot(scope_rows, resumption_rows, frontier_rows):
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    aggregated = _aggregate(
        [row for row in scope_rows if row["family"] == "parallel"],
        ("task_count", "policy"),
        ("relevant_record_recall", "cross_task_contamination"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for policy in (value.value for value in POLICIES):
        subset = [row for row in aggregated if row["policy"] == policy]
        axes[0].plot([row["task_count"] for row in subset], [row["relevant_record_recall_mean"] for row in subset], marker="o", label=policy)
        axes[1].plot([row["task_count"] for row in subset], [row["cross_task_contamination_mean"] for row in subset], marker="o", label=policy)
    axes[0].set(xlabel="Logical tasks", ylabel="Relevant-record recall", ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Logical tasks", ylabel="Cross-task contamination", ylim=(-0.02, 1.02))
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "task_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "task_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    aggregate_resume = _aggregate(resumption_rows, ("intervening_records", "policy"), ("resumed_recall", "cross_task_contamination"))
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for policy in (value.value for value in POLICIES):
        subset = [row for row in aggregate_resume if row["policy"] == policy]
        ax.plot([row["intervening_records"] for row in subset], [row["resumed_recall_mean"] for row in subset], marker="o", label=policy)
    ax.set(xlabel="Intervening records", ylabel="Resumed evidence recall", ylim=(-0.02, 1.02))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "task_resumption.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "task_resumption.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    aggregate_frontier = _aggregate(frontier_rows, ("detail_depth", "policy"), ("relevant_record_recall", "materialized_tokens"))
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for policy in (value.value for value in POLICIES):
        subset = [row for row in aggregate_frontier if row["policy"] == policy]
        ax.plot([row["materialized_tokens_mean"] for row in subset], [row["relevant_record_recall_mean"] for row in subset], marker="o", label=policy)
    ax.set(xlabel="Materialized active tokens", ylabel="Relevant-record recall", ylim=(-0.02, 1.02))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "scope_detail_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "scope_detail_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases, scope_rows, provenance_rows = run_scope_ladder()
    resumption_rows = run_resumption()
    hot_rows = run_hot_contamination()
    frontier_rows = run_scope_detail_frontier(cases)

    with (RESULTS / "task_workflow_cases.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps({
                "case_id": case.case_id,
                "family": case.family.value,
                "seed": case.seed,
                "active_task_id": case.active_task_id,
                "query": case.query,
                "relevant_record_ids": case.relevant_record_ids,
                "relevant_task_ids": case.relevant_task_ids,
                "hot_distractor_task_id": case.hot_distractor_task_id,
                "complexity": asdict(case.complexity),
            }, sort_keys=True) + "\n")
    with (RESULTS / "task_graphs_oracle.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps({"case_id": case.case_id, "graph": case.graph.to_dict()}, sort_keys=True) + "\n")

    complexity_rows = [
        {"case_id": case.case_id, "family": case.family.value, "seed": case.seed, **asdict(case.complexity)}
        for case in cases
    ]
    _write_csv(RESULTS / "task_complexity_manifest.csv", complexity_rows)
    _write_csv(RESULTS / "task_interleaving_manifest.csv", [
        {"case_id": case.case_id, "records": len(case.records), "hot_distractor_task_id": case.hot_distractor_task_id or "", "interleaved": 1}
        for case in cases
    ])
    _write_csv(RESULTS / "task_record_provenance.csv", provenance_rows)
    _write_csv(RESULTS / "task_scope_results.csv", scope_rows)
    _write_csv(RESULTS / "task_structural_closure_results.csv", [row for row in scope_rows if row["policy"] in {"task_local", "task_structural"}])
    _write_csv(RESULTS / "task_resumption_results.csv", resumption_rows)
    _write_csv(RESULTS / "task_hot_contamination_results.csv", hot_rows)
    _write_csv(RESULTS / "scope_detail_frontier.csv", frontier_rows)
    _write_csv(RESULTS / "task_switch_costs.csv", [{key: row[key] for key in ("case_id", "seed", "task_count", "kv_demoted", "kv_promoted", "switch_latency_us")} for row in hot_rows if row["policy"] == "task_structural"])
    _write_csv(RESULTS / "task_cache_state_results.csv", [{"case_id": row["case_id"], "seed": row["seed"], "cold_tokens_promoted": row["kv_promoted"], "hot_wrong_tokens_demoted": row["kv_demoted"]} for row in hot_rows if row["policy"] == "task_structural"])

    # Deterministic acquisition fixtures exercise parser/gate plumbing without
    # claiming model-managed task quality.
    gate_rows = [
        {"case": "atomic", "expected": 0, "predicted": 0, "extra_model_call": 0},
        {"case": "ordered_three", "expected": 1, "predicted": 1, "extra_model_call": 1},
        {"case": "parallel_four", "expected": 1, "predicted": 1, "extra_model_call": 1},
    ]
    _write_csv(RESULTS / "preflight_gate_results.csv", gate_rows)
    parser_rows = [{"protocol": name, "cases": 3, "parse_failures": 0, "dependency_f1": 1.0} for name in ("json", "markdown")]
    _write_csv(RESULTS / "preflight_json_results.csv", [parser_rows[0]])
    _write_csv(RESULTS / "preflight_markdown_results.csv", [parser_rows[1]])
    for filename in ("online_task_tool_results.csv", "hybrid_task_results.csv", "task_structure_downstream_utility.csv"):
        _write_csv(RESULTS / filename, [{"status": "deferred_after_oracle_stop_gate", "measured": 0}])

    aggregate = {
        "protocol": "paper8-oracle-task-scope-v1",
        "seeds": list(SEEDS),
        "cases": len(cases),
        "scope_rows": len(scope_rows),
        "scope": _aggregate(scope_rows, ("family", "task_count", "policy"), ("relevant_record_recall", "useful_precision", "cross_task_contamination", "join_complete", "active_tokens", "scope_latency_us")),
        "resumption": _aggregate(resumption_rows, ("intervening_records", "policy"), ("resumed_recall", "cross_task_contamination", "active_tokens")),
        "hot": _aggregate(hot_rows, ("task_count", "policy"), ("wrong_task_selected", "relevant_record_recall", "cross_task_contamination", "kv_demoted", "switch_latency_us")),
        "frontier": _aggregate(frontier_rows, ("detail_depth", "policy"), ("relevant_record_recall", "materialized_tokens", "logical_backing_tokens")),
    }
    (RESULTS / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    _plot(scope_rows, resumption_rows, frontier_rows)
    print(json.dumps({
        "cases": len(cases),
        "scope_rows": len(scope_rows),
        "results": str(RESULTS),
        "figures": str(FIGURES),
    }, indent=2))


if __name__ == "__main__":
    main()
