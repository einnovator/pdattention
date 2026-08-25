"""Run Paper 6.5 M5 speculative capability-disclosure experiments.

The deterministic portion evaluates all required P0--P9 policies with graded
relevance and graph provenance. The pretrained portion adds only P6/P8 static
graph disclosures and reuses frozen M4 direct, reactive, and oracle traces.
This preserves a matched comparison rather than regenerating convenient
baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks
from experiments.paper6_5_tools.run_m2_m4_pretrained import (
    MODEL_ID,
    MODEL_REVISION,
    SEEDS,
    FrozenToolModel,
    _prompt_variant,
    _write_csv,
    call_matches,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.agent_disclosure import (
    DisclosureMode,
    DisclosureProvenance,
    DisclosureTrace,
    ToolCapabilityGraph,
    ToolDisclosurePolicy,
)
from pra_hf.agent_execution import ExecutionAuthorization, parse_tool_call, resource_tool_schema
from pra_hf.agent_resources import DiscoveryRequest, PersistentResourceIndex


POLICIES = (
    "p0_all_eager",
    "p1_direct_top1",
    "p2_direct_topk",
    "p3_family_category",
    "p4_tag_keyword",
    "p5_schema_graph",
    "p6_combined_graph",
    "p7_reactive_jit",
    "p8_speculative_planning",
    "p9_oracle_capabilities",
)


def _trace_for_uris(policy: str, roots, uris, *, source: str) -> DisclosureTrace:
    return DisclosureTrace(
        policy,
        policy,
        tuple(roots),
        tuple(uris),
        tuple(
            DisclosureProvenance(uri, source, roots[0] if roots else None, direct_rank=index)
            for index, uri in enumerate(uris, start=1)
        ),
        0,
        0,
        0,
    )


def _retrieval_topk(index: PersistentResourceIndex, query: str, k: int) -> tuple[str, ...]:
    request = DiscoveryRequest(query=query, tenant_id="paper6_5", top_k=k)
    rows = index.score(request, channels=("hybrid",))
    return tuple(row.uri for row in sorted(rows, key=lambda row: (-row.hybrid, row.uri))[:k])


def policy_trace(
    policy: str,
    graph: ToolCapabilityGraph,
    index: PersistentResourceIndex,
    task,
) -> tuple[DisclosureTrace, tuple[str, ...]]:
    """Return initial disclosure and total task-level disclosure for P0--P9."""

    by_name = {resource.name: resource for resource in graph.resources}
    roots = (by_name[task.required_tools[0]].uri,)
    required = tuple(by_name[name].uri for name in task.required_tools)
    horizon = len(required)
    if policy == "p0_all_eager":
        uris = tuple(resource.uri for resource in graph.resources)
        return _trace_for_uris(policy, roots, uris, source="all_eager"), uris
    if policy == "p1_direct_top1":
        return _trace_for_uris(policy, roots, roots, source="direct"), roots
    if policy == "p2_direct_topk":
        uris = _retrieval_topk(index, task.query, horizon)
        return _trace_for_uris(policy, roots, uris, source="retrieval_topk"), uris
    if policy == "p3_family_category":
        trace = graph.disclose(roots, ToolDisclosurePolicy("local", max_tools=horizon, family_k=max(horizon - 1, 0), tag_k=0, schema_successor_k=0, schema_predecessor_k=0, schema_depth=0))
        return trace, trace.disclosed_uris
    if policy == "p4_tag_keyword":
        trace = graph.disclose(roots, ToolDisclosurePolicy("local", max_tools=horizon, family_k=0, tag_k=max(horizon - 1, 0), schema_successor_k=0, schema_predecessor_k=0, schema_depth=0))
        return trace, trace.disclosed_uris
    if policy == "p5_schema_graph":
        trace = graph.disclose(roots, ToolDisclosurePolicy("planning", max_tools=horizon, family_k=0, tag_k=0, schema_successor_k=horizon * 2, schema_predecessor_k=horizon, schema_depth=4))
        return trace, trace.disclosed_uris
    if policy == "p6_combined_graph":
        trace = graph.disclose(roots, ToolDisclosurePolicy("planning", max_tools=10, family_k=3, tag_k=2, schema_successor_k=8, schema_predecessor_k=3, schema_depth=4))
        return trace, trace.disclosed_uris
    if policy == "p7_reactive_jit":
        return _trace_for_uris(policy, roots, roots, source="direct"), required
    if policy == "p8_speculative_planning":
        trace = graph.disclose(roots, ToolDisclosurePolicy("planning", max_tools=horizon, family_k=2, tag_k=1, schema_successor_k=horizon * 2, schema_predecessor_k=1, schema_depth=4))
        return trace, trace.disclosed_uris
    if policy == "p9_oracle_capabilities":
        return _trace_for_uris(policy, roots, required, source="oracle"), required
    raise ValueError(policy)


def _set_metrics(task, graph, trace: DisclosureTrace, total_uris: Sequence[str]) -> dict[str, object]:
    names = {graph.by_uri[uri].name for uri in trace.disclosed_uris}
    total_names = {graph.by_uri[uri].name for uri in total_uris}
    required = set(task.required_tools)
    useful = required | set(task.useful_tools)
    related = set(task.related_tools)
    unsafe = set(task.unsafe_tools)
    required_recall = len(names & required) / len(required)
    useful_recall = len(names & useful) / max(len(useful), 1)
    useful_precision = len(names & useful) / max(len(names), 1)
    irrelevant = names - useful - related
    return {
        "initial_disclosed_tools": len(names),
        "total_disclosed_tools": len(total_names),
        "required_recall_at_k": required_recall,
        "useful_recall_at_k": useful_recall,
        "useful_precision_at_k": useful_precision,
        "related_unnecessary_fraction": len(names & related) / max(len(names), 1),
        "irrelevant_tool_fraction": len(irrelevant) / max(len(names), 1),
        "unsafe_exposure_count": len(names & unsafe),
        "unsafe_exposure_fraction": len(names & unsafe) / max(len(names), 1),
        "plan_coverage": required_recall,
        "initial_capability_success": required <= names,
        "task_level_capability_success": required <= total_names,
    }


def deterministic_study(tokenizer) -> tuple[list[dict], dict[str, object]]:
    resources = realistic_tool_catalog()
    started = time.perf_counter()
    graph = ToolCapabilityGraph(resources)
    graph_seconds = time.perf_counter() - started
    index = PersistentResourceIndex(resources)
    definition_tokens = {
        resource.uri: len(tokenizer.encode(json.dumps(resource_tool_schema(resource))))
        for resource in resources
    }
    rows = []
    for task in (task for task in workflow_tasks() if len(task.steps) > 1):
        for policy in POLICIES:
            policy_started = time.perf_counter()
            trace, total = policy_trace(policy, graph, index, task)
            latency = time.perf_counter() - policy_started
            metrics = _set_metrics(task, graph, trace, total)
            rows.append({
                "task_id": task.task_id,
                "family": task.family,
                "plan_horizon": len(task.steps),
                "policy": policy,
                **metrics,
                "initial_definition_tokens": sum(definition_tokens[uri] for uri in trace.disclosed_uris),
                "total_definition_tokens": sum(definition_tokens[uri] for uri in total),
                "graph_expansions": trace.graph_expansions,
                "candidate_edges_considered": trace.candidate_edges_considered,
                "unsafe_suppressed": trace.unsafe_suppressed,
                "disclosure_seconds": latency,
                "disclosed_uris": " ".join(trace.disclosed_uris),
                "provenance": json.dumps([row.__dict__ for row in trace.provenance], sort_keys=True),
            })
    graph_meta = {
        "resources": len(resources),
        "edges": len(graph.edges),
        "density": graph.density,
        "construction_seconds": graph_seconds,
    }
    return rows, graph_meta


def _static_execution(model, graph, task, disclosed_uris, seed: int) -> dict[str, object]:
    disclosed = tuple(graph.by_uri[uri] for uri in disclosed_uris)
    executor = workflow_executor(graph.resources, task)
    messages: list[dict[str, str]] = [{"role": "user", "content": _prompt_variant(task.query, seed) + " Execute one step at a time."}]
    observations = []
    completed = 0
    definition_tokens = prompt_tokens = generated_tokens = 0
    seconds = 0.0
    failure = ""
    for index, step in enumerate(task.steps):
        text, cost = model.generate(messages, disclosed)
        call = parse_tool_call(text)
        _tool_ok, arguments_ok = call_matches(call, step)
        result = executor.execute(
            call,
            selected_uris=disclosed_uris,
            authorization=ExecutionAuthorization(frozenset(disclosed_uris), allow_writes=True),
            prior_observations=observations,
            call_id=f"m5-{task.task_id}-seed{seed}-step{index + 1}",
        )
        definition_tokens += int(cost["tool_definition_tokens"])
        prompt_tokens += int(cost["prompt_tokens"])
        generated_tokens += int(cost["generated_tokens"])
        seconds += float(cost["generation_seconds"])
        if not arguments_ok:
            failure = "wrong_call"
            break
        if not result.executed or result.observation is None:
            failure = result.reason
            break
        completed += 1
        observations.append(result.observation)
        messages.extend((
            {"role": "assistant", "content": text},
            {"role": "tool", "content": result.observation.content},
            {"role": "user", "content": "Continue the workflow with exactly one next tool call."},
        ))
    return {
        "completed_steps": completed,
        "task_success": completed == len(task.steps),
        "failure_reason": failure,
        "cumulative_definition_tokens": definition_tokens,
        "cumulative_prompt_tokens": prompt_tokens,
        "cumulative_generated_tokens": generated_tokens,
        "generation_seconds": seconds,
    }


def _load_m4_baselines(path: Path) -> dict[tuple[int, str, str], dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["milestone"] == "M4-summary"]
    names = {
        "no_refresh": "p1_direct_top1",
        "reactive_jit": "p7_reactive_jit",
        "eager_required": "p9_oracle_capabilities",
    }
    return {
        (int(row["seed"]), row["task_id"], names[row["condition"]]): row
        for row in rows
    }


def pretrained_study(model, seeds, m4_path: Path) -> list[dict]:
    resources = realistic_tool_catalog()
    graph = ToolCapabilityGraph(resources)
    index = PersistentResourceIndex(resources)
    baselines = _load_m4_baselines(m4_path)
    rows = []
    for seed in seeds:
        for task in (task for task in workflow_tasks() if len(task.steps) > 1):
            for policy in ("p1_direct_top1", "p6_combined_graph", "p7_reactive_jit", "p8_speculative_planning", "p9_oracle_capabilities"):
                trace, _total = policy_trace(policy, graph, index, task)
                base = {
                    "seed": seed,
                    "task_id": task.task_id,
                    "family": task.family,
                    "plan_horizon": len(task.steps),
                    "policy": policy,
                    "initial_disclosed_tools": len(trace.disclosed_uris),
                    "initial_required_recall": _set_metrics(task, graph, trace, trace.disclosed_uris)["required_recall_at_k"],
                }
                if policy in {"p1_direct_top1", "p7_reactive_jit", "p9_oracle_capabilities"}:
                    frozen = baselines[(seed, task.task_id, policy)]
                    result = {
                        "completed_steps": len(task.steps) if frozen["executed"] == "True" else 0,
                        "task_success": frozen["executed"] == "True",
                        "failure_reason": frozen["failure_reason"],
                        "cumulative_definition_tokens": int(frozen["cumulative_definition_tokens"]),
                        "cumulative_prompt_tokens": int(frozen["cumulative_prompt_tokens"]),
                        "cumulative_generated_tokens": int(frozen["cumulative_generated_tokens"]),
                        "generation_seconds": float(frozen["cumulative_generation_seconds"]),
                    }
                    source = "frozen_m4"
                else:
                    result = _static_execution(model, graph, task, trace.disclosed_uris, seed)
                    source = "m5_static_run"
                rows.append({**base, **result, "result_source": source})
                print(f"[M5 seed={seed} task={task.task_id} {policy}] success={result['task_success']}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper6_5_tools/m5_disclosure")
    parser.add_argument("--m4-rows", type=Path, default=ROOT / "docs/papers/shared/results/paper6_5_tools/pretrained_bridge/m4_rows.csv")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, local_files_only=True)
    deterministic, graph_meta = deterministic_study(tokenizer)
    _write_csv(args.output_dir / "m5_policy_rows.csv", deterministic)
    model_rows = []
    if not args.skip_model:
        device = torch.device(args.device)
        model = FrozenToolModel(args.model_id, args.revision, device)
        seeds = tuple(int(value) for value in args.seeds.split(",") if value)
        model_rows = pretrained_study(model, seeds, args.m4_rows)
        _write_csv(args.output_dir / "m5_model_rows.csv", model_rows)
    manifest = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_frozen": True,
        "policies": list(POLICIES),
        "graph": graph_meta,
        "deterministic_rows": len(deterministic),
        "model_rows": len(model_rows),
        "m4_baselines_reused": str(args.m4_rows.relative_to(ROOT)),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": runtime_metadata(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
