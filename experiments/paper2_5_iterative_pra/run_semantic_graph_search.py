"""Calibrate and evaluate terminal-query semantic graph search for Paper 2.5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_grounded_facet_gate import FacetConfig, build_facets
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    canonical_oracle_parent_indices,
    evidence_parent_groups,
    validation_partition,
)
from pra_hf.query_facets import score_semantic_query_facets
from pra_hf.semantic_graph_search import (
    SemanticGraphSearchConfig,
    build_native_parent_adjacency,
    search_semantic_graph,
)
from pra_torch.hf import load_hf_routing_projection


K_VALUES = (1, 2, 3, 4, 5, 6, 8, 11)
B_VALUES = (2, 4, 6, 8, 16, None)
H_VALUES = (1, 2, 3, 4)
STRATEGIES = ("breadth_first", "best_first", "beam")
FALSE_GOAL_LIMIT = 0.25
MIN_TRUE_CLOSURE = 0.50
MIN_LATER_RECALL = 0.50
EXPECTED_K_CURVE = {
    1: 3 / 17,
    2: 7 / 17,
    3: 11 / 17,
    4: 13 / 17,
    5: 15 / 17,
    6: 16 / 17,
    8: 16 / 17,
    11: 1.0,
}


@dataclass
class PreparedExample:
    dataset: str
    example_id: str
    partition: str
    question: str
    parent_spans: tuple[tuple[int, int], ...]
    parent_texts: tuple[str, ...]
    groups: tuple[frozenset[int], ...]
    oracle: frozenset[int]
    edge_scores: torch.Tensor
    goal_scores: dict[int, torch.Tensor]
    goal_winning_facets: dict[int, torch.Tensor]
    facet_count: int
    query_support_tokens: int
    source_tokens: int
    native_dot_products: int
    local_pair_count: int
    adjacency_build_seconds: float
    root_projection_seconds: dict[int, float]
    h2d_transfer_bytes: int
    h2d_transfer_seconds: float
    cpu_reference_cache_bytes: int
    adjacency_cache_bytes: int
    goal_cache_bytes: dict[int, int]
    prep_baseline_gpu_allocated_bytes: int
    prep_peak_gpu_allocated_bytes: int
    prep_peak_gpu_reserved_bytes: int


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    """Replace non-finite floats before strict JSON serialization."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _quantile_ladder(values: list[float], quantiles: tuple[float, ...], prefix: str):
    tensor = torch.tensor(values, dtype=torch.float64)
    output = []
    for quantile in quantiles:
        value = float(torch.quantile(tensor, quantile))
        if not any(math.isclose(value, row["value"], abs_tol=1e-12) for row in output):
            output.append({"label": f"{prefix}{int(100 * quantile):02d}", "value": value})
    return output


def _ordered(scores: torch.Tensor, excluded: set[int]) -> list[int]:
    candidates = [
        index
        for index in range(scores.numel())
        if index not in excluded and math.isfinite(float(scores[index]))
    ]
    return sorted(candidates, key=lambda index: (-float(scores[index]), index))


def _chain_complete(groups: tuple[frozenset[int], ...], visited: set[int]) -> float:
    return float(bool(groups) and all(set(group) & visited for group in groups))


def _set_metrics(visited: set[int], oracle: set[int], roots: set[int]) -> dict:
    intersection = visited & oracle
    later = oracle - roots
    later_found = (visited - roots) & later
    return {
        "oracle_recall": len(intersection) / len(oracle) if oracle else 1.0,
        "oracle_precision": len(intersection) / len(visited) if visited else 0.0,
        "complete_oracle": float(oracle <= visited),
        "later_oracle_recall": len(later_found) / len(later) if later else 1.0,
    }


def _prepare_examples(args, device: torch.device) -> list[PreparedExample]:
    print("[semantic-graph] loading frozen source/query caches", flush=True)
    source_features = torch.load(args.source_feature_file, map_location="cpu", weights_only=False)
    query_features = torch.load(args.query_feature_file, map_location="cpu", weights_only=False)
    query_by_id = {row["example_id"]: row for row in query_features}
    gate = json.loads(args.facet_gate_file.read_text(encoding="utf-8"))
    facet_config = FacetConfig(**gate["selected_facet_config"])
    support_mode = gate["selected_query_support"]
    projections = {
        seed: load_hf_routing_projection(
            args.projection_dir
            / "checkpoints"
            / f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt",
            device=device,
        )
        for seed in args.seeds
    }
    source_file_bytes = args.source_feature_file.stat().st_size
    prepared = []
    for index, feature in enumerate(source_features, start=1):
        print(
            f"[semantic-graph prep {index}/{len(source_features)}] begin "
            f"{feature['dataset']} {feature['example_id']}",
            flush=True,
        )
        query_feature = query_by_id[feature["example_id"]]
        facets = build_facets(query_feature, facet_config, support_mode, None)
        support = query_feature.get("latest_message_span", query_feature["question_span"])
        if device.type == "cuda":
            torch.cuda.empty_cache()
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        else:
            baseline_allocated = 0
        h2d_started = time.perf_counter()
        local_q = feature["local_pre_query"].to(device)
        local_k = feature["local_pre_key"].to(device)
        local_mask = feature["local_token_mask"].to(device)
        local_parents = feature["local_parent_indices"].to(device)
        parent_hidden = feature["parent_hidden"].to(device)
        facet_hidden = facets.hidden.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        h2d_seconds = time.perf_counter() - h2d_started
        adjacency_started = time.perf_counter()
        adjacency = build_native_parent_adjacency(
            local_q,
            local_k,
            local_mask,
            local_parents,
            len(feature["parent_spans"]),
            token_reduction="top_m_mean",
            head_reduction="top_m_mean",
            top_m=4,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        adjacency_seconds = time.perf_counter() - adjacency_started
        goal_scores, goal_facets, root_times = {}, {}, {}
        for seed, projection in projections.items():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            root_started = time.perf_counter()
            projected_query = projection.project_query(facet_hidden)
            projected_memory = projection.project_memory(parent_hidden)
            scored = score_semantic_query_facets(projected_query, projected_memory)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            root_times[seed] = time.perf_counter() - root_started
            goal_scores[seed] = scored.component_scores[:, 0, :].detach().cpu()
            goal_facets[seed] = scored.winning_facet.detach().cpu()
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        groups = tuple(frozenset(group) for group in evidence_parent_groups(feature))
        oracle = frozenset(canonical_oracle_parent_indices(feature))
        h2d_bytes = sum(
            _tensor_bytes(tensor)
            for tensor in (
                feature["local_pre_query"],
                feature["local_pre_key"],
                feature["local_token_mask"],
                feature["local_parent_indices"],
                feature["parent_hidden"],
                facets.hidden,
            )
        )
        prepared.append(
            PreparedExample(
                dataset=feature["dataset"],
                example_id=feature["example_id"],
                partition=validation_partition(feature["example_id"]),
                question=query_feature["question"],
                parent_spans=tuple(tuple(map(int, span)) for span in feature["parent_spans"]),
                parent_texts=tuple(
                    f"[source token span {int(start)}:{int(end)}]"
                    for start, end in feature["parent_spans"]
                ),
                groups=groups,
                oracle=oracle,
                edge_scores=adjacency.scores.detach().cpu(),
                goal_scores=goal_scores,
                goal_winning_facets=goal_facets,
                facet_count=len(facets.provenance),
                query_support_tokens=int(support[1]) - int(support[0]),
                source_tokens=int(feature["source_tokens"]),
                native_dot_products=adjacency.dot_products,
                local_pair_count=adjacency.local_pair_count,
                adjacency_build_seconds=adjacency_seconds,
                root_projection_seconds=root_times,
                h2d_transfer_bytes=h2d_bytes,
                h2d_transfer_seconds=h2d_seconds,
                cpu_reference_cache_bytes=source_file_bytes,
                adjacency_cache_bytes=_tensor_bytes(adjacency.scores),
                goal_cache_bytes={seed: _tensor_bytes(scores) for seed, scores in goal_scores.items()},
                prep_baseline_gpu_allocated_bytes=baseline_allocated,
                prep_peak_gpu_allocated_bytes=peak_allocated,
                prep_peak_gpu_reserved_bytes=peak_reserved,
            )
        )
        print(
            f"[semantic-graph prep {index}/{len(source_features)}] "
            f"{feature['dataset']} {feature['example_id']} parents={len(feature['parent_spans'])}",
            flush=True,
        )
    return prepared


def _edge_ladder(prepared: list[PreparedExample]) -> list[dict]:
    values = []
    for example in prepared:
        if example.dataset != "hotpotqa" or example.partition != "validation":
            continue
        matrix = example.edge_scores
        values.extend(float(value) for value in matrix[torch.isfinite(matrix)])
    return [{"label": "open", "value": float("-inf")}, *_quantile_ladder(
        values, (0.25, 0.50, 0.65, 0.75, 0.85, 0.90), "q"
    )]


def _goal_ladder(prepared: list[PreparedExample]) -> list[dict]:
    values = []
    for example in prepared:
        if example.partition != "validation" or not example.groups:
            continue
        for seed in example.goal_scores:
            values.extend(float(value) for value in example.goal_scores[seed].reshape(-1))
    return [*_quantile_ladder(
        values, (0.50, 0.65, 0.75, 0.85, 0.90, 0.95), "q"
    ), {"label": "closed", "value": float("inf")}]


def _transition_rows(prepared: list[PreparedExample], edge_ladder: list[dict]) -> list[dict]:
    rows = []
    for example in prepared:
        if example.dataset != "hotpotqa" or len(example.groups) < 2:
            continue
        for transition, (source, targets) in enumerate(zip(example.groups, example.groups[1:])):
            scores = example.edge_scores[list(source)].amax(dim=0)
            ordered = _ordered(scores, set(source))
            for k in K_VALUES:
                candidates = ordered[:k]
                for threshold in edge_ladder:
                    admitted = [
                        parent
                        for parent in candidates
                        if float(scores[parent]) >= threshold["value"]
                    ]
                    rows.append(
                        {
                            "partition": example.partition,
                            "dataset": example.dataset,
                            "example_id": example.example_id,
                            "transition": transition,
                            "K_search": k,
                            "theta_edge_label": threshold["label"],
                            "theta_edge": (
                                threshold["value"]
                                if math.isfinite(threshold["value"])
                                else None
                            ),
                            "source_parents": json.dumps(sorted(source)),
                            "oracle_successors": json.dumps(sorted(targets)),
                            "raw_candidates": len(candidates),
                            "edge_admitted_candidates": len(admitted),
                            "successor_recall": float(bool(set(admitted) & set(targets))),
                            "native_dot_products": example.native_dot_products,
                        }
                    )
    open_rows = [row for row in rows if row["theta_edge_label"] == "open"]
    for k, expected in EXPECTED_K_CURVE.items():
        samples = [row["successor_recall"] for row in open_rows if row["K_search"] == k]
        observed = statistics.fmean(samples)
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise AssertionError(f"Native K={k} recovery changed: {observed} != {expected}")
    return rows


def _aggregate(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    metric_names = (
        "root_available",
        "goal_triggered",
        "goal_correct",
        "false_goal",
        "goal_precision",
        "oracle_recall",
        "later_oracle_recall",
        "oracle_precision",
        "complete_oracle",
        "chain_complete",
        "visited_count",
        "nodes_expanded",
        "raw_proposals",
        "edge_admitted_candidates",
        "duplicate_proposals",
        "cycles_prevented",
        "peak_frontier",
        "goal_tests",
        "goal_comparisons",
        "adjacency_lookup_comparisons",
        "native_dot_products",
        "search_seconds",
        "goal_test_seconds",
        "cpu_dedup_seconds",
        "total_routing_seconds",
        "candidate_tensor_bytes",
        "conceptual_active_parent_count",
        "materialized_parent_count",
        "materialized_native_kv_tokens",
        "active_native_kv_fraction",
        "routing_cache_bytes",
        "h2d_transfer_bytes",
        "h2d_transfer_seconds",
        "prep_peak_gpu_allocated_bytes",
        "prep_peak_gpu_reserved_bytes",
    )
    output = []
    for key, values in grouped.items():
        record = dict(zip(dimensions, key))
        record["rows"] = len(values)
        record["identities"] = len({row["example_id"] for row in values})
        record["seeds"] = len({row["seed"] for row in values})
        for metric in metric_names:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            if samples:
                record[metric] = statistics.fmean(samples)
                if metric in {"visited_count", "nodes_expanded", "total_routing_seconds"}:
                    ordered = sorted(samples)
                    record[f"{metric}_p95"] = ordered[
                        min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
                    ]
                    record[f"{metric}_max"] = max(ordered)
        triggered = sum(float(row["goal_triggered"]) for row in values)
        correct = sum(float(row["goal_correct"]) for row in values)
        record["goal_precision"] = correct / triggered if triggered else 1.0
        output.append(record)
    return sorted(output, key=lambda row: tuple(str(row[key]) for key in dimensions))


def _search_row(
    example: PreparedExample,
    seed: int,
    roots: tuple[int, ...],
    entry_facets: dict[int, int],
    config: SemanticGraphSearchConfig,
    *,
    root_mode: str,
    root_available: bool,
    device: torch.device,
) -> tuple[dict, tuple]:
    edge_scores = example.edge_scores.to(device)
    goal_scores = example.goal_scores[seed].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    result = search_semantic_graph(
        edge_scores,
        goal_scores,
        roots,
        config,
        entry_facets=entry_facets,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    visited = set(result.visited)
    root_set = set(roots)
    terminal_correct = bool(
        result.terminal_parent is not None
        and result.terminal_parent in (set(example.oracle) - root_set)
    )
    metrics = _set_metrics(visited, set(example.oracle), root_set)
    terminal_text = (
        example.parent_texts[result.terminal_parent]
        if result.terminal_parent is not None
        else None
    )
    if terminal_correct:
        classification = "oracle_evidence"
    elif result.goal_triggered:
        classification = "false_semantic_closure"
    else:
        classification = "no_terminal"
    b_label = "none" if config.max_visited_parents is None else str(config.max_visited_parents)
    row = {
        "partition": example.partition,
        "dataset": example.dataset,
        "example_id": example.example_id,
        "seed": seed,
        "root_mode": root_mode,
        "root_parent": json.dumps(roots),
        "entry_facet": json.dumps(entry_facets, sort_keys=True),
        "root_available": float(root_available),
        "K_search": config.successor_k,
        "B_ceiling": config.max_visited_parents,
        "B_label": b_label,
        "theta_edge": config.edge_threshold if math.isfinite(config.edge_threshold) else None,
        "theta_goal": config.goal_threshold if math.isfinite(config.goal_threshold) else None,
        "max_hops": config.max_hops,
        "strategy": config.strategy,
        "different_facet_goal": config.different_facet_goal,
        "goal_triggered": float(result.goal_triggered),
        "goal_correct": float(terminal_correct),
        "false_goal": float(result.goal_triggered and not terminal_correct),
        "goal_precision": float(terminal_correct) if result.goal_triggered else None,
        "terminal_parent": result.terminal_parent,
        "terminal_facet": result.terminal_facet,
        "terminal_classification": classification,
        "terminal_text": terminal_text,
        "question": example.question,
        "path": json.dumps(result.path),
        "path_depth": len(result.path) - 1 if result.path else None,
        "stop_reason": result.stop_reason,
        "visited": json.dumps(result.visited),
        "visited_count": len(result.visited),
        "frontier_size": result.peak_frontier,
        "nodes_expanded": result.nodes_expanded,
        "raw_proposals": result.raw_proposals,
        "edge_admitted_candidates": result.edge_admitted_proposals,
        "duplicate_proposals": result.duplicate_proposals,
        "cycles_prevented": result.cycles_prevented,
        "peak_frontier": result.peak_frontier,
        "goal_tests": result.goal_tests,
        "goal_comparisons": result.goal_comparisons,
        "adjacency_lookup_comparisons": result.nodes_expanded * max(0, len(example.parent_spans) - 1),
        "native_dot_products": example.native_dot_products,
        "oracle_parent": json.dumps(sorted(example.oracle)),
        **metrics,
        "chain_complete": _chain_complete(example.groups, visited),
        "search_seconds": result.search_seconds,
        "goal_test_seconds": result.goal_test_seconds,
        "cpu_dedup_seconds": result.cpu_dedup_seconds,
        "root_routing_seconds": example.root_projection_seconds[seed],
        "graph_index_build_seconds": example.adjacency_build_seconds,
        "total_routing_seconds": (
            example.root_projection_seconds[seed]
            + example.adjacency_build_seconds
            + result.search_seconds
        ),
        "candidate_tensor_bytes": result.peak_candidate_tensor_bytes,
        "logical_reference_tokens": example.source_tokens,
        "logical_parent_count": len(example.parent_spans),
        "conceptual_active_parent_count": len(result.visited),
        "conceptual_active_parent_fraction": len(result.visited) / len(example.parent_spans),
        "materialized_parent_count": 0,
        "materialized_native_kv_tokens": 0,
        "active_native_kv_fraction": 0.0,
        "native_kv_bytes": 0,
        "peak_active_parent_count": len(result.visited),
        "query_facet_count": example.facet_count,
        "query_support_tokens": example.query_support_tokens,
        "cpu_reference_cache_bytes": example.cpu_reference_cache_bytes,
        "adjacency_cache_bytes": example.adjacency_cache_bytes,
        "goal_cache_bytes": example.goal_cache_bytes[seed],
        "routing_cache_bytes": example.adjacency_cache_bytes + example.goal_cache_bytes[seed],
        "h2d_transfer_bytes": example.h2d_transfer_bytes,
        "h2d_transfer_seconds": example.h2d_transfer_seconds,
        "prep_baseline_gpu_allocated_bytes": example.prep_baseline_gpu_allocated_bytes,
        "prep_peak_gpu_allocated_bytes": example.prep_peak_gpu_allocated_bytes,
        "prep_peak_gpu_reserved_bytes": example.prep_peak_gpu_reserved_bytes,
        "search_peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "search_peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "active_kv_tokens": 0,
        "generation_performed": False,
        "tpot": None,
    }
    return row, result.decisions


def _select_edge_policy(transition_rows: list[dict], edge_ladder: list[dict]) -> dict:
    validation = [
        row
        for row in transition_rows
        if row["partition"] == "validation" and row["K_search"] == max(K_VALUES)
    ]
    records = []
    for threshold in edge_ladder:
        values = [row for row in validation if row["theta_edge_label"] == threshold["label"]]
        records.append(
            {
                "theta_edge_label": threshold["label"],
                "theta_edge": threshold["value"],
                "successor_recall": statistics.fmean(row["successor_recall"] for row in values),
                "branching_factor": statistics.fmean(
                    row["edge_admitted_candidates"] for row in values
                ),
                "transitions": len(values),
            }
        )
    baseline = next(row for row in records if row["theta_edge_label"] == "open")
    eligible = [
        row for row in records if row["successor_recall"] >= baseline["successor_recall"] - 0.10
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["branching_factor"],
            -row["successor_recall"],
            -row["theta_edge"] if math.isfinite(row["theta_edge"]) else float("inf"),
        ),
    )
    return {"selected": selected, "audit": records}


def _roots(example: PreparedExample, mode: str, seed: int, count: int = 1):
    if mode == "oracle":
        if not example.groups:
            return (), {}, False
        roots = tuple(sorted(example.groups[0]))
        winners = example.goal_scores[seed].argmax(dim=0)
        return roots, {root: int(winners[root]) for root in roots}, True
    scores = example.goal_scores[seed].amax(dim=0)
    ordered = _ordered(scores, set())[:count]
    winners = example.goal_scores[seed].argmax(dim=0)
    first_group = set(example.groups[0]) if example.groups else set()
    return (
        tuple(ordered),
        {root: int(winners[root]) for root in ordered},
        bool(first_group & set(ordered)),
    )


def _run_conditions(
    prepared: list[PreparedExample],
    args,
    device: torch.device,
    conditions: list[dict],
    *,
    root_mode: str,
    root_count: int = 1,
    collect_decisions: bool = False,
) -> tuple[list[dict], list[dict]]:
    rows, node_rows = [], []
    for example in prepared:
        if not example.groups:
            continue
        for seed in args.seeds:
            roots, entry_facets, root_available = _roots(
                example, root_mode, seed, root_count
            )
            if not roots:
                continue
            for condition in conditions:
                ceiling = condition["B"]
                if ceiling is not None and len(roots) > ceiling:
                    continue
                config = SemanticGraphSearchConfig(
                    successor_k=condition["K"],
                    max_visited_parents=ceiling,
                    edge_threshold=condition["theta_edge"],
                    goal_threshold=condition["theta_goal"],
                    max_hops=condition["H"],
                    strategy=condition["strategy"],
                    beam_width=condition.get("beam_width", 4),
                    max_expanded_nodes=args.max_expanded_nodes,
                    different_facet_goal=condition.get("different_facet_goal", False),
                )
                row, decisions = _search_row(
                    example,
                    seed,
                    roots,
                    entry_facets,
                    config,
                    root_mode=root_mode,
                    root_available=root_available,
                    device=device,
                )
                row["theta_edge_label"] = condition["theta_edge_label"]
                row["theta_goal_label"] = condition["theta_goal_label"]
                row["condition_id"] = condition["condition_id"]
                row["root_count_requested"] = root_count
                rows.append(row)
                if collect_decisions:
                    for decision in decisions:
                        node_rows.append(
                            {
                                "partition": example.partition,
                                "dataset": example.dataset,
                                "example_id": example.example_id,
                                "seed": seed,
                                "root_mode": root_mode,
                                "root_parent": json.dumps(roots),
                                "entry_facet": json.dumps(entry_facets, sort_keys=True),
                                "K_search": config.successor_k,
                                "B_ceiling": config.max_visited_parents,
                                "theta_edge": row["theta_edge"],
                                "theta_goal": row["theta_goal"],
                                "hop": decision.hop,
                                "source_parent": decision.source_parent,
                                "candidate_parent": decision.candidate_parent,
                                "native_edge_rank": decision.native_rank,
                                "native_edge_score": decision.native_score,
                                "passed_edge_threshold": decision.passed_edge_threshold,
                                "duplicate": decision.duplicate,
                                "cycle": decision.cycle,
                                "admitted": decision.admitted,
                                "goal_best_facet": decision.goal_best_facet,
                                "goal_score": decision.goal_score,
                                "goal_triggered": decision.goal_triggered,
                                "goal_correct": bool(
                                    decision.goal_triggered
                                    and decision.candidate_parent
                                    in (set(example.oracle) - set(roots))
                                ),
                                "path": json.dumps(decision.path),
                                "path_quality": decision.path_quality,
                                "visited_count": row["visited_count"],
                                "frontier_size": row["frontier_size"],
                                "oracle_recall": row["oracle_recall"],
                                "oracle_precision": row["oracle_precision"],
                                "complete_oracle": row["complete_oracle"],
                                "chain_complete": row["chain_complete"],
                                "raw_proposals": row["raw_proposals"],
                                "search_comparisons": row["adjacency_lookup_comparisons"],
                                "goal_comparisons": row["goal_comparisons"],
                                "search_time": row["search_seconds"],
                                "peak_gpu_memory": row["search_peak_gpu_allocated_bytes"],
                                "active_kv_tokens": 0,
                            }
                        )
    return rows, node_rows


def _select_goal_policy(
    calibration_rows: list[dict], goal_ladder: list[dict]
) -> dict:
    validation = [row for row in calibration_rows if row["partition"] == "validation"]
    summary = _aggregate(
        validation,
        ("strategy", "theta_goal_label", "dataset"),
    )
    records = []
    for strategy in STRATEGIES:
        for threshold in goal_ladder:
            values = [
                row
                for row in validation
                if row["strategy"] == strategy
                and row["theta_goal_label"] == threshold["label"]
            ]
            hotpot = [row for row in values if row["dataset"] == "hotpotqa"]
            false_rate = statistics.fmean(row["false_goal"] for row in values)
            records.append(
                {
                    "strategy": strategy,
                    "theta_goal_label": threshold["label"],
                    "theta_goal": threshold["value"],
                    "hotpot_true_closure": statistics.fmean(
                        row["goal_correct"] for row in hotpot
                    ),
                    "false_goal_rate": false_rate,
                    "hotpot_oracle_recall": statistics.fmean(
                        row["oracle_recall"] for row in hotpot
                    ),
                    "mean_nodes_expanded": statistics.fmean(
                        row["nodes_expanded"] for row in values
                    ),
                    "goal_precision": (
                        sum(row["goal_correct"] for row in values)
                        / max(1.0, sum(row["goal_triggered"] for row in values))
                    ),
                }
            )
    eligible = [row for row in records if row["false_goal_rate"] <= FALSE_GOAL_LIMIT]
    pool = eligible or records
    selected = max(
        pool,
        key=lambda row: (
            row["hotpot_true_closure"],
            row["goal_precision"],
            row["hotpot_oracle_recall"],
            -row["false_goal_rate"],
            -row["mean_nodes_expanded"],
            row["strategy"] == "breadth_first",
        ),
    )
    return {"selected": selected, "audit": records, "summary": summary}


def _select_operating_point(surface_rows: list[dict]) -> dict:
    validation = [row for row in surface_rows if row["partition"] == "validation"]
    candidates = []
    keys = {
        (
            row["K_search"],
            row["B_label"],
            row["theta_edge_label"],
            row["max_hops"],
            row["strategy"],
        )
        for row in validation
        if row["K_search"] <= 6
        and row["B_label"] in {"2", "4", "6", "8"}
        and row["max_hops"] == 4
    }
    for key in keys:
        values = [
            row
            for row in validation
            if (
                row["K_search"],
                row["B_label"],
                row["theta_edge_label"],
                row["max_hops"],
                row["strategy"],
            )
            == key
        ]
        hotpot = [row for row in values if row["dataset"] == "hotpotqa"]
        candidates.append(
            {
                "K_search": key[0],
                "B_label": key[1],
                "B_ceiling": int(key[1]),
                "theta_edge_label": key[2],
                "theta_edge": next(row["theta_edge"] for row in values),
                "max_hops": key[3],
                "strategy": key[4],
                "hotpot_true_closure": statistics.fmean(
                    row["goal_correct"] for row in hotpot
                ),
                "false_goal_rate": statistics.fmean(row["false_goal"] for row in values),
                "hotpot_later_oracle_recall": statistics.fmean(
                    row["later_oracle_recall"] for row in hotpot
                ),
                "hotpot_oracle_recall": statistics.fmean(
                    row["oracle_recall"] for row in hotpot
                ),
                "mean_nodes_expanded": statistics.fmean(
                    row["nodes_expanded"] for row in values
                ),
            }
        )
    eligible = [row for row in candidates if row["false_goal_rate"] <= FALSE_GOAL_LIMIT]
    selected = max(
        eligible or candidates,
        key=lambda row: (
            row["hotpot_true_closure"],
            row["hotpot_later_oracle_recall"],
            row["hotpot_oracle_recall"],
            -row["false_goal_rate"],
            -row["mean_nodes_expanded"],
            -row["K_search"],
            -row["B_ceiling"],
        ),
    )
    return {"selected": selected, "audit": candidates}


def _synthetic_controls() -> dict:
    inf = float("-inf")
    edge = torch.tensor(
        [
            [inf, 0.95, 0.20, 0.10, 0.15],
            [0.10, inf, 0.90, 0.85, 0.30],
            [0.10, 0.90, inf, 0.95, 0.20],
            [0.10, 0.20, 0.80, inf, 0.10],
            [0.10, 0.10, 0.10, 0.10, inf],
        ]
    )
    # Parent 2 is deliberately weak against every query facet; parent 3 closes.
    goal = torch.tensor(
        [[0.90, 0.20, 0.05, 0.10, 0.15], [0.05, 0.10, 0.05, 0.95, 0.96]]
    )
    # The root cannot jump directly to parent 3. Native local association must
    # carry the search through query-dissimilar parent 2 before the goal fires.
    edge[1, 3] = 0.30
    base = SemanticGraphSearchConfig(2, 5, 0.5, 0.9, max_hops=3)
    bridge = search_semantic_graph(edge, goal, [1], base, entry_facets={1: 0})
    false_edge = edge.clone()
    false_edge[1, 4] = 0.99
    false_terminal = search_semantic_graph(
        false_edge,
        goal,
        [1],
        SemanticGraphSearchConfig(1, 5, 0.5, 0.9, max_hops=1),
        entry_facets={1: 0},
    )
    cycle = search_semantic_graph(
        edge[:3, :3],
        torch.zeros(1, 3),
        [1],
        SemanticGraphSearchConfig(2, None, 0.0, float("inf"), max_hops=3),
    )
    open_search = search_semantic_graph(
        edge,
        torch.zeros_like(goal),
        [0],
        SemanticGraphSearchConfig(4, None, float("-inf"), float("inf"), max_hops=3),
    )
    bounded_search = search_semantic_graph(
        edge,
        torch.zeros_like(goal),
        [0],
        SemanticGraphSearchConfig(4, 2, 0.8, float("inf"), max_hops=3),
    )
    early = search_semantic_graph(
        edge,
        goal,
        [2],
        SemanticGraphSearchConfig(4, None, 0.5, 0.9, max_hops=3),
    )
    return {
        "low_query_similarity_bridge": {
            "passed": bridge.path == (1, 2, 3),
            "path": bridge.path,
            "bridge_goal_score": float(goal[:, 2].max()),
        },
        "false_terminal_match": {
            "passed": false_terminal.goal_triggered,
            "terminal": false_terminal.terminal_parent,
            "demonstrates_false_goal_risk": false_terminal.terminal_parent not in {2, 3},
        },
        "cycle": {
            "passed": cycle.cycles_prevented > 0 and len(cycle.visited) == len(set(cycle.visited)),
            "cycles_prevented": cycle.cycles_prevented,
        },
        "branch_explosion": {
            "passed": len(bounded_search.visited) < len(open_search.visited),
            "open_visited": len(open_search.visited),
            "bounded_visited": len(bounded_search.visited),
        },
        "early_valid_path": {
            "passed": early.goal_triggered and early.nodes_expanded == 1,
            "path": early.path,
            "nodes_expanded": early.nodes_expanded,
        },
    }


def _mean(rows: list[dict], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows)


def _plot_quality_cost(
    surface_rows: list[dict], goal_audit: list[dict], selected: dict, output_dir: Path
) -> None:
    test = [
        row
        for row in surface_rows
        if row["partition"] == "test"
        and row["dataset"] == "hotpotqa"
        and row["theta_edge_label"] == selected["theta_edge_label"]
        and row["max_hops"] == selected["max_hops"]
        and row["strategy"] == selected["strategy"]
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    b_rows = []
    for label in ("2", "4", "6", "8", "16", "none"):
        values = [
            row for row in test
            if row["B_label"] == label and row["K_search"] == selected["K_search"]
        ]
        if values:
            b_rows.append((label, _mean(values, "oracle_recall"), _mean(values, "nodes_expanded")))
    axes[0, 0].plot(
        [row[2] for row in b_rows], [row[1] for row in b_rows], marker="o", color="#2878B5"
    )
    axes[0, 0].set(xlabel="Nodes expanded", ylabel="Oracle recall")
    for label, recall, nodes in b_rows:
        axes[0, 0].annotate(label, (nodes, recall), fontsize=8)

    k_rows = []
    for k in K_VALUES:
        values = [
            row for row in test
            if row["K_search"] == k and row["B_label"] == selected["B_label"]
        ]
        if values:
            k_rows.append((k, _mean(values, "goal_correct"), _mean(values, "nodes_expanded"), _mean(values, "oracle_recall"), _mean(values, "search_seconds"), _mean(values, "routing_cache_bytes")))

    goal_rows = [row for row in goal_audit if row["strategy"] == selected["strategy"]]
    goal_rows.sort(
        key=lambda row: (
            math.inf if row["theta_goal"] is None else float(row["theta_goal"])
        )
    )
    axes[0, 1].plot(
        [row["mean_nodes_expanded"] for row in goal_rows],
        [row["hotpot_true_closure"] for row in goal_rows],
        marker="o",
        color="#D95F02",
    )
    axes[0, 1].set(xlabel="Validation nodes expanded", ylabel="Correct path closure")
    for row in goal_rows:
        axes[0, 1].annotate(
            row["theta_goal_label"],
            (row["mean_nodes_expanded"], row["hotpot_true_closure"]),
            fontsize=7,
        )

    axes[0, 2].plot(
        range(len(goal_rows)), [row["false_goal_rate"] for row in goal_rows], marker="o", color="#A34832"
    )
    axes[0, 2].set_xticks(range(len(goal_rows)), [row["theta_goal_label"] for row in goal_rows], rotation=35)
    axes[0, 2].set(xlabel="Goal threshold", ylabel="Validation false-goal rate")

    axes[1, 0].plot([row[0] for row in k_rows], [row[3] for row in k_rows], marker="o", color="#2878B5")
    axes[1, 0].set(xlabel="Native successor breadth K", ylabel="Oracle recall")
    axes[1, 0].set_xticks(K_VALUES)

    axes[1, 1].scatter([row[4] for row in k_rows], [row[3] for row in k_rows], c=[row[0] for row in k_rows], cmap="viridis")
    axes[1, 1].set(xlabel="CPU graph-search time (s)", ylabel="Oracle recall")

    axes[1, 2].scatter([row[5] for row in k_rows], [row[3] for row in k_rows], c=[row[0] for row in k_rows], cmap="plasma")
    axes[1, 2].set(xlabel="Routing cache bytes", ylabel="Oracle recall")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"semantic_graph_quality_cost.{suffix}", dpi=180)
    plt.close(figure)


def _plot_surfaces(surface_rows: list[dict], selected: dict, output_dir: Path) -> None:
    test = [
        row
        for row in surface_rows
        if row["partition"] == "test"
        and row["dataset"] == "hotpotqa"
        and row["max_hops"] == selected["max_hops"]
        and row["strategy"] == selected["strategy"]
    ]
    b_labels = ("2", "4", "6", "8", "16", "none")
    edge_labels = list(dict.fromkeys(row["theta_edge_label"] for row in test))
    kb = torch.zeros(len(b_labels), len(K_VALUES))
    be = torch.zeros(len(b_labels), len(edge_labels))
    for i, b in enumerate(b_labels):
        for j, k in enumerate(K_VALUES):
            values = [
                row for row in test
                if row["B_label"] == b
                and row["K_search"] == k
                and row["theta_edge_label"] == selected["theta_edge_label"]
            ]
            kb[i, j] = _mean(values, "oracle_recall") if values else float("nan")
        for j, edge in enumerate(edge_labels):
            values = [
                row for row in test
                if row["B_label"] == b
                and row["K_search"] == selected["K_search"]
                and row["theta_edge_label"] == edge
            ]
            be[i, j] = _mean(values, "oracle_recall") if values else float("nan")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, matrix, columns, title in (
        (axes[0], kb, K_VALUES, "Oracle recall: K x B"),
        (axes[1], be, edge_labels, "Oracle recall: edge threshold x B"),
    ):
        image = axis.imshow(matrix.numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(columns)), columns, rotation=35 if axis is axes[1] else 0)
        axis.set_yticks(range(len(b_labels)), b_labels)
        axis.set_xlabel("K" if axis is axes[0] else "theta_edge")
        axis.set_ylabel("B ceiling")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"semantic_graph_surfaces.{suffix}", dpi=180)
    plt.close(figure)


def _condition(
    *,
    k: int,
    b: int | None,
    edge: dict,
    goal: dict,
    hops: int,
    strategy: str,
    suffix: str = "",
    different_facet_goal: bool = False,
) -> dict:
    b_label = "none" if b is None else str(b)
    return {
        "condition_id": (
            f"K{k}_B{b_label}_E{edge['label']}_G{goal['label']}_H{hops}_{strategy}{suffix}"
        ),
        "K": k,
        "B": b,
        "theta_edge": edge["value"],
        "theta_edge_label": edge["label"],
        "theta_goal": goal["value"],
        "theta_goal_label": goal["label"],
        "H": hops,
        "strategy": strategy,
        "different_facet_goal": different_facet_goal,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    calibration_device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_examples(args, device)
    edge_ladder = _edge_ladder(prepared)
    goal_ladder = _goal_ladder(prepared)
    transition_rows = _transition_rows(prepared, edge_ladder)
    edge_selection = _select_edge_policy(transition_rows, edge_ladder)
    selected_edge = next(
        row
        for row in edge_ladder
        if row["label"] == edge_selection["selected"]["theta_edge_label"]
    )

    goal_conditions = [
        _condition(
            k=11,
            b=None,
            edge=selected_edge,
            goal=goal,
            hops=4,
            strategy=strategy,
            suffix="_goal_calibration",
        )
        for strategy in STRATEGIES
        for goal in goal_ladder
    ]
    goal_rows, _ = _run_conditions(
        prepared, args, calibration_device, goal_conditions, root_mode="oracle"
    )
    goal_selection = _select_goal_policy(goal_rows, goal_ladder)
    selected_goal = next(
        row
        for row in goal_ladder
        if row["label"] == goal_selection["selected"]["theta_goal_label"]
    )
    selected_strategy = goal_selection["selected"]["strategy"]

    surface_conditions = [
        _condition(
            k=k,
            b=b,
            edge=edge,
            goal=selected_goal,
            hops=4,
            strategy=selected_strategy,
            suffix="_surface",
        )
        for k in K_VALUES
        for b in B_VALUES
        for edge in edge_ladder
    ]
    surface_rows, _ = _run_conditions(
        prepared, args, calibration_device, surface_conditions, root_mode="oracle"
    )
    operating = _select_operating_point(surface_rows)
    selected = operating["selected"]
    operating_edge = next(
        row for row in edge_ladder if row["label"] == selected["theta_edge_label"]
    )

    hop_conditions = [
        _condition(
            k=selected["K_search"],
            b=selected["B_ceiling"],
            edge=operating_edge,
            goal=selected_goal,
            hops=hops,
            strategy=selected["strategy"],
            suffix="_hop",
        )
        for hops in H_VALUES
    ]
    hop_rows, _ = _run_conditions(
        prepared, args, calibration_device, hop_conditions, root_mode="oracle"
    )

    selected_condition = _condition(
        k=selected["K_search"],
        b=selected["B_ceiling"],
        edge=operating_edge,
        goal=selected_goal,
        hops=selected["max_hops"],
        strategy=selected["strategy"],
        suffix="_selected",
    )
    selected_rows, selected_node_rows = _run_conditions(
        prepared,
        args,
        device,
        [selected_condition],
        root_mode="oracle",
        collect_decisions=True,
    )
    different_condition = _condition(
        k=selected["K_search"],
        b=selected["B_ceiling"],
        edge=operating_edge,
        goal=selected_goal,
        hops=selected["max_hops"],
        strategy=selected["strategy"],
        suffix="_different_facet",
        different_facet_goal=True,
    )
    different_rows, _ = _run_conditions(
        prepared, args, calibration_device, [different_condition], root_mode="oracle"
    )

    heldout = [row for row in selected_rows if row["partition"] == "test"]
    heldout_hotpot = [row for row in heldout if row["dataset"] == "hotpotqa"]
    heldout_false_goal = _mean(heldout, "false_goal")
    heldout_true_closure = _mean(heldout_hotpot, "goal_correct")
    heldout_later_recall = _mean(heldout_hotpot, "later_oracle_recall")
    goal_gate_passed = heldout_false_goal <= FALSE_GOAL_LIMIT
    oracle_root_gate_passed = (
        goal_gate_passed
        and heldout_true_closure >= MIN_TRUE_CLOSURE
        and heldout_later_recall >= MIN_LATER_RECALL
    )

    routed_rows, routed_node_rows = [], []
    if oracle_root_gate_passed:
        for root_count in (1, 2, 4):
            routed_b = max(selected["B_ceiling"], root_count + 1)
            conditions = [
                _condition(
                    k=k,
                    b=routed_b,
                    edge=operating_edge,
                    goal=selected_goal,
                    hops=selected["max_hops"],
                    strategy=selected["strategy"],
                    suffix=f"_routed_R{root_count}",
                )
                for k in K_VALUES
            ]
            rows, nodes = _run_conditions(
                prepared,
                args,
                calibration_device,
                conditions,
                root_mode="routed",
                root_count=root_count,
                collect_decisions=(root_count == 4),
            )
            routed_rows.extend(rows)
            routed_node_rows.extend(nodes)

    synthetic = _synthetic_controls()
    if not all(record["passed"] for record in synthetic.values()):
        raise AssertionError(f"Synthetic semantic-graph control failed: {synthetic}")

    surface_summary = _aggregate(
        surface_rows,
        (
            "partition",
            "dataset",
            "K_search",
            "B_label",
            "theta_edge_label",
            "max_hops",
            "strategy",
        ),
    )
    hop_summary = _aggregate(
        hop_rows, ("partition", "dataset", "max_hops", "strategy")
    )
    selected_summary = _aggregate(
        selected_rows + different_rows,
        ("partition", "dataset", "different_facet_goal"),
    )
    routed_summary = (
        _aggregate(
            routed_rows,
            ("partition", "dataset", "root_count_requested", "K_search"),
        )
        if routed_rows
        else []
    )
    _write_csv(args.output_dir / "native_k_recovery_rows.csv", transition_rows)
    _write_csv(args.output_dir / "edge_threshold_audit.csv", edge_selection["audit"])
    _write_csv(args.output_dir / "goal_calibration_rows.csv", goal_rows)
    _write_csv(args.output_dir / "goal_threshold_audit.csv", goal_selection["audit"])
    _write_csv(args.output_dir / "oracle_surface_rows.csv", surface_rows)
    _write_csv(args.output_dir / "oracle_surface_summary.csv", surface_summary)
    _write_csv(args.output_dir / "hop_depth_rows.csv", hop_rows)
    _write_csv(args.output_dir / "hop_depth_summary.csv", hop_summary)
    _write_csv(args.output_dir / "selected_oracle_rows.csv", selected_rows + different_rows)
    _write_csv(args.output_dir / "selected_oracle_summary.csv", selected_summary)
    _write_csv(args.output_dir / "selected_oracle_node_rows.csv", selected_node_rows)
    _write_csv(args.output_dir / "routed_root_rows.csv", routed_rows)
    _write_csv(args.output_dir / "routed_root_summary.csv", routed_summary)
    _write_csv(args.output_dir / "routed_root_node_rows.csv", routed_node_rows)
    _plot_quality_cost(surface_rows, goal_selection["audit"], selected, args.output_dir)
    _plot_surfaces(surface_rows, selected, args.output_dir)

    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "production_default_changed": False,
        "training_performed": False,
        "generation_performed": False,
        "execution_policy": {
            "feature_extraction_and_index_build": str(device),
            "calibration_and_parameter_surfaces": str(calibration_device),
            "selected_operating_point_profile": str(device),
            "reason": (
                "Small graph-search tensors are calibrated on CPU to avoid one CUDA "
                "synchronization per condition; the selected condition is rerun on CUDA."
            ),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seeds": list(args.seeds),
        "query_facet_policy": {
            "facet_config": "w2_s1",
            "query_support": "latest_message",
            "goal_reduction": "maximum_over_all_facets",
        },
        "native_edge_policy": "layer_27_pre_rope_top4_token_and_head_mean",
        "candidate_ks": list(K_VALUES),
        "active_parent_ceilings": [*B_VALUES[:-1], "none"],
        "max_hops": list(H_VALUES),
        "strategies": list(STRATEGIES),
        "known_k_curve_reproduced_exactly": True,
        "expected_k_curve": EXPECTED_K_CURVE,
        "edge_threshold_ladder": edge_ladder,
        "edge_threshold_selection": edge_selection,
        "goal_threshold_ladder": goal_ladder,
        "goal_threshold_selection": goal_selection,
        "selected_operating_point": selected,
        "heldout_oracle_root": {
            "hotpot_true_closure": heldout_true_closure,
            "hotpot_later_oracle_recall": heldout_later_recall,
            "all_false_goal_rate": heldout_false_goal,
            "goal_gate_passed": goal_gate_passed,
            "oracle_root_gate_passed": oracle_root_gate_passed,
        },
        "routed_root_run": oracle_root_gate_passed,
        "routed_root_reason": (
            "oracle_and_goal_gates_passed"
            if oracle_root_gate_passed
            else "stopped_by_predeclared_oracle_or_goal_gate"
        ),
        "intermediate_query_filtering_used": False,
        "query_used_only_for_root_and_terminal_goal": True,
        "one_path_stop": True,
        "native_kv_materialization_performed": False,
        "conceptual_search_and_native_kv_separated": True,
        "synthetic_controls": synthetic,
        "row_counts": {
            "native_k_recovery": len(transition_rows),
            "goal_calibration": len(goal_rows),
            "oracle_surface": len(surface_rows),
            "hop_depth": len(hop_rows),
            "selected_oracle": len(selected_rows) + len(different_rows),
            "selected_oracle_nodes": len(selected_node_rows),
            "routed_root": len(routed_rows),
        },
        "serving_metric_deferrals": {
            "ttft": "routing-only diagnostic; no token generated",
            "tpot": "no generation",
            "throughput": "not an end-to-end serving path",
            "concurrency": "deferred to Paper 3.5 systems work",
            "dollar_per_million_tokens": "no cloud pricing run",
        },
    }
    (args.output_dir / "semantic_graph_search_results.json").write_text(
        json.dumps(_json_safe(artifact), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--examples-per-dataset", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--max-expanded-nodes", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=result_root / "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--query-feature-file",
        type=Path,
        default=result_root / "query_entry_facets/query_entry_features.pt",
    )
    parser.add_argument(
        "--facet-gate-file",
        type=Path,
        default=result_root / "grounded_query_facets/grounded_facet_gate_results.json",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_root / "semantic_graph_search",
    )
    args = parser.parse_args()
    args.seeds = tuple(int(seed) for seed in args.seeds.split(","))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "selected_operating_point": result["selected_operating_point"],
                "heldout_oracle_root": result["heldout_oracle_root"],
                "routed_root_run": result["routed_root_run"],
            },
            indent=2,
        )
    )
