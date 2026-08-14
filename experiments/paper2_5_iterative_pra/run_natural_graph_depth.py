"""Evaluate frozen native graph search on MuSiQue and 2WikiMultiHopQA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
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
from experiments.paper2_5_iterative_pra.run_oracle_convergence import SEEDS
from pra_hf.chunk_granularity import (
    facet_parent_statistics,
    incremental_facet_coverage,
    normalize_facet_scores,
    path_facet_coverage,
)
from pra_hf.natural_reasoning_graph import (
    AnnotatedEvidenceNode,
    NaturalReasoningExample,
    map_example_to_parents,
)
from pra_hf.query_facets import score_semantic_query_facets
from pra_hf.semantic_graph_search import (
    SemanticGraphSearchConfig,
    build_native_parent_adjacency,
    search_semantic_graph,
)
from pra_torch.hf import load_hf_routing_projection


CHUNK_SIZES = (64, 128, 256)
PRIMARY_CHUNK = 128
K_VALUES = (1, 2, 4, 6)
RANK_K_VALUES = (1, 2, 3, 4, 5, 6, 8, 11)
H_VALUES = (1, 2, 3, 4)
B_VALUES = (6, 16, None)
STRATEGY = "best_first"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) if rows else float("nan")


def _feature_example(feature: dict) -> NaturalReasoningExample:
    nodes = tuple(AnnotatedEvidenceNode(**node) for node in feature["nodes"])
    return NaturalReasoningExample(
        dataset=feature["dataset"],
        example_id=feature["example_id"],
        question=feature["question"],
        answer="",
        question_type=feature["question_type"],
        annotated_hops=int(feature["annotated_hops"]),
        graph_type=feature["graph_type"],
        source="",
        nodes=nodes,
        raw_annotation={},
    )


def _parent_hidden(token_hidden: torch.Tensor, spans) -> torch.Tensor:
    return torch.stack([token_hidden[start:end].float().mean(dim=0) for start, end in spans])


def _search(scores: torch.Tensor, roots, k: int, h: int, b: int | None):
    return search_semantic_graph(
        scores,
        torch.zeros(1, scores.shape[0]),
        roots,
        SemanticGraphSearchConfig(
            successor_k=k,
            max_visited_parents=b,
            edge_threshold=float("-inf"),
            goal_threshold=float("inf"),
            max_hops=h,
            strategy=STRATEGY,
            max_expanded_nodes=64,
        ),
    )


def _node_depths(example: NaturalReasoningExample) -> dict[str, int]:
    depths = {}
    for node in example.nodes:
        depths[node.node_id] = 1 + max((depths.get(parent, 0) for parent in node.dependencies), default=0)
    return depths


def _node_recovery(visited, mapping) -> tuple[float, bool, dict[str, bool]]:
    visited = set(map(int, visited))
    hit = {
        node: bool(visited.intersection(parents))
        for node, parents in mapping.node_parent_groups.items()
    }
    return sum(hit.values()) / len(hit), all(hit.values()), hit


def _transition_rank(scores: torch.Tensor, source_group, target_group) -> int | None:
    ranks = []
    novel_targets = set(map(int, target_group)) - set(map(int, source_group))
    for source in source_group:
        values = scores[source].clone()
        values[source] = float("-inf")
        order = torch.argsort(values, descending=True, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        for target in novel_targets:
            if torch.isfinite(values[target]):
                ranks.append(int(inverse[target]) + 1)
    return min(ranks) if ranks else None


def _native_successors(scores: torch.Tensor, source: int, k: int) -> tuple[int, ...]:
    values = scores[source].clone()
    values[source] = float("-inf")
    order = torch.argsort(values, descending=True, stable=True)
    return tuple(int(value) for value in order[: min(k, len(order))] if torch.isfinite(values[value]))


def _shortest_native_path(scores, source_group, target_group, k: int, max_hops: int = 4):
    """Return bounded native distance without using targets during expansion."""
    frontier = set(map(int, source_group))
    # Shared parents are a chunk-level contraction, not a recovered transition.
    targets = set(map(int, target_group)) - frontier
    if not targets:
        return None
    visited = set(frontier)
    for depth in range(1, max_hops + 1):
        next_frontier = {
            candidate
            for source in frontier
            for candidate in _native_successors(scores, source, k)
            if candidate not in visited
        }
        if next_frontier.intersection(targets):
            return depth
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return None


def _transition_rows(feature: dict, example, mapping, scores: torch.Tensor) -> list[dict]:
    rows = []
    for source_node, target_node in example.annotated_edges:
        source_group = mapping.node_parent_groups.get(source_node, ())
        target_group = mapping.node_parent_groups.get(target_node, ())
        rank = _transition_rank(scores, source_group, target_group)
        if not source_group or not target_group:
            status = "unmappable"
        elif rank is None:
            status = "collapsed"
        else:
            status = "preserved"
        row = {
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "partition": feature["partition"],
                "question_type": feature["question_type"],
                "annotated_hops": feature["annotated_hops"],
                "graph_type": feature["graph_type"],
                "chunk_size": mapping.parent_spans[0][1] - mapping.parent_spans[0][0],
                "source_node": source_node,
                "target_node": target_node,
                "source_parent_group": json.dumps(source_group),
                "target_parent_group": json.dumps(target_group),
                "mapping_status": status,
                "partial_parent_overlap": int(bool(set(source_group).intersection(target_group))),
                "target_native_rank": "" if rank is None else rank,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                **{f"recovered_at_{k}": int(rank is not None and rank <= k) for k in RANK_K_VALUES},
            }
        for k in RANK_K_VALUES:
            distance = _shortest_native_path(scores, source_group, target_group, k)
            row[f"shortest_path_at_{k}"] = "" if distance is None else distance
            row[f"path_distortion_at_{k}"] = "" if distance is None else distance - 1
        rows.append(row)
    return rows


def _selected_token_metrics(visited, mapping, source_tokens: int) -> dict:
    selected = sum(mapping.parent_spans[parent][1] - mapping.parent_spans[parent][0] for parent in visited)
    return {
        "logical_reference_tokens": source_tokens,
        "conceptual_active_parents": len(visited),
        "counterfactual_native_kv_tokens": selected,
        "active_kv_fraction": selected / source_tokens,
        "native_kv_tokens": 0,
        "materialization_performed": False,
    }


def _bootstrap_identity_ci(rows: list[dict], field: str, *, replicates: int = 2000) -> dict:
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["example_id"]].append(float(row[field]))
    identities = sorted(by_id)
    if not identities:
        return {"mean": None, "ci_low": None, "ci_high": None, "identities": 0}
    identity_values = [statistics.fmean(by_id[identity]) for identity in identities]
    generator = torch.Generator().manual_seed(20260814)
    samples = []
    for _ in range(replicates):
        indices = torch.randint(len(identity_values), (len(identity_values),), generator=generator)
        samples.append(statistics.fmean(identity_values[int(index)] for index in indices))
    samples.sort()
    return {
        "mean": statistics.fmean(identity_values),
        "ci_low": samples[int(0.025 * replicates)],
        "ci_high": samples[int(0.975 * replicates) - 1],
        "identities": len(identities),
    }


def _prepare(args, device):
    features = torch.load(args.feature_file, map_location="cpu", weights_only=False)
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
    prepared, mapping_rows, transition_rows, systems, facet_rows = [], [], [], [], []
    for feature_index, feature in enumerate(features, start=1):
        example = _feature_example(feature)
        facets = build_facets(feature, facet_config, support_mode, None)
        for chunk_size in args.chunk_sizes:
            mapping = map_example_to_parents(
                example,
                int(feature["source_tokens"]),
                feature["node_token_spans"],
                chunk_size=chunk_size,
            )
            if not mapping.root_parent_ids:
                raise ValueError(f"No oracle root for {feature['example_id']}.")
            parent_hidden = _parent_hidden(feature["token_hidden"], mapping.parent_spans)
            local_parents = torch.tensor(
                [int(start) // chunk_size for start, _ in feature["local_spans"]], dtype=torch.long
            )
            q, k, mask = (
                feature["local_pre_query"],
                feature["local_pre_key"],
                feature["local_token_mask"],
            )
            h2d_bytes = sum(value.numel() * value.element_size() for value in (q, k, mask, local_parents))
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            h2d_started = time.perf_counter()
            qd, kd, md, pd = q.to(device), k.to(device), mask.to(device), local_parents.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - h2d_started
            adjacency_started = time.perf_counter()
            adjacency = build_native_parent_adjacency(
                qd,
                kd,
                md,
                pd,
                len(mapping.parent_spans),
                token_reduction="top_m_mean",
                head_reduction="top_m_mean",
                top_m=4,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            adjacency_seconds = time.perf_counter() - adjacency_started
            edge_scores = adjacency.scores.detach().cpu()
            item_transitions = _transition_rows(feature, example, mapping, edge_scores)
            transition_rows.extend(item_transitions)
            semantic_scores = {}
            for seed, projection in projections.items():
                scored = score_semantic_query_facets(
                    projection.project_query(facets.hidden.to(device)),
                    projection.project_memory(parent_hidden.to(device)),
                )
                scores = scored.component_scores[:, 0, :].detach().cpu()
                semantic_scores[seed] = scores
                if chunk_size == PRIMARY_CHUNK:
                    normalized = normalize_facet_scores(scores)
                    stats = facet_parent_statistics(scores)
                    ordered_nodes = [node.node_id for node in example.nodes]
                    path = []
                    for node_id in ordered_nodes:
                        group = mapping.node_parent_groups.get(node_id, ())
                        if not group:
                            continue
                        parent = group[0]
                        previous = tuple(path)
                        path.append(parent)
                        facet_rows.append(
                            {
                                "dataset": feature["dataset"],
                                "example_id": feature["example_id"],
                                "partition": feature["partition"],
                                "annotated_hops": feature["annotated_hops"],
                                "graph_type": feature["graph_type"],
                                "seed": seed,
                                "node_id": node_id,
                                "parent_id": parent,
                                "facet_count": scores.shape[0],
                                "max_facet_score": stats[parent].maximum,
                                "top2_facet_score": stats[parent].top2_mean,
                                "winning_facet": stats[parent].winning_facet,
                                "path_facet_coverage": path_facet_coverage(normalized, path),
                                "incremental_facet_coverage": incremental_facet_coverage(
                                    normalized, previous, parent
                                ),
                            }
                        )
            prepared.append(
                {
                    "feature": feature,
                    "example": example,
                    "mapping": mapping,
                    "edge_scores": edge_scores,
                    "semantic_scores": semantic_scores,
                    "transition_rows": item_transitions,
                    "adjacency_seconds": adjacency_seconds,
                    "native_dot_products": adjacency.dot_products,
                    "feature_build_seconds": feature["capture_seconds"],
                    "h2d_bytes": h2d_bytes,
                    "h2d_seconds": h2d_seconds,
                }
            )
            mapping_rows.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "partition": feature["partition"],
                    "question_type": feature["question_type"],
                    "annotated_hops": feature["annotated_hops"],
                    "graph_type": feature["graph_type"],
                    "chunk_size": chunk_size,
                    "source_tokens": feature["source_tokens"],
                    "parent_count": len(mapping.parent_spans),
                    "oracle_parent_count": len(mapping.oracle_parent_ids),
                    "annotated_node_count": len(example.nodes),
                    "annotated_transition_count": len(example.annotated_edges),
                    "preserved_parent_transition_count": len(mapping.preserved_edges),
                    "collapsed_transition_count": len(mapping.collapsed_node_edges),
                    "partially_collapsed_transition_count": sum(
                        bool(
                            set(mapping.node_parent_groups.get(source, ())).intersection(
                                mapping.node_parent_groups.get(target, ())
                            )
                        )
                        for source, target in example.annotated_edges
                    ),
                    "unmappable_transition_count": len(mapping.unmappable_node_edges),
                    "root_parent_count": len(mapping.root_parent_ids),
                }
            )
            systems.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "chunk_size": chunk_size,
                    "source_tokens": feature["source_tokens"],
                    "parent_count": len(mapping.parent_spans),
                    "local_count": len(feature["local_spans"]),
                    "native_dot_products": adjacency.dot_products,
                    "adjacency_build_seconds": adjacency_seconds,
                    "feature_build_seconds": feature["capture_seconds"],
                    "routing_search_cache_bytes": edge_scores.numel() * edge_scores.element_size(),
                    "h2d_bytes": h2d_bytes,
                    "h2d_seconds": h2d_seconds,
                    "peak_gpu_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                    "peak_gpu_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
                    ),
                }
            )
        print(
            f"[natural-adjacency {feature_index}/{len(features)}] "
            f"{feature['dataset']} {feature['example_id']}",
            flush=True,
        )
    return prepared, mapping_rows, transition_rows, systems, facet_rows


def _oracle_rows(prepared):
    rows, survival = [], []
    for item in prepared:
        feature, example, mapping = item["feature"], item["example"], item["mapping"]
        depths = _node_depths(example)
        for k in K_VALUES:
            for b in B_VALUES:
                for h in H_VALUES:
                    result = _search(item["edge_scores"], mapping.root_parent_ids, k, h, b)
                    recall, complete, node_hit = _node_recovery(result.visited, mapping)
                    preserved = [row for row in item["transition_rows"] if row["mapping_status"] == "preserved"]
                    edge_recall = (
                        statistics.fmean(float(row[f"recovered_at_{k}"]) for row in preserved)
                        if preserved
                        else 1.0
                    )
                    token_metrics = _selected_token_metrics(
                        result.visited, mapping, int(feature["source_tokens"])
                    )
                    row = {
                            "dataset": feature["dataset"],
                            "example_id": feature["example_id"],
                            "partition": feature["partition"],
                            "question_type": feature["question_type"],
                            "annotated_hops": feature["annotated_hops"],
                            "graph_type": feature["graph_type"],
                            "chunk_size": mapping.parent_spans[0][1] - mapping.parent_spans[0][0],
                            "root_mode": "oracle_annotated_entry",
                            "R_root": len(mapping.root_parent_ids),
                            "K": k,
                            "H": h,
                            "B": "none" if b is None else b,
                            "parent_count": len(mapping.parent_spans),
                            "oracle_parent_count": len(mapping.oracle_parent_ids),
                            "annotated_transition_count": len(example.annotated_edges),
                            "preserved_transition_count": len(preserved),
                            "visited_count": len(result.visited),
                            "nodes_expanded": result.nodes_expanded,
                            "peak_frontier": result.peak_frontier,
                            "search_comparisons": result.raw_proposals,
                            "duplicates": result.duplicate_proposals,
                            "cycles": result.cycles_prevented,
                            "candidate_tensor_bytes": result.peak_candidate_tensor_bytes,
                            "oracle_node_recall": recall,
                            "annotated_edge_recall": edge_recall,
                            "complete_graph": int(complete),
                            "complete_native_path": int(all(row[f"recovered_at_{k}"] for row in preserved)),
                            "extra_non_oracle_visited": len(set(result.visited) - set(mapping.oracle_parent_ids)),
                            "branching_overhead": result.raw_proposals / max(1, result.nodes_expanded),
                            "adjacency_build_seconds": item["adjacency_seconds"],
                            "search_seconds": result.search_seconds,
                            "total_routing_seconds": item["adjacency_seconds"] + result.search_seconds,
                            **token_metrics,
                        }
                    rows.append(row)
                    admitted_by_hop = defaultdict(set)
                    for decision in result.decisions:
                        if decision.admitted:
                            admitted_by_hop[decision.hop].add(decision.candidate_parent)
                    for node in example.nodes:
                        required_hop = depths[node.node_id] - 1
                        group = mapping.node_parent_groups.get(node.node_id, ())
                        exact = set(mapping.root_parent_ids) if required_hop == 0 else admitted_by_hop[required_hop]
                        survival.append(
                            {
                                "dataset": feature["dataset"],
                                "example_id": feature["example_id"],
                                "partition": feature["partition"],
                                "annotated_hops": feature["annotated_hops"],
                                "graph_type": feature["graph_type"],
                                "chunk_size": mapping.parent_spans[0][1] - mapping.parent_spans[0][0],
                                "K": k,
                                "H": h,
                                "B": "none" if b is None else b,
                                "node_id": node.node_id,
                                "annotated_step": depths[node.node_id],
                                "required_search_hop": required_hop,
                                "frontier_survival": int(bool(set(group).intersection(exact))),
                                "cumulative_recovery": int(node_hit.get(node.node_id, False)),
                            }
                        )
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["example_id"],
            row["chunk_size"],
            row["K"],
            row["B"],
        )
        grouped[key].append(row)
    minimum_depth = {}
    for key, values in grouped.items():
        complete = sorted(int(row["H"]) for row in values if row["complete_graph"])
        minimum_depth[key] = complete[0] if complete else "unrecovered"
    for row in rows:
        key = (
            row["dataset"],
            row["example_id"],
            row["chunk_size"],
            row["K"],
            row["B"],
        )
        row["minimum_recovery_depth"] = minimum_depth[key]
    return rows, survival


def _select_operating_point(rows):
    candidates = []
    for k in K_VALUES:
        for b in B_VALUES:
            selected = [
                row
                for row in rows
                if row["partition"] == "validation"
                and row["chunk_size"] == PRIMARY_CHUNK
                and row["H"] == 4
                and row["K"] == k
                and row["B"] == ("none" if b is None else b)
            ]
            candidates.append(
                {
                    "K": k,
                    "B": "none" if b is None else b,
                    "complete_graph": _mean(selected, "complete_graph"),
                    "oracle_node_recall": _mean(selected, "oracle_node_recall"),
                    "mean_visited": _mean(selected, "visited_count"),
                }
            )
    selected = max(
        candidates,
        key=lambda row: (row["complete_graph"], row["oracle_node_recall"], -row["mean_visited"]),
    )
    return selected, candidates


def _routed_rows(prepared, operating_point, gates):
    rows = []
    k = int(operating_point["K"])
    b = None if operating_point["B"] == "none" else int(operating_point["B"])
    for item in prepared:
        feature, mapping = item["feature"], item["mapping"]
        if mapping.parent_spans[0][1] - mapping.parent_spans[0][0] != PRIMARY_CHUNK:
            continue
        if not gates[feature["dataset"]]["passed"]:
            continue
        for seed, scores in item["semantic_scores"].items():
            parent_scores = scores.max(dim=0).values
            order = torch.argsort(parent_scores, descending=True, stable=True)
            for root_breadth in (1, 2, 4):
                roots = tuple(int(value) for value in order[:root_breadth])
                result = _search(item["edge_scores"], roots, k, 4, b)
                recall, complete, _ = _node_recovery(result.visited, mapping)
                root_present = bool(set(roots).intersection(mapping.root_parent_ids))
                rows.append(
                    {
                        "dataset": feature["dataset"],
                        "example_id": feature["example_id"],
                        "partition": feature["partition"],
                        "seed": seed,
                        "root_mode": "bounded_contextual_facets",
                        "R_root": root_breadth,
                        "K": k,
                        "H": 4,
                        "B": operating_point["B"],
                        "correct_root_present": int(root_present),
                        "oracle_node_recall": recall,
                        "complete_graph": int(complete),
                        "conditional_complete_graph": int(complete) if root_present else "",
                        "visited_count": len(result.visited),
                        "nodes_expanded": result.nodes_expanded,
                        "search_comparisons": result.raw_proposals,
                        "search_seconds": result.search_seconds,
                    }
                )
    return rows


def _aggregate(mapping_rows, transitions, oracle, survival, systems, facets, routed, op, candidates):
    selected = [
        row
        for row in oracle
        if row["chunk_size"] == PRIMARY_CHUNK
        and row["K"] == op["K"]
        and row["H"] == 4
        and row["B"] == op["B"]
    ]
    gates = {}
    headline = []
    for dataset in ("musique", "2wikimultihopqa"):
        heldout = [row for row in selected if row["dataset"] == dataset and row["partition"] == "test"]
        ci = _bootstrap_identity_ci(heldout, "complete_graph")
        recall_ci = _bootstrap_identity_ci(heldout, "oracle_node_recall")
        passed = ci["mean"] >= 0.5 and recall_ci["mean"] >= 0.6
        gates[dataset] = {
            "passed": passed,
            "criterion": "held-out complete graph >=0.50 and node recall >=0.60",
            "complete_graph": ci,
            "oracle_node_recall": recall_ci,
        }
        headline.append(
            {
                "dataset": dataset,
                "complete_graph": ci["mean"],
                "complete_ci_low": ci["ci_low"],
                "complete_ci_high": ci["ci_high"],
                "oracle_node_recall": recall_ci["mean"],
                "recall_ci_low": recall_ci["ci_low"],
                "recall_ci_high": recall_ci["ci_high"],
                "mean_visited": _mean(heldout, "visited_count"),
                "mean_search_ms": 1000 * _mean(heldout, "search_seconds"),
            }
        )
    transition_curve = []
    wiki_transitions = [
        row
        for row in transitions
        if row["dataset"] == "2wikimultihopqa"
        and row["partition"] == "test"
        and row["chunk_size"] == PRIMARY_CHUNK
        and row["mapping_status"] == "preserved"
    ]
    for k in RANK_K_VALUES:
        identities = defaultdict(list)
        for row in wiki_transitions:
            identities[row["example_id"]].append(int(row[f"recovered_at_{k}"]))
        path_survival = statistics.fmean(all(values) for values in identities.values())
        reachable_distances = [
            float(row[f"shortest_path_at_{k}"])
            for row in wiki_transitions
            if row[f"shortest_path_at_{k}"] != ""
        ]
        transition_curve.append(
            {
                "K": k,
                "transition_recall": _mean(wiki_transitions, f"recovered_at_{k}"),
                "MRR": _mean(wiki_transitions, "reciprocal_rank"),
                "complete_path_survival": path_survival,
                "reachable_within_H4": len(reachable_distances) / len(wiki_transitions),
                "mean_shortest_path_if_reachable": statistics.fmean(reachable_distances),
                "mean_search_comparisons_per_source": _mean(
                    [
                        {
                            "comparisons": next(
                                row["parent_count"] - 1
                                for row in mapping_rows
                                if row["example_id"] == transition["example_id"]
                                and row["chunk_size"] == PRIMARY_CHUNK
                            )
                        }
                        for transition in wiki_transitions
                    ],
                    "comparisons",
                ),
                "transitions": len(wiki_transitions),
                "paths": len(identities),
            }
        )
    depth_rows = []
    for depth in (2, 3, 4):
        values = [
            row
            for row in selected
            if row["dataset"] == "musique"
            and row["partition"] == "test"
            and row["annotated_hops"] == depth
        ]
        depth_rows.append(
            {
                "annotated_hops": depth,
                "oracle_node_recall": _mean(values, "oracle_node_recall"),
                "complete_graph": _mean(values, "complete_graph"),
                "nodes_expanded": _mean(values, "nodes_expanded"),
            }
        )
    topology = []
    for chunk in CHUNK_SIZES:
        for dataset in ("musique", "2wikimultihopqa"):
            values = [row for row in mapping_rows if row["dataset"] == dataset and row["chunk_size"] == chunk]
            topology.append(
                {
                    "dataset": dataset,
                    "chunk_size": chunk,
                    "mean_parent_count": _mean(values, "parent_count"),
                    "mean_oracle_parent_count": _mean(values, "oracle_parent_count"),
                    "collapsed_transitions": sum(row["collapsed_transition_count"] for row in values),
                    "unmappable_transitions": sum(row["unmappable_transition_count"] for row in values),
                }
            )
    systems_summary = []
    for chunk in CHUNK_SIZES:
        for dataset in ("musique", "2wikimultihopqa"):
            values = [row for row in systems if row["dataset"] == dataset and row["chunk_size"] == chunk]
            systems_summary.append(
                {
                    "dataset": dataset,
                    "chunk_size": chunk,
                    "mean_parent_count": _mean(values, "parent_count"),
                    "mean_adjacency_seconds": _mean(values, "adjacency_build_seconds"),
                    "mean_native_dot_products": _mean(values, "native_dot_products"),
                    "mean_cache_bytes": _mean(values, "routing_search_cache_bytes"),
                    "mean_peak_gpu_allocated": _mean(values, "peak_gpu_allocated_bytes"),
                    "mean_feature_build_seconds": _mean(values, "feature_build_seconds"),
                }
            )
    budget_summary = []
    for dataset in ("musique", "2wikimultihopqa"):
        for k in (4, 6):
            for b in ("6", "16"):
                values = [
                    row
                    for row in oracle
                    if row["dataset"] == dataset
                    and row["partition"] == "test"
                    and row["chunk_size"] == PRIMARY_CHUNK
                    and row["K"] == k
                    and row["H"] == 4
                    and str(row["B"]) == b
                ]
                budget_summary.append(
                    {
                        "dataset": dataset,
                        "K": k,
                        "B": b,
                        "oracle_node_recall": _mean(values, "oracle_node_recall"),
                        "complete_graph": _mean(values, "complete_graph"),
                        "mean_visited": _mean(values, "visited_count"),
                        "mean_active_kv_fraction": _mean(values, "active_kv_fraction"),
                        "mean_parent_count": _mean(values, "parent_count"),
                    }
                )
    hop_scaling = []
    for dataset in ("musique", "2wikimultihopqa"):
        for h in H_VALUES:
            values = [
                row
                for row in oracle
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["chunk_size"] == PRIMARY_CHUNK
                and row["K"] == op["K"]
                and row["B"] == op["B"]
                and row["H"] == h
            ]
            hop_scaling.append(
                {
                    "dataset": dataset,
                    "H": h,
                    "oracle_node_recall": _mean(values, "oracle_node_recall"),
                    "complete_graph": _mean(values, "complete_graph"),
                    "mean_visited": _mean(values, "visited_count"),
                }
            )
    routed_summary = []
    for dataset in ("musique", "2wikimultihopqa"):
        for root_breadth in (1, 2, 4):
            values = [
                row
                for row in routed
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["R_root"] == root_breadth
            ]
            conditional = [row for row in values if row["conditional_complete_graph"] != ""]
            if values:
                routed_summary.append(
                    {
                        "dataset": dataset,
                        "R_root": root_breadth,
                        "root_recall": _mean(values, "correct_root_present"),
                        "conditional_complete_graph": _mean(
                            conditional, "conditional_complete_graph"
                        ),
                        "end_to_end_complete_graph": _mean(values, "complete_graph"),
                        "oracle_node_recall": _mean(values, "oracle_node_recall"),
                        "mean_visited": _mean(values, "visited_count"),
                    }
                )
    graph_type_summary = []
    for dataset in ("musique", "2wikimultihopqa"):
        types = sorted(
            {
                row["graph_type"]
                for row in selected
                if row["dataset"] == dataset and row["partition"] == "test"
            }
        )
        for graph_type in types:
            values = [
                row
                for row in selected
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["graph_type"] == graph_type
            ]
            graph_type_summary.append(
                {
                    "dataset": dataset,
                    "graph_type": graph_type,
                    "examples": len(values),
                    "oracle_node_recall": _mean(values, "oracle_node_recall"),
                    "complete_graph": _mean(values, "complete_graph"),
                    "mean_visited": _mean(values, "visited_count"),
                }
            )
    return {
        "selected_operating_point": op,
        "validation_operating_points": candidates,
        "oracle_gate": gates,
        "headline_heldout": headline,
        "2wiki_transition_curve": transition_curve,
        "musique_depth_scaling": depth_rows,
        "topology_mapping": topology,
        "systems": systems_summary,
        "budget_tradeoff": budget_summary,
        "search_depth_scaling": hop_scaling,
        "routed_root_decomposition": routed_summary,
        "graph_type_summary": graph_type_summary,
        "row_counts": {
            "mapping": len(mapping_rows),
            "transitions": len(transitions),
            "oracle_search": len(oracle),
            "hop_survival": len(survival),
            "facets": len(facets),
            "routed": len(routed),
        },
    }, gates


def _plots(aggregate, oracle, survival, output_dir):
    selected = aggregate["selected_operating_point"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    curve = aggregate["2wiki_transition_curve"]
    axes[0].plot([row["K"] for row in curve], [row["transition_recall"] for row in curve], marker="o")
    axes[0].set(xlabel="Native successor K", ylabel="Transition recall", title="2Wiki path edges", ylim=(0, 1.02))
    for depth in (2, 3, 4):
        values = []
        for step in range(1, depth + 1):
            rows = [
                row
                for row in survival
                if row["dataset"] == "musique"
                and row["partition"] == "test"
                and row["annotated_hops"] == depth
                and row["chunk_size"] == PRIMARY_CHUNK
                and row["K"] == selected["K"]
                and row["H"] == 4
                and row["B"] == selected["B"]
                and row["annotated_step"] == step
            ]
            values.append(_mean(rows, "frontier_survival"))
        axes[1].plot(range(1, depth + 1), values, marker="o", label=f"D={depth}")
    axes[1].set(xlabel="Annotated evidence step", ylabel="Exact-hop frontier survival", title="MuSiQue depth survival", ylim=(0, 1.02))
    axes[1].legend(frameon=False)
    depths = aggregate["musique_depth_scaling"]
    axes[2].plot([row["annotated_hops"] for row in depths], [row["complete_graph"] for row in depths], marker="o", label="Complete")
    axes[2].plot([row["annotated_hops"] for row in depths], [row["oracle_node_recall"] for row in depths], marker="s", label="Node recall")
    axes[2].set(xlabel="Annotated task depth", ylabel="Held-out recovery", title="Depth-normalized quality", ylim=(0, 1.02))
    axes[2].legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"natural_graph_depth.{suffix}", dpi=180)
    plt.close(figure)


def run(args):
    device = torch.device(args.device)
    prepared, mappings, transitions, systems, facets = _prepare(args, device)
    oracle, survival = _oracle_rows(prepared)
    operating_point, candidates = _select_operating_point(oracle)
    provisional, gates = _aggregate(
        mappings, transitions, oracle, survival, systems, facets, [], operating_point, candidates
    )
    routed = _routed_rows(prepared, operating_point, gates)
    aggregate, gates = _aggregate(
        mappings, transitions, oracle, survival, systems, facets, routed, operating_point, candidates
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("natural_graph_mapping_rows.csv", mappings),
        ("natural_graph_transition_rows.csv", transitions),
        ("natural_graph_oracle_search_rows.csv", oracle),
        ("natural_graph_hop_survival_rows.csv", survival),
        ("natural_graph_system_rows.csv", systems),
        ("natural_graph_facet_rows.csv", facets),
        ("natural_graph_routed_rows.csv", routed),
    ):
        _write_csv(args.output_dir / name, rows)
    _plots(aggregate, oracle, survival, args.output_dir)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "training_performed": False,
        "generation_performed": False,
        "oracle_labels_available_during_search": False,
        "oracle_evaluation_post_hoc": True,
        "terminal_query_stopping": False,
        "search_strategy": STRATEGY,
        "chunk_sizes": list(args.chunk_sizes),
        "primary_chunk_size": PRIMARY_CHUNK,
        "K_values": list(K_VALUES),
        "H_values": list(H_VALUES),
        "B_values": ["none" if value is None else value for value in B_VALUES],
        "rank_K_values": list(RANK_K_VALUES),
        "facet_policy": "frozen w2_s1 latest-message facets; root/diagnostic only",
        "native_edge_policy": "exact dense layer-27 pre-RoPE QK Top-4 token/head mean",
        "native_kv_materialization_performed": False,
        "aggregate": aggregate,
    }
    (args.output_dir / "natural_graph_depth_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser()
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth"
    parser.add_argument("--feature-file", type=Path, default=output / "natural_graph_features.pt")
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=list(CHUNK_SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--facet-gate-file",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/grounded_query_facets/grounded_facet_gate_results.json",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
