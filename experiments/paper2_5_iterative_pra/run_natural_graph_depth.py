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
from pra_hf.cross_dataset_diagnostics import evidence_token_metrics
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


CHUNK_SIZES = (16, 32, 64, 128, 256)
PRIMARY_CHUNK = 128
K_VALUES = (1, 2, 4, 6, 8)
RANK_K_VALUES = (1, 2, 3, 4, 5, 6, 8, 11)
H_VALUES = (0, 1, 2, 3, 4)
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


def _atomic_native(feature: dict, chunk_size: int) -> tuple[torch.Tensor, ...]:
    """Map contextual 32-token native blocks to exact 16+ token parents."""
    if chunk_size >= 32:
        spans = [tuple(map(int, span)) for span in feature["local_spans"]]
        return (
            feature["local_pre_query"],
            feature["local_pre_key"],
            feature["local_token_mask"],
            torch.tensor([start // chunk_size for start, _ in spans], dtype=torch.long),
        )
    queries, keys, masks, parent_ids = [], [], [], []
    for index, (start, end) in enumerate(feature["local_spans"]):
        start, end = int(start), int(end)
        for offset in range(0, 32, chunk_size):
            piece_start = start + offset
            if piece_start >= end:
                continue
            queries.append(feature["local_pre_query"][index, offset : offset + chunk_size])
            keys.append(feature["local_pre_key"][index, offset : offset + chunk_size])
            masks.append(feature["local_token_mask"][index, offset : offset + chunk_size])
            parent_ids.append(piece_start // chunk_size)
    return (
        torch.stack(queries),
        torch.stack(keys),
        torch.stack(masks),
        torch.tensor(parent_ids, dtype=torch.long),
    )


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
    depths = _node_depths(example)
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
                "source_annotated_step": depths[source_node],
                "target_annotated_step": depths[target_node],
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


def _strict_path_survival(rows: list[dict], k: int) -> float:
    """Require every annotated edge to remain distinct and rank within K."""
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["example_id"]].append(row)
    eligible = [values for values in by_id.values() if values]
    return (
        statistics.fmean(
            all(
                row["mapping_status"] == "preserved" and int(row[f"recovered_at_{k}"])
                for row in values
            )
            for values in eligible
        )
        if eligible
        else 1.0
    )


def _preserved_path_survival(rows: list[dict], k: int) -> float:
    """Measure paths conditional on every reported edge remaining distinct."""
    by_id = defaultdict(list)
    for row in rows:
        if row["mapping_status"] == "preserved":
            by_id[row["example_id"]].append(int(row[f"recovered_at_{k}"]))
    return statistics.fmean(all(values) for values in by_id.values()) if by_id else 1.0


def _product_path_survival(rows: list[dict], k: int) -> float:
    """Estimate strict path survival from independent per-step edge rates."""
    by_step = defaultdict(list)
    by_id = defaultdict(list)
    for row in rows:
        success = int(
            row["mapping_status"] == "preserved" and int(row[f"recovered_at_{k}"])
        )
        by_step[int(row["target_annotated_step"])].append(success)
        by_id[row["example_id"]].append(int(row["target_annotated_step"]))
    rates = {step: statistics.fmean(values) for step, values in by_step.items()}
    products = [
        math.prod(rates.get(step, 0.0) for step in steps)
        for steps in by_id.values()
        if steps
    ]
    return statistics.fmean(products) if products else 1.0


def _pareto_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return aggregate sparse-recovery points and their quality/cost frontier."""
    points = []
    for dataset in ("musique", "2wikimultihopqa"):
        for chunk in CHUNK_SIZES:
            for k in K_VALUES:
                for b in B_VALUES:
                    selected = [
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["partition"] == "test"
                        and row["chunk_size"] == chunk
                        and row["K"] == k
                        and row["H"] == 4
                        and row["B"] == ("none" if b is None else b)
                    ]
                    points.append(
                        {
                            "dataset": dataset,
                            "chunk_size": chunk,
                            "K": k,
                            "B": "none" if b is None else b,
                            "complete_recovery": _mean(selected, "complete_graph"),
                            "node_recall": _mean(selected, "oracle_node_recall"),
                            "selected_source_fraction": _mean(selected, "active_kv_fraction"),
                            "evidence_density": _mean(selected, "evidence_density"),
                            "nodes_expanded": _mean(selected, "nodes_expanded"),
                        }
                    )
    frontier = []
    for point in points:
        dominated = any(
            other["dataset"] == point["dataset"]
            and other["complete_recovery"] >= point["complete_recovery"]
            and other["selected_source_fraction"] <= point["selected_source_fraction"]
            and (
                other["complete_recovery"] > point["complete_recovery"]
                or other["selected_source_fraction"] < point["selected_source_fraction"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return points, frontier


def _granularity_aggregate(mapping_rows, transitions, oracle, systems, op) -> dict:
    """Build the cross-dataset Gate-1 tables from frozen execution rows."""
    central, musique_depth, transition_by_g, path_by_g = [], [], [], []
    decomposition, systems_by_g, graph_type_by_g = [], [], []
    for dataset in ("musique", "2wikimultihopqa"):
        for chunk in CHUNK_SIZES:
            mapped = [
                row
                for row in mapping_rows
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["chunk_size"] == chunk
            ]
            selected = [
                row
                for row in oracle
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["chunk_size"] == chunk
                and row["K"] == op["K"]
                and row["H"] == 4
                and row["B"] == op["B"]
            ]
            edge_rows = [
                row
                for row in transitions
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["chunk_size"] == chunk
            ]
            preserved_edges = [
                row for row in edge_rows if row["mapping_status"] == "preserved"
            ]
            edge_r6 = (
                _mean(preserved_edges, "recovered_at_6") if preserved_edges else 1.0
            )
            central.append(
                {
                    "dataset": dataset,
                    "chunk_size": chunk,
                    "G_encode": 256,
                    "G_search": chunk,
                    "mean_parent_count": _mean(mapped, "parent_count"),
                    "mean_oracle_parents": _mean(mapped, "oracle_parent_count"),
                    "root_evidence_fraction": _mean(mapped, "root_evidence_fraction"),
                    "root_contains_all_evidence": _mean(mapped, "root_contains_all_evidence"),
                    "mean_evidence_group_collisions": _mean(
                        mapped, "evidence_group_collisions"
                    ),
                    "edge_R_at_6": edge_r6,
                    "complete_recovery": _mean(selected, "complete_graph"),
                    "mean_minimum_native_depth": _mean(
                        [
                            {"depth": row["minimum_recovery_depth"]}
                            for row in selected
                            if row["minimum_recovery_depth"] != "unrecovered"
                        ],
                        "depth",
                    ),
                    "selected_source_fraction": _mean(selected, "active_kv_fraction"),
                    "evidence_density": _mean(selected, "evidence_density"),
                    "nodes_expanded": _mean(selected, "nodes_expanded"),
                    "preserved_transitions": sum(
                        row["mapping_status"] == "preserved" for row in edge_rows
                    ),
                    "collapsed_transitions": sum(
                        int(row["collapsed_transition_count"]) for row in mapped
                    ),
                    "unmappable_transitions": sum(
                        int(row["unmappable_transition_count"]) for row in mapped
                    ),
                }
            )
            for k in (1, 2, 4, 6, 8, 11):
                transition_by_g.append(
                    {
                        "dataset": dataset,
                        "chunk_size": chunk,
                        "K": k,
                        "edge_recall": (
                            _mean(preserved_edges, f"recovered_at_{k}")
                            if preserved_edges
                            else 1.0
                        ),
                        "strict_complete_path_survival": _strict_path_survival(edge_rows, k),
                        "preserved_path_survival": _preserved_path_survival(edge_rows, k),
                        "product_model_path_survival": _product_path_survival(edge_rows, k),
                        "mean_preserved_path_length": (
                            sum(row["mapping_status"] == "preserved" for row in edge_rows)
                            / len({row["example_id"] for row in edge_rows})
                            if edge_rows
                            else 0.0
                        ),
                    }
                )
            selected_systems = [
                row
                for row in systems
                if row["dataset"] == dataset and row["chunk_size"] == chunk
            ]
            systems_by_g.append(
                {
                    "dataset": dataset,
                    "chunk_size": chunk,
                    "mean_parent_count": _mean(selected_systems, "parent_count"),
                    "mean_adjacency_seconds": _mean(selected_systems, "adjacency_build_seconds"),
                    "mean_native_dot_products": _mean(selected_systems, "native_dot_products"),
                    "mean_local_pair_count": _mean(selected_systems, "local_pair_count"),
                    "mean_candidate_tensor_bytes": _mean(selected_systems, "candidate_tensor_bytes"),
                    "mean_cache_bytes": _mean(selected_systems, "routing_search_cache_bytes"),
                    "mean_peak_cuda_allocated": _mean(selected_systems, "peak_gpu_allocated_bytes"),
                    "mean_peak_cuda_reserved": _mean(selected_systems, "peak_gpu_reserved_bytes"),
                    "mean_search_seconds": _mean(selected, "search_seconds"),
                }
            )
            for graph_type in sorted({row["graph_type"] for row in mapped}):
                type_rows = [row for row in selected if row["graph_type"] == graph_type]
                type_mapped = [row for row in mapped if row["graph_type"] == graph_type]
                graph_type_by_g.append(
                    {
                        "dataset": dataset,
                        "chunk_size": chunk,
                        "graph_type": graph_type,
                        "examples": len(type_rows),
                        "oracle_parents": _mean(type_mapped, "oracle_parent_count"),
                        "node_recall": _mean(type_rows, "oracle_node_recall"),
                        "complete_recovery": _mean(type_rows, "complete_graph"),
                        "nodes_expanded": _mean(type_rows, "nodes_expanded"),
                    }
                )

    for depth in (2, 3, 4):
        for chunk in CHUNK_SIZES:
            base = [
                row
                for row in oracle
                if row["dataset"] == "musique"
                and row["partition"] == "test"
                and row["annotated_hops"] == depth
                and row["chunk_size"] == chunk
                and row["K"] == op["K"]
                and row["B"] == op["B"]
            ]
            mapped = [
                row
                for row in mapping_rows
                if row["dataset"] == "musique"
                and row["partition"] == "test"
                and row["annotated_hops"] == depth
                and row["chunk_size"] == chunk
            ]
            h_rows = {h: [row for row in base if row["H"] == h] for h in H_VALUES}
            h4 = h_rows[4]
            musique_depth.append(
                {
                    "annotated_depth": depth,
                    "chunk_size": chunk,
                    "oracle_parents": _mean(mapped, "oracle_parent_count"),
                    "root_evidence_fraction": _mean(mapped, "root_evidence_fraction"),
                    **{f"H{h}_recall": _mean(h_rows[h], "oracle_node_recall") for h in H_VALUES},
                    "complete_recovery": _mean(h4, "complete_graph"),
                    "mean_minimum_native_depth": _mean(
                        [
                            {"depth": row["minimum_recovery_depth"]}
                            for row in h4
                            if row["minimum_recovery_depth"] != "unrecovered"
                        ],
                        "depth",
                    ),
                    "nodes_expanded": _mean(h4, "nodes_expanded"),
                    "selected_source_fraction": _mean(h4, "active_kv_fraction"),
                }
            )

    for dataset in ("musique", "2wikimultihopqa"):
        for chunk in CHUNK_SIZES:
            edge_rows = [
                row
                for row in transitions
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["chunk_size"] == chunk
            ]
            for k in K_VALUES:
                observed_rows = [
                    row
                    for row in oracle
                    if row["dataset"] == dataset
                    and row["partition"] == "test"
                    and row["chunk_size"] == chunk
                    and row["K"] == k
                    and row["H"] == 4
                    and row["B"] == op["B"]
                ]
                preserved_edges = [
                    row for row in edge_rows if row["mapping_status"] == "preserved"
                ]
                local = (
                    _mean(preserved_edges, f"recovered_at_{k}")
                    if preserved_edges
                    else 1.0
                )
                product = _product_path_survival(preserved_edges, k)
                observed = _mean(observed_rows, "complete_graph")
                decomposition.append(
                    {
                        "dataset": dataset,
                        "chunk_size": chunk,
                        "K": k,
                        "local_edge_recall": local,
                        "product_expected_path_survival": product,
                        "observed_search_complete_recovery": observed,
                        "extra_search_loss": product - observed,
                        "nodes_expanded": _mean(observed_rows, "nodes_expanded"),
                    }
                )

    points, frontier = _pareto_rows(oracle)
    operating_points = []
    for dataset in ("musique", "2wikimultihopqa"):
        values = [row for row in points if row["dataset"] == dataset]
        conservative_pool = [row for row in values if row["complete_recovery"] >= 0.5]
        conservative = min(
            conservative_pool or values,
            key=lambda row: (row["selected_source_fraction"], -row["complete_recovery"]),
        )
        balanced = max(
            values,
            key=lambda row: (
                row["complete_recovery"] - 0.5 * row["selected_source_fraction"],
                row["node_recall"],
            ),
        )
        high_recall = max(
            values,
            key=lambda row: (row["complete_recovery"], -row["selected_source_fraction"]),
        )
        for label, row in (
            ("conservative", conservative),
            ("balanced", balanced),
            ("high_recall", high_recall),
        ):
            operating_points.append({"operating_point": label, **row})
    fine = {(row["dataset"], row["chunk_size"]): row for row in central}
    fine_edge_collapse = any(
            fine[(dataset, size)]["edge_R_at_6"]
            < fine[(dataset, PRIMARY_CHUNK)]["edge_R_at_6"] - 0.15
            for dataset in ("musique", "2wikimultihopqa")
            for size in (16, 32)
        )
    d4 = {
        row["chunk_size"]: row
        for row in musique_depth
        if row["annotated_depth"] == 4
    }
    h2_persists = all(
        math.isclose(d4[size]["H2_recall"], d4[size]["H4_recall"])
        for size in (16, 32)
    )
    fine_payload_gain = all(
        fine[(dataset, 16)]["selected_source_fraction"]
        < 0.5 * fine[(dataset, PRIMARY_CHUNK)]["selected_source_fraction"]
        and fine[(dataset, 16)]["complete_recovery"] >= 0.5
        for dataset in ("musique", "2wikimultihopqa")
    )
    classifications = {
        "A_strong_fine_edges_weak_paths": False,
        "B_fine_edge_representation_collapse": fine_edge_collapse,
        "C_H2_saturation_persists_at_fine_granularity": h2_persists,
        "D_required_H_rises_toward_annotated_depth": False,
        "E_fine_granularity_payload_gain_at_acceptable_recovery": fine_payload_gain,
        "classification": [
            label
            for label, passed in (
                ("B", fine_edge_collapse),
                ("C", h2_persists),
                ("E", fine_payload_gain),
            )
            if passed
        ],
        "contextual_fine_node_control": (
            "active by construction: G_encode=256 and G_search in {16,32}; "
            "no tiny chunk was independently encoded"
        ),
    }
    return {
        "cross_dataset_granularity": central,
        "musique_depth_by_granularity": musique_depth,
        "transition_and_path_by_granularity": transition_by_g,
        "edge_search_decomposition": decomposition,
        "systems_by_granularity": systems_by_g,
        "graph_type_by_granularity": graph_type_by_g,
        "sparse_recovery_points": points,
        "sparse_recovery_frontier": frontier,
        "recommended_operating_points": operating_points,
        "gate1_classification": classifications,
    }


def _selected_token_metrics(visited, mapping, feature: dict) -> dict:
    source_tokens = int(feature["source_tokens"])
    selected = sum(mapping.parent_spans[parent][1] - mapping.parent_spans[parent][0] for parent in visited)
    evidence = evidence_token_metrics(
        tuple(feature["node_token_spans"].values()),
        mapping.parent_spans,
        visited,
        mapping.root_parent_ids,
    )
    return {
        "logical_reference_tokens": source_tokens,
        "conceptual_active_parents": len(visited),
        "counterfactual_native_kv_tokens": selected,
        "active_kv_fraction": selected / source_tokens,
        "native_kv_tokens": 0,
        "materialization_performed": False,
        **evidence,
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
            q, k, mask, local_parents = _atomic_native(feature, chunk_size)
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
                    "evidence_group_collisions": sum(
                        bool(
                            set(mapping.node_parent_groups.get(left.node_id, ())).intersection(
                                mapping.node_parent_groups.get(right.node_id, ())
                            )
                        )
                        for left_index, left in enumerate(example.nodes)
                        for right in example.nodes[left_index + 1 :]
                    ),
                    **evidence_token_metrics(
                        tuple(feature["node_token_spans"].values()),
                        mapping.parent_spans,
                        mapping.oracle_parent_ids,
                        mapping.root_parent_ids,
                    ),
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
                    "local_pair_count": adjacency.local_pair_count,
                    "candidate_tensor_bytes": edge_scores.numel() * edge_scores.element_size(),
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
                    transition_rows = item["transition_rows"]
                    preserved = [row for row in transition_rows if row["mapping_status"] == "preserved"]
                    edge_recall = (
                        statistics.fmean(float(row[f"recovered_at_{k}"]) for row in transition_rows)
                        if transition_rows
                        else 1.0
                    )
                    token_metrics = _selected_token_metrics(
                        result.visited, mapping, feature
                    )
                    later_nodes = [
                        node.node_id for node in example.nodes if depths[node.node_id] > 1
                    ]
                    later_recall = (
                        statistics.fmean(float(node_hit[node]) for node in later_nodes)
                        if later_nodes
                        else 1.0
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
                            "later_evidence_recall": later_recall,
                            "annotated_edge_recall": edge_recall,
                            "complete_graph": int(complete),
                            "complete_native_path": int(
                                all(
                                    row["mapping_status"] == "preserved"
                                    and row[f"recovered_at_{k}"]
                                    for row in transition_rows
                                )
                            ),
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
    # Keep the operating point frozen by the preceding MuSiQue/2Wiki gate.
    # K=8 remains a ceiling diagnostic and must not silently retune the baseline.
    selected = next(row for row in candidates if row["K"] == 6 and row["B"] == 16)
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
    granularity = _granularity_aggregate(mapping_rows, transitions, oracle, systems, op)
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
        **granularity,
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


def _granularity_plots(aggregate: dict, output_dir: Path) -> None:
    central = aggregate["cross_dataset_granularity"]
    transition = aggregate["transition_and_path_by_granularity"]
    systems = aggregate["systems_by_granularity"]
    depth = aggregate["musique_depth_by_granularity"]
    for dataset, label in (("musique", "MuSiQue"), ("2wikimultihopqa", "2Wiki")):
        rows = [row for row in central if row["dataset"] == dataset]
        figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
        axes[0, 0].plot(
            [row["chunk_size"] for row in rows],
            [row["complete_recovery"] for row in rows],
            marker="o",
        )
        axes[0, 0].set(xlabel="Search parent tokens", ylabel="Complete recovery")
        for k in (2, 4, 6, 8):
            values = [
                row for row in transition if row["dataset"] == dataset and row["K"] == k
            ]
            axes[0, 1].plot(
                [row["chunk_size"] for row in values],
                [row["edge_recall"] for row in values],
                marker="o",
                label=f"K={k}",
            )
        axes[0, 1].set(xlabel="Search parent tokens", ylabel="Local edge recall")
        axes[0, 1].legend(frameon=False, fontsize=8)
        axes[0, 2].plot(
            [row["selected_source_fraction"] for row in rows],
            [row["complete_recovery"] for row in rows],
            marker="o",
        )
        axes[0, 2].set(xlabel="Selected source fraction", ylabel="Complete recovery")
        axes[1, 0].plot(
            [row["nodes_expanded"] for row in rows],
            [row["complete_recovery"] for row in rows],
            marker="o",
        )
        axes[1, 0].set(xlabel="Nodes expanded", ylabel="Complete recovery")
        if dataset == "musique":
            for annotated_depth in (2, 3, 4):
                values = [row for row in depth if row["annotated_depth"] == annotated_depth]
                axes[1, 1].plot(
                    [row["chunk_size"] for row in values],
                    [row["mean_minimum_native_depth"] for row in values],
                    marker="o",
                    label=f"D={annotated_depth}",
                )
            axes[1, 1].set(xlabel="Search parent tokens", ylabel="Mean minimum native H")
            axes[1, 1].legend(frameon=False, fontsize=8)
        else:
            for k in (2, 4, 6, 8):
                values = [
                    row for row in transition if row["dataset"] == dataset and row["K"] == k
                ]
                axes[1, 1].plot(
                    [row["chunk_size"] for row in values],
                    [row["strict_complete_path_survival"] for row in values],
                    marker="o",
                    label=f"K={k}",
                )
            axes[1, 1].set(xlabel="Search parent tokens", ylabel="Strict path survival")
            axes[1, 1].legend(frameon=False, fontsize=8)
        system_rows = [row for row in systems if row["dataset"] == dataset]
        axes[1, 2].plot(
            [row["mean_parent_count"] for row in system_rows],
            [row["mean_adjacency_seconds"] for row in system_rows],
            marker="o",
        )
        axes[1, 2].set(xlabel="Mean parent count", ylabel="Dense adjacency seconds")
        for axis in axes.flat:
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.25)
        figure.suptitle(f"{label}: frozen native graph granularity", fontsize=12)
        figure.tight_layout()
        stem = "musique_granularity" if dataset == "musique" else "2wiki_granularity"
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{stem}.{suffix}", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    frontier = aggregate["sparse_recovery_frontier"]
    for axis, (dataset, label) in zip(
        axes, (("musique", "MuSiQue"), ("2wikimultihopqa", "2Wiki"))
    ):
        points = [row for row in frontier if row["dataset"] == dataset]
        axis.scatter(
            [row["selected_source_fraction"] for row in points],
            [row["complete_recovery"] for row in points],
            c=[row["chunk_size"] for row in points],
            cmap="viridis",
            s=45,
        )
        axis.set(
            xlabel="Selected source fraction",
            ylabel="Complete recovery",
            title=label,
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"natural_graph_sparse_frontier.{suffix}", dpi=180)
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
        ("cross_dataset_granularity.csv", aggregate["cross_dataset_granularity"]),
        ("musique_depth_by_granularity.csv", aggregate["musique_depth_by_granularity"]),
        ("transition_path_by_granularity.csv", aggregate["transition_and_path_by_granularity"]),
        ("edge_search_decomposition.csv", aggregate["edge_search_decomposition"]),
        ("systems_by_granularity.csv", aggregate["systems_by_granularity"]),
        ("graph_type_by_granularity.csv", aggregate["graph_type_by_granularity"]),
        ("sparse_recovery_points.csv", aggregate["sparse_recovery_points"]),
        ("sparse_recovery_frontier.csv", aggregate["sparse_recovery_frontier"]),
        ("recommended_operating_points.csv", aggregate["recommended_operating_points"]),
    ):
        _write_csv(args.output_dir / name, rows)
    _plots(aggregate, oracle, survival, args.output_dir)
    _granularity_plots(aggregate, args.output_dir)
    canonical = {
        row["K"]: row
        for row in aggregate["transition_and_path_by_granularity"]
        if row["dataset"] == "2wikimultihopqa"
        and row["chunk_size"] == PRIMARY_CHUNK
        and row["K"] in (4, 6, 8)
    }
    expected = {4: (0.72, 10 / 17), 6: (0.88, 14 / 17), 8: (1.0, 1.0)}
    canonical_exact = all(
        math.isclose(canonical[k]["edge_recall"], edge, abs_tol=1e-12)
        and math.isclose(
            canonical[k]["preserved_path_survival"], path, abs_tol=1e-12
        )
        for k, (edge, path) in expected.items()
    )
    if not canonical_exact:
        raise AssertionError(f"Canonical 2Wiki transition curve changed: {canonical}.")
    artifact = {
        "schema_version": "2.0",
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
        "encoding_granularity_tokens": 256,
        "search_granularity_tokens": list(args.chunk_sizes),
        "overlap_tokens": 0,
        "primary_chunk_size": PRIMARY_CHUNK,
        "K_values": list(K_VALUES),
        "H_values": list(H_VALUES),
        "B_values": ["none" if value is None else value for value in B_VALUES],
        "rank_K_values": list(RANK_K_VALUES),
        "facet_policy": "frozen w2_s1 latest-message facets; root/diagnostic only",
        "native_edge_policy": "exact dense layer-27 pre-RoPE QK Top-4 token/head mean",
        "native_kv_materialization_performed": False,
        "canonical_2wiki_reproduction": {
            "exact": canonical_exact,
            "preserved_transition_R_at_4_6_8": [
                canonical[k]["edge_recall"] for k in (4, 6, 8)
            ],
            "preserved_path_survival_at_4_6_8": [
                canonical[k]["preserved_path_survival"] for k in (4, 6, 8)
            ],
            "strict_path_survival_counts_collapsed_as_failure": [
                canonical[k]["strict_complete_path_survival"] for k in (4, 6, 8)
            ],
        },
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
