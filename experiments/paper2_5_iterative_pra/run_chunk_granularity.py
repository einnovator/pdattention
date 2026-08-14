"""Measure Hotpot evidence topology and oracle-root discovery across chunk sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_grounded_facet_gate import FacetConfig, build_facets
from experiments.paper2_5_iterative_pra.run_oracle_convergence import SEEDS, validation_partition
from pra_hf.chunk_granularity import (
    chunk_spans,
    contracted_chain_depth,
    evaluate_oracle_recovery,
    evidence_topology,
    facet_parent_statistics,
    incremental_facet_coverage,
    minimum_recovery_depth,
    normalize_facet_scores,
    path_facet_coverage,
)
from pra_hf.query_facets import score_semantic_query_facets
from pra_hf.semantic_graph_search import (
    SemanticGraphSearchConfig,
    build_native_parent_adjacency,
    search_semantic_graph,
)
from pra_torch.hf import load_hf_routing_projection


CHUNK_SIZES = (16, 32, 64, 128, 256)
K_VALUES = (1, 2, 4, 6)
H_VALUES = (0, 1, 2, 3, 4)
B_VALUES = (6, 16, None)
SELECTED_K = 4
SELECTED_H = 4
SELECTED_B = 6
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
    return statistics.fmean(float(row[field]) for row in rows)


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _hotpot_metadata(cache_dir: Path) -> dict[str, dict]:
    candidates = sorted(
        cache_dir.glob("hotpotqa___hotpot_qa/distractor/*/*/hotpot_qa-validation.arrow")
    )
    if not candidates:
        raise FileNotFoundError("Cached Hotpot validation Arrow file was not found.")
    dataset = Dataset.from_file(str(candidates[-1]))
    return {
        row["id"]: {"question_type": row["type"], "level": row["level"]}
        for row in dataset
    }


def _prior_manual_labels(path: Path) -> dict[tuple[str, int], str]:
    if not path.exists():
        return {}
    labels = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "hotpotqa":
                continue
            label = {
                "non_oracle_but_plausibly_relevant": "plausibly_relevant_non_oracle",
                "false_semantic_closure": "clear_distractor",
            }.get(row["classification"])
            if label:
                labels[(row["example_id"], int(row["terminal_parent"]))] = label
    return labels


def _atomic_native(feature: dict, chunk_size: int) -> tuple[torch.Tensor, ...]:
    """Return exact 16/32-token native windows and their new parent IDs."""
    if chunk_size >= 32:
        spans = [tuple(map(int, span)) for span in feature["local_spans"]]
        parent_ids = torch.tensor([start // chunk_size for start, _ in spans])
        return (
            feature["local_pre_query"],
            feature["local_pre_key"],
            feature["local_token_mask"],
            parent_ids,
        )
    queries, keys, masks, parent_ids = [], [], [], []
    for index, (start, end) in enumerate(feature["local_spans"]):
        start, end = int(start), int(end)
        for offset in (0, 16):
            piece_start = start + offset
            if piece_start >= end:
                continue
            queries.append(feature["local_pre_query"][index, offset : offset + 16])
            keys.append(feature["local_pre_key"][index, offset : offset + 16])
            masks.append(feature["local_token_mask"][index, offset : offset + 16])
            parent_ids.append(piece_start // chunk_size)
    return (
        torch.stack(queries),
        torch.stack(keys),
        torch.stack(masks),
        torch.tensor(parent_ids, dtype=torch.long),
    )


def _parent_hidden(token_hidden: torch.Tensor, spans: tuple[tuple[int, int], ...]) -> torch.Tensor:
    return torch.stack([token_hidden[start:end].float().mean(dim=0) for start, end in spans])


def _parent_class(
    parent: int,
    root: int,
    oracle: set[int],
    discovered: set[int],
    prior_label: str | None,
) -> str | None:
    if parent == root:
        return "oracle_root"
    if parent in oracle:
        return "oracle_other"
    if prior_label is not None:
        return prior_label
    if parent in discovered:
        return "other_discovered"
    return None


def _search(edge_scores: torch.Tensor, root: int, k: int, h: int, b: int | None):
    return search_semantic_graph(
        edge_scores,
        torch.zeros(1, edge_scores.shape[0]),
        [root],
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


def _selected_token_metrics(visited: tuple[int, ...], topology, source_tokens: int) -> dict:
    selected_tokens = sum(
        topology.parent_spans[parent][1] - topology.parent_spans[parent][0]
        for parent in visited
    )
    evidence_tokens = sum(topology.evidence_tokens_per_parent[parent] for parent in visited)
    return {
        "logical_reference_tokens": source_tokens,
        "selected_parent_tokens": selected_tokens,
        "selected_evidence_tokens": evidence_tokens,
        "non_evidence_selected_tokens": selected_tokens - evidence_tokens,
        "evidence_density": evidence_tokens / selected_tokens if selected_tokens else 0.0,
        "active_kv_fraction": selected_tokens / source_tokens,
        "conceptual_active_parents": len(visited),
        "materialized_parent_count": 0,
        "native_kv_tokens": 0,
        "counterfactual_native_kv_tokens": selected_tokens,
        "materialization_performed": False,
    }


def _prepare(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    source = torch.load(args.source_feature_file, map_location="cpu", weights_only=False)
    source = [row for row in source if row["dataset"] == "hotpotqa"]
    hidden_rows = torch.load(args.token_hidden_file, map_location="cpu", weights_only=False)
    hidden_by_id = {row["example_id"]: row for row in hidden_rows}
    query_rows = torch.load(args.query_feature_file, map_location="cpu", weights_only=False)
    query_by_id = {row["example_id"]: row for row in query_rows}
    metadata = _hotpot_metadata(args.cache_dir)
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
    prior_labels = _prior_manual_labels(args.false_goal_review)
    prepared, topology_rows, system_rows, facet_rows = [], [], [], []
    for example_index, feature in enumerate(source, start=1):
        example_id = feature["example_id"]
        hidden_feature = hidden_by_id[example_id]
        query_feature = query_by_id[example_id]
        if int(feature["source_tokens"]) != int(hidden_feature["source_tokens"]):
            raise ValueError(f"Token-hidden alignment failed for {example_id}.")
        facets = build_facets(query_feature, facet_config, support_mode, None)
        for chunk_size in args.chunk_sizes:
            source_tokens = int(feature["source_tokens"])
            topology = evidence_topology(
                source_tokens,
                feature["evidence_spans"],
                chunk_size=chunk_size,
            )
            if chunk_size == 256 and tuple(topology.parent_spans) != tuple(
                tuple(map(int, span)) for span in feature["parent_spans"]
            ):
                raise AssertionError(f"Canonical boundaries changed for {example_id}.")
            pooled_started = time.perf_counter()
            parent_hidden = _parent_hidden(
                hidden_feature["token_hidden"], topology.parent_spans
            )
            pooling_seconds = time.perf_counter() - pooled_started
            q, k, mask, local_parents = _atomic_native(feature, chunk_size)
            h2d_bytes = sum(_tensor_bytes(value) for value in (q, k, mask, local_parents))
            if device.type == "cuda":
                torch.cuda.empty_cache()
                baseline_allocated = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            else:
                baseline_allocated = 0
            h2d_started = time.perf_counter()
            q_device, k_device = q.to(device), k.to(device)
            mask_device, local_parent_device = mask.to(device), local_parents.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - h2d_started
            adjacency_started = time.perf_counter()
            adjacency = build_native_parent_adjacency(
                q_device,
                k_device,
                mask_device,
                local_parent_device,
                len(topology.parent_spans),
                token_reduction="top_m_mean",
                head_reduction="top_m_mean",
                top_m=4,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            adjacency_seconds = time.perf_counter() - adjacency_started
            edge_scores = adjacency.scores.detach().cpu()
            peak_allocated = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            )
            selected_search = _search(
                edge_scores,
                topology.root_parent_id,
                SELECTED_K,
                SELECTED_H,
                SELECTED_B,
            )
            discovered = set(selected_search.visited) - set(topology.oracle_parent_ids)
            semantic_scores = {}
            parent_hidden_device = parent_hidden.to(device)
            facet_hidden_device = facets.hidden.to(device)
            for seed, projection in projections.items():
                scored = score_semantic_query_facets(
                    projection.project_query(facet_hidden_device),
                    projection.project_memory(parent_hidden_device),
                )
                scores = scored.component_scores[:, 0, :].detach().cpu()
                semantic_scores[seed] = scores
                normalized = normalize_facet_scores(scores)
                stats = facet_parent_statistics(scores)
                root_winner = stats[topology.root_parent_id].winning_facet
                relevant = set(topology.oracle_parent_ids) | discovered
                if chunk_size == 256:
                    relevant.update(
                        parent
                        for (label_id, parent), _ in prior_labels.items()
                        if label_id == example_id
                    )
                visited_prefix = list(selected_search.visited)
                for parent in sorted(relevant):
                    label = prior_labels.get((example_id, parent)) if chunk_size == 256 else None
                    parent_class = _parent_class(
                        parent,
                        topology.root_parent_id,
                        set(topology.oracle_parent_ids),
                        discovered,
                        label,
                    )
                    if parent_class is None:
                        continue
                    if parent in visited_prefix:
                        path = visited_prefix[: visited_prefix.index(parent) + 1]
                    else:
                        path = [topology.root_parent_id, parent]
                    facet_rows.append(
                        {
                            "example_id": example_id,
                            "partition": validation_partition(example_id),
                            "question_type": metadata[example_id]["question_type"],
                            "chunk_size": chunk_size,
                            "overlap": 0,
                            "seed": seed,
                            "facet_count": scores.shape[0],
                            "parent_id": parent,
                            "parent_class": parent_class,
                            "max_facet_score": stats[parent].maximum,
                            "top2_facet_score": stats[parent].top2_mean,
                            "winning_facet": stats[parent].winning_facet,
                            "normalized_facet_entropy": stats[parent].normalized_entropy,
                            "facet_concentration": stats[parent].concentration,
                            "winning_facet_matches_root": int(
                                stats[parent].winning_facet == root_winner
                            ),
                            "root_path_coverage": path_facet_coverage(
                                normalized, [topology.root_parent_id]
                            ),
                            "path_coverage": path_facet_coverage(normalized, path),
                            "incremental_coverage": incremental_facet_coverage(
                                normalized, [topology.root_parent_id], parent
                            ),
                            "manual_label_reused": bool(label),
                            "boundaries_identical_to_prior": chunk_size == 256,
                        }
                    )
            common = {
                "example_id": example_id,
                "partition": validation_partition(example_id),
                "question_type": metadata[example_id]["question_type"],
                "level": metadata[example_id]["level"],
                "chunk_size": chunk_size,
                "overlap": 0,
                "parent_count": len(topology.parent_spans),
                "oracle_parent_count": len(topology.oracle_parent_ids),
                "later_oracle_parent_count": len(topology.later_oracle_parent_ids),
                "root_parent_id": topology.root_parent_id,
                "root_oracle_fraction": topology.root_oracle_fraction,
                "root_contains_all_evidence": topology.root_contains_all_evidence,
                "root_contains_multiple_groups": topology.root_contains_multiple_groups,
                "root_contains_only_initial_evidence": topology.root_contains_only_initial_evidence,
                "evidence_group_count": len(topology.evidence_parent_groups),
                "evidence_group_collisions": topology.evidence_group_collisions,
                "oracle_parent_ids": " ".join(map(str, topology.oracle_parent_ids)),
                "evidence_parent_groups": ";".join(
                    " ".join(map(str, group)) for group in topology.evidence_parent_groups
                ),
                "evidence_tokens": sum(topology.evidence_tokens_per_parent),
                "source_tokens": source_tokens,
            }
            topology_rows.append(common)
            system_rows.append(
                {
                    **common,
                    "feature_index_build_seconds": pooling_seconds + adjacency_seconds,
                    "parent_pooling_seconds": pooling_seconds,
                    "adjacency_build_seconds": adjacency_seconds,
                    "native_dot_products": adjacency.dot_products,
                    "local_pair_count": adjacency.local_pair_count,
                    "atomic_local_count": q.shape[0],
                    "routing_search_cache_bytes": _tensor_bytes(edge_scores),
                    "h2d_bytes": h2d_bytes,
                    "h2d_seconds": h2d_seconds,
                    "gpu_baseline_allocated_bytes": baseline_allocated,
                    "peak_gpu_allocated_bytes": peak_allocated,
                    "peak_gpu_reserved_bytes": peak_reserved,
                    "native_kv_tokens": 0,
                    "materialization_performed": False,
                }
            )
            prepared.append(
                {
                    **common,
                    "topology": topology,
                    "edge_scores": edge_scores,
                    "semantic_scores": semantic_scores,
                    "adjacency_build_seconds": adjacency_seconds,
                    "native_dot_products": adjacency.dot_products,
                    "routing_search_cache_bytes": _tensor_bytes(edge_scores),
                    "peak_gpu_allocated_bytes": peak_allocated,
                }
            )
            print(
                f"[chunk prep {example_index}/{len(source)}] {example_id} "
                f"size={chunk_size} parents={len(topology.parent_spans)} "
                f"adj={adjacency_seconds:.3f}s",
                flush=True,
            )
    return prepared, topology_rows, system_rows, facet_rows


def _discovery_rows(prepared: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, minimum_rows = [], []
    for example in prepared:
        topology = example["topology"]
        depth_visits = {}
        for k in K_VALUES:
            for h in H_VALUES:
                for b in B_VALUES:
                    result = _search(
                        example["edge_scores"], topology.root_parent_id, k, h, b
                    )
                    recovery = evaluate_oracle_recovery(result.visited, topology)
                    visited_set = set(result.visited)
                    chain_complete = all(
                        visited_set.intersection(group)
                        for group in topology.evidence_parent_groups
                    )
                    token_metrics = _selected_token_metrics(
                        result.visited, topology, example["source_tokens"]
                    )
                    if k == 6 and b is None:
                        depth_visits[h] = result.visited
                    rows.append(
                        {
                            **{
                                key: value
                                for key, value in example.items()
                                if key
                                not in {"topology", "edge_scores", "semantic_scores"}
                            },
                            "K": k,
                            "H": h,
                            "B": "none" if b is None else b,
                            "visited_parent_count": len(result.visited),
                            "nodes_expanded": result.nodes_expanded,
                            "successors_proposed": result.raw_proposals,
                            "raw_proposals": result.raw_proposals,
                            "admitted_proposals": result.edge_admitted_proposals,
                            "duplicates": result.duplicate_proposals,
                            "cycles": result.cycles_prevented,
                            "peak_frontier": result.peak_frontier,
                            "search_comparisons": result.raw_proposals,
                            "candidate_tensor_bytes": result.peak_candidate_tensor_bytes,
                            "search_seconds": result.search_seconds,
                            "stop_reason": result.stop_reason,
                            "oracle_recall": recovery.oracle_recall,
                            "later_oracle_recall": recovery.later_oracle_recall,
                            "complete_oracle": recovery.complete_oracle,
                            "chain_complete": chain_complete,
                            "recall_per_visited_parent": (
                                recovery.oracle_recall / len(result.visited)
                            ),
                            "recall_per_search_comparison": (
                                recovery.oracle_recall / result.raw_proposals
                                if result.raw_proposals
                                else recovery.oracle_recall
                            ),
                            **token_metrics,
                        }
                    )
        depth = minimum_recovery_depth(depth_visits, topology)
        minimum_rows.append(
            {
                "example_id": example["example_id"],
                "partition": example["partition"],
                "question_type": example["question_type"],
                "chunk_size": example["chunk_size"],
                "oracle_parent_count": example["oracle_parent_count"],
                "minimum_recovery_depth": "unrecovered" if depth is None else depth,
                "minimum_recovery_depth_numeric": 5 if depth is None else depth,
                "unrecovered_by_h4": depth is None,
                "diagnostic_K": 6,
                "diagnostic_B": "none",
            }
        )
    depth_map = {
        (row["example_id"], row["chunk_size"]): row["minimum_recovery_depth"]
        for row in minimum_rows
    }
    for row in rows:
        row["minimum_recovery_depth"] = depth_map[(row["example_id"], row["chunk_size"])]
    return rows, minimum_rows


def _synthetic_rows() -> list[dict]:
    rows = []
    for task_depth in (1, 2, 3, 4, 8):
        for nodes_per_chunk in (1, 2, 4):
            observed = contracted_chain_depth(task_depth, nodes_per_chunk)
            parents = observed + 1
            edge = torch.full((parents, parents), float("-inf"))
            for parent in range(parents - 1):
                edge[parent, parent + 1] = 1.0
            measured = None
            for h in range(0, 9):
                result = _search(edge, 0, 1, h, None)
                if parents - 1 in result.visited:
                    measured = h
                    break
            rows.append(
                {
                    "task_edge_depth": task_depth,
                    "nodes_per_chunk": nodes_per_chunk,
                    "observed_required_depth": observed,
                    "measured_recovery_depth": measured,
                    "passed": measured == observed,
                }
            )
    return rows


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positive
        for right in negative
    )
    return wins / (len(positive) * len(negative))


def _aggregate(
    prepared: list[dict],
    discovery: list[dict],
    minimum: list[dict],
    facet_rows: list[dict],
) -> dict:
    central, size_k, containment, systems, depth_distribution = [], [], [], [], []
    oracle_distribution, type_summary, density_summary = [], [], []
    for size in CHUNK_SIZES:
        topology = [row for row in prepared if row["chunk_size"] == size]
        selected = [
            row
            for row in discovery
            if row["chunk_size"] == size
            and row["K"] == SELECTED_K
            and row["B"] == SELECTED_B
        ]
        by_h = {h: [row for row in selected if row["H"] == h] for h in H_VALUES}
        central.append(
            {
                "chunk_size": size,
                "mean_oracle_parents": _mean(topology, "oracle_parent_count"),
                "mean_root_evidence_fraction": _mean(topology, "root_oracle_fraction"),
                **{f"H{h}_oracle_recall": _mean(by_h[h], "oracle_recall") for h in (0, 1, 2, 3)},
                "complete_recovery_H4": _mean(by_h[4], "complete_oracle"),
            }
        )
        for k in K_VALUES:
            values = [
                row
                for row in discovery
                if row["chunk_size"] == size
                and row["K"] == k
                and row["H"] == 4
                and row["B"] == SELECTED_B
            ]
            size_k.append(
                {
                    "chunk_size": size,
                    "K": k,
                    "mean_oracle_recall": _mean(values, "oracle_recall"),
                    "complete_recovery": _mean(values, "complete_oracle"),
                    "mean_visited_parents": _mean(values, "visited_parent_count"),
                }
            )
        containment.append(
            {
                "chunk_size": size,
                "root_has_all_evidence": _mean(topology, "root_contains_all_evidence"),
                "root_has_multiple_groups": _mean(topology, "root_contains_multiple_groups"),
                "root_has_only_initial_evidence": _mean(
                    topology, "root_contains_only_initial_evidence"
                ),
                "mean_oracle_parents": _mean(topology, "oracle_parent_count"),
                "mean_later_evidence_parents": _mean(
                    topology, "later_oracle_parent_count"
                ),
            }
        )
        systems.append(
            {
                "chunk_size": size,
                "mean_parent_count": _mean(topology, "parent_count"),
                "mean_adjacency_build_seconds": _mean(topology, "adjacency_build_seconds"),
                "mean_native_dot_products": _mean(topology, "native_dot_products"),
                "mean_adjacency_cache_bytes": _mean(
                    topology, "routing_search_cache_bytes"
                ),
                "mean_peak_gpu_allocated_bytes": _mean(
                    topology, "peak_gpu_allocated_bytes"
                ),
            }
        )
        size_depth = [row for row in minimum if row["chunk_size"] == size]
        counts = Counter(str(row["minimum_recovery_depth"]) for row in size_depth)
        for depth in ("0", "1", "2", "3", "4", "unrecovered"):
            depth_distribution.append(
                {
                    "chunk_size": size,
                    "minimum_recovery_depth": depth,
                    "fraction": counts[depth] / len(size_depth),
                }
            )
        oracle_counts = torch.tensor(
            [float(row["oracle_parent_count"]) for row in topology], dtype=torch.float64
        )
        oracle_distribution.append(
            {
                "chunk_size": size,
                "mean": float(oracle_counts.mean()),
                "median": float(torch.quantile(oracle_counts, 0.5)),
                "p90": float(torch.quantile(oracle_counts, 0.9)),
                "maximum": float(oracle_counts.max()),
                "distribution": " ".join(
                    f"{key}:{value}"
                    for key, value in sorted(
                        Counter(int(value) for value in oracle_counts.tolist()).items()
                    )
                ),
            }
        )
        selected_h4 = [row for row in selected if row["H"] == SELECTED_H]
        density_summary.append(
            {
                "chunk_size": size,
                "mean_oracle_recall": _mean(selected_h4, "oracle_recall"),
                "mean_visited_parents": _mean(selected_h4, "visited_parent_count"),
                "mean_selected_parent_tokens": _mean(selected_h4, "selected_parent_tokens"),
                "mean_selected_evidence_tokens": _mean(selected_h4, "selected_evidence_tokens"),
                "mean_non_evidence_selected_tokens": _mean(
                    selected_h4, "non_evidence_selected_tokens"
                ),
                "mean_evidence_density": _mean(selected_h4, "evidence_density"),
                "mean_active_kv_fraction": _mean(selected_h4, "active_kv_fraction"),
                "actual_native_kv_tokens": 0,
            }
        )
        for question_type in ("bridge", "comparison"):
            values = [row for row in selected_h4 if row["question_type"] == question_type]
            if values:
                type_summary.append(
                    {
                        "chunk_size": size,
                        "question_type": question_type,
                        "examples": len(values),
                        "mean_oracle_recall": _mean(values, "oracle_recall"),
                        "complete_recovery": _mean(values, "complete_oracle"),
                        "mean_nodes_expanded": _mean(values, "nodes_expanded"),
                    }
                )
    facet_summary, facet_auc = [], []
    metrics = (
        "max_facet_score",
        "top2_facet_score",
        "incremental_coverage",
        "normalized_facet_entropy",
        "winning_facet_matches_root",
    )
    for size in CHUNK_SIZES:
        size_rows = [row for row in facet_rows if int(row["chunk_size"]) == size]
        for parent_class in sorted({row["parent_class"] for row in size_rows}):
            values = [row for row in size_rows if row["parent_class"] == parent_class]
            facet_summary.append(
                {
                    "chunk_size": size,
                    "parent_class": parent_class,
                    "rows": len(values),
                    **{f"mean_{metric}": _mean(values, metric) for metric in metrics},
                }
            )
        oracle = [row for row in size_rows if row["parent_class"] == "oracle_other"]
        for negative_class in (
            "plausibly_relevant_non_oracle",
            "clear_distractor",
            "other_discovered",
        ):
            negative = [row for row in size_rows if row["parent_class"] == negative_class]
            if not negative:
                continue
            for metric in ("max_facet_score", "top2_facet_score", "incremental_coverage"):
                facet_auc.append(
                    {
                        "chunk_size": size,
                        "positive_class": "oracle_other",
                        "negative_class": negative_class,
                        "metric": metric,
                        "auc": _auc(
                            [float(row[metric]) for row in oracle],
                            [float(row[metric]) for row in negative],
                        ),
                        "positive_rows": len(oracle),
                        "negative_rows": len(negative),
                    }
                )
    return {
        "central": central,
        "chunk_k": size_k,
        "containment": containment,
        "systems": systems,
        "depth_distribution": depth_distribution,
        "oracle_parent_distribution": oracle_distribution,
        "question_type_summary": type_summary,
        "evidence_density": density_summary,
        "facet_class_summary": facet_summary,
        "facet_auc": facet_auc,
    }


def _plots(discovery: list[dict], minimum: list[dict], aggregate: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.8))
    central = aggregate["central"]
    axes[0, 0].plot(
        CHUNK_SIZES,
        [row["mean_oracle_parents"] for row in central],
        marker="o",
        color="#2878B5",
    )
    axes[0, 0].set(xlabel="Parent size (tokens)", ylabel="Mean oracle parents")
    axes[0, 1].plot(
        CHUNK_SIZES,
        [row["mean_root_evidence_fraction"] for row in central],
        marker="o",
        color="#D95F02",
    )
    axes[0, 1].set(xlabel="Parent size (tokens)", ylabel="Evidence fraction in root")
    for size in CHUNK_SIZES:
        selected = [
            row
            for row in discovery
            if row["chunk_size"] == size
            and row["K"] == SELECTED_K
            and row["B"] == SELECTED_B
        ]
        axes[0, 2].plot(
            H_VALUES,
            [_mean([row for row in selected if row["H"] == h], "oracle_recall") for h in H_VALUES],
            marker="o",
            label=str(size),
        )
    axes[0, 2].set(xlabel="Hop limit H", ylabel="Oracle evidence recall")
    axes[0, 2].legend(title="Tokens", fontsize=8)
    depth_rows = aggregate["depth_distribution"]
    bottoms = [0.0] * len(CHUNK_SIZES)
    depth_colors = ("#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C", "#777777")
    for depth, color in zip(("0", "1", "2", "3", "4", "unrecovered"), depth_colors):
        values = [
            next(
                row["fraction"]
                for row in depth_rows
                if row["chunk_size"] == size and row["minimum_recovery_depth"] == depth
            )
            for size in CHUNK_SIZES
        ]
        axes[1, 0].bar(
            [str(size) for size in CHUNK_SIZES],
            values,
            bottom=bottoms,
            label=depth,
            color=color,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axes[1, 0].set(xlabel="Parent size (tokens)", ylabel="Fraction of examples")
    axes[1, 0].legend(title="Minimum H", fontsize=7, ncol=2)
    selected = [
        row
        for row in discovery
        if row["K"] == SELECTED_K and row["B"] == SELECTED_B
    ]
    for size in CHUNK_SIZES:
        values = [row for row in selected if row["chunk_size"] == size]
        axes[1, 1].scatter(
            [row["nodes_expanded"] for row in values],
            [row["oracle_recall"] for row in values],
            s=12,
            alpha=0.45,
            label=str(size),
        )
    axes[1, 1].set(xlabel="Nodes expanded", ylabel="Oracle evidence recall")
    frontier = [row for row in selected if row["H"] == SELECTED_H]
    axes[1, 2].scatter(
        [row["counterfactual_native_kv_tokens"] for row in frontier],
        [row["oracle_recall"] for row in frontier],
        c=[row["chunk_size"] for row in frontier],
        cmap="viridis",
        s=22,
        alpha=0.65,
    )
    axes[1, 2].set(xlabel="Counterfactual selected native-KV tokens", ylabel="Oracle recall")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"chunk_granularity_discovery.{suffix}", dpi=180)
    plt.close(figure)

    facet = aggregate["facet_class_summary"]
    classes = ("oracle_other", "other_discovered")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for parent_class, color in zip(classes, ("#2878B5", "#D95F02")):
        values = [row for row in facet if row["parent_class"] == parent_class]
        axes[0].plot(
            [row["chunk_size"] for row in values],
            [row["mean_incremental_coverage"] for row in values],
            marker="o",
            label=parent_class.replace("_", " "),
            color=color,
        )
        axes[1].plot(
            [row["chunk_size"] for row in values],
            [row["mean_max_facet_score"] for row in values],
            marker="o",
            label=parent_class.replace("_", " "),
            color=color,
        )
    axes[0].set(xlabel="Parent size (tokens)", ylabel="Incremental facet coverage")
    axes[1].set(xlabel="Parent size (tokens)", ylabel="Maximum facet score")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"chunk_granularity_facets.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared, topology_rows, system_rows, facet_rows = _prepare(args, device)
    discovery_rows, minimum_rows = _discovery_rows(prepared)
    synthetic_rows = _synthetic_rows()
    if not all(row["passed"] for row in synthetic_rows):
        raise AssertionError("Synthetic chain-contraction control failed.")
    aggregate = _aggregate(prepared, discovery_rows, minimum_rows, facet_rows)
    canonical = [
        row
        for row in discovery_rows
        if row["partition"] == "test"
        and row["chunk_size"] == 256
        and row["K"] == SELECTED_K
        and row["H"] == SELECTED_H
        and row["B"] == SELECTED_B
    ]
    canonical_recall = _mean(canonical, "oracle_recall")
    canonical_later = _mean(canonical, "later_oracle_recall")
    canonical_exact = math.isclose(canonical_recall, 1.0) and math.isclose(
        canonical_later, 1.0
    )
    if not canonical_exact:
        raise AssertionError(
            f"Canonical result changed: recall={canonical_recall}, later={canonical_later}."
        )
    _write_csv(args.output_dir / "evidence_topology_rows.csv", topology_rows)
    _write_csv(args.output_dir / "discovery_surface_rows.csv", discovery_rows)
    _write_csv(args.output_dir / "minimum_recovery_depth_rows.csv", minimum_rows)
    _write_csv(args.output_dir / "facet_parent_rows.csv", facet_rows)
    _write_csv(args.output_dir / "systems_scaling_rows.csv", system_rows)
    _write_csv(args.output_dir / "synthetic_chain_contraction.csv", synthetic_rows)
    _write_csv(args.output_dir / "central_chunk_depth_table.csv", aggregate["central"])
    _write_csv(args.output_dir / "chunk_k_table.csv", aggregate["chunk_k"])
    _write_csv(args.output_dir / "root_containment_table.csv", aggregate["containment"])
    _write_csv(args.output_dir / "adjacency_scaling_table.csv", aggregate["systems"])
    _write_csv(
        args.output_dir / "oracle_parent_distribution.csv",
        aggregate["oracle_parent_distribution"],
    )
    _write_csv(
        args.output_dir / "discovery_by_question_type.csv",
        aggregate["question_type_summary"],
    )
    _write_csv(
        args.output_dir / "evidence_density_table.csv", aggregate["evidence_density"]
    )
    _write_csv(
        args.output_dir / "facet_class_summary.csv", aggregate["facet_class_summary"]
    )
    _write_csv(args.output_dir / "facet_auc_summary.csv", aggregate["facet_auc"])
    _write_csv(
        args.output_dir / "minimum_depth_distribution.csv",
        aggregate["depth_distribution"],
    )
    _plots(discovery_rows, minimum_rows, aggregate, args.output_dir)
    result = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "diagnostic_only": True,
        "backbone_frozen": True,
        "training_performed": False,
        "generation_performed": False,
        "dataset": "hotpotqa",
        "examples": len({row["example_id"] for row in topology_rows}),
        "chunk_sizes": list(args.chunk_sizes),
        "overlap": 0,
        "K_values": list(K_VALUES),
        "H_values": list(H_VALUES),
        "B_values": [6, 16, "none"],
        "search_strategy": STRATEGY,
        "edge_threshold": "open",
        "terminal_query_stopping": False,
        "oracle_root_only": True,
        "oracle_labels_available_during_search": False,
        "oracle_evaluation_post_hoc": True,
        "query_used_during_intermediate_search": False,
        "materialization_performed": False,
        "native_kv_tokens_are_counterfactual_only": True,
        "semantic_parent_representation": "exact_mean_of_layer_27_token_hidden_states",
        "native_edge_representation": "layer_27_pre_rope_top4_token_and_head_mean",
        "query_facet_policy": {
            "configuration": "w2_s1",
            "support": "latest_message",
            "projection_seeds": list(args.seeds),
        },
        "canonical_256_reproduction": {
            "condition": "K4_B6_H4_best_first_open_edge",
            "heldout_oracle_recall": canonical_recall,
            "heldout_later_oracle_recall": canonical_later,
            "exact": canonical_exact,
        },
        "row_counts": {
            "topology": len(topology_rows),
            "discovery": len(discovery_rows),
            "facet_parent": len(facet_rows),
            "systems": len(system_rows),
            "synthetic": len(synthetic_rows),
        },
        "aggregate": aggregate,
    }
    (args.output_dir / "chunk_granularity_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--chunk-sizes", default=",".join(map(str, CHUNK_SIZES)))
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=result_root / "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--token-hidden-file",
        type=Path,
        default=result_root / "chunk_granularity/chunk_granularity_token_hidden.pt",
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
        "--false-goal-review",
        type=Path,
        default=result_root / "semantic_graph_search/false_goal_review.csv",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=result_root / "chunk_granularity"
    )
    args = parser.parse_args()
    args.seeds = tuple(int(seed) for seed in args.seeds.split(","))
    args.chunk_sizes = tuple(int(size) for size in args.chunk_sizes.split(","))
    return args


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output["canonical_256_reproduction"], indent=2))
