"""Analyze native graph topology and measured contextualization across Qwen depth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_natural_graph_depth import (
    _feature_example,
    _node_recovery,
    _product_path_survival,
    _search,
    _strict_path_survival,
    _transition_rows,
)
from pra_hf.layerwise_graph import (
    bootstrap_mean_ci,
    pearson,
    shortest_distance,
    spearman,
    topk_neighbors,
    topology_metrics,
)
from pra_hf.natural_reasoning_graph import map_example_to_parents
from pra_hf.semantic_graph_search import build_native_parent_adjacency


PRIMARY_CHUNK = 128
PRIMARY_K = 6
PRIMARY_B = 16
MAX_HOPS = 4
RANK_K = (1, 2, 4, 6, 8, 11)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: "" if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in row.items()
            }
            for row in rows
        )


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean(rows, field):
    values = [
        float(row[field])
        for row in rows
        if row.get(field, "") not in ("", None)
        and not math.isnan(float(row[field]))
    ]
    return statistics.fmean(values) if values else float("nan")


def _atomic_native(layer_feature: dict) -> tuple[torch.Tensor, ...]:
    spans = [tuple(map(int, span)) for span in layer_feature["local_spans"]]
    return (
        layer_feature["local_pre_query"],
        layer_feature["local_pre_key"],
        layer_feature["local_token_mask"],
        torch.tensor([start // PRIMARY_CHUNK for start, _ in spans], dtype=torch.long),
    )


def _lexical_overlap(feature, source_node: str, target_node: str) -> float:
    spans = feature["node_token_spans"]
    source_span = spans.get(source_node)
    target_span = spans.get(target_node)
    if source_span is None or target_span is None:
        return float("nan")
    ids = feature["source_token_ids"]
    source = set(map(int, ids[slice(*source_span)].tolist()))
    target = set(map(int, ids[slice(*target_span)].tolist()))
    return len(source.intersection(target)) / max(len(source.union(target)), 1)


def _annotated_distance(example, source: str, target: str) -> int | None:
    outgoing = defaultdict(list)
    for left, right in example.annotated_edges:
        outgoing[left].append(right)
    queue = deque([(source, 0)])
    visited = {source}
    while queue:
        node, depth = queue.popleft()
        for successor in outgoing[node]:
            if successor == target:
                return depth + 1
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, depth + 1))
    return None


def _shortcut_metrics(example, mapping, scores):
    neighbors = topk_neighbors(scores, PRIMARY_K)
    evidence_distances = []
    unreachable = 0
    for source, target in example.annotated_edges:
        distance = shortest_distance(
            neighbors,
            mapping.node_parent_groups.get(source, ()),
            mapping.node_parent_groups.get(target, ()),
            max_hops=MAX_HOPS,
        )
        if distance is None:
            unreachable += 1
        else:
            evidence_distances.append(distance)
    shortcut_pairs = 0
    eligible_pairs = 0
    node_ids = [node.node_id for node in example.nodes]
    for source in node_ids:
        for target in node_ids:
            annotated = _annotated_distance(example, source, target)
            if annotated is None or annotated < 2:
                continue
            eligible_pairs += 1
            native = shortest_distance(
                neighbors,
                mapping.node_parent_groups.get(source, ()),
                mapping.node_parent_groups.get(target, ()),
                max_hops=1,
            )
            shortcut_pairs += native == 1
    return {
        "mean_evidence_native_distance": (
            statistics.fmean(evidence_distances) if evidence_distances else float("nan")
        ),
        "unreachable_evidence_fraction": unreachable / max(len(example.annotated_edges), 1),
        "shortcut_rate": shortcut_pairs / max(eligible_pairs, 1),
        "shortcut_pairs": shortcut_pairs,
        "shortcut_eligible_pairs": eligible_pairs,
    }


def _minimum_recovery(feature, example, mapping, scores):
    rows = []
    minimum = None
    final_recall = 0.0
    final_complete = False
    for hops in range(MAX_HOPS + 1):
        result = _search(scores, mapping.root_parent_ids, PRIMARY_K, hops, PRIMARY_B)
        recall, complete, _ = _node_recovery(result.visited, mapping)
        rows.append(
            {
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "partition": feature["partition"],
                "annotated_hops": feature["annotated_hops"],
                "graph_type": feature["graph_type"],
                "H": hops,
                "node_recall": recall,
                "complete_recovery": int(complete),
                "visited_count": len(result.visited),
                "nodes_expanded": result.nodes_expanded,
                "search_seconds": result.search_seconds,
            }
        )
        if complete and minimum is None:
            minimum = hops
        if hops == MAX_HOPS:
            final_recall = recall
            final_complete = complete
    return rows, minimum, final_recall, final_complete


def _process_feature(feature, layer, device):
    # Schema 1.0 caches produced before the question metadata patch remain valid:
    # graph construction and oracle-root traversal do not consume question text.
    example = _feature_example({**feature, "question": feature.get("question", "")})
    mapping = map_example_to_parents(
        example,
        int(feature["source_tokens"]),
        feature["node_token_spans"],
        chunk_size=PRIMARY_CHUNK,
    )
    native = feature["layers"][layer]
    q, k, mask, parents = _atomic_native(native)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    transfer_started = time.perf_counter()
    q, k, mask, parents = q.to(device), k.to(device), mask.to(device), parents.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_started
    adjacency_started = time.perf_counter()
    adjacency = build_native_parent_adjacency(
        q,
        k,
        mask,
        parents,
        len(mapping.parent_spans),
        token_reduction="top_m_mean",
        head_reduction="top_m_mean",
        top_m=4,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    adjacency_seconds = time.perf_counter() - adjacency_started
    scores = adjacency.scores.detach().cpu()
    transitions = _transition_rows(feature, example, mapping, scores)
    for row in transitions:
        row["layer"] = layer
        overlap = _lexical_overlap(feature, row["source_node"], row["target_node"])
        row["lexical_overlap"] = overlap
        row["lexical_group"] = (
            "unmapped" if math.isnan(overlap) else ("high" if overlap >= 0.10 else "low")
        )
    hop_rows, minimum, final_recall, final_complete = _minimum_recovery(
        feature, example, mapping, scores
    )
    for row in hop_rows:
        row["layer"] = layer
    topology = topology_metrics(scores, PRIMARY_K)
    topology.update(_shortcut_metrics(example, mapping, scores))
    graph_row = {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "partition": feature["partition"],
        "layer": layer,
        "annotated_hops": feature["annotated_hops"],
        "graph_type": feature["graph_type"],
        "parent_count": len(mapping.parent_spans),
        "oracle_parent_count": len(mapping.oracle_parent_ids),
        "minimum_native_recovery_depth": "" if minimum is None else minimum,
        "depth_contraction": "" if minimum is None else int(feature["annotated_hops"]) - minimum,
        "complete_recovery": int(final_complete),
        "node_recall": final_recall,
        **topology,
    }
    systems = {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "partition": feature["partition"],
        "layer": layer,
        "parent_count": len(mapping.parent_spans),
        "cache_tensor_bytes": sum(
            value.numel() * value.element_size()
            for key, value in native.items()
            if isinstance(value, torch.Tensor)
        ),
        "candidate_tensor_bytes": scores.numel() * scores.element_size(),
        "native_dot_products": adjacency.dot_products,
        "h2d_seconds": transfer_seconds,
        "adjacency_build_seconds": adjacency_seconds,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    return transitions, hop_rows, graph_row, systems


def _transition_aggregate(transitions, layers):
    rows = []
    for layer in layers:
        selected = [
            row
            for row in transitions
            if row["dataset"] == "2wikimultihopqa"
            and row["partition"] == "test"
            and row["layer"] == layer
            and row["mapping_status"] == "preserved"
        ]
        all_edges = [
            row
            for row in transitions
            if row["dataset"] == "2wikimultihopqa"
            and row["partition"] == "test"
            and row["layer"] == layer
        ]
        row = {
            "layer": layer,
            "MRR": _mean(selected, "reciprocal_rank"),
            "preserved_transitions": len(selected),
            "all_annotated_transitions": len(all_edges),
        }
        for k in RANK_K:
            row[f"R_at_{k}"] = _mean(selected, f"recovered_at_{k}")
        identities = defaultdict(list)
        for edge in selected:
            identities[edge["example_id"]].append(edge)
        for k in (2, 4, 6, 8):
            row[f"path_survival_K{k}"] = statistics.fmean(
                all(int(edge[f"recovered_at_{k}"]) for edge in values)
                for values in identities.values()
            )
            row[f"strict_path_survival_K{k}"] = _strict_path_survival(all_edges, k)
            row[f"independent_edge_product_K{k}"] = _product_path_survival(all_edges, k)
        rows.append(row)
    return rows


def _depth_aggregate(graph_rows, layers):
    rows = []
    for layer in layers:
        for depth in (2, 3, 4):
            selected = [
                row
                for row in graph_rows
                if row["dataset"] == "musique"
                and row["partition"] == "test"
                and row["layer"] == layer
                and int(row["annotated_hops"]) == depth
            ]
            completed = [row for row in selected if row["minimum_native_recovery_depth"] != ""]
            depths = [float(row["minimum_native_recovery_depth"]) for row in completed]
            rows.append(
                {
                    "layer": layer,
                    "annotated_depth": depth,
                    "mean_minimum_native_depth_if_complete": (
                        statistics.fmean(depths) if depths else ""
                    ),
                    "median_minimum_native_depth_if_complete": (
                        statistics.median(depths) if depths else ""
                    ),
                    "p90_minimum_native_depth_if_complete": (
                        float(torch.quantile(torch.tensor(depths), 0.9)) if depths else ""
                    ),
                    "complete_recovery": _mean(selected, "complete_recovery"),
                    "node_recall": _mean(selected, "node_recall"),
                    "examples": len(selected),
                }
            )
    return rows


def _topology_aggregate(graph_rows, layers):
    fields = (
        "mean_out_degree",
        "effective_branching_factor",
        "unique_neighbor_count",
        "duplicate_neighbor_rate",
        "connected_component_count",
        "giant_component_fraction",
        "graph_density",
        "reciprocal_edge_rate",
        "mean_evidence_native_distance",
        "shortcut_rate",
        "unreachable_evidence_fraction",
        "depth_contraction",
        "complete_recovery",
        "node_recall",
    )
    rows = []
    for dataset in ("musique", "2wikimultihopqa", "all"):
        for layer in layers:
            selected = [
                row
                for row in graph_rows
                if row["partition"] == "test"
                and row["layer"] == layer
                and (dataset == "all" or row["dataset"] == dataset)
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "layer": layer,
                    **{field: _mean(selected, field) for field in fields},
                    "examples": len(selected),
                }
            )
    return rows


def _lexical_aggregate(transitions, layers):
    rows = []
    for layer in layers:
        for group in ("low", "high"):
            selected = [
                row
                for row in transitions
                if row["dataset"] == "2wikimultihopqa"
                and row["partition"] == "test"
                and row["layer"] == layer
                and row["mapping_status"] == "preserved"
                and row["lexical_group"] == group
            ]
            rows.append(
                {
                    "layer": layer,
                    "lexical_group": group,
                    "mean_lexical_overlap": _mean(selected, "lexical_overlap"),
                    "R_at_4": _mean(selected, "recovered_at_4"),
                    "R_at_6": _mean(selected, "recovered_at_6"),
                    "R_at_8": _mean(selected, "recovered_at_8"),
                    "MRR": _mean(selected, "reciprocal_rank"),
                    "transitions": len(selected),
                }
            )
    return rows


def _context_aggregate(context_rows, layers):
    metrics = (
        "attention_contribution_ratio",
        "post_attention_displacement",
        "post_attention_rotation",
        "ffn_contribution_ratio",
        "attention_entropy",
        "effective_attention_support",
        "self_attention_fraction",
        "local_attention_fraction",
        "distant_attention_fraction",
        "evidence_attention_fraction",
        "intervention_displacement",
        "intervention_rotation",
    )
    rows = []
    for layer in layers:
        for radius in ("full", "1", "16", "32"):
            selected = [
                row
                for row in context_rows
                if row["partition"] == "test"
                and int(row["layer"]) == layer
                and row["token_class"] == "all"
                and row["context_radius"] == radius
            ]
            aggregate = {"layer": layer, "context_radius": radius, "examples": len(selected)}
            for metric in metrics:
                aggregate[metric] = _mean(selected, metric)
                values = [float(row[metric]) for row in selected if row.get(metric, "") != ""]
                ci = bootstrap_mean_ci(values)
                aggregate[f"{metric}_ci_low"] = ci["ci_low"]
                aggregate[f"{metric}_ci_high"] = ci["ci_high"]
            rows.append(aggregate)
    return rows


def _joined_and_correlations(context, transition, topology, layers):
    by_transition = {row["layer"]: row for row in transition}
    by_topology = {
        row["layer"]: row for row in topology if row["dataset"] == "all"
    }
    full = {
        row["layer"]: row
        for row in context
        if row["context_radius"] == "32"
    }
    rows = []
    for layer in layers:
        context_row = full[layer]
        transition_row = by_transition[layer]
        topology_row = by_topology[layer]
        rows.append(
            {
                "layer": layer,
                "attention_contribution_ratio": context_row["attention_contribution_ratio"],
                "attention_rotation": context_row["post_attention_rotation"],
                "intervention_contextualization": context_row["intervention_displacement"],
                "attention_entropy": context_row["attention_entropy"],
                "effective_attention_support": context_row["effective_attention_support"],
                "ffn_contribution_ratio": context_row["ffn_contribution_ratio"],
                "edge_R4": transition_row["R_at_4"],
                "edge_R6": transition_row["R_at_6"],
                "edge_R8": transition_row["R_at_8"],
                "edge_MRR": transition_row["MRR"],
                "complete_path_K6": transition_row["path_survival_K6"],
                "complete_path_K8": transition_row["path_survival_K8"],
                "depth_contraction": topology_row["depth_contraction"],
                "branching_factor": topology_row["effective_branching_factor"],
                "shortcut_rate": topology_row["shortcut_rate"],
                "unreachable_evidence_fraction": topology_row["unreachable_evidence_fraction"],
                "complete_recovery": topology_row["complete_recovery"],
            }
        )
    correlations = []
    contextual = (
        "attention_contribution_ratio",
        "attention_rotation",
        "intervention_contextualization",
        "attention_entropy",
    )
    graph = (
        "edge_R4",
        "edge_R6",
        "edge_R8",
        "edge_MRR",
        "complete_path_K6",
        "depth_contraction",
        "branching_factor",
        "shortcut_rate",
        "unreachable_evidence_fraction",
        "complete_recovery",
    )
    for left in contextual:
        for right in graph:
            x = [row[left] for row in rows]
            y = [row[right] for row in rows]
            correlations.append(
                {
                    "context_metric": left,
                    "graph_metric": right,
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                    "layers": len(rows),
                }
            )
    return rows, correlations


def _plots(output, transition, path, depth, topology, context, joined):
    layers = [row["layer"] for row in transition]
    plt.figure(figsize=(7.2, 4.4))
    for k in (1, 2, 4, 6, 8, 11):
        plt.plot(layers, [row[f"R_at_{k}"] for row in transition], marker="o", label=f"R@{k}")
    plt.xlabel("Decoder layer (zero based)")
    plt.ylabel("Preserved-transition recall")
    plt.ylim(0, 1.03)
    plt.grid(alpha=0.25)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output / "layerwise_edge_recall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    for k in (2, 4, 6, 8):
        plt.plot(layers, [row[f"path_survival_K{k}"] for row in transition], marker="o", label=f"K={k}")
    plt.xlabel("Decoder layer (zero based)")
    plt.ylabel("Complete preserved-path survival")
    plt.ylim(0, 1.03)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "layerwise_path_survival.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    for annotated in (2, 3, 4):
        selected = [row for row in depth if row["annotated_depth"] == annotated]
        plt.plot(
            [row["layer"] for row in selected],
            [float(row["mean_minimum_native_depth_if_complete"]) if row["mean_minimum_native_depth_if_complete"] != "" else math.nan for row in selected],
            marker="o",
            label=f"annotated D={annotated}",
        )
    plt.xlabel("Decoder layer (zero based)")
    plt.ylabel("Minimum native depth (completed cases)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "layerwise_musique_native_depth.png", dpi=180)
    plt.close()

    all_topology = [row for row in topology if row["dataset"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(layers, [row["depth_contraction"] for row in all_topology], marker="o", color="#b24a35")
    axes[0].set(xlabel="Decoder layer", ylabel="Mean depth contraction")
    axes[1].plot(layers, [row["reciprocal_edge_rate"] for row in all_topology], marker="o", label="reciprocal edge rate", color="#2b6f9c")
    axes[1].plot(layers, [row["duplicate_neighbor_rate"] for row in all_topology], marker="s", label="repeated destination rate", color="#558b2f")
    axes[1].set(xlabel="Decoder layer", ylabel="Topology fraction")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "layerwise_contraction_topology.png", dpi=180)
    plt.close(fig)

    radius32 = [row for row in context if row["context_radius"] == "32"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(layers, [row["attention_contribution_ratio"] for row in radius32], marker="o", label="attention contribution")
    axes[0].plot(layers, [row["post_attention_rotation"] for row in radius32], marker="s", label="post-attention rotation")
    axes[0].plot(layers, [row["intervention_displacement"] for row in radius32], marker="^", label="full vs radius 32")
    axes[0].set(xlabel="Decoder layer", ylabel="Normalized contextualization metric")
    axes[0].legend()
    axes[1].plot(layers, [row["ffn_contribution_ratio"] for row in radius32], marker="o", color="#7b4b94")
    axes[1].set(xlabel="Decoder layer", ylabel="FFN contribution ratio")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "layerwise_contextualization.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for axis, field, label in zip(
        axes,
        ("edge_R6", "depth_contraction", "shortcut_rate"),
        ("2Wiki edge R@6", "Depth contraction", "Shortcut rate"),
    ):
        x = [row["intervention_contextualization"] for row in joined]
        y = [row[field] for row in joined]
        axis.scatter(x, y, color="#2b6f9c")
        for row in joined:
            axis.annotate(str(row["layer"]), (row["intervention_contextualization"], row[field]), fontsize=8)
        axis.set(xlabel="Full vs radius-32 displacement", ylabel=label)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "contextualization_graph_correlations.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7.2, 4.4))
    for layer in (layers[0], layers[len(layers) // 2], layers[-1]):
        selected = [row for row in context if row["layer"] == layer and row["context_radius"] != "full"]
        selected.sort(key=lambda row: int(row["context_radius"]))
        plt.plot(
            [int(row["context_radius"]) for row in selected],
            [row["intervention_displacement"] for row in selected],
            marker="o",
            label=f"layer {layer}",
        )
    plt.xlabel("Accessible causal radius (tokens)")
    plt.ylabel("Full vs restricted displacement")
    plt.xscale("log", base=2)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "layerwise_context_radius.png", dpi=180)
    plt.close()


def run(args):
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    layers = tuple(map(int, manifest["selected_layers"]))
    transitions, hop_rows, graph_rows, system_rows = [], [], [], []
    for index, entry in enumerate(manifest["entries"], start=1):
        feature = torch.load(
            args.output_dir / entry["path"], map_location="cpu", weights_only=False
        )
        if tuple(sorted(map(int, feature["layers"]))) != layers:
            raise ValueError("Feature layer identities do not match the manifest.")
        for layer in layers:
            edge, hops, graph, systems = _process_feature(feature, layer, device)
            transitions.extend(edge)
            hop_rows.extend(hops)
            graph_rows.append(graph)
            system_rows.append(systems)
        print(
            f"[layerwise-graph {index}/{len(manifest['entries'])}] "
            f"{feature['dataset']} {feature['example_id']}",
            flush=True,
        )
    transition_aggregate = _transition_aggregate(transitions, layers)
    depth_aggregate = _depth_aggregate(graph_rows, layers)
    topology_aggregate = _topology_aggregate(graph_rows, layers)
    lexical_aggregate = _lexical_aggregate(transitions, layers)
    context_rows = _read_csv(args.output_dir / "layerwise_context_rows.csv")
    context_aggregate = _context_aggregate(context_rows, layers)
    joined, correlations = _joined_and_correlations(
        context_aggregate, transition_aggregate, topology_aggregate, layers
    )
    canonical = next(row for row in transition_aggregate if row["layer"] == 27)
    canonical_exact = (
        math.isclose(canonical["R_at_4"], 0.72, abs_tol=1e-12)
        and math.isclose(canonical["R_at_6"], 0.88, abs_tol=1e-12)
        and math.isclose(canonical["R_at_8"], 1.0, abs_tol=1e-12)
        and math.isclose(canonical["path_survival_K6"], 14 / 17, abs_tol=1e-12)
    )
    if not canonical_exact:
        raise AssertionError(f"Canonical layer-27 graph changed: {canonical}")
    for name, rows in (
        ("layerwise_transition_rows.csv", transitions),
        ("layerwise_hop_rows.csv", hop_rows),
        ("layerwise_graph_rows.csv", graph_rows),
        ("layerwise_system_rows.csv", system_rows),
        ("layerwise_transition_summary.csv", transition_aggregate),
        ("layerwise_musique_depth_summary.csv", depth_aggregate),
        ("layerwise_topology_summary.csv", topology_aggregate),
        ("layerwise_lexical_summary.csv", lexical_aggregate),
        ("layerwise_context_summary.csv", context_aggregate),
        ("layerwise_joined_summary.csv", joined),
        ("layerwise_correlations.csv", correlations),
    ):
        _write_csv(args.output_dir / name, rows)
    _plots(
        args.output_dir,
        transition_aggregate,
        transition_aggregate,
        depth_aggregate,
        topology_aggregate,
        context_aggregate,
        joined,
    )
    best_edge = max(transition_aggregate, key=lambda row: (row["R_at_6"], row["MRR"]))
    result = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "training_performed": False,
        "oracle_labels_available_during_search": False,
        "oracle_root_granted": True,
        "post_hoc_evidence_evaluation": True,
        "layer_indexing": manifest["layer_indexing"],
        "layers": list(layers),
        "fixed_policy": {
            "encoding_block_tokens": 256,
            "search_chunk_tokens": PRIMARY_CHUNK,
            "overlap_tokens": 0,
            "K": PRIMARY_K,
            "B": PRIMARY_B,
            "H": MAX_HOPS,
            "edge_scorer": "exact pre-RoPE native Q/K Top-4 token/head mean",
            "root": "deterministic oracle annotated entry",
            "terminal_goal": "disabled",
        },
        "canonical_layer27_exact_reproduction": canonical_exact,
        "transition_summary": transition_aggregate,
        "musique_depth_summary": depth_aggregate,
        "topology_summary": topology_aggregate,
        "lexical_summary": lexical_aggregate,
        "contextualization_summary": context_aggregate,
        "joined_summary": joined,
        "correlations": correlations,
        "provisional_recommendation": {
            "best_edge_layer": best_edge["layer"],
            "rule": "prefer the highest held-out R@6/MRR layer; retain full curve and do not fuse layers",
        },
    }
    (args.output_dir / "layerwise_graph_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/layerwise_graph"
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--manifest", type=Path, default=output / "layerwise_feature_manifest.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
