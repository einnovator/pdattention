"""Held-out controlled facet recovery and cluster-ablation study."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import (  # noqa: E402
    controlled_manifest,
    controlled_queries,
    git_metadata,
    write_csv,
    write_json,
)
from experiments.paper2_7_query_graph.run_algorithm_cross import EDGE_WEIGHTS  # noqa: E402
from pra_hf.query_graph import build_query_graph  # noqa: E402
from pra_hf.query_graph_cluster import (  # noqa: E402
    ClusterResult,
    deterministic_kmeans,
    facet_recovery_metrics,
    connected_components,
    weighted_label_propagation,
)
from pra_hf.query_graph_facets import pool_hard_graph_facets  # noqa: E402


METHODS = (
    "global",
    "fixed_window",
    "clause_proxy",
    "embedding_kmeans",
    "graph_cc",
    "graph_label_propagation",
)


def _fixed_labels(count: int, width: int) -> torch.Tensor:
    return torch.arange(count, dtype=torch.long) // max(1, int(width))


def _select_window(validation_cases) -> int:
    scores = {}
    for width in (2, 4, 8):
        values = [
            facet_recovery_metrics(
                _fixed_labels(case.hidden.shape[0], width), case.target_labels
            ).ari
            for case in validation_cases
        ]
        scores[width] = sum(values) / len(values)
    return max(scores, key=lambda width: (scores[width], -width))


def _graph(case, policy):
    alpha, beta, delta = EDGE_WEIGHTS[policy["edge_family"]]
    return build_query_graph(
        case.hidden,
        lexical_features=case.lexical,
        provenance=case.provenance,
        contextual_weight=alpha,
        lexical_weight=beta,
        position_weight=delta,
        top_k=int(policy["top_k"]),
        threshold=float(policy["threshold"]),
        policy=str(policy["policy"]),
    )


def _labels(case, method, graph, fixed_window):
    count = int(case.hidden.shape[0])
    if method == "global":
        return ClusterResult(torch.zeros(count, dtype=torch.long), 1, True, method)
    if method == "fixed_window":
        return ClusterResult(_fixed_labels(count, fixed_window), 1, True, method)
    if method == "clause_proxy":
        return ClusterResult(_fixed_labels(count, 4), 1, True, method)
    if method == "embedding_kmeans":
        cluster_count = max(1, min(6, round(math.sqrt(count / 2.0))))
        return deterministic_kmeans(case.hidden, cluster_count)
    if method == "graph_cc":
        return connected_components(graph)
    if method == "graph_label_propagation":
        return weighted_label_propagation(graph)
    raise ValueError(method)


def _ablation_rows(case, graph, clusters, rng):
    facets = pool_hard_graph_facets(
        graph, case.hidden, clusters, include_global=False
    )
    target_count = int(case.target_labels.max()) + 1
    target_states = torch.stack(
        [case.hidden[case.target_labels == index].mean(0) for index in range(target_count)]
    )
    baseline = F.normalize(facets.hidden, dim=-1) @ F.normalize(target_states, dim=-1).T
    rows = []
    for target in range(target_count):
        overlap = facets.membership[case.target_labels == target].sum(0)
        selected_facet = int(torch.argmax(overlap))
        normal = float(baseline[:, target].max())
        keep_facets = [index for index in range(facets.hidden.shape[0]) if index != selected_facet]
        suppressed = (
            float(baseline[keep_facets, target].max()) if keep_facets else 0.0
        )
        member_count = int((facets.membership[:, selected_facet] > 0).sum())
        removable = list(range(case.hidden.shape[0]))
        random_removed = set(rng.sample(removable, min(member_count, len(removable))))
        random_keep = [index for index in removable if index not in random_removed]
        random_score = (
            float(
                (
                    F.normalize(case.hidden[random_keep], dim=-1)
                    @ F.normalize(target_states[target], dim=-1)
                ).max()
            )
            if random_keep
            else 0.0
        )
        rows.append(
            {
                "split": case.split,
                "seed": case.seed,
                "example_id": case.example_id,
                "target_facet": target,
                "discovered_facet": selected_facet,
                "suppressed_units": member_count,
                "normal_similarity": normal,
                "cluster_suppressed_similarity": suppressed,
                "random_unit_suppressed_similarity": random_score,
                "cluster_damage": normal - suppressed,
                "random_damage": normal - random_score,
                "selective_damage_advantage": (normal - suppressed) - (normal - random_score),
            }
        )
    return rows


def _summaries(rows):
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "examples": len(group),
                "mean_ari": sum(row["ari"] for row in group) / len(group),
                "mean_nmi": sum(row["nmi"] for row in group) / len(group),
                "mean_pairwise_f1": sum(row["pairwise_f1"] for row in group) / len(group),
                "mean_boundary_f1": sum(row["boundary_f1"] for row in group) / len(group),
                "mean_cluster_count_error": sum(row["cluster_count_error"] for row in group) / len(group),
                "interleaved_ari": sum(row["ari"] for row in group if row["interleaved"]) / max(sum(row["interleaved"] for row in group), 1),
                "contiguous_ari": sum(row["ari"] for row in group if not row["interleaved"]) / max(sum(not row["interleaved"] for row in group), 1),
            }
        )
    return output


def _plots(rows, summaries, output):
    colors = ["#59656f", "#d1495b", "#e09f3e", "#7a5195", "#2f6690", "#16817a"]
    fig, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    axis.bar(
        range(len(METHODS)),
        [next(row["mean_ari"] for row in summaries if row["method"] == method) for method in METHODS],
        color=colors,
    )
    axis.set(
        xticks=range(len(METHODS)),
        xticklabels=[method.replace("_", "\n") for method in METHODS],
        ylabel="Adjusted Rand index",
        ylim=(-0.1, 1.0),
        title="Controlled held-out facet recovery",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "controlled_facet_recovery.png", dpi=180)
    fig.savefig(output / "controlled_facet_recovery.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for method, color in (("fixed_window", colors[1]), ("embedding_kmeans", colors[3]), ("graph_cc", colors[4]), ("graph_label_propagation", colors[5])):
        means = []
        for facets in range(1, 7):
            group = [row for row in rows if row["method"] == method and row["facets"] == facets]
            means.append(sum(row["ari"] for row in group) / len(group))
        axis.plot(range(1, 7), means, marker="o", label=method.replace("_", " "), color=color)
    axis.set(xlabel="Latent facet count", ylabel="Adjusted Rand index", ylim=(-0.1, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "controlled_scaling_by_facets.png", dpi=180)
    fig.savefig(output / "controlled_scaling_by_facets.pdf")
    plt.close(fig)


def run(args):
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    validation = controlled_queries("validation", examples_per_seed=args.examples_per_seed)
    test = controlled_queries("test", examples_per_seed=args.examples_per_seed)
    fixed_window = _select_window(validation)
    rows = []
    ablations = []
    rng = random.Random(args.seed)
    for case in test:
        graph = _graph(case, policy)
        results = {}
        for method in METHODS:
            result = _labels(case, method, graph, fixed_window)
            results[method] = result
            metrics = facet_recovery_metrics(result.labels, case.target_labels)
            rows.append(
                {
                    "split": "test",
                    "seed": case.seed,
                    "example_id": case.example_id,
                    "method": method,
                    "facets": case.facet_count,
                    "units": int(case.hidden.shape[0]),
                    "interleaved": int(case.interleaved),
                    "shared_entity": int(case.shared_entity),
                    "ari": metrics.ari,
                    "nmi": metrics.nmi,
                    "pairwise_f1": metrics.pairwise_f1,
                    "boundary_f1": metrics.boundary_f1,
                    "cluster_count_error": metrics.cluster_count_error,
                    "predicted_clusters": result.cluster_count,
                    "iterations": result.iterations,
                    "converged": int(result.converged),
                    "edges": graph.edge_count if method.startswith("graph_") else 0,
                }
            )
        ablations.extend(_ablation_rows(case, graph, results["graph_cc"], rng))
    summaries = _summaries(rows)
    graph_ari = next(row["mean_ari"] for row in summaries if row["method"] == "graph_cc")
    fixed_ari = next(row["mean_ari"] for row in summaries if row["method"] == "fixed_window")
    kmeans_ari = next(row["mean_ari"] for row in summaries if row["method"] == "embedding_kmeans")
    gate2 = graph_ari > fixed_ari and graph_ari > kmeans_ari
    selective = sum(row["selective_damage_advantage"] for row in ablations) / len(ablations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "controlled_facet_rows.csv", rows)
    write_csv(args.output_dir / "controlled_facet_summary.csv", summaries)
    write_csv(args.output_dir / "cluster_ablation.csv", ablations)
    findings = {
        "schema_version": "1.0",
        "git": git_metadata(),
        "selected_graph_policy": policy,
        "fixed_window_selected_on_validation": fixed_window,
        "validation_manifest": controlled_manifest(validation),
        "test_manifest": controlled_manifest(test),
        "summary": summaries,
        "gate2_pass": gate2,
        "gate2_rule": "graph_cc ARI > validation-selected fixed-window ARI and non-oracle k-means ARI",
        "mean_selective_cluster_ablation_advantage": selective,
    }
    write_json(args.output_dir / "controlled_findings.json", findings)
    _plots(rows, summaries, args.output_dir)
    return findings


def parse_args():
    parser = argparse.ArgumentParser()
    base = ROOT / "docs/papers/shared/results/paper2_7_query_graph"
    parser.add_argument("--policy", type=Path, default=base / "algorithm_cross/selected_graph_policy.json")
    parser.add_argument("--examples-per-seed", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output-dir", type=Path, default=base / "controlled")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
