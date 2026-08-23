"""Validation-only staged query-graph policy selection for Paper 2.7."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

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
from pra_hf.query_graph import build_query_graph  # noqa: E402
from pra_hf.query_graph_cluster import (  # noqa: E402
    connected_components,
    facet_recovery_metrics,
    weighted_label_propagation,
)


EDGE_WEIGHTS = {
    "contextual": (1.0, 0.0, 0.0),
    "lexical": (0.0, 1.0, 0.0),
    "contextual_lexical": (0.75, 0.25, 0.0),
    "contextual_lexical_position": (0.72, 0.23, 0.05),
}


def _evaluate(cases, config: dict, stage: str) -> list[dict]:
    rows = []
    alpha, beta, delta = EDGE_WEIGHTS[config["edge_family"]]
    for case in cases:
        started = time.perf_counter()
        graph = build_query_graph(
            case.hidden,
            lexical_features=case.lexical,
            provenance=case.provenance,
            contextual_weight=alpha,
            lexical_weight=beta,
            position_weight=delta,
            top_k=config["top_k"],
            threshold=config["threshold"],
            policy=config["policy"],
        )
        graph_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        if config["method"] == "cc":
            result = connected_components(graph)
        else:
            result = weighted_label_propagation(graph)
        cluster_ms = (time.perf_counter() - started) * 1000.0
        metrics = facet_recovery_metrics(result.labels, case.target_labels)
        rows.append(
            {
                "stage": stage,
                "config_id": config["config_id"],
                **config,
                "split": case.split,
                "seed": case.seed,
                "example_id": case.example_id,
                "facets": case.facet_count,
                "units": graph.node_count,
                "edges": graph.edge_count,
                "interleaved": int(case.interleaved),
                "shared_entity": int(case.shared_entity),
                "ari": metrics.ari,
                "nmi": metrics.nmi,
                "pairwise_f1": metrics.pairwise_f1,
                "boundary_f1": metrics.boundary_f1,
                "cluster_count_error": metrics.cluster_count_error,
                "graph_ms": graph_ms,
                "cluster_ms": cluster_ms,
                "iterations": result.iterations,
                "converged": int(result.converged),
            }
        )
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["config_id"])].append(row)
    output = []
    for (stage, config_id), group in grouped.items():
        first = group[0]
        output.append(
            {
                "stage": stage,
                "config_id": config_id,
                **{key: first[key] for key in ("method", "edge_family", "top_k", "threshold", "policy")},
                "examples": len(group),
                "mean_ari": sum(row["ari"] for row in group) / len(group),
                "mean_nmi": sum(row["nmi"] for row in group) / len(group),
                "mean_pairwise_f1": sum(row["pairwise_f1"] for row in group) / len(group),
                "mean_boundary_f1": sum(row["boundary_f1"] for row in group) / len(group),
                "mean_cluster_count_error": sum(row["cluster_count_error"] for row in group) / len(group),
                "mean_graph_ms": sum(row["graph_ms"] for row in group) / len(group),
                "mean_cluster_ms": sum(row["cluster_ms"] for row in group) / len(group),
                "convergence_rate": sum(row["converged"] for row in group) / len(group),
            }
        )
    return output


def _best(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["mean_ari"],
            row["mean_pairwise_f1"],
            -row["mean_cluster_count_error"],
            -row["mean_graph_ms"] - row["mean_cluster_ms"],
            row["config_id"],
        ),
    )


def _plot_ablation(summaries: list[dict], output_dir: Path) -> None:
    """Render the validation-only staged sweep without implying a full grid."""

    stages = (
        ("edge_method", "Edge family / method"),
        ("top_k", "Neighbors per node"),
        ("threshold", "Edge threshold"),
        ("policy", "Direction policy"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (stage, title) in zip(axes.flat, stages, strict=True):
        rows = [row for row in summaries if row["stage"] == stage]
        if stage == "edge_method":
            family_labels = {
                "contextual": "ctx",
                "lexical": "lex",
                "contextual_lexical": "ctx+lex",
                "contextual_lexical_position": "ctx+lex+pos",
            }
            labels = [
                f"{family_labels[row['edge_family']]}\n{'CC' if row['method'] == 'cc' else 'LP'}"
                for row in rows
            ]
        elif stage == "top_k":
            labels = [str(row["top_k"]) for row in rows]
        elif stage == "threshold":
            labels = [f"{float(row['threshold']):.2f}" for row in rows]
        else:
            labels = [str(row["policy"]).replace("reciprocal_average", "reciprocal avg") for row in rows]
        axis.bar(range(len(rows)), [row["mean_ari"] for row in rows], color="#2f6690")
        axis.set_xticks(range(len(rows)), labels, rotation=0)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Validation ARI")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "validation_ablation.png", dpi=180)
    fig.savefig(output_dir / "validation_ablation.pdf")
    plt.close(fig)


def run(args) -> dict:
    cases = controlled_queries("validation", examples_per_seed=args.examples_per_seed)
    all_rows = []
    base = {"method": "cc", "edge_family": "contextual_lexical", "top_k": 4, "threshold": 0.55, "policy": "union"}

    edge_configs = []
    for method in ("cc", "label_propagation"):
        for edge_family in EDGE_WEIGHTS:
            config = {**base, "method": method, "edge_family": edge_family}
            config["config_id"] = f"edge-{method}-{edge_family}"
            edge_configs.append(config)
    for config in edge_configs:
        all_rows.extend(_evaluate(cases, config, "edge_method"))
    edge_summary = _summary(all_rows)
    chosen = _best(edge_summary)

    k_rows = []
    for top_k in (2, 4, 8, 16, 32):
        config = {key: chosen[key] for key in ("method", "edge_family", "threshold", "policy")}
        config.update(top_k=top_k, config_id=f"k-{top_k}")
        k_rows.extend(_evaluate(cases, config, "top_k"))
    all_rows.extend(k_rows)
    chosen_k = _best(_summary(k_rows))

    threshold_rows = []
    for threshold in (0.35, 0.45, 0.55, 0.65, 0.75):
        config = {key: chosen_k[key] for key in ("method", "edge_family", "top_k", "policy")}
        config.update(threshold=threshold, config_id=f"threshold-{threshold:.2f}")
        threshold_rows.extend(_evaluate(cases, config, "threshold"))
    all_rows.extend(threshold_rows)
    chosen_threshold = _best(_summary(threshold_rows))

    policy_rows = []
    for policy in ("directed", "union", "mutual", "reciprocal_average"):
        config = {key: chosen_threshold[key] for key in ("method", "edge_family", "top_k", "threshold")}
        config.update(policy=policy, config_id=f"policy-{policy}")
        policy_rows.extend(_evaluate(cases, config, "policy"))
    all_rows.extend(policy_rows)
    summaries = _summary(all_rows)
    selected_row = _best(_summary(policy_rows))
    selected = {key: selected_row[key] for key in ("method", "edge_family", "top_k", "threshold", "policy")}
    selected["selection_split"] = "controlled_validation"
    selected["selection_metric"] = "mean_ari_then_pairwise_f1"
    selected["oracle_test_cluster_count"] = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "algorithm_cross_rows.csv", all_rows)
    write_csv(args.output_dir / "algorithm_cross_summary.csv", summaries)
    _plot_ablation(summaries, args.output_dir)
    write_json(args.output_dir / "selected_graph_policy.json", selected)
    findings = {
        "schema_version": "1.0",
        "git": git_metadata(),
        "controlled_validation": controlled_manifest(cases),
        "selected_policy": selected,
        "selected_validation_metrics": selected_row,
        "stages": ["edge_method", "top_k", "threshold", "policy"],
        "attention_edges_available": False,
        "residual_edges_available": False,
    }
    write_json(args.output_dir / "algorithm_cross_findings.json", findings)
    return findings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-per-seed", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_7_query_graph/algorithm_cross",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
