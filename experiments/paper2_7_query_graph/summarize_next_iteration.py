"""Synthesize the standalone Paper 2.7 natural-validation iteration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_7_query_graph.helpers import write_csv, write_json


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _strata(rows, metric, left, right):
    paired = defaultdict(dict)
    metadata = {}
    for row in rows:
        key = (row["dataset"], row["example_id"])
        paired[key][row.get("method", row.get("condition"))] = float(row[metric])
        metadata[key] = row
    output = []
    for stratum, predicate in (
        ("two_facets", lambda row: int(float(row.get("target_facets", row.get("facet_count", 0)))) == 2),
        ("three_plus_facets", lambda row: int(float(row.get("target_facets", row.get("facet_count", 0)))) >= 3),
        ("contiguous", lambda row: float(row["has_non_contiguous_facet"]) == 0),
        ("non_contiguous", lambda row: float(row["has_non_contiguous_facet"]) == 1),
    ):
        values = [methods[left] - methods[right] for key, methods in paired.items() if left in methods and right in methods and predicate(metadata[key])]
        output.append({"metric": metric, "left": left, "right": right, "stratum": stratum, "examples": len(values), "mean_delta": sum(values) / len(values) if values else None})
    return output


def run(args):
    facet_findings = [_json(path) for path in args.facet_findings]
    retrieval_findings = [_json(path) for path in args.retrieval_findings]
    facet_rows = [row for path in args.facet_rows for row in _csv(path)]
    retrieval_rows = [row for path in args.retrieval_rows for row in _csv(path)]
    llm = _json(args.llm)
    strata = []
    for model in sorted({row["model_id"] for row in facet_rows}):
        group = [row for row in facet_rows if row["model_id"] == model]
        for comparison in (("graph_cc", "fixed_window"), ("graph_cc", "embedding_kmeans"), ("graph_cc", "llm")):
            strata.extend({"model_id": model, **row} for row in _strata(group, "ari", *comparison))
    for model in sorted({row["query_graph_model"] for row in retrieval_rows}):
        group = [row for row in retrieval_rows if row["query_graph_model"] == model]
        for comparison in (("graph_cc_semantic", "global_semantic"), ("graph_cc_hybrid", "lexical_semantic_hybrid"), ("graph_cc_semantic", "llm_semantic")):
            strata.extend({"model_id": model, **row} for row in _strata(group, "evidence_recall", *comparison))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "complexity_strata.csv", strata)
    artifact = {
        "schema_version": "1.0",
        "benchmark_examples": facet_findings[0]["examples"],
        "test_examples": facet_findings[0]["test_examples"],
        "models": [row["model_id"] for row in facet_findings],
        "facet_findings": facet_findings,
        "retrieval_findings": retrieval_findings,
        "llm": {key: llm.get(key) for key in ("model", "temperature", "max_generated_tokens", "examples", "failures", "strict_json_examples", "partial_json_recoveries", "mean_latency_ms", "total_prompt_tokens", "total_generated_tokens")},
        "native_kv_gate_pass": any(row["native_kv_gate_pass"] for row in retrieval_findings),
        "outcome": "C: natural facets are partly recoverable, but graph decomposition does not improve matched-budget retrieval",
        "complexity_strata": strata,
    }
    write_json(args.output_dir / "next_iteration_findings.json", artifact)

    method_order = ["global", "fixed_window", "syntax", "embedding_kmeans", "graph_cc", "graph_lp", "llm"]
    fig, axes = plt.subplots(1, len(facet_findings), figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for axis, finding in zip(axes, facet_findings):
        aggregate = []
        for method in method_order:
            values = [row["mean_ari"] for row in finding["summary"] if row["method"] == method]
            if values:
                aggregate.append((method, sum(values) / len(values)))
        axis.bar(range(len(aggregate)), [value for _, value in aggregate], color=["#2878b5" if name == "graph_cc" else "#68737d" for name, _ in aggregate])
        axis.set_xticks(range(len(aggregate)), [name.replace("_", "\n") for name, _ in aggregate], fontsize=7)
        axis.set_title(finding["model_id"].split("/")[-1])
        axis.set_ylabel("Mean natural-facet ARI")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "natural_facet_cross_model.pdf")
    fig.savefig(args.output_dir / "natural_facet_cross_model.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(retrieval_findings), figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for axis, finding in zip(axes, retrieval_findings):
        comparisons = [row for row in finding["paired_recall"] if (row["left"], row["right"]) in {("graph_cc_semantic", "global_semantic"), ("graph_cc_hybrid", "lexical_semantic_hybrid")}]
        axis.axhline(0, color="black", linewidth=0.8)
        axis.bar(range(len(comparisons)), [row["mean_recall_delta"] for row in comparisons], color="#b24c42")
        axis.set_xticks(range(len(comparisons)), [f"{row['dataset']}\n{row['left'].replace('graph_cc_', '')}" for row in comparisons], fontsize=7)
        axis.set_title(finding["query_graph_model"].split("/")[-1])
        axis.set_ylabel("Graph recall delta")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "fresh_retrieval_cross_model.pdf")
    fig.savefig(args.output_dir / "fresh_retrieval_cross_model.png", dpi=180)
    plt.close(fig)
    return artifact


def parse_args():
    base = ROOT / "docs/papers/shared/results/paper2_7_query_graph"
    parser = argparse.ArgumentParser()
    parser.add_argument("--facet-findings", type=Path, nargs="+", default=[base / "natural_facets/qwen/natural_facet_findings.json", base / "natural_facets/smollm/natural_facet_findings.json"])
    parser.add_argument("--facet-rows", type=Path, nargs="+", default=[base / "natural_facets/qwen/natural_facet_rows.csv", base / "natural_facets/smollm/natural_facet_rows.csv"])
    parser.add_argument("--retrieval-findings", type=Path, nargs="+", default=[base / "fresh_retrieval/qwen/fresh_retrieval_findings.json", base / "fresh_retrieval/smollm/fresh_retrieval_findings.json"])
    parser.add_argument("--retrieval-rows", type=Path, nargs="+", default=[base / "fresh_retrieval/qwen/fresh_retrieval_rows.csv", base / "fresh_retrieval/smollm/fresh_retrieval_rows.csv"])
    parser.add_argument("--llm", type=Path, default=base / "natural_facets/llm_decomposition.json")
    parser.add_argument("--output-dir", type=Path, default=base / "next_iteration")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
