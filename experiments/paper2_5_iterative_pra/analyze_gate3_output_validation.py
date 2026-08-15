"""Summarize and plot the frozen Gate-3 end-to-end generation benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


CONDITIONS = (
    "native_bounded",
    "one_shot",
    "graph_sparse",
    "graph_balanced",
    "graph_high",
    "oracle_evidence",
    "native_full_context",
)
LABELS = {
    "native_bounded": "Native, no source",
    "one_shot": "One-shot PRA",
    "graph_sparse": "Graph sparse",
    "graph_balanced": "Graph balanced",
    "graph_high": "Graph broad",
    "oracle_evidence": "Oracle evidence",
    "native_full_context": "Native full context",
}
COLORS = {
    "native_bounded": "#555555",
    "one_shot": "#2676b8",
    "graph_sparse": "#3e9b76",
    "graph_balanced": "#d17c18",
    "graph_high": "#b64545",
    "oracle_evidence": "#7c55a5",
    "native_full_context": "#111111",
}
METRICS = (
    "exact_match",
    "token_f1",
    "normalized_answer_accuracy",
    "oracle_evidence_recall",
    "complete_evidence_recovery",
    "selected_source_fraction",
    "materialized_unique_tokens",
    "native_kv_token_states",
    "native_kv_bytes",
    "active_native_kv_token_states",
    "active_native_kv_bytes",
    "evidence_attention_mass",
    "non_evidence_attention_mass",
    "normalized_attention_entropy",
    "ttft_seconds",
    "tpot_seconds",
    "total_generation_seconds",
    "routing_search_seconds",
    "materialization_seconds",
    "peak_gpu_allocated_bytes",
)


def _mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Average numeric measurements over exact categorical groups."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for identity, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        result.append(
            {
                **dict(zip(keys, identity)),
                "n": len(group),
                **{field: _mean(group, field) for field in METRICS},
            }
        )
    return result


def paired_bootstrap(
    rows: list[dict],
    target: str,
    baseline: str = "one_shot",
    *,
    replicates: int = 2000,
    seed: int = 2505,
) -> dict:
    """Bootstrap paired example-level F1 and accuracy deltas."""
    indexed = {(row["example_id"], row["condition"]): row for row in rows}
    identities = sorted(
        example_id
        for example_id in {row["example_id"] for row in rows}
        if (example_id, target) in indexed and (example_id, baseline) in indexed
    )
    rng = random.Random(seed)
    result = {"target": target, "baseline": baseline, "n": len(identities)}
    for field in ("token_f1", "normalized_answer_accuracy"):
        deltas = [
            float(indexed[(identity, target)][field])
            - float(indexed[(identity, baseline)][field])
            for identity in identities
        ]
        samples = sorted(
            statistics.fmean(rng.choice(deltas) for _ in deltas)
            for _ in range(replicates)
        ) if deltas else []
        result[f"delta_{field}"] = statistics.fmean(deltas) if deltas else None
        result[f"delta_{field}_ci_low"] = samples[int(0.025 * replicates)] if samples else None
        result[f"delta_{field}_ci_high"] = samples[int(0.975 * replicates)] if samples else None
        result[f"{field}_wins"] = sum(delta > 0 for delta in deltas)
        result[f"{field}_ties"] = sum(delta == 0 for delta in deltas)
        result[f"{field}_losses"] = sum(delta < 0 for delta in deltas)
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def correlations(rows: list[dict]) -> list[dict]:
    result = []
    memory = [row for row in rows if row["condition"] in CONDITIONS[1:6]]
    for dataset in sorted({row["dataset"] for row in memory}):
        group = [row for row in memory if row["dataset"] == dataset]
        for source in (
            "oracle_evidence_recall",
            "complete_evidence_recovery",
            "selected_source_fraction",
            "native_kv_token_states",
            "evidence_attention_mass",
            "non_evidence_attention_mass",
        ):
            pairs = [
                (float(row[source]), float(row["token_f1"]))
                for row in group
                if row.get(source) is not None
            ]
            result.append(
                {
                    "dataset": dataset,
                    "source_metric": source,
                    "quality_metric": "token_f1",
                    "n": len(pairs),
                    "pearson": _pearson(
                        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                    ),
                }
            )
    return result


def _save(figure, output: Path, stem: str) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _frontier_plot(summary: list[dict], output: Path, x: str, stem: str, xlabel: str) -> None:
    datasets = sorted({row["dataset"] for row in summary})
    figure, axes = plt.subplots(1, len(datasets), figsize=(10.2, 4.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        for row in [value for value in summary if value["dataset"] == dataset]:
            condition = row["condition"]
            value = row.get(x)
            if value is None:
                continue
            axis.scatter(value, row["token_f1"], color=COLORS[condition], s=55, zorder=3)
            axis.annotate(LABELS[condition], (value, row["token_f1"]), xytext=(4, 4),
                          textcoords="offset points", fontsize=7)
        axis.set_title("MuSiQue" if dataset == "musique" else "2Wiki")
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Token F1")
    _save(figure, output, stem)


def plots(heldout: list[dict], layer_rows: list[dict], summary: list[dict], output: Path) -> None:
    _frontier_plot(
        summary, output, "selected_source_fraction", "gate3_quality_source",
        "Selected source fraction",
    )
    _frontier_plot(
        summary, output, "active_native_kv_token_states", "gate3_quality_active_kv",
        "Total active native K/V token states",
    )

    band_summary = aggregate(
        [row for row in layer_rows if row["selection"] in {"graph_balanced", "oracle_evidence"}],
        ("dataset", "selection", "materialization_band"),
    )
    order = ["late_1", "late_4", "late_8", "middle_4", "layer_12", "topology_sparse", "all_28"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharey=True)
    for axis, dataset in zip(axes, ("musique", "2wikimultihopqa")):
        for selection, marker in (("graph_balanced", "o"), ("oracle_evidence", "s")):
            index = {row["materialization_band"]: row for row in band_summary
                     if row["dataset"] == dataset and row["selection"] == selection}
            axis.plot(range(len(order)), [index[name]["token_f1"] for name in order],
                      marker=marker, label=selection.replace("_", " "))
        axis.set_xticks(range(len(order)), order, rotation=35, ha="right")
        axis.set_title("MuSiQue" if dataset == "musique" else "2Wiki")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Validation token F1")
    axes[0].legend(frameon=False)
    _save(figure, output, "gate3_materialization_layers")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    mu = aggregate(
        [row for row in heldout if row["dataset"] == "musique"],
        ("annotated_hops", "condition"),
    )
    for condition in ("one_shot", "graph_balanced", "graph_high", "oracle_evidence", "native_full_context"):
        values = sorted((row for row in mu if row["condition"] == condition), key=lambda row: row["annotated_hops"])
        axes[0].plot([row["annotated_hops"] for row in values], [row["token_f1"] for row in values],
                     marker="o", color=COLORS[condition], label=LABELS[condition])
    axes[0].set_xlabel("MuSiQue annotated depth")
    axes[0].set_ylabel("Token F1")
    axes[0].set_xticks(sorted({row["annotated_hops"] for row in mu}))
    wiki = aggregate(
        [row for row in heldout if row["dataset"] == "2wikimultihopqa"],
        ("graph_type", "condition"),
    )
    types = sorted({str(row["graph_type"]) for row in wiki})
    for condition in ("one_shot", "graph_balanced", "graph_high", "oracle_evidence", "native_full_context"):
        index = {str(row["graph_type"]): row for row in wiki if row["condition"] == condition}
        axes[1].plot(range(len(types)), [index[name]["token_f1"] for name in types],
                     marker="o", color=COLORS[condition])
    axes[1].set_xticks(range(len(types)), types, rotation=30, ha="right")
    axes[1].set_xlabel("2Wiki graph type")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7)
    _save(figure, output, "gate3_depth_path_quality")

    memory = [row for row in heldout if row["condition"] in CONDITIONS[1:6]]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for axis, dataset in zip(axes, ("musique", "2wikimultihopqa")):
        group = [row for row in memory if row["dataset"] == dataset]
        for condition in CONDITIONS[1:6]:
            values = [row for row in group if row["condition"] == condition]
            axis.scatter([row["oracle_evidence_recall"] for row in values],
                         [row["token_f1"] for row in values], alpha=0.65, s=22,
                         color=COLORS[condition], label=LABELS[condition])
        axis.set_xlabel("Annotated-evidence recall")
        axis.set_title("MuSiQue" if dataset == "musique" else "2Wiki")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Token F1")
    _save(figure, output, "gate3_evidence_recall_quality")


def run(args: argparse.Namespace) -> dict:
    artifact = json.loads(args.results.read_text(encoding="utf-8"))
    rows = artifact["rows"]
    heldout = [row for row in rows if row["phase"] == "heldout"]
    layer_rows = [row for row in rows if row["phase"] == "layer_sweep"]
    if not heldout or not layer_rows:
        raise ValueError("both layer_sweep and heldout rows are required")
    expected = len({(row["dataset"], row["example_id"]) for row in heldout}) * len(CONDITIONS)
    if len(heldout) != expected:
        raise ValueError(f"heldout matrix is incomplete: {len(heldout)} != {expected}")
    if any(row["native_limit_violations"] for row in rows):
        raise AssertionError("a native-operation limit was violated")

    summary = aggregate(heldout, ("dataset", "condition", "materialization_band"))
    strata = aggregate(heldout, ("dataset", "annotated_hops", "graph_type", "condition"))
    paired = [
        {"dataset": dataset, **paired_bootstrap(
            [row for row in heldout if row["dataset"] == dataset], condition
        )}
        for dataset in sorted({row["dataset"] for row in heldout})
        for condition in ("graph_sparse", "graph_balanced", "graph_high", "oracle_evidence", "native_full_context")
    ]
    correlation_rows = correlations(heldout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "gate3_output_summary.csv", summary)
    _write_csv(args.output_dir / "gate3_output_strata.csv", strata)
    _write_csv(args.output_dir / "gate3_output_paired_bootstrap.csv", paired)
    _write_csv(args.output_dir / "gate3_output_correlations.csv", correlation_rows)
    plots(heldout, layer_rows, summary, args.output_dir)
    report = {
        "schema_version": "1.0",
        "generation_artifact": str(args.results),
        "heldout_examples": len({(row["dataset"], row["example_id"]) for row in heldout}),
        "generation_rows": len(rows),
        "summary": summary,
        "paired_bootstrap": paired,
        "correlations": correlation_rows,
    }
    (args.output_dir / "gate3_output_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("docs/papers/shared/results/paper2_5_iterative_pra/output_validation")
    parser.add_argument("--results", type=Path, default=default / "gate3_generation_results.json")
    parser.add_argument("--output-dir", type=Path, default=default)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"rows": result["generation_rows"], "heldout": result["heldout_examples"]}, indent=2))
