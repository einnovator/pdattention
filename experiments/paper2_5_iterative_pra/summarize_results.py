"""Derive paired diagnostics and publication figures from closure artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"


def _primary(rows, dataset, method):
    return {
        (row["seed"], row["example_id"]): row
        for row in rows
        if row["dataset"] == dataset
        and row["method"] == method
        and row["fraction"] == 0.10
        and (
            method == "one_shot"
            or (
                row["depth"] == 2
                and row["alpha"] == 0.25
                and row["frontier"] == "direct"
                and row["path_score"] == "product"
            )
        )
    }


def relational_gap_rows(rows):
    output = []
    for dataset in ("hotpotqa", "qasper"):
        one_shot = _primary(rows, dataset, "one_shot")
        iterative = _primary(rows, dataset, "iterative")
        paired = []
        for key, baseline in one_shot.items():
            if baseline["relational_gap"] is None:
                continue
            closure = iterative[key]
            paired.append(
                {
                    "dataset": dataset,
                    "seed": key[0],
                    "example_id": key[1],
                    "relational_gap": baseline["relational_gap"],
                    "delta_any": closure["any_evidence"] - baseline["any_evidence"],
                    "delta_all": closure["all_evidence"] - baseline["all_evidence"],
                    "delta_chain": closure["chain_completion"] - baseline["chain_completion"],
                    "delta_coverage": closure["evidence_coverage"] - baseline["evidence_coverage"],
                }
            )
        ordered = sorted(paired, key=lambda row: row["relational_gap"])
        for index, row in enumerate(ordered):
            row["gap_quartile"] = min(4, 1 + 4 * index // len(ordered))
        output.extend(ordered)
    return output


def quartiles(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["gap_quartile"])].append(row)
    output = []
    for (dataset, quartile), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "gap_quartile": quartile,
                "pairs": len(values),
                **{
                    key: statistics.fmean(row[key] for row in values)
                    for key in ("relational_gap", "delta_any", "delta_all", "delta_chain", "delta_coverage")
                },
            }
        )
    return output


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gap_plot(rows):
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    colors = {"hotpotqa": "#1f6f8b", "qasper": "#c05746"}
    for axis, metric, title in zip(
        axes,
        ("delta_any", "delta_coverage"),
        ("Any-evidence effect", "Evidence-coverage effect"),
    ):
        for dataset in ("hotpotqa", "qasper"):
            selected = [row for row in rows if row["dataset"] == dataset]
            axis.scatter(
                [row["relational_gap"] for row in selected],
                [row[metric] for row in selected],
                alpha=0.45,
                s=18,
                color=colors[dataset],
                label=dataset,
            )
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xlabel(r"Relational gap $s(A,B)-s(q,B)$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"Closure $-$ one-shot at 10\%")
    axes[1].legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(RESULTS / f"relational_gap_effect.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _cost_plot(aggregates):
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = {"hotpotqa": "#1f6f8b", "qasper": "#c05746"}
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for method, depth, label, marker in (
            ("one_shot", 1, "One-shot", "o"),
            ("iterative", 2, "Closure D=2", "s"),
            ("iterative", 3, "Closure D=3", "^"),
            ("iterative", 4, "Closure D=4", "D"),
        ):
            selected = [
                row for row in aggregates
                if row["dataset"] == dataset and row["method"] == method
                and row["depth"] == depth
                and (
                    method == "one_shot"
                    or (row["alpha"] == 0.25 and row["frontier"] == "direct" and row["path_score"] == "product")
                )
            ]
            selected.sort(key=lambda row: row["fraction"])
            axis.plot(
                [100 * row["fraction"] for row in selected],
                [row["gist_comparisons"] for row in selected],
                marker=marker,
                label=label,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Final unique-chunk budget (%)")
        axis.set_ylabel("Gist comparisons per example")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(RESULTS / f"closure_search_cost.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run():
    artifact = json.loads((RESULTS / "iterative_closure_results.json").read_text(encoding="utf-8"))
    paired = relational_gap_rows(artifact["rows"])
    quartile_rows = quartiles(paired)
    _write_csv(RESULTS / "relational_gap_pairs.csv", paired)
    _write_csv(RESULTS / "relational_gap_quartiles.csv", quartile_rows)
    _gap_plot(paired)
    _cost_plot(artifact["aggregates"])
    summary = {
        "protocol": "paired D=2 closure minus matched one-shot Top-B at 10%",
        "quartiles": quartile_rows,
    }
    (RESULTS / "relational_gap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
