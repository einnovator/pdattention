"""Summarize the Paper 2.8 query-conditioned selector sweep."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_query_conditioned_study import (
    OBJECTIVES,
    PRIMARY_CONFIGURATION,
    _paired_effects,
    _write_csv,
)


RESULTS = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
OUTPUT = RESULTS / "query_conditioned"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _typed_rows(rows: list[dict]) -> list[dict]:
    integer_fields = {"rank", "m", "seed", "parameter_count"}
    float_fields = {
        "evidence_recall",
        "teacher_top4_overlap",
        "selection_ms",
        "materialized_kv_tokens",
        "native_dots",
    }
    return [
        {
            key: (
                int(value)
                if key in integer_fields
                else float(value)
                if key in float_fields
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def _mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _configuration_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["objective"], row["rank"], row["m"])].append(row)
    return [
        {
            "dataset": key[0],
            "objective": key[1],
            "rank": key[2],
            "m": key[3],
            "evidence_recall": _mean(group, "evidence_recall"),
            "teacher_top4_overlap": _mean(group, "teacher_top4_overlap"),
            "selection_ms": _mean(group, "selection_ms"),
            "parameter_count": group[0]["parameter_count"],
        }
        for key, group in sorted(grouped.items())
    ]


def _seed_stability(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["objective"],
                row["rank"],
                row["m"],
                row["seed"],
            )
        ].append(row)
    return [
        {
            "dataset": key[0],
            "objective": key[1],
            "rank": key[2],
            "m": key[3],
            "seed": key[4],
            "evidence_recall": _mean(group, "evidence_recall"),
            "teacher_top4_overlap": _mean(group, "teacher_top4_overlap"),
            "selection_ms": _mean(group, "selection_ms"),
        }
        for key, group in sorted(grouped.items())
    ]


def _response_recovery(summary: list[dict]) -> list[dict]:
    original = _read(RESULTS / "summary.csv")
    baseline = {}
    for dataset in ("hotpotqa", "qasper"):
        for method, m in (("mean", 1), ("greedy_oracle", 4), ("greedy_oracle", 8)):
            row = next(
                item
                for item in original
                if item["split"] == "test"
                and item["dataset"] == dataset
                and item["method"] == method
                and int(item["m"]) == m
            )
            baseline[(dataset, method, m)] = float(row["teacher_top4_overlap"])
    output = []
    configurations = sorted(
        {(row["objective"], row["rank"], row["m"]) for row in summary}
    )
    for objective, rank, m in configurations:
        learned = _mean(
            [
                row
                for row in summary
                if row["objective"] == objective
                and row["rank"] == rank
                and row["m"] == m
            ],
            "teacher_top4_overlap",
        )
        mean_overlap = sum(
            baseline[(dataset, "mean", 1)] for dataset in ("hotpotqa", "qasper")
        ) / 2
        oracle_overlap = sum(
            baseline[(dataset, "greedy_oracle", m)]
            for dataset in ("hotpotqa", "qasper")
        ) / 2
        output.append(
            {
                "objective": objective,
                "rank": rank,
                "m": m,
                "learned_top4_overlap": learned,
                "mean_top4_overlap": mean_overlap,
                "oracle_top4_overlap": oracle_overlap,
                "oracle_gain_recovered": (learned - mean_overlap)
                / max(oracle_overlap - mean_overlap, 1e-12),
            }
        )
    return output


def _baseline_values() -> dict[tuple[str, str], float]:
    original = _read(RESULTS / "summary.csv")
    values = {}
    for dataset in ("hotpotqa", "qasper"):
        values[(dataset, "QK mean")] = float(
            next(
                row["evidence_recall"]
                for row in original
                if row["split"] == "test"
                and row["dataset"] == dataset
                and row["method"] == "mean"
            )
        )
        values[(dataset, "key-only m=8")] = float(
            next(
                row["evidence_recall"]
                for row in original
                if row["split"] == "test"
                and row["dataset"] == dataset
                and row["method"] == "learned"
                and int(row["m"]) == 8
            )
        )
    paper26 = _read(RESULTS / "paper2_6_matched_comparison.csv")
    for dataset in ("hotpotqa", "qasper"):
        for condition, label in (
            ("B0_gist", "gist"),
            ("B1_bm25", "BM25"),
            ("B2_exact", "exact"),
            ("H5_iterative_hybrid", "hybrid"),
        ):
            values[(dataset, label)] = float(
                next(
                    row["evidence_recall"]
                    for row in paper26
                    if row["dataset"] == dataset and row["condition"] == condition
                )
            )
    return values


def _plots(summary: list[dict], seed_rows: list[dict], output: Path) -> None:
    colors = {
        "oracle_imitation": "#457b9d",
        "listwise": "#edae49",
        "combined": "#2a9d8f",
        "decision_aware": "#d1495b",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for objective in OBJECTIVES:
            selected = [
                row
                for row in summary
                if row["dataset"] == dataset and row["objective"] == objective
            ]
            axis.scatter(
                [row["teacher_top4_overlap"] for row in selected],
                [row["evidence_recall"] for row in selected],
                label=objective.replace("_", " "),
                color=colors[objective],
                alpha=0.8,
            )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Full-QK teacher top-four overlap")
        axis.set_ylabel("Evidence recall at four chunks")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"recall_vs_preservation.{suffix}", dpi=180)
    plt.close(figure)

    baseline = _baseline_values()
    primary = {
        row["dataset"]: row
        for row in summary
        if row["objective"] == PRIMARY_CONFIGURATION["objective"]
        and row["rank"] == PRIMARY_CONFIGURATION["rank"]
        and row["m"] == PRIMARY_CONFIGURATION["m"]
    }
    exploratory = max(
        (row for row in summary if row["dataset"] == "qasper"),
        key=lambda row: row["evidence_recall"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        labels = ["QK mean", "key-only m=8", "gist", "BM25", "exact", "hybrid", "primary"]
        values = [baseline[(dataset, label)] for label in labels[:-1]]
        values.append(primary[dataset]["evidence_recall"])
        if dataset == "qasper":
            labels.append("exploratory")
            values.append(exploratory["evidence_recall"])
        axis.bar(labels, values, color="#457b9d")
        axis.set_title(dataset.upper())
        axis.set_ylabel("Evidence recall at four chunks")
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"primary_exploratory_comparison.{suffix}", dpi=180)
    plt.close(figure)

    selected_configs = (
        ("combined", 16, 4, "primary"),
        ("decision_aware", 32, 8, "QASPER exploratory"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for objective, rank, m, label in selected_configs:
            values = [
                row
                for row in seed_rows
                if row["dataset"] == dataset
                and row["objective"] == objective
                and row["rank"] == rank
                and row["m"] == m
            ]
            axis.plot(
                [row["seed"] for row in values],
                [row["evidence_recall"] for row in values],
                marker="o",
                label=label,
            )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Seed")
        axis.set_ylabel("Evidence recall")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"selected_seed_stability.{suffix}", dpi=180)
    plt.close(figure)


def run() -> dict:
    rows = _typed_rows(_read(OUTPUT / "per_example.csv"))
    summary = _configuration_summary(rows)
    seed_rows = _seed_stability(rows)
    paired = _paired_effects(rows, 20260824)
    recovery = _response_recovery(summary)
    _write_csv(OUTPUT / "summary.csv", summary)
    _write_csv(OUTPUT / "seed_stability.csv", seed_rows)
    _write_csv(OUTPUT / "paired_effects.csv", paired)
    _write_csv(OUTPUT / "response_recovery.csv", recovery)
    _plots(summary, seed_rows, OUTPUT)

    primary = [
        row
        for row in summary
        if row["objective"] == PRIMARY_CONFIGURATION["objective"]
        and row["rank"] == PRIMARY_CONFIGURATION["rank"]
        and row["m"] == PRIMARY_CONFIGURATION["m"]
    ]
    exploratory = {
        dataset: max(
            (row for row in summary if row["dataset"] == dataset),
            key=lambda row: row["evidence_recall"],
        )
        for dataset in ("hotpotqa", "qasper")
    }
    best_recovery = max(recovery, key=lambda row: row["oracle_gain_recovered"])
    preservation_recall_correlation = {}
    for dataset in ("hotpotqa", "qasper"):
        selected = [row for row in summary if row["dataset"] == dataset]
        preservation_recall_correlation[dataset] = float(
            np.corrcoef(
                [row["teacher_top4_overlap"] for row in selected],
                [row["evidence_recall"] for row in selected],
            )[0, 1]
        )
    findings = {
        "primary_configuration": PRIMARY_CONFIGURATION,
        "primary_results": primary,
        "exploratory_best_by_dataset": exploratory,
        "best_response_recovery": best_recovery,
        "preservation_recall_pearson": preservation_recall_correlation,
        "interpretation": {
            "primary_is_uniform_frontier": False,
            "qasper_exploratory_near_exact": True,
            "test_selected_results_are_exploratory": True,
        },
    }
    (OUTPUT / "findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
