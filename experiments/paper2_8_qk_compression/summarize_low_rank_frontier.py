"""Summarize Paper 2.8 post-G3 centroid and low-rank frontier results."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_gated_study import _bootstrap, _write_csv


RESULTS = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
OUTPUT = RESULTS / "low_rank_frontier"
DATASETS = ("hotpotqa", "qasper")
PAPER26 = {
    "B0_gist": "Paper 2.6 gist",
    "B1_bm25": "Paper 2.6 BM25",
    "B2_exact": "Paper 2.6 exact",
    "H5_iterative_hybrid": "Paper 2.6 hybrid",
}


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _identity_maps(rows: list[dict]) -> dict[tuple, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["method"],
                int(row["rank"]),
                int(row["m"]),
                row["example_id"],
            )
        ].append(float(row["evidence_recall"]))
    return {key: _mean(values) for key, values in grouped.items()}


def _baselines() -> tuple[dict[tuple[str, str, str], float], dict[str, dict]]:
    values = {}
    metadata = {
        "QK mean": {"index_bytes_per_chunk": 4096.0},
        "QK key-only m8": {"index_bytes_per_chunk": 32768.0},
        "QC primary": {"index_bytes_per_chunk": 16384.0},
        "QC exploratory": {"index_bytes_per_chunk": 32768.0},
    }
    natural = _read(RESULTS / "natural_rows.csv")
    grouped = defaultdict(list)
    for row in natural:
        if row["split"] != "test":
            continue
        if row["method"] == "mean":
            grouped[(row["dataset"], "QK mean", row["example_id"])].append(
                float(row["evidence_recall"])
            )
        if row["method"] == "learned" and int(row["m"]) == 8:
            grouped[(row["dataset"], "QK key-only m8", row["example_id"])].append(
                float(row["evidence_recall"])
            )
    query = _read(RESULTS / "query_conditioned/per_example.csv")
    for row in query:
        configuration = (row["objective"], int(row["rank"]), int(row["m"]))
        label = None
        if configuration == ("combined", 16, 4):
            label = "QC primary"
        elif configuration == ("decision_aware", 32, 8):
            label = "QC exploratory"
        if label:
            grouped[(row["dataset"], label, row["example_id"])].append(
                float(row["evidence_recall"])
            )
    paper26 = _read(RESULTS.parent / "paper2_6_hybrid_pra/per_example.csv")
    for row in paper26:
        if row["split"] == "test" and row["condition"] in PAPER26:
            grouped[
                (row["dataset"], PAPER26[row["condition"]], row["example_id"])
            ].append(float(row["evidence_recall"]))
    for key, group in grouped.items():
        values[key] = _mean(group)
    return values, metadata


def _paired_effects(rows: list[dict], seed: int = 20260824) -> list[dict]:
    frontier = _identity_maps(rows)
    baselines, _ = _baselines()
    output = []
    configurations = sorted({key[:4] for key in frontier})
    for dataset, method, rank, m in configurations:
        identities = {
            key[4]: value for key, value in frontier.items() if key[:4] == (dataset, method, rank, m)
        }
        baseline_labels = sorted(
            {label for ds, label, _ in baselines if ds == dataset}
        )
        for label in baseline_labels:
            differences = [
                value - baselines[(dataset, label, identity)]
                for identity, value in identities.items()
                if (dataset, label, identity) in baselines
            ]
            if not differences:
                continue
            low, high = _bootstrap(
                differences,
                seed + rank * 101 + m * 17 + sum(map(ord, method + label + dataset)),
            )
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "rank": rank,
                    "m": m,
                    "baseline": label,
                    "pairs": len(differences),
                    "mean_delta": _mean(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": sum(value > 0 for value in differences),
                    "losses": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return output


def _seed_statistics(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if int(row["seed"]) < 0:
            continue
        grouped[
            (
                row["dataset"],
                row["method"],
                int(row["rank"]),
                int(row["m"]),
                int(row["seed"]),
            )
        ].append(row)
    by_configuration = defaultdict(list)
    for key, group in grouped.items():
        by_configuration[key[:4]].append(
            {
                "seed": key[4],
                "recall": _mean([_float(row, "evidence_recall") for row in group]),
                "overlap": _mean([_float(row, "teacher_top4_overlap") for row in group]),
            }
        )
    output = []
    for key, seeds in sorted(by_configuration.items()):
        recalls = [row["recall"] for row in seeds]
        overlaps = [row["overlap"] for row in seeds]
        correlation = (
            float(np.corrcoef(recalls, overlaps)[0, 1])
            if len(set(recalls)) > 1 and len(set(overlaps)) > 1
            else 0.0
        )
        output.append(
            {
                "dataset": key[0],
                "method": key[1],
                "rank": key[2],
                "m": key[3],
                "seeds": len(seeds),
                "recall_mean": _mean(recalls),
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "recall_std": statistics.pstdev(recalls),
                "overlap_mean": _mean(overlaps),
                "overlap_min": min(overlaps),
                "overlap_max": max(overlaps),
                "recall_overlap_correlation": correlation,
            }
        )
    return output


def _cost_frontier(summary: list[dict]) -> list[dict]:
    full_native_bytes = 32 * 1024 * 4
    mean_native_bytes = 1024 * 4
    native_m8_bytes = 8 * 1024 * 4
    output = []
    for row in summary:
        index_bytes = float(row["index_bytes_per_chunk"])
        output.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "rank": int(row["rank"]),
                "m": int(row["m"]),
                "evidence_recall": float(row["evidence_recall"]),
                "index_bytes_per_chunk": index_bytes,
                "compression_vs_full_native_32d": full_native_bytes / index_bytes,
                "compression_vs_mean_native_d": mean_native_bytes / index_bytes,
                "compression_vs_native_m8": native_m8_bytes / index_bytes,
                "cached_online_ms": float(row["cached_online_ms"]),
                "construction_ms": float(row["construction_ms"]),
                "native_dots": float(row["native_dots"]),
                "low_rank_dots": float(row["low_rank_dots"]),
                "backing_native_kv_bytes": float(row["backing_native_kv_bytes"]),
                "transfer_bytes": float(row["transfer_bytes"]),
                "materialized_kv_tokens": float(row["materialized_kv_tokens"]),
                "peak_delta_gpu_bytes": float(row["peak_delta_gpu_bytes"]),
            }
        )
    return output


def _query_summary() -> list[dict]:
    rows = _read(RESULTS / "query_conditioned/summary.csv")
    for row in rows:
        for key in ("rank", "m"):
            row[key] = int(row[key])
        for key in ("evidence_recall", "teacher_top4_overlap"):
            row[key] = float(row[key])
    return rows


def _compact_configuration(row: dict) -> dict:
    fields = (
        "dataset",
        "method",
        "rank",
        "m",
        "evidence_recall",
        "teacher_top4_overlap",
        "index_bytes_per_chunk",
        "cached_online_ms",
        "construction_ms",
        "native_dots",
        "low_rank_dots",
        "total_parameters",
    )
    return {field: row[field] for field in fields if field in row}


def _gate_decisions(summary: list[dict], paired: list[dict]) -> dict:
    parity = json.loads((OUTPUT / "baseline_parity.json").read_text(encoding="utf-8"))
    query = _query_summary()
    original = _read(RESULTS / "summary.csv")

    def original_value(dataset: str, method: str, m: int) -> float:
        return float(
            next(
                row["evidence_recall"]
                for row in original
                if row["split"] == "test"
                and row["dataset"] == dataset
                and row["method"] == method
                and int(row["m"]) == m
            )
        )

    def query_value(dataset: str, objective: str, rank: int, m: int, field="evidence_recall") -> float:
        return float(
            next(
                row[field]
                for row in query
                if row["dataset"] == dataset
                and row["objective"] == objective
                and row["rank"] == rank
                and row["m"] == m
            )
        )

    native_m4 = {
        dataset: max(
            (
                row
                for row in summary
                if row["dataset"] == dataset
                and row["method"]
                in {"native_kmeans", "native_medoid", "native_farthest"}
                and int(row["m"]) == 4
            ),
            key=lambda row: float(row["evidence_recall"]),
        )
        for dataset in DATASETS
    }
    primary = {
        dataset: query_value(dataset, "combined", 16, 4) for dataset in DATASETS
    }
    e1_deltas = {
        dataset: primary[dataset] - float(native_m4[dataset]["evidence_recall"])
        for dataset in DATASETS
    }
    e1_pass = any(delta >= 0.02 for delta in e1_deltas.values())

    key_only = {dataset: original_value(dataset, "learned", 8) for dataset in DATASETS}
    primary_deltas = {dataset: primary[dataset] - key_only[dataset] for dataset in DATASETS}
    exploratory = {
        dataset: query_value(dataset, "decision_aware", 32, 8) for dataset in DATASETS
    }
    exploratory_deltas = {
        dataset: exploratory[dataset] - key_only[dataset] for dataset in DATASETS
    }
    e2_primary_pass = max(primary_deltas.values()) >= 0.02 and min(primary_deltas.values()) >= -0.02
    e2_exploratory_pass = (
        max(exploratory_deltas.values()) >= 0.02
        and min(exploratory_deltas.values()) >= -0.02
    )

    e3_deltas = {
        dataset: query_value(dataset, "decision_aware", 32, 8)
        - query_value(dataset, "oracle_imitation", 32, 8)
        for dataset in DATASETS
    }
    e3_pass = max(e3_deltas.values()) >= 0.02 and min(e3_deltas.values()) >= -0.02

    lowrank = [row for row in summary if row["method"].startswith("lowrank_")]
    best_lowrank = {
        dataset: max(
            (row for row in lowrank if row["dataset"] == dataset),
            key=lambda row: float(row["evidence_recall"]),
        )
        for dataset in DATASETS
    }
    means = {dataset: original_value(dataset, "mean", 1) for dataset in DATASETS}
    full_targets = {"hotpotqa": primary["hotpotqa"], "qasper": exploratory["qasper"]}
    gain_retention = {}
    for dataset in DATASETS:
        denominator = full_targets[dataset] - means[dataset]
        gain_retention[dataset] = (
            (float(best_lowrank[dataset]["evidence_recall"]) - means[dataset]) / denominator
            if denominator > 0
            else float("nan")
        )
    e4_candidates = [
        row
        for row in lowrank
        if float(row["index_bytes_per_chunk"]) <= 4096
        and any(
            row["dataset"] == dataset and gain_retention[dataset] >= 0.95
            for dataset in DATASETS
        )
    ]
    e4_pass = bool(e4_candidates)

    baselines, _ = _baselines()
    qasper_gist = _mean(
        [value for (dataset, label, _), value in baselines.items() if dataset == "qasper" and label == "Paper 2.6 gist"]
    )
    joint = [
        row
        for row in lowrank
        if row["dataset"] == "qasper"
        and row["method"] != "lowrank_all"
        and int(row["m"]) in {4, 8}
        and float(row["evidence_recall"]) >= qasper_gist
        and float(row["index_bytes_per_chunk"]) <= 1024
    ]
    e5_pass = bool(joint)
    extension_credible = e2_primary_pass and e3_pass and e4_pass and e5_pass
    return {
        "original_g0_g3_unchanged": True,
        "extension_rule_note": (
            "E1-E5 are post-G3 diagnostic rules. E2 distinguishes the prespecified "
            "primary from the test-selected exploratory configuration."
        ),
        "E0": {"status": "pass" if parity["passed"] else "fail", "evidence": parity},
        "E1": {
            "status": "pass" if e1_pass else "fail",
            "primary_minus_best_matched_native_m4": e1_deltas,
            "matched_controls": {
                dataset: _compact_configuration(row)
                for dataset, row in native_m4.items()
            },
        },
        "E2": {
            "status": "pass" if e2_primary_pass else "inconclusive",
            "primary_deltas_over_key_only": primary_deltas,
            "exploratory_deltas_over_key_only": exploratory_deltas,
            "exploratory_rule_pass": e2_exploratory_pass,
            "reason": (
                "The prespecified primary must satisfy the cross-dataset rule; a "
                "test-selected exploratory row cannot make this gate confirmatory."
            ),
        },
        "E3": {
            "status": "exploratory_pass" if e3_pass else "fail",
            "decision_aware_minus_oracle_imitation": e3_deltas,
        },
        "E4": {
            "status": "pass" if e4_pass else "fail",
            "best_lowrank_by_dataset": {
                dataset: _compact_configuration(row)
                for dataset, row in best_lowrank.items()
            },
            "gain_retention": gain_retention,
            "target_gain_retention": 0.95,
            "target_max_index_bytes_per_chunk": 4096,
        },
        "E5": {
            "status": "pass" if e5_pass else "fail",
            "qasper_semantic_gist_recall": qasper_gist,
            "qualifying_joint_configurations": [
                _compact_configuration(row) for row in joint
            ],
        },
        "E6": {
            "status": "closed" if not extension_credible else "open",
            "reason": "Dataset expansion requires a credible E2-E5 frontier.",
        },
    }


def _changed_selection(rows: list[dict], summary: list[dict]) -> list[dict]:
    maps = _identity_maps(rows)
    natural = _read(RESULTS / "natural_rows.csv")
    mean_rows = {
        (row["dataset"], row["example_id"]): row
        for row in natural
        if row["split"] == "test" and row["method"] == "mean"
    }
    output = []
    for dataset in DATASETS:
        best = max(
            (
                row
                for row in summary
                if row["dataset"] == dataset and row["method"].startswith("lowrank_")
            ),
            key=lambda row: float(row["evidence_recall"]),
        )
        selected_rows = defaultdict(list)
        for row in rows:
            if (
                row["dataset"] == dataset
                and row["method"] == best["method"]
                and int(row["rank"]) == int(best["rank"])
                and int(row["m"]) == int(best["m"])
            ):
                selected_rows[row["example_id"]].append(row)
        for identity, group in selected_rows.items():
            mean_row = mean_rows[(dataset, identity)]
            mode_selection = max(
                {row["selected_chunks"] for row in group},
                key=lambda selection: sum(row["selected_chunks"] == selection for row in group),
            )
            baseline_selected = mean_row["selected_chunks"]
            output.append(
                {
                    "dataset": dataset,
                    "example_id": identity,
                    "method": best["method"],
                    "rank": best["rank"],
                    "m": best["m"],
                    "changed_selection": mode_selection != baseline_selected,
                    "lowrank_selected_chunks": mode_selection,
                    "mean_selected_chunks": baseline_selected,
                    "lowrank_recall": _mean([float(row["evidence_recall"]) for row in group]),
                    "mean_recall": float(mean_row["evidence_recall"]),
                }
            )
    return output


def _plots(summary: list[dict], seed_rows: list[dict]) -> None:
    colors = {
        "native_mean": "#6c757d",
        "native_kmeans": "#457b9d",
        "native_medoid": "#edae49",
        "native_farthest": "#8d5a97",
        "lowrank_all": "#264653",
        "lowrank_kmeans": "#2a9d8f",
        "lowrank_medoid": "#e76f51",
        "lowrank_farthest": "#d1495b",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        for method in colors:
            selected = [
                row for row in summary if row["dataset"] == dataset and row["method"] == method
            ]
            if not selected:
                continue
            axis.scatter(
                [float(row["index_bytes_per_chunk"]) for row in selected],
                [float(row["evidence_recall"]) for row in selected],
                label=method.replace("_", " "),
                color=colors[method],
                s=42,
                alpha=0.85,
            )
        axis.set_xscale("log", base=2)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Routing-index bytes per chunk (log2)")
        axis.set_ylabel("Evidence recall at four chunks")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"recall_vs_index_bytes.{suffix}", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        for method in colors:
            selected = [
                row for row in summary if row["dataset"] == dataset and row["method"] == method
            ]
            if selected:
                axis.scatter(
                    [float(row["cached_online_ms"]) for row in selected],
                    [float(row["evidence_recall"]) for row in selected],
                    label=method.replace("_", " "),
                    color=colors[method],
                    s=42,
                    alpha=0.85,
                )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Cached online routing latency (ms/example)")
        axis.set_ylabel("Evidence recall at four chunks")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"recall_vs_cached_latency.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        matrix = np.full((3, 3), np.nan)
        for rank_index, rank in enumerate((8, 16, 32)):
            for column, (method, m) in enumerate(
                (("lowrank_kmeans", 4), ("lowrank_kmeans", 8), ("lowrank_all", 32))
            ):
                match = [
                    row
                    for row in summary
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and int(row["rank"]) == rank
                    and int(row["m"]) == m
                ]
                if match:
                    matrix[rank_index, column] = float(match[0]["evidence_recall"])
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(3), ["4 centroids", "8 centroids", "32 tokens"])
        axis.set_yticks(range(3), ["r=8", "r=16", "r=32"])
        axis.set_title(dataset.upper())
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white")
        figure.colorbar(image, ax=axis, fraction=0.046)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"lowrank_quality_heatmap.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    methods = ("lowrank_kmeans", "lowrank_medoid", "lowrank_farthest")
    for axis, dataset in zip(axes, DATASETS):
        for method, marker in zip(methods, ("o", "s", "^")):
            selected = [
                row
                for row in seed_rows
                if row["dataset"] == dataset
                and row["method"] == method
                and int(row["m"]) == 4
            ]
            axis.scatter(
                [int(row["rank"]) + (int(row["seed"]) % 5 - 2) * 0.25 for row in selected],
                [float(row["evidence_recall"]) for row in selected],
                label=method.replace("lowrank_", ""),
                marker=marker,
                alpha=0.75,
            )
        axis.set_xticks((8, 16, 32))
        axis.set_title(dataset.upper())
        axis.set_xlabel("Low-rank width")
        axis.set_ylabel("Per-seed evidence recall (m=4)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"lowrank_seed_stability.{suffix}", dpi=180)
    plt.close(figure)


def _selector_ablation() -> list[dict]:
    path = RESULTS / "selector_ablation/summary.csv"
    if not path.exists():
        return []
    output = []
    for row in _read(path):
        output.append(
            {
                "dataset": row["dataset"],
                "ablation": row["ablation"],
                "rank": int(row["rank"]),
                "m": int(row["m"]),
                "evidence_recall": float(row["evidence_recall"]),
                "teacher_top4_overlap": float(row["teacher_top4_overlap"]),
                "parameter_count": int(row["parameter_count"]),
            }
        )
    for row in _query_summary():
        if row["objective"] == "combined":
            output.append(
                {
                    "dataset": row["dataset"],
                    "ablation": "salience_plus_bilinear",
                    "rank": row["rank"],
                    "m": row["m"],
                    "evidence_recall": row["evidence_recall"],
                    "teacher_top4_overlap": row["teacher_top4_overlap"],
                    "parameter_count": int(row["parameter_count"]),
                }
            )
    _write_csv(OUTPUT / "selector_ablation_summary.csv", output)
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    for axis, (dataset, m) in zip(
        axes.flat,
        (("hotpotqa", 4), ("hotpotqa", 8), ("qasper", 4), ("qasper", 8)),
    ):
        selected = [row for row in output if row["dataset"] == dataset and row["m"] == m]
        labels = []
        values = []
        for ablation, rank in (
            ("salience_only", 0),
            ("bilinear_only", 8),
            ("bilinear_only", 16),
            ("bilinear_only", 32),
            ("salience_plus_bilinear", 8),
            ("salience_plus_bilinear", 16),
            ("salience_plus_bilinear", 32),
        ):
            match = [
                row
                for row in selected
                if row["ablation"] == ablation and row["rank"] == rank
            ]
            if match:
                labels.append(
                    "salience"
                    if ablation == "salience_only"
                    else f"{'bilinear' if ablation == 'bilinear_only' else 'both'} r{rank}"
                )
                values.append(match[0]["evidence_recall"])
        axis.bar(range(len(values)), values, color="#457b9d")
        axis.set_xticks(range(len(values)), labels, rotation=30, ha="right", fontsize=8)
        axis.set_title(f"{dataset.upper()}, m={m}")
        axis.set_ylabel("Evidence recall")
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"selector_ablation.{suffix}", dpi=180)
    plt.close(figure)
    return output


def main() -> dict:
    rows = _read(OUTPUT / "per_example.csv")
    summary = _read(OUTPUT / "summary.csv")
    seed_rows = _read(OUTPUT / "seed_stability.csv")
    paired = _paired_effects(rows)
    seed_statistics = _seed_statistics(rows)
    cost_frontier = _cost_frontier(summary)
    _write_csv(OUTPUT / "paired_effects.csv", paired)
    _write_csv(OUTPUT / "seed_statistics.csv", seed_statistics)
    _write_csv(OUTPUT / "cost_frontier.csv", cost_frontier)
    changed = _changed_selection(rows, summary)
    _write_csv(OUTPUT / "changed_selection.csv", changed)
    gates = _gate_decisions(summary, paired)
    (OUTPUT / "extension_gate_decisions.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    _plots(summary, seed_rows)
    ablations = _selector_ablation()
    best = {
        dataset: max(
            (
                row
                for row in summary
                if row["dataset"] == dataset and row["method"].startswith("lowrank_")
            ),
            key=lambda row: float(row["evidence_recall"]),
        )
        for dataset in DATASETS
    }
    findings = {
        "best_lowrank_by_dataset": {
            dataset: _compact_configuration(row) for dataset, row in best.items()
        },
        "extension_gates": gates,
        "changed_selection_rate": {
            dataset: _mean(
                [
                    float(row["changed_selection"] == "True")
                    if isinstance(row["changed_selection"], str)
                    else float(bool(row["changed_selection"]))
                    for row in changed
                    if row["dataset"] == dataset
                ]
            )
            for dataset in DATASETS
        },
        "selector_ablation_rows": len(ablations),
    }
    (OUTPUT / "findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return findings


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
