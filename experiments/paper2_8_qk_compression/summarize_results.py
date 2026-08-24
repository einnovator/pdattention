"""Create Paper 2.8 comparison tables and publication plots from row artifacts."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return sum(values) / max(len(values), 1)


def _aggregate(rows: list[dict], dimensions: tuple[str, ...], metrics: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in dimensions)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        output.append(
            {
                **dict(zip(dimensions, key)),
                "rows": len(group),
                "identities": len({row.get("example_id", "") for row in group}),
                **{metric: _mean(group, metric) for metric in metrics},
            }
        )
    return output


def _bootstrap(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    )
    return draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def run(output_dir: Path) -> dict:
    synthetic = _read(output_dir / "synthetic_rows.csv")
    natural = _read(output_dir / "natural_rows.csv")
    inherited = _read(output_dir / "paper2_6_inherited_rows.csv")
    gates = json.loads((output_dir / "gate_decisions.json").read_text(encoding="utf-8"))

    synthetic_scaling = _aggregate(
        synthetic,
        ("chunk_tokens", "method", "m"),
        (
            "evidence_recall",
            "teacher_top4_overlap",
            "spearman",
            "rmse",
            "kl",
            "compression_ms",
            "native_dots",
        ),
    )
    _write(output_dir / "synthetic_scaling.csv", synthetic_scaling)

    learned = [row for row in natural if row["method"] == "learned"]
    seed_stability = _aggregate(
        learned,
        ("dataset", "m", "seed"),
        ("evidence_recall", "chain_completion", "teacher_top4_overlap", "compression_ms"),
    )
    _write(output_dir / "learned_seed_stability.csv", seed_stability)

    comparison = []
    grouped_inherited = _aggregate(
        inherited,
        ("dataset", "condition"),
        ("evidence_recall", "precision", "mrr", "chain_completion", "requested_chunks"),
    )
    for row in grouped_inherited:
        comparison.append(
            {
                **row,
                "family": "Paper 2.6 inherited",
                "deployable": row["condition"] != "O1_oracle",
                "candidate_chunk_tokens": 32,
                "comparison_scope": "same frozen identities and four-chunk budget",
            }
        )
    qk_specs = {
        ("mean", "1"): "QK mean",
        ("full_k", "0"): "QK full-K oracle",
        ("farthest", "2"): "QK farthest m=2",
        ("greedy_oracle", "4"): "QK greedy oracle m=4",
        ("greedy_oracle", "8"): "QK greedy oracle m=8",
        ("learned", "4"): "QK learned m=4",
        ("learned", "8"): "QK learned m=8",
    }
    for dataset in ("hotpotqa", "qasper"):
        for (method, m), condition in qk_specs.items():
            group = [
                row
                for row in natural
                if row["dataset"] == dataset and row["method"] == method and row["m"] == m
            ]
            comparison.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "rows": len(group),
                    "identities": len({row["example_id"] for row in group}),
                    "evidence_recall": _mean(group, "evidence_recall"),
                    "precision": _mean(group, "evidence_precision"),
                    "mrr": _mean(group, "mrr"),
                    "chain_completion": _mean(group, "chain_completion"),
                    "requested_chunks": 4,
                    "family": "Paper 2.8",
                    "deployable": method not in {"full_k", "greedy_oracle"},
                    "candidate_chunk_tokens": 32,
                    "comparison_scope": "same frozen identities and four-chunk budget",
                }
            )
    _write(output_dir / "paper2_6_matched_comparison.csv", comparison)

    paired_paper26 = []
    for dataset in ("hotpotqa", "qasper"):
        for method, m in (("greedy_oracle", "4"), ("learned", "8")):
            candidate = defaultdict(list)
            for row in natural:
                if row["dataset"] == dataset and row["method"] == method and row["m"] == m:
                    candidate[row["example_id"]].append(float(row["evidence_recall"]))
            for baseline_name in ("B0_gist", "B1_bm25", "B2_exact", "H5_iterative_hybrid"):
                baseline = {
                    row["example_id"]: float(row["evidence_recall"])
                    for row in inherited
                    if row["dataset"] == dataset and row["condition"] == baseline_name
                }
                differences = [
                    sum(values) / len(values) - baseline[identity]
                    for identity, values in candidate.items()
                    if identity in baseline
                ]
                low, high = _bootstrap(differences, 20260824 + int(m))
                paired_paper26.append(
                    {
                        "dataset": dataset,
                        "method": f"{method}_m{m}",
                        "baseline": baseline_name,
                        "pairs": len(differences),
                        "mean_recall_delta": sum(differences) / len(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": sum(value > 0 for value in differences),
                        "losses": sum(value < 0 for value in differences),
                        "ties": sum(value == 0 for value in differences),
                    }
                )
    _write(output_dir / "paper2_6_paired_effects.csv", paired_paper26)

    paper25_path = (
        ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure/"
        "gate3_native_qk_aggregate.csv"
    )
    paper25 = [
        row
        for row in _read(paper25_path)
        if float(row["fraction"]) == 0.2
        and row["condition"]
        in {"one_shot_parent", "local_gist_closure", "native_qk_top4_topk_p10"}
    ]
    historical = [
        {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "chain_completion": row["chain_completion"],
            "evidence_coverage": row["evidence_coverage"],
            "any_evidence": row["any_evidence"],
            "native_qk_dot_products": row["native_qk_dot_products"],
            "candidate_unit_tokens": 256,
            "final_budget": "20% of parents",
            "comparison_scope": "historical context only; not numerically matched to Paper 2.8",
        }
        for row in paper25
    ]
    _write(output_dir / "paper2_5_historical_controls.csv", historical)

    colors = {
        "mean": "#6c757d",
        "farthest": "#457b9d",
        "greedy_oracle": "#2a9d8f",
        "learned": "#6a4c93",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("teacher_top4_overlap", "evidence_recall"),
        ("Teacher top-4 preservation", "Controlled evidence recall"),
    ):
        for method in colors:
            values = [row for row in synthetic_scaling if row["method"] == method and row["m"] in {"1", "4", "8"}]
            if method == "mean":
                values = [row for row in values if row["m"] == "1"]
            for m in sorted({row["m"] for row in values}, key=int):
                subset = sorted(
                    (row for row in values if row["m"] == m),
                    key=lambda row: int(row["chunk_tokens"]),
                )
                label = "mean" if method == "mean" else f"{method.replace('_', ' ')} m={m}"
                axis.plot(
                    [int(row["chunk_tokens"]) for row in subset],
                    [float(row[metric]) for row in subset],
                    marker="o",
                    color=colors[method],
                    alpha=0.65 if m != "8" else 1.0,
                    label=label,
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks((32, 64, 128, 256), ("32", "64", "128", "256"))
        axis.set_xlabel("Chunk tokens")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"synthetic_qk_scaling.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    labels = ("B0_gist", "B1_bm25", "B2_exact", "H5_iterative_hybrid", "QK mean", "QK full-K oracle", "QK greedy oracle m=4", "QK learned m=8")
    x = list(range(len(labels)))
    width = 0.38
    for offset, dataset, color in ((-width / 2, "hotpotqa", "#457b9d"), (width / 2, "qasper", "#d1495b")):
        values = {
            row["condition"]: float(row["evidence_recall"])
            for row in comparison
            if row["dataset"] == dataset
        }
        axis.bar([value + offset for value in x], [values[label] for label in labels], width, label=dataset)
    axis.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=30, ha="right")
    axis.set_ylabel("Evidence recall at four chunks")
    axis.set_title("Matched frozen-cohort comparison with Paper 2.6")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"paper2_6_matched_comparison.{suffix}", dpi=180)
    plt.close(figure)

    gate_rows = []
    for name, gate in gates.items():
        gate_rows.append(
            {
                "gate": name.split("_")[0],
                "description": name.split("_", 1)[1],
                "run": gate.get("run", True),
                "passed": gate.get("passed", False),
                "reason": gate.get("reason", ""),
            }
        )
    _write(output_dir / "gate_status.csv", gate_rows)
    return {
        "synthetic_scaling_rows": len(synthetic_scaling),
        "matched_comparison_rows": len(comparison),
        "paper2_6_paired_rows": len(paired_paper26),
        "paper2_5_historical_rows": len(historical),
        "learned_seed_rows": len(seed_stability),
        "gates": gates,
    }


if __name__ == "__main__":
    output = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
    print(json.dumps(run(output), indent=2, sort_keys=True))
