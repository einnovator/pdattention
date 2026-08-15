"""Aggregate the toy-motivated Qwen materialization confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER3_ROOT = ROOT / "docs/papers/shared/results/paper3_kv_materialization"
DEFAULT_ROOT = PAPER3_ROOT / "pretrained_confirmation"
TOY_ROOT = PAPER3_ROOT / "toy_materialization"


def _number(value):
    if value in {"", None, "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: _number(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else float("nan")


def _paired_ci(values: list[float], *, seed: int = 303, samples: int = 10_000):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    estimates.sort()
    return (
        estimates[int(.025 * (samples - 1))],
        estimates[int(.975 * (samples - 1))],
    )


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["dataset"], row["condition"])].append(row)
    metrics = (
        "materialized_unique_tokens",
        "kv_reduction_vs_whole",
        "evidence_coverage",
        "evidence_density",
        "gold_mean_token_logprob",
        "gold_mean_logprob_delta_vs_none",
        "gold_mean_logprob_delta_vs_whole",
        "memory_attention_mass",
        "evidence_attention_mass",
        "non_evidence_attention_mass",
        "exact_match",
        "token_f1",
        "normalized_answer_accuracy",
        "ttft_seconds",
    )
    frontier = [
        {
            "phase": key[0],
            "dataset": key[1],
            "condition": key[2],
            "examples": len(group),
            **{metric: _mean(group, metric) for metric in metrics},
        }
        for key, group in sorted(grouped.items())
    ]
    paired = []
    heldout = [row for row in rows if row["phase"] == "heldout"]
    by_example = defaultdict(dict)
    for row in heldout:
        by_example[(row["dataset"], row["example_id"])][row["condition"]] = row
    conditions = sorted({row["condition"] for row in heldout})
    for dataset in sorted({row["dataset"] for row in heldout}):
        for condition in conditions:
            if condition == "M1_whole_parent":
                continue
            pairs = [
                values
                for (row_dataset, _example), values in by_example.items()
                if row_dataset == dataset
                and condition in values
                and "M1_whole_parent" in values
            ]
            if not pairs:
                continue
            record = {
                "dataset": dataset,
                "condition": condition,
                "reference": "M1_whole_parent",
                "examples": len(pairs),
            }
            for metric in (
                "gold_mean_token_logprob",
                "token_f1",
                "normalized_answer_accuracy",
                "materialized_unique_tokens",
                "evidence_coverage",
            ):
                deltas = [
                    float(pair[condition][metric])
                    - float(pair["M1_whole_parent"][metric])
                    for pair in pairs
                ]
                low, high = _paired_ci(deltas)
                record[f"{metric}_delta"] = statistics.fmean(deltas)
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            paired.append(record)
    return frontier, paired


def _plots(frontier: list[dict], toy_frontier: list[dict], output: Path) -> None:
    heldout = [row for row in frontier if row["phase"] == "heldout"]
    colors = {"musique": "#4c78a8", "2wikimultihopqa": "#e45756"}
    short_names = {
        "M_none": "none",
        "M0_native_gist": "gist",
        "M1_whole_parent": "parent",
        "M2_evidence_only": "exact",
        "M3_radius_2": "r2",
    }
    offsets = {
        ("musique", "M_none"): (4, 7),
        ("musique", "M0_native_gist"): (4, -15),
        ("musique", "M1_whole_parent"): (-25, 14),
        ("musique", "M2_evidence_only"): (4, -16),
        ("musique", "M3_radius_2"): (4, 8),
        ("2wikimultihopqa", "M_none"): (4, 8),
        ("2wikimultihopqa", "M0_native_gist"): (4, -15),
        ("2wikimultihopqa", "M1_whole_parent"): (-42, 7),
        ("2wikimultihopqa", "M2_evidence_only"): (4, 8),
        ("2wikimultihopqa", "M3_radius_2"): (4, -16),
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for axis, (dataset, color) in zip(axes, colors.items()):
        rows = [row for row in heldout if row["dataset"] == dataset]
        axis.scatter(
            [row["materialized_unique_tokens"] for row in rows],
            [row["gold_mean_logprob_delta_vs_none"] for row in rows],
            color=color,
            s=34,
        )
        for row in rows:
            axis.annotate(
                short_names.get(str(row["condition"]), str(row["condition"])),
                (row["materialized_unique_tokens"], row["gold_mean_logprob_delta_vs_none"]),
                fontsize=8,
                xytext=offsets.get((dataset, str(row["condition"])), (4, 4)),
                textcoords="offset points",
            )
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title("MuSiQue" if dataset == "musique" else "2WikiMultiHopQA")
        axis.set_xlabel("materialized native K/V tokens")
        axis.margins(x=.08, y=.15)
        axis.grid(alpha=.25)
    axes[0].set_ylabel("gold log-probability change vs no memory")
    figure.tight_layout(); figure.savefig(output / "pretrained_kv_quality_frontier.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    for dataset, color in colors.items():
        rows = [row for row in heldout if row["dataset"] == dataset]
        axis.scatter(
            [row["evidence_coverage"] for row in rows],
            [row["evidence_density"] for row in rows],
            label=dataset,
            color=color,
            s=42,
        )
    axis.set(xlabel="evidence coverage", ylabel="evidence density")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    figure.tight_layout(); figure.savefig(output / "density_vs_coverage.png", dpi=180); plt.close(figure)

    toy_heldout = [
        row for row in toy_frontier if row["partition"] == "heldout"
    ]
    toy_by_policy = defaultdict(list)
    for row in toy_heldout:
        toy_by_policy[row["policy"]].append(row)
    toy_exact_parent = (
        _mean(toy_by_policy["T1_radius_0"], "correct_margin")
        - _mean(toy_by_policy["T7_whole_parent"], "correct_margin")
    )
    toy_radius_exact = (
        _mean(toy_by_policy["T2_radius_2"], "correct_margin")
        - _mean(toy_by_policy["T1_radius_0"], "correct_margin")
    )
    labels = ["Toy exact-parent", "Toy r2-exact"]
    values = [toy_exact_parent, toy_radius_exact]
    for dataset in ("musique", "2wikimultihopqa"):
        rows = [row for row in heldout if row["dataset"] == dataset]
        by_condition = {row["condition"]: row for row in rows}
        labels.extend([f"{dataset} exact-parent", f"{dataset} r2-exact"])
        values.extend(
            [
                by_condition["M2_evidence_only"]["gold_mean_token_logprob"]
                - by_condition["M1_whole_parent"]["gold_mean_token_logprob"],
                by_condition["M3_radius_2"]["gold_mean_token_logprob"]
                - by_condition["M2_evidence_only"]["gold_mean_token_logprob"],
            ]
        )
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.bar(range(len(values)), values, color=["#72b7b2", "#54a24b", "#4c78a8", "#9ecae9", "#e45756", "#f2a07b"])
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("quality difference (native metric units)")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(output / "toy_prediction_vs_pretrained.png", dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--toy-root", type=Path, default=TOY_ROOT)
    parser.add_argument("--paper3-root", type=Path, default=PAPER3_ROOT)
    args = parser.parse_args()
    rows = []
    for phase in ("validation", "heldout"):
        rows.extend(_read(args.root / f"pretrained_confirmation_{phase}_rows.csv"))
    for row in rows:
        if not row.get("geometry_source"):
            row["geometry_source"] = "frozen_paper2_5_discovery_manifest"
    frontier, paired = summarize(rows)
    _write(args.root / "pretrained_confirmation_rows.csv", rows)
    _write(args.root / "pretrained_materialization_frontier.csv", frontier)
    _write(args.root / "pretrained_confirmation_paired.csv", paired)
    toy_frontier = _read(args.toy_root / "toy_materialization_frontier.csv")
    _plots(frontier, toy_frontier, args.root)
    findings_path = args.paper3_root / "paper3_findings.json"
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    findings["toy_first_iteration"] = json.loads(
        (args.toy_root / "toy_materialization_summary.json").read_text(encoding="utf-8")
    )
    cohort_sizes = {
        phase: {
            dataset: len(
                {
                    row["example_id"]
                    for row in rows
                    if row["phase"] == phase and row["dataset"] == dataset
                }
            )
            for dataset in sorted({row["dataset"] for row in rows})
        }
        for phase in ("validation", "heldout")
    }
    findings["pretrained_confirmation"] = {
        "cohort_sizes": cohort_sizes,
        "frontier": frontier,
        "paired_effects": paired,
        "claim_boundary": (
            "MuSiQue is a dataset-specific materialization mismatch; controlled oracle "
            "interventions reject a universal frozen-consumer limitation"
        ),
    }
    findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "paired": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
