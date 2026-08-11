"""Build Paper 1.5 recall-sparsity tables from labeled routed QA rankings.

Oracle, all-memory, and fixed-selection positional conditions are intentionally excluded.
Each routed row contributes its per-layer reference ranking. Within a model tier the layer
count is fixed, so this reproduces the paper's mean-across-layers routing convention.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.recall_sparsity import recall_sparsity_curve  # noqa: E402


VALIDATION = (
    ROOT / "docs" / "papers" / "shared" / "results" / "paper1_5_rope" / "validation"
)
OUTPUT = (
    ROOT / "docs" / "papers" / "shared" / "results" / "recall_sparsity" / "paper1_5"
)
FIGURES = ROOT / "docs" / "papers" / "shared" / "figures"


def _reference_length(candidate: dict) -> int:
    return sum(
        int(chunk["token_end"]) - int(chunk["token_start"])
        for chunk in candidate["chunks"]
    )


def _groups() -> dict[tuple[str, str, str, str], dict[str, list]]:
    groups: dict[tuple[str, str, str, str], dict[str, list]] = defaultdict(
        lambda: {"rankings": [], "evidence": [], "lengths": []}
    )
    for dataset in ("hotpotqa", "qasper"):
        artifact = json.loads(
            (VALIDATION / f"{dataset}_position_validation.json").read_text()
        )
        for row in artifact["rows"]:
            if row["condition"] != "native_routed":
                continue
            target = set(row["oracle_selected_reference_uris"])
            key = (
                dataset,
                row["model_tier"],
                row["position_mode"],
                row["stage"],
            )
            for layer_id in sorted(row["candidate_rankings_by_layer"], key=int):
                candidates = row["candidate_rankings_by_layer"][layer_id]
                ranking = [candidate["reference_uri"] for candidate in candidates]
                groups[key]["rankings"].append(ranking)
                groups[key]["evidence"].append(target)
                groups[key]["lengths"].append(
                    [_reference_length(candidate) for candidate in candidates]
                )
    return groups


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (dataset, tier, position, stage), values in sorted(_groups().items()):
        result = recall_sparsity_curve(
            values["rankings"],
            values["evidence"],
            candidate_token_lengths=values["lengths"],
            require_complete_endpoint=True,
        )
        for point in result["curve"]:
            curve_rows.append(
                {
                    "dataset": dataset,
                    "model_tier": tier,
                    "position_mode": position,
                    "stage": stage,
                    "layer_rankings": result["examples"],
                    **point,
                }
            )
        by_fraction = {float(row["fraction"]): row for row in result["curve"]}
        summary_rows.append(
            {
                "dataset": dataset,
                "model_tier": tier,
                "position_mode": position,
                "stage": stage,
                "layer_rankings": result["examples"],
                "r_at_5pct": by_fraction[0.05]["recall"],
                "r_at_10pct": by_fraction[0.10]["recall"],
                "r_at_20pct": by_fraction[0.20]["recall"],
                "r_at_30pct": by_fraction[0.30]["recall"],
                "f80": result["inverse"]["f80"],
                "f90": result["inverse"]["f90"],
                "auc_0_30": result["auc_0_30"],
                "recall_at_1": result["fixed_k"]["1"]["recall"],
                "recall_at_3": result["fixed_k"]["3"]["recall"],
                "kv_fraction_exact": result["kv_fraction_exact"],
                "endpoint_complete": result["endpoint_complete"],
            }
        )
    return curve_rows, summary_rows


def _plot_requested_fraction(curves: list[dict[str, object]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    styles = {
        "absolute": ("#2f6f9f", "o", "Learned absolute"),
        "sinusoidal": ("#3f8f5f", "s", "Sinusoidal"),
        "rope": ("#b44b4b", "^", "RoPE"),
    }
    for axis, dataset in zip(axes, ("hotpotqa", "qasper"), strict=True):
        for position, (color, marker, label) in styles.items():
            raw = [
                row
                for row in curves
                if row["dataset"] == dataset
                and row["model_tier"] == "small"
                and row["position_mode"] == position
                and row["stage"] == "offset_overlap"
            ]
            points = sorted(
                {
                    (
                        round(float(row["selected_chunk_fraction"]), 12),
                        float(row["recall"]),
                    )
                    for row in raw
                }
            )
            axis.plot(
                [100 * point[0] for point in points],
                [point[1] for point in points],
                marker=marker,
                color=color,
                label=label,
            )
        axis.set_title("HotpotQA-derived" if dataset == "hotpotqa" else "QASPER-derived")
        axis.set_xlabel("Realized selected parent references (%)")
        axis.set_xlim(20, 105)
        axis.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("Programmatic target-reference recall")
    axes[1].legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"rope_qa_recall_sparsity.{suffix}", dpi=220)
    plt.close(fig)


def _plot_kv_fraction(curves: list[dict[str, object]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    styles = {
        "absolute": ("#2f6f9f", "o", "Learned absolute"),
        "sinusoidal": ("#3f8f5f", "s", "Sinusoidal"),
        "rope": ("#b44b4b", "^", "RoPE"),
    }
    for axis, dataset in zip(axes, ("hotpotqa", "qasper"), strict=True):
        for position, (color, marker, label) in styles.items():
            raw = [
                row
                for row in curves
                if row["dataset"] == dataset
                and row["model_tier"] == "small"
                and row["position_mode"] == position
                and row["stage"] == "offset_overlap"
            ]
            points = sorted(
                {
                    (
                        round(float(row["selected_kv_token_fraction"]), 12),
                        float(row["recall"]),
                    )
                    for row in raw
                }
            )
            axis.plot(
                [100 * point[0] for point in points],
                [point[1] for point in points],
                marker=marker,
                color=color,
                label=label,
            )
        axis.set_title("HotpotQA-derived" if dataset == "hotpotqa" else "QASPER-derived")
        axis.set_xlabel("Measured materialized native-K/V tokens (%)")
        axis.set_xlim(20, 105)
        axis.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("Programmatic target-reference recall")
    axes[1].legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"rope_qa_recall_kv_fraction.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves, summaries = _summaries()
    _write_csv(OUTPUT / "routing_curves.csv", curves)
    _write_csv(OUTPUT / "routing_summary.csv", summaries)
    _plot_requested_fraction(curves)
    _plot_kv_fraction(curves)
    print(
        json.dumps(
            {
                "conditions": len(summaries),
                "endpoint_failures": sum(
                    not bool(row["endpoint_complete"]) for row in summaries
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
