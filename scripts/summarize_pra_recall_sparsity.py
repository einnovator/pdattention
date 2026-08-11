"""Re-express Paper 1 routing coverage against the selected candidate fraction.

The historical scale artifact stores aggregate coverage at fixed k values rather than
complete per-example rankings. This script therefore reports only measured k / N points;
it never interpolates missing requested fractions or approximates physical K/V fractions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results"
OUTPUT = RESULTS / "recall_sparsity" / "paper1"
FIGURES = ROOT / "docs" / "papers" / "shared" / "figures"
KS = (1, 2, 4, 8, 16, 32)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scale_rows() -> list[dict[str, object]]:
    artifact = json.loads((RESULTS / "pra_scale_sensitivity.json").read_text())
    rows: list[dict[str, object]] = []
    for aggregate in artifact["aggregate"]:
        # The k=8 row is canonical; coverage-at-k fields describe its complete stored
        # ranking diagnostics and are duplicated by the other tiny-tier selections.
        if aggregate["top_k_references"] != 8:
            continue
        candidates = int(aggregate["source_unit_count"])
        for k in KS:
            rows.append(
                {
                    "dataset": aggregate["dataset"],
                    "model_tier": aggregate["model_tier"],
                    "split_count": aggregate["split_count"],
                    "candidate_references": candidates,
                    "k": k,
                    "selected_reference_fraction": k / candidates,
                    "target_coverage": aggregate[f"fraction_targets_covered_at_{k}_mean"],
                    "all_targets_hit": aggregate[f"all_targets_hit_at_{k}_mean"],
                    "artifact": "pra_scale_sensitivity.json",
                    "encoding": "native historical slice",
                }
            )
    return rows


def _split32_rows() -> list[dict[str, object]]:
    artifact = json.loads(
        (RESULTS / "pra_parameter_sensitivity_topk.json").read_text()
    )
    rows: list[dict[str, object]] = []
    for aggregate in artifact["aggregate"]:
        if aggregate["split_count"] != 32 or aggregate["top_k_references"] != 8:
            continue
        candidates = 31
        for k in KS:
            rows.append(
                {
                    "dataset": aggregate["dataset"],
                    "model_tier": "independent-encoding screen",
                    "split_count": 32,
                    "candidate_references": candidates,
                    "k": k,
                    "selected_reference_fraction": min(k, candidates) / candidates,
                    "target_coverage": aggregate[f"fraction_targets_covered_at_{k}_mean"],
                    "all_targets_hit": aggregate[f"all_targets_hit_at_{k}_mean"],
                    "artifact": "pra_parameter_sensitivity_topk.json",
                    "encoding": "independent",
                }
            )
    return rows


def _first_fraction(
    rows: list[dict[str, object]], threshold: float
) -> float | None:
    for row in sorted(rows, key=lambda item: float(item["selected_reference_fraction"])):
        if float(row["target_coverage"]) >= threshold:
            return float(row["selected_reference_fraction"])
    return None


def _inverse_rows(curves: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups = {
        (str(row["dataset"]), str(row["model_tier"]), int(row["split_count"]))
        for row in curves
    }
    for dataset, tier, split_count in sorted(groups):
        group = [
            row
            for row in curves
            if row["dataset"] == dataset
            and row["model_tier"] == tier
            and row["split_count"] == split_count
        ]
        output.append(
            {
                "dataset": dataset,
                "model_tier": tier,
                "split_count": split_count,
                "f70_first_measured": _first_fraction(group, 0.70),
                "f80_first_measured": _first_fraction(group, 0.80),
                "f90_first_measured": _first_fraction(group, 0.90),
                "f95_first_measured": _first_fraction(group, 0.95),
                "auc_0_30": "unavailable_without_interpolation",
            }
        )
    return output


def _comparison_rows(curves: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ("hotpotqa", "qasper"):
        for split_count, fraction_k in ((64, 8), (128, 16), (256, 32)):
            group = [
                row
                for row in curves
                if row["dataset"] == dataset
                and row["model_tier"] == "small"
                and row["split_count"] == split_count
            ]
            fixed = next(row for row in group if row["k"] == 8)
            fraction = next(row for row in group if row["k"] == fraction_k)
            rows.append(
                {
                    "dataset": dataset,
                    "split_count": split_count,
                    "candidate_references": fixed["candidate_references"],
                    "fixed_k": 8,
                    "fixed_k_fraction": fixed["selected_reference_fraction"],
                    "fixed_k_coverage": fixed["target_coverage"],
                    "matched_fraction_k": fraction_k,
                    "matched_fraction": fraction["selected_reference_fraction"],
                    "matched_fraction_coverage": fraction["target_coverage"],
                }
            )
    return rows


def _plot_curves(curves: list[dict[str, object]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    colors = {64: "#2f6f9f", 128: "#d56a32", 256: "#3f8f5f"}
    for axis, dataset in zip(axes, ("hotpotqa", "qasper"), strict=True):
        for split_count in (64, 128, 256):
            group = sorted(
                (
                    row
                    for row in curves
                    if row["dataset"] == dataset
                    and row["model_tier"] == "small"
                    and row["split_count"] == split_count
                    and float(row["selected_reference_fraction"]) <= 0.30
                ),
                key=lambda row: float(row["selected_reference_fraction"]),
            )
            axis.plot(
                [100 * float(row["selected_reference_fraction"]) for row in group],
                [float(row["target_coverage"]) for row in group],
                marker="o",
                color=colors[split_count],
                label=f"{split_count} units",
            )
        axis.axvline(12.5, color="#777777", linestyle="--", linewidth=1)
        axis.set_title("HotpotQA-derived" if dataset == "hotpotqa" else "QASPER-derived")
        axis.set_xlabel("Selected references (%)")
        axis.set_xlim(0, 30)
        axis.set_ylim(0.45, 0.92)
    axes[0].set_ylabel("Annotated target coverage")
    axes[1].legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"pra_recall_sparsity_small.{suffix}", dpi=220)
    plt.close(fig)


def _plot_scaling(comparison: list[dict[str, object]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper"), strict=True):
        group = [row for row in comparison if row["dataset"] == dataset]
        x = [int(row["split_count"]) for row in group]
        axis.plot(
            x,
            [float(row["fixed_k_coverage"]) for row in group],
            marker="o",
            color="#b44b4b",
            label="fixed k=8",
        )
        axis.plot(
            x,
            [float(row["matched_fraction_coverage"]) for row in group],
            marker="s",
            color="#2f6f9f",
            label="about 12.5% selected",
        )
        axis.set_title("HotpotQA-derived" if dataset == "hotpotqa" else "QASPER-derived")
        axis.set_xlabel("Nominal source units")
        axis.set_xticks(x)
        axis.set_ylim(0.62, 0.84)
    axes[0].set_ylabel("Annotated target coverage")
    axes[1].legend(loc="lower left", frameon=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"pra_fixed_k_vs_fraction_scaling.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves = _scale_rows()
    split32 = _split32_rows()
    comparison = _comparison_rows(curves)
    _write_csv(OUTPUT / "native_slice_measured_curves.csv", curves)
    _write_csv(OUTPUT / "split32_independent_screen.csv", split32)
    _write_csv(OUTPUT / "fixed_k_vs_fraction.csv", comparison)
    _write_csv(OUTPUT / "inverse_metrics.csv", _inverse_rows(curves + split32))
    _plot_curves(curves)
    _plot_scaling(comparison)
    print(json.dumps({"scale_points": len(curves), "split32_points": len(split32)}))


if __name__ == "__main__":
    main()
