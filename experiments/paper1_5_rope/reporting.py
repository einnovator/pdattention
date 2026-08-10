"""Aggregate and plot Paper 1.5 machine-readable result rows."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def aggregate_fragmentation(rows: list[dict]) -> list[dict]:
    """Aggregate matched seed rows by tier, position mode, and split count."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_tier"], row["position_mode"], row["split_count"])].append(row)
    metrics = (
        "sa_full_loss",
        "sa_tail_loss",
        "native_all_loss",
        "native_oracle_loss",
        "native_routed_loss",
        "transport_gap",
        "rcb_all",
        "rcb_oracle",
        "rcb_routed",
        "active_fraction_all",
    )
    output = []
    for (tier, mode, splits), values in sorted(grouped.items()):
        row = {
            "model_tier": tier,
            "position_mode": mode,
            "split_count": splits,
            "seed_count": len(values),
        }
        for metric in metrics:
            observed = [
                float(value[metric])
                for value in values
                if value.get(metric) is not None and math.isfinite(float(value[metric]))
            ]
            if observed:
                row[f"{metric}_mean"] = statistics.fmean(observed)
                row[f"{metric}_std"] = statistics.pstdev(observed)
        output.append(row)
    return output


def plot_translation(rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 3.7))
    for width in sorted({row["head_dim"] for row in rows}):
        values = [row for row in rows if row["head_dim"] == width]
        axis.plot(
            [row["translation"] for row in values],
            [row["attention_logit_rmse"] for row in values],
            marker="o",
            label=f"head dim {width}",
        )
    axis.set_xscale("symlog", linthresh=1)
    axis.set_yscale("log")
    axis.set_xlabel("Common position translation")
    axis.set_ylabel("Attention-logit RMSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_fragmentation(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    colors = {"absolute": "#245A8D", "rope": "#A34832"}
    markers = {"tiny": "o", "small": "s"}
    for tier in sorted({row["model_tier"] for row in rows}):
        for mode in sorted({row["position_mode"] for row in rows}):
            values = [
                row for row in rows
                if row["model_tier"] == tier and row["position_mode"] == mode
            ]
            if not values:
                continue
            label = f"{tier} {mode}"
            x = [row["split_count"] for row in values]
            axes[0].plot(
                x,
                [row["native_all_loss_mean"] for row in values],
                color=colors[mode],
                marker=markers[tier],
                linestyle="-" if tier == "small" else "--",
                label=label,
            )
            axes[1].plot(
                x,
                [row.get("rcb_all_mean", float("nan")) for row in values],
                color=colors[mode],
                marker=markers[tier],
                linestyle="-" if tier == "small" else "--",
                label=label,
            )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Source split count")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Native-all answer loss")
    axes[1].set_ylabel("Recovered context benefit")
    axes[1].axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
