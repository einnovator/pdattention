"""Generate publication figures from the five-seed Phase 1 reports.

The plotted runs use the historical cross-attention transport.  Native-KV results
are deliberately excluded because its long-context quality study is still pending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = ROOT / "out" / "reports" / "phase1_followup_v2_5seed"
FINDINGS_PATH = (
    ROOT / "out" / "reports" / "phase1_followup_v2_5seed_findings" / "report.json"
)
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

BLUE = "#245A8D"
GREEN = "#2D7A68"
GOLD = "#A66A18"
GRAY = "#5F6B76"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _aggregate(group: str) -> dict[str, Any]:
    path = REPORT_ROOT / "aggregates" / group / "report.json"
    return _load_json(path)["aggregate"]


def _metric_values(group: str, metric: str) -> np.ndarray:
    return np.asarray(_aggregate(group)[metric]["values"], dtype=float)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D8DEE4", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / name,
        bbox_inches="tight",
        metadata={"Creator": "generate_result_figures.py"},
    )
    plt.close(figure)


def plot_paired_plain_loss() -> None:
    """Show each PRA seed beside its parameter-matched SelfAttention seed."""

    comparisons = (
        (
            "Full PRA",
            _metric_values("plain_pra", "test_loss"),
            _metric_values("plain_sa_match_full", "test_loss"),
        ),
        (
            "Hybrid PRA",
            _metric_values("plain_hybrid", "test_loss"),
            _metric_values("plain_sa_match_hybrid", "test_loss"),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)

    for axis, (name, pra_loss, sa_loss) in zip(axes, comparisons):
        for pra_value, sa_value in zip(pra_loss, sa_loss):
            axis.plot(
                [0, 1],
                [sa_value, pra_value],
                color=GRAY,
                marker="o",
                markersize=4.5,
                linewidth=1.0,
                alpha=0.8,
            )
        axis.scatter([0], [sa_loss.mean()], color=BLUE, marker="D", s=38, zorder=4)
        axis.scatter([1], [pra_loss.mean()], color=GREEN, marker="D", s=38, zorder=4)
        axis.set_xticks([0, 1], ["Matched SA", name])
        axis.set_title(f"{name}: mean change {np.mean(pra_loss - sa_loss):+.4f}")
        _style_axis(axis)

    axes[0].text(
        0.03,
        0.05,
        "each line = one paired seed; diamonds = means",
        transform=axes[0].transAxes,
        fontsize=7,
        color=GRAY,
    )
    axes[0].set_ylabel("WikiText-2 test loss (lower is better)")
    figure.suptitle("Paired ordinary-language results (five seeds)", fontsize=11)
    figure.tight_layout()
    _save(figure, "phase1_plain_paired.pdf")


def plot_adaptation_and_causality() -> None:
    """Contrast adaptation order and content-sensitive memory interventions."""

    findings = _load_json(FINDINGS_PATH)
    rows = {
        (row["architecture"], row["mode"]): row
        for row in findings["training_order"]
        if row["architecture"] in {"pra", "hybrid"}
    }
    orders = ("scratch", "frozen_refpath", "joint")
    order_labels = ("Scratch", "Frozen path", "Joint")
    architectures = (("pra", "Full PRA", GREEN), ("hybrid", "Hybrid PRA", GOLD))

    figure, (adapt_axis, causal_axis) = plt.subplots(1, 2, figsize=(7.1, 3.15))
    x = np.arange(len(orders), dtype=float)
    width = 0.34
    for offset, (key, label, color) in zip((-width / 2, width / 2), architectures):
        values = [rows[(key, order)]["valid_loss"] for order in orders]
        adapt_axis.bar(x + offset, values, width, label=label, color=color)
        for xpos, value in zip(x + offset, values):
            adapt_axis.text(xpos, value + 0.025, f"{value:.2f}", ha="center", fontsize=7)
    adapt_axis.set_xticks(x, order_labels)
    adapt_axis.set_ylim(4.4, 6.65)
    adapt_axis.set_ylabel("Reference-task validation loss")
    adapt_axis.set_title("A. Training order")
    adapt_axis.legend(frameon=False, fontsize=8, loc="upper center")
    _style_axis(adapt_axis)

    interventions = (
        ("disabled_loss", "Disabled"),
        ("shuffled_loss", "Shuffled"),
        ("irrelevant_loss", "Irrelevant"),
        ("oracle_loss", "Oracle"),
    )
    x = np.arange(len(interventions), dtype=float)
    for offset, (key, label, color) in zip((-width / 2, width / 2), architectures):
        valid = rows[(key, "frozen_refpath")]["valid_loss"]
        deltas = [rows[(key, "frozen_refpath")][metric] - valid for metric, _ in interventions]
        causal_axis.bar(x + offset, deltas, width, label=label, color=color)
    causal_axis.axhline(0, color="#30363D", linewidth=0.9)
    causal_axis.set_xticks(x, [label for _, label in interventions], rotation=18)
    causal_axis.set_ylabel(r"Loss change relative to valid refs")
    causal_axis.set_title("B. Frozen-path counterfactuals")
    _style_axis(causal_axis)

    figure.suptitle("Historical cross-attention adaptation results", fontsize=11)
    figure.tight_layout()
    _save(figure, "phase1_adaptation_causality.pdf")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    plot_paired_plain_loss()
    plot_adaptation_and_causality()


if __name__ == "__main__":
    main()
