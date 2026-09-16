"""Plot fixed-trajectory instruction-epoch opportunity without mixing denominators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    points = evidence["points"]
    observed_n = [int(point["issue_count"]) for point in points]
    observed_interaction = [
        100.0 * float(point["assistant_tool_saving_fraction"]) for point in points
    ]
    observed_whole = [
        100.0 * float(point["gross_saving_fraction"]) for point in points
    ]

    analytic_n = list(range(1, 11))
    final_request = [100.0 * (1.0 - 1.0 / n) for n in analytic_n]
    cumulative_session = [100.0 * (1.0 - 2.0 / (n + 1.0)) for n in analytic_n]

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(
        analytic_n,
        final_request,
        color="0.35",
        linestyle="--",
        label="Equal-size final-request interaction opportunity",
    )
    ax.plot(
        analytic_n,
        cumulative_session,
        color="0.45",
        linestyle=":",
        label="Equal-size cumulative-session interaction opportunity",
    )
    ax.plot(
        observed_n,
        observed_interaction,
        marker="s",
        linewidth=2,
        label="Observed cumulative interaction-history saving",
    )
    ax.plot(
        observed_n,
        observed_whole,
        marker="o",
        linewidth=2,
        label="Observed cumulative whole-request saving",
    )
    ax.set_xlim(0.55, 10.45)
    ax.set_ylim(0, 100)
    ax.set_xticks(analytic_n)
    ax.set_yticks(range(0, 101, 20), labels=[f"{value}%" for value in range(0, 101, 20)])
    ax.set_xlabel("Independent issues accumulated in one session (N)")
    ax.set_ylabel("Input-token saving")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
