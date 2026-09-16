"""Plot empirical instruction-epoch opportunity against analytic scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def curve_series(evidence: Mapping[str, Any], *, maximum_n: int = 10) -> dict[str, list[float]]:
    if maximum_n < 1:
        raise ValueError("maximum_n must be positive")
    points = list(evidence.get("points") or ())
    if not points:
        raise ValueError("evidence has no curve points")
    observed_instruction_fraction = float(points[-1]["instruction_floor_fraction"])
    issue_counts = list(range(1, maximum_n + 1))
    interaction_ideal = [1.0 - 1.0 / count for count in issue_counts]
    return {
        "issue_counts": issue_counts,
        "interaction_ideal": interaction_ideal,
        "fixed_instruction_fraction_illustration": [
            (1.0 - observed_instruction_fraction) * value
            for value in interaction_ideal
        ],
        "empirical_issue_counts": [int(row["issue_count"]) for row in points],
        "empirical_whole": [float(row["gross_saving_fraction"]) for row in points],
        "empirical_interaction": [
            float(row["assistant_tool_saving_fraction"]) for row in points
        ],
        "observed_instruction_fraction": [observed_instruction_fraction],
    }


def plot(evidence: Mapping[str, Any], output: Path, *, maximum_n: int = 10) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    series = curve_series(evidence, maximum_n=maximum_n)
    figure, axis = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    axis.plot(
        series["issue_counts"],
        series["interaction_ideal"],
        linestyle="--",
        color="#5b5b5b",
        label="Equal-size interaction-history ideal",
    )
    p = series["observed_instruction_fraction"][0]
    axis.plot(
        series["issue_counts"],
        series["fixed_instruction_fraction_illustration"],
        linestyle=":",
        color="#8c6d31",
        label=f"Illustration if instruction share stays {p:.1%}",
    )
    axis.plot(
        series["empirical_issue_counts"],
        series["empirical_interaction"],
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Observed interaction-history saving",
    )
    axis.plot(
        series["empirical_issue_counts"],
        series["empirical_whole"],
        marker="o",
        linewidth=2,
        color="#d95f02",
        label="Observed whole-request saving",
    )
    axis.set_xlabel("Independent issues accumulated in one session (N)")
    axis.set_ylabel("Cumulative input-token saving")
    axis.set_xticks(series["issue_counts"])
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", linewidth=0.5, alpha=0.35)
    axis.legend(frameon=False, loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-n", type=int, default=10)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    plot(evidence, args.output, maximum_n=args.maximum_n)
    print(json.dumps({"output": str(args.output.resolve()), "maximum_n": args.maximum_n}))


if __name__ == "__main__":
    main()
