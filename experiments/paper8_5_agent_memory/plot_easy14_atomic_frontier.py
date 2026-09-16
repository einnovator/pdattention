"""Plot the Easy-14 persistent-session quality--saving frontier by prefix.

The input is the audited ``evidence.json`` emitted by
``export_multi_issue_evidence``.  Prefix points are descriptive observations
from one ordered sequence, not independent samples or confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def prefix_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in evidence.get("runs") or ():
        issues = list(run.get("paired_issues") or ())
        if not issues:
            continue
        candidate_tokens = 0
        control_tokens = 0
        candidate_calls = 0
        control_calls = 0
        candidate_resolved = 0
        control_resolved = 0
        lost_successes = 0
        for index, issue in enumerate(issues, 1):
            candidate_tokens += int(issue["candidate_materialized_tokens"])
            control_tokens += int(issue["control_full_tokens"])
            candidate_calls += int(issue["candidate_calls"])
            control_calls += int(issue["control_calls"])
            candidate_resolved += bool(issue["candidate_official_resolved"])
            control_resolved += bool(issue["control_official_resolved"])
            lost_successes += bool(issue["lost_persistent_full_success"])
            raw_saving = (
                1 - candidate_tokens / control_tokens if control_tokens else 0.0
            )
            rows.append({
                "sequence_id": run["sequence_id"],
                "repeat": int(run["repeat"]),
                "strategy_id": run["strategy_id"],
                "strategy_config_id": run["strategy_config_id"],
                "prefix_issue_count": index,
                "candidate_resolved_count": int(candidate_resolved),
                "control_resolved_count": int(control_resolved),
                "candidate_resolution_fraction": candidate_resolved / index,
                "control_resolution_fraction": control_resolved / index,
                "resolution_delta": candidate_resolved - control_resolved,
                "candidate_materialized_tokens": candidate_tokens,
                "control_full_tokens": control_tokens,
                "raw_saving_vs_persistent_full": raw_saving,
                "failure_aware_saving_vs_persistent_full": (
                    0.0 if lost_successes else raw_saving
                ),
                "lost_persistent_full_successes": int(lost_successes),
                "candidate_calls": candidate_calls,
                "control_calls": control_calls,
                "calls_delta_vs_persistent_full": candidate_calls - control_calls,
            })
    return rows


def _label(row: Mapping[str, Any]) -> str:
    config = str(row["strategy_config_id"])
    for key, label in (
        ("quality_", "Atomic E3 / Quality"),
        ("balanced_", "Atomic E2 / Balanced"),
        ("economy_", "Atomic E1 / Economy"),
    ):
        if config.startswith(key):
            return label
    return f"{row['strategy_id']} / {config}"


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "sequence_id", "repeat", "strategy_id", "strategy_config_id",
        "prefix_issue_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("no complete paired candidate runs in evidence")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["strategy_id"]), str(row["strategy_config_id"])), []
        ).append(row)

    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    for points in grouped.values():
        points = sorted(points, key=lambda row: int(row["prefix_issue_count"]))
        x = [int(row["prefix_issue_count"]) for row in points]
        label = _label(points[0])
        axes[0].plot(
            x,
            [100 * float(row["candidate_resolution_fraction"]) for row in points],
            marker="o", label=label,
        )
        axes[1].plot(
            x,
            [100 * float(row["raw_saving_vs_persistent_full"]) for row in points],
            marker="o", label=label,
        )
        axes[1].plot(
            x,
            [
                100 * float(row["failure_aware_saving_vs_persistent_full"])
                for row in points
            ],
            linestyle="--", alpha=0.65,
        )
        axes[2].plot(
            x,
            [int(row["calls_delta_vs_persistent_full"]) for row in points],
            marker="o", label=label,
        )
    control = sorted(
        next(iter(grouped.values())),
        key=lambda row: int(row["prefix_issue_count"]),
    )
    axes[0].plot(
        [int(row["prefix_issue_count"]) for row in control],
        [100 * float(row["control_resolution_fraction"]) for row in control],
        color="black", linestyle=":", label="Persistent Full",
    )
    axes[0].set_ylabel("Official resolution (%)")
    axes[1].set_ylabel("Paired input saving (%)")
    axes[2].set_ylabel("Cumulative call delta")
    axes[2].set_xlabel("Issues completed in one continuous session")
    axes[1].axhspan(30, 50, color="#d8ecd2", alpha=0.45)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle=":")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    figure.suptitle("Easy-14 atomic instruction-epoch frontier (descriptive)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    rows = prefix_rows(evidence)
    write_csv(rows, args.output / "prefix_curves.csv")
    plot(rows, args.output / "easy14_atomic_prefix_frontier.pdf")
    print(json.dumps({"prefix_rows": len(rows), "strategies": len({
        (row["strategy_id"], row["strategy_config_id"]) for row in rows
    })}))


if __name__ == "__main__":
    main()
