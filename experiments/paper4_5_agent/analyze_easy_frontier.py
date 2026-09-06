"""Summarize paired Easy-50 outcomes and render the preregistered plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sum_present(rows: Iterable[dict[str, Any]], field: str) -> int | float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return sum(values) if values else None


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(root: Path) -> dict[str, Any]:
    """Build condition and paired summaries from normalized per-task rows."""

    condition_rows = {
        directory.name: _rows(directory / "results.jsonl")
        for directory in sorted(root.iterdir())
        if directory.is_dir() and (directory / "results.jsonl").is_file()
    }
    summaries = []
    for condition, rows in condition_rows.items():
        resolved = sum(bool(row.get("resolved")) for row in rows)
        physical = _sum_present(rows, "physical_input_tokens")
        logical = _sum_present(rows, "logical_input_tokens")
        summaries.append({
            "condition": condition,
            "mode": rows[0].get("mode") if rows else None,
            "context_budget_fraction": rows[0].get("context_budget_fraction") if rows else None,
            "tasks": len(rows),
            "resolved": resolved,
            "success": resolved / len(rows) if rows else None,
            "wilson_95": _wilson(resolved, len(rows)),
            "physical_input_tokens": physical,
            "logical_input_tokens": logical,
            "cumulative_prompt_tokens": _sum_present(rows, "cumulative_prompt_tokens"),
            "output_tokens": _sum_present(rows, "output_tokens"),
            "wall_time_s": _sum_present(rows, "wall_time_s"),
            "grader_time_s": _sum_present(rows, "grader_time_s"),
            "invalid_patches": sum(row.get("invalid_patch") is True for row in rows),
            "errors": sum(row.get("error_type") is not None for row in rows),
            "token_saving_fraction": (
                1 - physical / logical if physical is not None and logical else None
            ),
        })
    baseline_rows = condition_rows.get("no_pra", [])
    baseline = {row["instance_id"]: row for row in baseline_rows}
    paired = []
    for condition, rows in condition_rows.items():
        if condition == "no_pra":
            continue
        for row in rows:
            base = baseline.get(row["instance_id"])
            if base is None:
                continue
            paired.append({
                "condition": condition,
                "mode": row.get("mode"),
                "context_budget_fraction": row.get("context_budget_fraction"),
                "instance_id": row["instance_id"],
                "baseline_resolved": bool(base.get("resolved")),
                "treatment_resolved": bool(row.get("resolved")),
                "outcome": (
                    "retained" if base.get("resolved") and row.get("resolved")
                    else "regressed" if base.get("resolved")
                    else "recovered" if row.get("resolved")
                    else "unchanged_failure"
                ),
                "baseline_physical_tokens": base.get("physical_input_tokens"),
                "treatment_physical_tokens": row.get("physical_input_tokens"),
                "baseline_cumulative_prompt_tokens": base.get("cumulative_prompt_tokens"),
                "treatment_cumulative_prompt_tokens": row.get("cumulative_prompt_tokens"),
                "baseline_trajectory_length": base.get("trajectory_length"),
                "baseline_wall_time_s": base.get("wall_time_s"),
            })
    matched_budget = []
    for budget in (0.5, 0.25, 0.125):
        truncation = next(
            (row for row in summaries if row["mode"] == "truncation" and row["context_budget_fraction"] == budget),
            None,
        )
        pra = next(
            (row for row in summaries if row["mode"] == "gateway-pra" and row["context_budget_fraction"] == budget),
            None,
        )
        if truncation is None or pra is None:
            continue
        matched_budget.append({
            "context_budget_fraction": budget,
            "truncation_condition": truncation["condition"],
            "pra_condition": pra["condition"],
            "success_delta": pra["success"] - truncation["success"],
            "resolved_delta": pra["resolved"] - truncation["resolved"],
            "physical_input_token_delta": (
                pra["physical_input_tokens"] - truncation["physical_input_tokens"]
                if pra["physical_input_tokens"] is not None
                and truncation["physical_input_tokens"] is not None
                else None
            ),
            "wall_time_delta_s": (
                pra["wall_time_s"] - truncation["wall_time_s"]
                if pra["wall_time_s"] is not None and truncation["wall_time_s"] is not None
                else None
            ),
        })
    return {"conditions": summaries, "paired": paired, "matched_budget": matched_budget}


def write_analysis(root: Path) -> dict[str, Any]:
    """Persist machine-readable summaries, paired rows, Markdown, and plots."""

    analysis = summarize(root)
    (root / "frontier_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    paired_path = root / "paired_outcomes.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "condition", "mode", "context_budget_fraction", "instance_id",
            "baseline_resolved", "treatment_resolved", "outcome",
            "baseline_physical_tokens", "treatment_physical_tokens",
            "baseline_cumulative_prompt_tokens", "treatment_cumulative_prompt_tokens",
            "baseline_trajectory_length", "baseline_wall_time_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis["paired"])
    lines = [
        "# Easy coding-agent frontier", "",
        "All treatment rows use the frozen Easy-50 task identities and the admitted No-PRA baseline.", "",
        "| Condition | Resolved | Success | Wilson 95% | Physical input | Cumulative prompt | Wall (s) | Invalid patches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["conditions"]:
        interval = row["wilson_95"]
        lines.append(
            f"| `{row['condition']}` | {row['resolved']}/{row['tasks']} | "
            f"{row['success']:.1%} | {interval[0]:.1%}--{interval[1]:.1%} | "
            f"{_display(row['physical_input_tokens'])} | "
            f"{_display(row['cumulative_prompt_tokens'])} | "
            f"{_display(row['wall_time_s'])} | {row['invalid_patches']} |"
        )
    lines.extend(["", "## Paired outcomes", "", "| Condition | Retained | Regressed | Recovered | Unchanged failure |", "| --- | ---: | ---: | ---: | ---: |"])
    for condition in sorted({row["condition"] for row in analysis["paired"]}):
        outcomes = [row["outcome"] for row in analysis["paired"] if row["condition"] == condition]
        lines.append(
            f"| `{condition}` | {outcomes.count('retained')} | {outcomes.count('regressed')} | "
            f"{outcomes.count('recovered')} | {outcomes.count('unchanged_failure')} |"
        )
    lines.extend([
        "", "## Matched-budget contrasts", "",
        "| Budget | PRA - truncation success | Resolved delta | Physical-token delta | Wall-time delta (s) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in analysis["matched_budget"]:
        lines.append(
            f"| {row['context_budget_fraction']:.1%} | {row['success_delta']:+.1%} | "
            f"{row['resolved_delta']:+d} | {_signed_display(row['physical_input_token_delta'])} | "
            f"{_signed_display(row['wall_time_delta_s'])} |"
        )
    if not analysis["matched_budget"]:
        lines.append("| N/R | N/R | N/R | N/R | N/R |")
    (root / "pra_frontier_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_difficulty_analysis(root, analysis["paired"])
    _plots(root, analysis["conditions"], analysis["paired"])
    return analysis


def _display(value: Any) -> str:
    return "N/R" if value is None else f"{value:,.0f}"


def _signed_display(value: Any) -> str:
    return "N/R" if value is None else f"{value:+,.0f}"


def _write_difficulty_analysis(root: Path, paired: list[dict[str, Any]]) -> None:
    """Report paired quality effects in predeclared context-demand buckets."""

    lines = [
        "# Difficulty and context-demand analysis", "",
        "Effects are paired by frozen task identity. Low, medium, and high buckets are "
        "tertiles of the No-PRA baseline rather than post-treatment task labels.", "",
        "| PRA condition | Measure | Low | Medium | High |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    pra_rows = [row for row in paired if row.get("mode") == "gateway-pra"]
    for condition in sorted({row["condition"] for row in pra_rows}):
        selected = [row for row in pra_rows if row["condition"] == condition]
        for field, label in (
            ("baseline_trajectory_length", "trajectory length"),
            ("baseline_cumulative_prompt_tokens", "cumulative prompt tokens"),
        ):
            effects = _bucket_effects(selected, field)
            lines.append(
                f"| `{condition}` | {label} | "
                + " | ".join(_effect_display(value) for value in effects)
                + " |"
            )
    if not pra_rows:
        lines.append("| Not available | matched PRA rows have not completed | N/R | N/R | N/R |")
    (root / "difficulty_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bucket_effects(rows: list[dict[str, Any]], field: str) -> list[float | None]:
    available = sorted((row for row in rows if row.get(field) is not None), key=lambda row: row[field])
    if not available:
        return [None, None, None]
    first = math.ceil(len(available) / 3)
    second = math.ceil(2 * len(available) / 3)
    buckets = [available[:first], available[first:second], available[second:]]
    return [
        sum(int(row["treatment_resolved"]) - int(row["baseline_resolved"]) for row in bucket) / len(bucket)
        if bucket else None
        for bucket in buckets
    ]


def _effect_display(value: float | None) -> str:
    return "N/R" if value is None else f"{100 * value:+.1f} pp"


def _plots(
    root: Path, rows: list[dict[str, Any]], paired: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    labels = [row["condition"] for row in rows]
    success = [100 * row["success"] for row in rows]
    physical = [row["physical_input_tokens"] for row in rows]
    available = [(label, x, y) for label, x, y in zip(labels, physical, success) if x is not None]
    if available:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for label, x, y in available:
            ax.scatter(x, y, s=55)
            ax.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Physical input tokens", ylabel="Official success (%)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / "success_vs_physical_tokens.png", dpi=180)
        plt.close(fig)
    for field, xlabel, filename in (
        ("cumulative_prompt_tokens", "Cumulative input tokens", "success_vs_cumulative_input_tokens.png"),
        ("wall_time_s", "Task wall time (s)", "success_vs_wall_clock_time.png"),
    ):
        available = [(row["condition"], row[field], 100 * row["success"]) for row in rows if row[field] is not None]
        if available:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for label, x, y in available:
                ax.scatter(x, y, s=55)
                ax.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
            ax.set(xlabel=xlabel, ylabel="Official success (%)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(root / filename, dpi=180)
            plt.close(fig)
    budget_rows = [row for row in rows if row["context_budget_fraction"] is not None]
    if budget_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for mode in sorted({row["mode"] for row in budget_rows}):
            selected = sorted(
                (row for row in budget_rows if row["mode"] == mode),
                key=lambda row: row["context_budget_fraction"],
            )
            ax.plot(
                [100 * row["context_budget_fraction"] for row in selected],
                [100 * row["success"] for row in selected], marker="o", label=mode,
            )
        ax.set(xlabel="Context budget (%)", ylabel="Official success (%)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "success_vs_context_budget.png", dpi=180)
        plt.close(fig)
    baseline = next((row for row in rows if row["condition"] == "no_pra"), None)
    if baseline and baseline["resolved"] and baseline["physical_input_tokens"]:
        comparison = [row for row in rows if row is not baseline and row["physical_input_tokens"] is not None]
        if comparison:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for row in comparison:
                saving = 100 * (1 - row["physical_input_tokens"] / baseline["physical_input_tokens"])
                retention = 100 * row["resolved"] / baseline["resolved"]
                ax.scatter(saving, retention, s=55)
                ax.annotate(row["condition"], (saving, retention), xytext=(4, 5), textcoords="offset points", fontsize=8)
            ax.axhline(100, color="black", linewidth=1, linestyle="--")
            ax.set(xlabel="Physical-token saving vs No-PRA (%)", ylabel="Solved-count retention (%)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(root / "token_saving_vs_success_retention.png", dpi=180)
            plt.close(fig)
        retained_rows = [row for row in rows if row["context_budget_fraction"] is not None]
        if retained_rows:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for mode in sorted({row["mode"] for row in retained_rows}):
                selected = sorted(
                    (row for row in retained_rows if row["mode"] == mode),
                    key=lambda row: row["context_budget_fraction"],
                )
                ax.plot(
                    [100 * row["context_budget_fraction"] for row in selected],
                    [100 * row["resolved"] / baseline["resolved"] for row in selected],
                    marker="o", label=mode,
                )
            ax.axhline(100, color="black", linewidth=1, linestyle="--")
            ax.set(xlabel="Context budget (%)", ylabel="Solved-count retention (%)")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(root / "success_retention_vs_context_budget.png", dpi=180)
            plt.close(fig)
    pra_rows = [row for row in paired if row.get("mode") == "gateway-pra"]
    for field, xlabel, filename in (
        ("baseline_trajectory_length", "No-PRA trajectory length", "pra_gain_vs_trajectory_length.png"),
        ("baseline_cumulative_prompt_tokens", "No-PRA cumulative input tokens", "pra_gain_vs_cumulative_context_load.png"),
    ):
        if not any(row.get(field) is not None for row in pra_rows):
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for condition in sorted({row["condition"] for row in pra_rows}):
            selected = [row for row in pra_rows if row["condition"] == condition]
            effects = _bucket_effects(selected, field)
            ax.plot((1, 2, 3), [100 * value if value is not None else math.nan for value in effects], marker="o", label=condition)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set(xlabel=f"{xlabel} bucket", ylabel="Paired success effect (pp)", xticks=(1, 2, 3), xticklabels=("Low", "Medium", "High"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_analysis(args.root), indent=2))


if __name__ == "__main__":
    main()
