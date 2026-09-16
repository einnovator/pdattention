"""Aggregate and plot Paper 8.5 multi-issue quality--saving frontiers.

The input is the strict output of :mod:`multi_issue_frontier`. Repetitions of
the same ordered issue family are averaged before uncertainty is computed, so
requests and executions are never treated as independent observations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
from typing import Any, Mapping, Sequence


METRICS = (
    "official_resolution",
    "failure_aware_saving_vs_persistent_full",
    "failure_aware_saving_vs_fresh_full",
    "calls_delta_vs_persistent_full",
    "calls_delta_vs_fresh_full",
    "successful_calls_delta_vs_persistent_full",
    "successful_calls_delta_vs_fresh_full",
    "rediscovery_delta_vs_fresh_full",
    "cost_per_resolved_issue",
    "first_action_divergence_rate",
    "first_divergence_preceding_saving",
    "selected_input_tokens",
)


def _mean_interval(
    values: Sequence[float], *, seed: int, samples: int
) -> dict[str, float | int | str | None]:
    if not values:
        return {"mean": None, "lower": None, "upper": None, "clusters": 0,
                "interval_status": "missing"}
    mean = fmean(values)
    if len(values) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "clusters": 1,
                "interval_status": "single_cluster_no_inference"}
    rng = random.Random(seed)
    draws = sorted(
        fmean(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    lower_index = max(0, math.floor(0.025 * (samples - 1)))
    upper_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "mean": mean,
        "lower": draws[lower_index],
        "upper": draws[upper_index],
        "clusters": len(values),
        "interval_status": "sequence_cluster_bootstrap",
    }


def aggregate_frontier(
    reduction: Mapping[str, Any], *, seed: int = 850, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    """Aggregate without pooling strata, issue counts, or strategy configs."""

    if reduction.get("schema_version") != 1:
        raise ValueError("frontier reduction requires schema_version=1")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rows = reduction.get("rows")
    if not isinstance(rows, list):
        raise ValueError("frontier reduction has no rows")

    cells: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("sequence_stratum"),
            row.get("agent_id"),
            row.get("agent_revision"),
            row.get("model_revision"),
            row.get("tokenizer_revision"),
            row.get("harness_revision"),
            int(row.get("issue_count") or 0),
            row.get("session_mode"),
            row.get("strategy_id"),
            row.get("strategy_config_id"),
            row.get("strategy_config_digest"),
        )
        if not all(key[:6]) or key[6] not in set(range(1, 11)) or not all(key[7:]):
            raise ValueError("row lacks a complete curve-cell identity")
        cells[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for key, cell_rows in sorted(cells.items(), key=lambda item: str(item[0])):
        by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            family = str(row.get("sequence_family_id") or "")
            if not family:
                raise ValueError("row lacks sequence_family_id")
            by_family[family].append(row)
        metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for family_rows in by_family.values():
            for metric in METRICS:
                values = [
                    float(row[metric]) for row in family_rows
                    if row.get(metric) is not None
                ]
                if values:
                    metric_values[metric].append(fmean(values))
        metric_summary = {
            metric: _mean_interval(
                values,
                seed=seed + sum(ord(char) for char in metric),
                samples=bootstrap_samples,
            )
            for metric, values in metric_values.items()
        }
        aggregated.append({
            "sequence_stratum": key[0],
            "agent_id": key[1],
            "agent_revision": key[2],
            "model_revision": key[3],
            "tokenizer_revision": key[4],
            "harness_revision": key[5],
            "issue_count": key[6],
            "session_mode": key[7],
            "strategy_id": key[8],
            "strategy_config_id": key[9],
            "strategy_config_digest": key[10],
            "strategy_config": dict(cell_rows[0].get("strategy_config") or {}),
            "execution_rows": len(cell_rows),
            "sequence_clusters": len(by_family),
            "metrics": metric_summary,
        })

    return {
        "schema_version": 1,
        "study": "paper8_5_multi_issue_quality_saving_curves",
        "uncertainty_unit": "ordered_issue_sequence_family",
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "pooling_rule": "never pool sequence strata, agents, models, tokenizer/harness revisions, issue counts, or configs",
        "cells": aggregated,
    }


def _metric(cell: Mapping[str, Any], name: str) -> float | None:
    value = cell["metrics"][name]["mean"]
    return None if value is None else float(value)


def _errors(cell: Mapping[str, Any], name: str) -> tuple[float, float]:
    summary = cell["metrics"][name]
    mean = float(summary["mean"])
    return mean - float(summary["lower"]), float(summary["upper"]) - mean


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _compact_identity_label(identity: tuple[str, ...]) -> str:
    """Keep audit identities in JSON while making plotted legends readable."""

    agent, _agent_revision, model, _tokenizer, _harness, strategy, config = identity
    if strategy == "S01_persistent_full" and config == "default":
        policy = "FULL"
    else:
        match = re.fullmatch(r"r(\d+)m(\d+)v(\d+)_tasks\d+", config)
        policy = (
            f"R{match.group(1)}/M{match.group(2)}/V{match.group(3)}"
            if match else f"{strategy}/{config}"
        )
    return f"{agent} / {model[:8]} / {policy}"


def _plot_xy(
    cells: Sequence[Mapping[str, Any]], *, x_metric: str, y_metric: str,
    xlabel: str, ylabel: str, title: str, output: Path
) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [cell for cell in cells if _metric(cell, x_metric) is not None
              and _metric(cell, y_metric) is not None]
    if not usable:
        return False
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    if x_metric == "failure_aware_saving_vs_persistent_full":
        axis.axvspan(
            0.30, 0.50, color="#2ca02c", alpha=0.08,
            label="primary 30--50% target region",
        )
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in usable:
        grouped[(
            str(cell["agent_id"]), str(cell["agent_revision"]),
            str(cell["model_revision"]), str(cell["tokenizer_revision"]),
            str(cell["harness_revision"]), str(cell["strategy_id"]),
            str(cell["strategy_config_id"]),
        )].append(cell)
    for identity, group in sorted(grouped.items()):
        group = sorted(group, key=lambda cell: int(cell["issue_count"]))
        x = [_metric(cell, x_metric) for cell in group]
        y = [_metric(cell, y_metric) for cell in group]
        x_errors = [_errors(cell, x_metric) for cell in group]
        y_errors = [_errors(cell, y_metric) for cell in group]
        axis.errorbar(
            x, y,
            xerr=([row[0] for row in x_errors], [row[1] for row in x_errors]),
            yerr=([row[0] for row in y_errors], [row[1] for row in y_errors]),
            marker="o", linewidth=1.2, capsize=2,
            label=_compact_identity_label(identity),
        )
        for cell, x_value, y_value in zip(group, x, y):
            axis.annotate(f"N={cell['issue_count']}", (x_value, y_value), fontsize=7)
    axis.axvline(0, color="0.75", linewidth=0.8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    if len(grouped) <= 12:
        axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return True


def render_frontier_plots(summary: Mapping[str, Any], output_directory: Path) -> dict[str, Any]:
    cells = summary.get("cells")
    if not isinstance(cells, list):
        raise ValueError("curve summary has no cells")
    emitted: list[str] = []
    plots = (
        ("quality_vs_persistent.png", "failure_aware_saving_vs_persistent_full",
         "official_resolution", "Failure-aware saving vs persistent FULL",
         "Official issue resolution", "Quality--saving frontier vs persistent FULL"),
        ("quality_vs_fresh.png", "failure_aware_saving_vs_fresh_full",
         "official_resolution", "Failure-aware saving vs fresh FULL",
         "Official issue resolution", "Quality--saving frontier vs fresh FULL"),
        ("calls_vs_persistent.png", "failure_aware_saving_vs_persistent_full",
         "calls_delta_vs_persistent_full", "Failure-aware saving vs persistent FULL",
         "Tool-call delta", "Interaction-cost frontier"),
        ("successful_calls_vs_persistent.png", "failure_aware_saving_vs_persistent_full",
         "successful_calls_delta_vs_persistent_full",
         "Failure-aware saving vs persistent FULL", "Tool-call delta (joint successes)",
         "Successful-run interaction frontier"),
        ("divergence_before_saving.png", "first_divergence_preceding_saving",
         "first_action_divergence_rate", "Saving before first divergence or terminal",
         "First-action divergence fraction", "Divergence is charged only prior saving"),
    )
    for stratum in sorted({str(cell["sequence_stratum"]) for cell in cells}):
        stratum_cells = [cell for cell in cells if cell["sequence_stratum"] == stratum]
        stratum_directory = output_directory / _safe_name(stratum)
        for filename, x_metric, y_metric, xlabel, ylabel, title in plots:
            destination = stratum_directory / filename
            if _plot_xy(stratum_cells, x_metric=x_metric, y_metric=y_metric,
                        xlabel=xlabel, ylabel=ylabel, title=title, output=destination):
                emitted.append(str(destination.relative_to(output_directory)))

    manifest = {
        "schema_version": 1,
        "study": summary.get("study"),
        "plot_count": len(emitted),
        "plots": emitted,
        "warning": "Plots are descriptive until multiple independent ordered sequence families exist.",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate and plot Paper 8.5 frontiers.")
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=850)
    args = parser.parse_args(argv)
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    summary = aggregate_frontier(
        reduction, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "curve_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_frontier_plots(summary, args.output_directory)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
