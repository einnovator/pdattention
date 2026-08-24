"""Summarize Paper 6.5 M0 policy, safety, index, and context results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values: list[float], *, seed: int, draws: int = 5000) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    center = mean(values)
    if len(values) == 1:
        return center, center, center
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    )
    return center, samples[round(0.025 * (draws - 1))], samples[round(0.975 * (draws - 1))]


def _ece(rows: list[dict[str, str]], bins: int = 10) -> float:
    selection = [row for row in rows if row["top1_correct"] != ""]
    if not selection:
        return math.nan
    total = len(selection)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        values = [
            row
            for row in selection
            if lower <= float(row["confidence"]) <= upper
            and (bin_index == bins - 1 or float(row["confidence"]) < upper)
        ]
        if not values:
            continue
        confidence = mean(float(row["confidence"]) for row in values)
        accuracy = mean(float(row["top1_correct"]) for row in values)
        error += len(values) / total * abs(accuracy - confidence)
    return error


def _brier(rows: list[dict[str, str]]) -> float:
    selection = [row for row in rows if row["top1_correct"] != ""]
    return mean(
        (float(row["confidence"]) - float(row["top1_correct"])) ** 2
        for row in selection
    ) if selection else math.nan


def _seed_policy_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    selection = [row for row in rows if row["top1_correct"] != ""]
    nonselection = [row for row in rows if row["top1_correct"] == ""]
    acted = [row for row in selection if row["decision"] == "select"]
    regrets = [float(row["quality_regret"]) for row in selection if row["quality_regret"] != ""]
    cost_regrets = [float(row["cost_regret"]) for row in selection if row["cost_regret"] != ""]
    return {
        "top1_accuracy": mean(float(row["top1_correct"]) for row in selection),
        "outcome_accuracy": mean(float(row["outcome_correct"]) for row in rows),
        "selection_coverage": len(acted) / max(len(selection), 1),
        "selective_accuracy": mean(float(row["top1_correct"]) for row in acted) if acted else 0.0,
        "safe_nonaction_rate": mean(1.0 - float(row["false_act"]) for row in nonselection)
        if nonselection else 1.0,
        "false_act_rate": mean(float(row["false_act"]) for row in nonselection)
        if nonselection else 0.0,
        "ece": _ece(rows),
        "brier": _brier(rows),
        "mean_retrieval_stages": mean(float(row["retrieval_stages"]) for row in rows),
        "mean_fallback_count": mean(float(row["fallback_count"]) for row in rows),
        "mean_policy_us": mean(float(row["policy_us"]) for row in rows),
        "estimated_discovery_ms": mean(float(row["estimated_discovery_ms"]) for row in rows),
        "quality_regret": mean(regrets) if regrets else 0.0,
        "cost_regret": mean(cost_regrets) if cost_regrets else 0.0,
        "queries": float(len(rows)),
    }


def summarize_policy(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test = [row for row in rows if row["split"] == "test" and row["calibrated"] == "1"]
    by_seed: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    by_stratum: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in test:
        size = int(row["catalog_size"])
        policy = row["policy"]
        by_seed[(size, policy, int(row["seed"]))].append(row)
        by_stratum[(size, policy, row["stratum"])].append(row)

    seed_metrics = {
        key: _seed_policy_metrics(values) for key, values in by_seed.items()
    }
    summary = []
    metrics = (
        "top1_accuracy",
        "outcome_accuracy",
        "selection_coverage",
        "selective_accuracy",
        "safe_nonaction_rate",
        "false_act_rate",
        "ece",
        "brier",
        "mean_retrieval_stages",
        "mean_fallback_count",
        "mean_policy_us",
        "estimated_discovery_ms",
        "quality_regret",
        "cost_regret",
    )
    for size, policy in sorted({(key[0], key[1]) for key in seed_metrics}):
        row: dict[str, object] = {"catalog_size": size, "policy": policy}
        matching = [value for (s, p, _), value in seed_metrics.items() if s == size and p == policy]
        row["seeds"] = len(matching)
        for metric_index, metric in enumerate(metrics):
            values = [float(value[metric]) for value in matching]
            center, low, high = _mean_ci(
                values,
                seed=size * 1000 + sum(ord(char) for char in policy) + metric_index,
            )
            row[metric] = center
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary.append(row)

    stratum_summary = []
    for (size, policy, stratum), values in sorted(by_stratum.items()):
        selection = [row for row in values if row["top1_correct"] != ""]
        stratum_summary.append(
            {
                "catalog_size": size,
                "policy": policy,
                "stratum": stratum,
                "examples": len(values),
                "top1_accuracy": mean(float(row["top1_correct"]) for row in selection)
                if selection else "",
                "outcome_accuracy": mean(float(row["outcome_correct"]) for row in values),
                "false_act_rate": mean(float(row["false_act"]) for row in values),
                "mean_confidence": mean(float(row["confidence"]) for row in values),
                "mean_stages": mean(float(row["retrieval_stages"]) for row in values),
            }
        )
    return summary, stratum_summary


def summarize_index(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["catalog_size"]), row["mode"])].append(row)
    summary = []
    for (size, mode), values in sorted(grouped.items()):
        summary.append(
            {
                "catalog_size": size,
                "mode": mode,
                "seeds": len(values),
                "build_ms": mean(float(row["build_ms"]) for row in values),
                "mutation_rebuild_ms": mean(float(row["mutation_rebuild_ms"]) for row in values),
                "index_bytes": mean(float(row["index_bytes"]) for row in values),
                "logical_definition_tokens": mean(
                    float(row["logical_definition_tokens"]) for row in values
                ),
                "active_materialized_tokens": mean(
                    float(row["active_materialized_tokens"]) for row in values
                ),
                "active_fraction": mean(float(row["active_fraction"]) for row in values),
                "warm_query_mean_ms": mean(float(row["warm_query_mean_ms"]) for row in values),
                "warm_query_median_ms": mean(float(row["warm_query_median_ms"]) for row in values),
                "mean_candidates_scored": mean(float(row["mean_candidates_scored"]) for row in values),
            }
        )
    return summary


def _plot_policy_scaling(output: Path, rows: list[dict[str, object]]) -> None:
    policies = ("fixed_token", "fixed_index", "fixed_semantic", "fixed_hybrid", "auto", "user_hint", "adaptive")
    plt.figure(figsize=(8.2, 4.8))
    for policy in policies:
        values = sorted((row for row in rows if row["policy"] == policy), key=lambda row: row["catalog_size"])
        plt.plot(
            [row["catalog_size"] for row in values],
            [row["top1_accuracy"] for row in values],
            marker="o",
            label=policy.replace("fixed_", ""),
        )
    plt.xscale("log", base=2)
    plt.ylim(0.0, 1.03)
    plt.xlabel("Catalog resources")
    plt.ylabel("Held-out top-1 selection accuracy")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(output / f"m0_policy_scaling.{suffix}", dpi=180)
    plt.close()


def _plot_index_latency(output: Path, rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(7.8, 4.6))
    for mode in ("token", "index", "semantic", "hybrid"):
        values = sorted((row for row in rows if row["mode"] == mode), key=lambda row: row["catalog_size"])
        plt.plot(
            [row["catalog_size"] for row in values],
            [row["warm_query_mean_ms"] for row in values],
            marker="o",
            label=mode,
        )
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Catalog resources")
    plt.ylabel("Warm discovery component latency (ms)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(output / f"m0_index_latency.{suffix}", dpi=180)
    plt.close()


def _plot_context_scaling(output: Path, rows: list[dict[str, object]]) -> None:
    values = sorted((row for row in rows if row["mode"] == "index"), key=lambda row: row["catalog_size"])
    figure, left = plt.subplots(figsize=(7.8, 4.6))
    right = left.twinx()
    left.plot(
        [row["catalog_size"] for row in values],
        [row["logical_definition_tokens"] for row in values],
        marker="o",
        color="#2563eb",
        label="logical catalog tokens",
    )
    right.plot(
        [row["catalog_size"] for row in values],
        [row["active_materialized_tokens"] for row in values],
        marker="s",
        color="#dc2626",
        label="selected definition tokens",
    )
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("Catalog resources")
    left.set_ylabel("Logical definition tokens", color="#2563eb")
    right.set_ylabel("Active selected-definition tokens", color="#dc2626")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="center left")
    left.grid(alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"m0_context_scaling.{suffix}", dpi=180)
    plt.close(figure)


def _plot_policy_frontier(output: Path, rows: list[dict[str, object]]) -> None:
    largest = max(int(row["catalog_size"]) for row in rows)
    values = [row for row in rows if int(row["catalog_size"]) == largest]
    plt.figure(figsize=(7.4, 4.8))
    for row in values:
        x = float(row["estimated_discovery_ms"])
        y = float(row["top1_accuracy"])
        plt.scatter(x, y, s=50)
        plt.annotate(str(row["policy"]).replace("fixed_", ""), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xscale("log")
    plt.xlabel("Estimated measured discovery-component cost (ms/query)")
    plt.ylabel("Held-out top-1 accuracy")
    plt.ylim(0.0, 1.03)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(output / f"m0_policy_frontier.{suffix}", dpi=180)
    plt.close()


def _plot_index_amortization(output: Path, rows: list[dict[str, object]]) -> None:
    largest = max(int(row["catalog_size"]) for row in rows)
    index = next(row for row in rows if int(row["catalog_size"]) == largest and row["mode"] == "index")
    queries = (1, 10, 100, 1000, 10000)
    values = [float(index["warm_query_mean_ms"]) + float(index["build_ms"]) / count for count in queries]
    plt.figure(figsize=(7.4, 4.5))
    plt.plot(queries, values, marker="o")
    plt.axhline(float(index["warm_query_mean_ms"]), color="#6b7280", linestyle="--", label="warm-query floor")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Queries per catalog rebuild")
    plt.ylabel("Amortized index discovery cost (ms/query)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(output / f"m0_index_amortization.{suffix}", dpi=180)
    plt.close()


def summarize(args: argparse.Namespace) -> None:
    output = Path(args.results)
    policy_rows = _read_csv(output / "m0_policy_per_query.csv")
    cost_rows = _read_csv(output / "m0_index_costs.csv")
    latency_groups: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in cost_rows:
        latency_groups[(int(row["catalog_size"]), int(row["seed"]), row["mode"])].append(
            float(row["warm_query_mean_ms"])
        )
    latency = {key: mean(values) for key, values in latency_groups.items()}
    for row in policy_rows:
        if row["expected_decision"] == "select":
            row["outcome_correct"] = str(
                int(row["top1_correct"] == "1" and row["decision"] == "select")
            )
        else:
            row["outcome_correct"] = str(int(row["decision"] in {"ask", "abstain"}))
        path = tuple(value for value in row["executed_path"].split(">") if value)
        row["estimated_discovery_ms"] = sum(
            latency.get((int(row["catalog_size"]), int(row["seed"]), mode), 0.0)
            for mode in path
        )
    # Persist the derived fields so machine readers see the same actionable
    # outcome and measured-path cost used by every summary and figure.
    _write_csv(output / "m0_policy_per_query.csv", policy_rows)
    policy_summary, stratum_summary = summarize_policy(policy_rows)
    index_summary = summarize_index(cost_rows)
    _write_csv(output / "m0_policy_summary.csv", policy_summary)
    _write_csv(output / "m0_stratum_summary.csv", stratum_summary)
    _write_csv(output / "m0_index_summary.csv", index_summary)

    largest = max(int(row["catalog_size"]) for row in policy_summary)
    test = [
        row
        for row in policy_rows
        if row["split"] == "test"
        and int(row["catalog_size"]) == largest
        and row["top1_correct"] != ""
    ]
    oracle_counts = Counter(row["oracle_policy"] for row in test if row["policy"] == "adaptive")
    oracle_rows = [
        {"catalog_size": largest, "oracle_policy": policy, "queries": count, "fraction": count / max(sum(oracle_counts.values()), 1)}
        for policy, count in sorted(oracle_counts.items())
    ]
    _write_csv(output / "m0_oracle_policy_distribution.csv", oracle_rows)

    _plot_policy_scaling(output, policy_summary)
    _plot_index_latency(output, index_summary)
    _plot_context_scaling(output, index_summary)
    _plot_policy_frontier(output, policy_summary)
    _plot_index_amortization(output, index_summary)

    largest_policy = {
        str(row["policy"]): row
        for row in policy_summary
        if int(row["catalog_size"]) == largest
    }
    largest_index = {
        str(row["mode"]): row
        for row in index_summary
        if int(row["catalog_size"]) == largest
    }
    findings = {
        "largest_catalog": largest,
        "policies": largest_policy,
        "index": largest_index,
        "oracle_policy_distribution": oracle_rows,
        "interpretation_constraints": [
            "M0 is deterministic selection-only evidence, not language-model or agent capability.",
            "The semantic channel is a declared concept-normalized signed-hash control.",
            "Mutation cost is a full immutable index rebuild, not an incremental-update claim.",
            "Component timings exclude tokenization, model forward passes, native-K/V transfer, and execution.",
        ],
    }
    (output / "m0_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="docs/papers/shared/results/paper6_5_tools",
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
