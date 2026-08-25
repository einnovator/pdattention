"""Summarize the five-seed Paper 6.5 M1 causal experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


METRICS = (
    "selected_identity_correct",
    "schema_exact",
    "argument_exact",
    "call_exact",
    "conditional_continuation_exact",
    "end_to_end_success",
    "teacher_nll",
    "native_prompt_tokens",
    "materialized_kv_tokens",
    "active_fraction",
    "cache_encode_ms",
    "call_generation_ms",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _values(rows, metric):
    return [float(row[metric]) for row in rows if row.get(metric, "") != ""]


def _mean_ci(values, *, seed, draws=10_000):
    values = list(values)
    if not values:
        return math.nan, math.nan, math.nan
    center = mean(values)
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(draws)
    )
    return center, samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]


def _sign_flip_paired(differences):
    values = [float(value) for value in differences if abs(float(value)) > 1e-12]
    if not values:
        return 1.0
    observed = abs(mean(values))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        extreme += int(statistic >= observed - 1e-12)
    return extreme / total


def summarize(root: Path) -> dict:
    rows = _read_csv(root / "m1_example_results.csv")
    training = _read_csv(root / "m1_training_history.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["catalog_size"]), row["condition"])].append(row)

    seed_rows = []
    for (seed, size, condition), group in sorted(grouped.items()):
        fit = [row for row in group if int(row["context_fit"])]
        summary = {
            "seed": seed,
            "catalog_size": size,
            "condition": condition,
            "examples": len(group),
            "context_fit_rate": len(fit) / len(group),
            "logical_catalog_tokens": mean(_values(group, "logical_catalog_tokens")),
        }
        for metric in METRICS:
            values = _values(fit, metric)
            summary[metric] = mean(values) if values else ""
        seed_rows.append(summary)
    _write_csv(root / "m1_seed_summary.csv", seed_rows)

    aggregate_rows = []
    aggregate = defaultdict(list)
    for row in seed_rows:
        aggregate[(row["catalog_size"], row["condition"])].append(row)
    for index, ((size, condition), group) in enumerate(sorted(aggregate.items())):
        result = {
            "catalog_size": size,
            "condition": condition,
            "seeds": len(group),
            "examples_per_seed": group[0]["examples"],
        }
        for metric in ("context_fit_rate", "logical_catalog_tokens", *METRICS):
            values = [float(row[metric]) for row in group if row.get(metric, "") != ""]
            center, low, high = _mean_ci(values, seed=6000 + index)
            result[f"{metric}_mean"] = center
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
            result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(result)
    _write_csv(root / "m1_summary.csv", aggregate_rows)

    by_key = {
        (int(row["seed"]), int(row["catalog_size"]), row["condition"]): row
        for row in seed_rows
    }
    paired_rows = []
    for size in sorted({int(row["catalog_size"]) for row in seed_rows}):
        seeds = sorted({int(row["seed"]) for row in seed_rows})
        for baseline in (
            "discovered_memory",
            "shuffled_memory",
            "disabled_memory",
            "direct_selected",
        ):
            differences = []
            for seed in seeds:
                oracle = by_key[(seed, size, "oracle_memory")]["end_to_end_success"]
                other = by_key[(seed, size, baseline)]["end_to_end_success"]
                differences.append(float(oracle) - float(other))
            paired_rows.append(
                {
                    "catalog_size": size,
                    "comparison": f"oracle_memory-minus-{baseline}",
                    "mean_difference": mean(differences),
                    "same_direction_seeds": sum(value > 0 for value in differences),
                    "zero_difference_seeds": sum(abs(value) <= 1e-12 for value in differences),
                    "exact_two_sided_sign_flip_p": _sign_flip_paired(differences),
                }
            )
    _write_csv(root / "m1_paired_effects.csv", paired_rows)

    def result(size, condition):
        return next(
            row
            for row in aggregate_rows
            if int(row["catalog_size"]) == size and row["condition"] == condition
        )

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    sizes = (8, 32, 128)
    colors = {
        "oracle_memory": "#006d77",
        "discovered_memory": "#2a9d8f",
        "shuffled_memory": "#c44536",
        "disabled_memory": "#6c757d",
        "direct_selected": "#7b2cbf",
        "eager_catalog": "#e9c46a",
    }
    labels = {
        "oracle_memory": "Oracle native K/V",
        "discovered_memory": "Discovered native K/V",
        "shuffled_memory": "Shuffled K/V",
        "disabled_memory": "Disabled memory",
        "direct_selected": "Direct selected definition",
        "eager_catalog": "Eager catalog",
    }
    fig, axis = plt.subplots(figsize=(7.4, 4.3))
    line_styles = {
        "oracle_memory": "--",
        "discovered_memory": "-",
    }
    markers = {
        "oracle_memory": "s",
        "discovered_memory": "o",
    }
    for condition in labels:
        points = [result(size, condition) for size in sizes]
        valid = [row for row in points if row["context_fit_rate_mean"] > 0]
        axis.errorbar(
            [row["catalog_size"] for row in valid],
            [row["end_to_end_success_mean"] for row in valid],
            yerr=[
                [row["end_to_end_success_mean"] - row["end_to_end_success_ci_low"] for row in valid],
                [row["end_to_end_success_ci_high"] - row["end_to_end_success_mean"] for row in valid],
            ],
            marker=markers.get(condition, "o"),
            linestyle=line_styles.get(condition, "-"),
            linewidth=1.8,
            capsize=3,
            color=colors[condition],
            label=labels[condition],
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, labels=[str(size) for size in sizes])
    axis.set_ylim(-0.04, 1.04)
    axis.set_xlabel("Catalog resources")
    axis.set_ylabel("End-to-end success")
    axis.grid(alpha=0.25)
    axis.legend(
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figures / "m1_end_to_end_scaling.pdf")
    fig.savefig(figures / "m1_end_to_end_scaling.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.1))
    logical = [result(size, "oracle_memory")["logical_catalog_tokens_mean"] for size in sizes]
    active = [result(size, "oracle_memory")["materialized_kv_tokens_mean"] for size in sizes]
    direct = [result(size, "direct_selected")["native_prompt_tokens_mean"] for size in sizes]
    axis.plot(sizes, logical, marker="o", color="#264653", label="Logical definitions")
    axis.plot(sizes, active, marker="o", color="#2a9d8f", label="PRA materialized K/V tokens")
    axis.plot(sizes, direct, marker="o", color="#7b2cbf", label="Direct selected prompt tokens")
    axis.axhline(512, color="#c44536", linestyle="--", linewidth=1.2, label="Native window")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks(sizes, labels=[str(size) for size in sizes])
    axis.set_xlabel("Catalog resources")
    axis.set_ylabel("Tokens")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "m1_context_accounting.pdf")
    fig.savefig(figures / "m1_context_accounting.png", dpi=180)
    plt.close(fig)

    training_grouped = defaultdict(list)
    for row in training:
        training_grouped[int(row["step"])].append(float(row["loss"]))
    window = 100
    curve_x = []
    curve_y = []
    ordered_steps = sorted(training_grouped)
    for stop in range(window, len(ordered_steps) + 1, window):
        selected = ordered_steps[stop - window : stop]
        curve_x.append(selected[-1])
        curve_y.append(mean(value for step in selected for value in training_grouped[step]))
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.plot(curve_x, curve_y, color="#006d77", linewidth=2)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Masked call/continuation loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "m1_training_curve.pdf")
    fig.savefig(figures / "m1_training_curve.png", dpi=180)
    plt.close(fig)

    findings = {
        "status": "m1_causal_gate_complete",
        "seeds": 5,
        "examples_per_seed_per_size": 8,
        "catalog_sizes": list(sizes),
        "discovered_e2e": {
            str(size): result(size, "discovered_memory")["end_to_end_success_mean"]
            for size in sizes
        },
        "oracle_e2e": {
            str(size): result(size, "oracle_memory")["end_to_end_success_mean"]
            for size in sizes
        },
        "direct_selected_e2e": {
            str(size): result(size, "direct_selected")["end_to_end_success_mean"]
            for size in sizes
        },
        "shuffled_e2e": {
            str(size): result(size, "shuffled_memory")["end_to_end_success_mean"]
            for size in sizes
        },
        "disabled_e2e": {
            str(size): result(size, "disabled_memory")["end_to_end_success_mean"]
            for size in sizes
        },
        "eager_8_e2e": result(8, "eager_catalog")["end_to_end_success_mean"],
        "eager_context_fit": {
            str(size): result(size, "eager_catalog")["context_fit_rate_mean"]
            for size in sizes
        },
        "discovery_oracle_parity": all(
            abs(
                result(size, "discovered_memory")["end_to_end_success_mean"]
                - result(size, "oracle_memory")["end_to_end_success_mean"]
            ) < 1e-12
            for size in sizes
        ),
        "interpretation": (
            "The toy decoder uses correctly selected native K/V causally; the result "
            "separates URI discovery from host-bound call construction and does not "
            "establish pretrained-agent or real-tool competence."
        ),
    }
    (root / "m1_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    macros = [
        rf"\newcommand{{\PaperSixFiveMOneSeeds}}{{{findings['seeds']}}}",
        rf"\newcommand{{\PaperSixFiveMOneExamples}}{{{findings['examples_per_seed_per_size']}}}",
    ]
    for size in sizes:
        suffix = {8: "Eight", 32: "ThirtyTwo", 128: "OneTwentyEight"}[size]
        macros.extend(
            (
                rf"\newcommand{{\PaperSixFiveMOneDiscovered{suffix}}}{{{findings['discovered_e2e'][str(size)]:.3f}}}",
                rf"\newcommand{{\PaperSixFiveMOneOracle{suffix}}}{{{findings['oracle_e2e'][str(size)]:.3f}}}",
                rf"\newcommand{{\PaperSixFiveMOneDirect{suffix}}}{{{findings['direct_selected_e2e'][str(size)]:.3f}}}",
                rf"\newcommand{{\PaperSixFiveMOneShuffled{suffix}}}{{{findings['shuffled_e2e'][str(size)]:.3f}}}",
                rf"\newcommand{{\PaperSixFiveMOneDisabled{suffix}}}{{{findings['disabled_e2e'][str(size)]:.3f}}}",
            )
        )
    macros.append(rf"\newcommand{{\PaperSixFiveMOneEagerEight}}{{{findings['eager_8_e2e']:.3f}}}")
    (root / "generated_m1_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("docs/papers/shared/results/paper6_5_tools/m1"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.input_dir), indent=2))


if __name__ == "__main__":
    main()
