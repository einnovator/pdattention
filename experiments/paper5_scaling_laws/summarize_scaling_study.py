"""Aggregate, fit, plot, and audit the Paper 5 scaling artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper5_scaling_laws.scaling_core import fit_candidate_laws, pareto_frontier


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({field for row in values for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def grouped_mean(
    rows: Sequence[dict[str, str]],
    keys: Sequence[str],
    value: str,
    predicate: Callable[[dict[str, str]], bool] = lambda row: True,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if predicate(row) and row.get(value, "") != "":
            groups[tuple(row[key] for key in keys)].append(float(row[value]))
    output = []
    for identity, values in groups.items():
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(
            {
                **dict(zip(keys, identity)),
                "mean": statistics.fmean(values),
                "sd": sd,
                "ci95": 1.96 * sd / math.sqrt(len(values)),
                "n": len(values),
            }
        )
    return sorted(output, key=lambda row: tuple(float(row[key]) if row[key].replace(".", "", 1).isdigit() else row[key] for key in keys))


def _fit_series(name: str, points: Sequence[dict[str, Any]], x_key: str = "logical_tokens") -> dict[str, Any]:
    ordered = sorted(points, key=lambda row: float(row[x_key]))
    x = [float(row[x_key]) for row in ordered]
    y = [float(row["mean"]) for row in ordered]
    fits = fit_candidate_laws(x, y)
    return {
        "series": name,
        "x": x,
        "y": y,
        "best_family_by_aic": fits[0].family,
        "fits": [fit.as_dict() for fit in fits],
        "interpretation": "descriptive five-point fit; no asymptotic claim",
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _error_plot(
    points: Sequence[dict[str, Any]],
    *,
    x_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    directory: Path,
    name: str,
    log_x: bool = False,
    color: str = "#167D9A",
) -> None:
    ordered = sorted(points, key=lambda row: float(row[x_key]))
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.errorbar(
        [float(row[x_key]) for row in ordered],
        [row["mean"] for row in ordered],
        yerr=[row["ci95"] for row in ordered],
        marker="o",
        capsize=3,
        color=color,
    )
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    _save(fig, directory, name)


def make_plots(
    output: Path,
    logical: list[dict[str, str]],
    active: list[dict[str, str]],
    adaptive: list[dict[str, str]],
    dispersion: list[dict[str, str]],
    serving: list[dict[str, str]],
    models: list[dict[str, str]],
    training: list[dict[str, str]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    exact = lambda row: row["backend"] == "exact_gemm"
    coarse = lambda row: row["backend"] == "coarse_to_fine"

    quality = grouped_mean(logical, ["logical_tokens"], "evidence_recall", exact)
    _error_plot(
        quality,
        x_key="logical_tokens",
        title="Fixed-working-set retrieval quality",
        xlabel="Logical reference tokens",
        ylabel="Evidence recall",
        directory=figures,
        name="quality_vs_logical_memory",
        log_x=True,
    )
    active_points = grouped_mean(logical, ["logical_tokens"], "active_native_kv_tokens", exact)
    _error_plot(
        active_points,
        x_key="logical_tokens",
        title="Physical attention state remains fixed by construction",
        xlabel="Logical reference tokens",
        ylabel="Active native-K/V tokens",
        directory=figures,
        name="active_kv_vs_logical_memory",
        log_x=True,
        color="#C64B3C",
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for label, predicate, color in (
        ("Exact GEMM", exact, "#167D9A"),
        ("Coarse-to-fine, 4 probes", coarse, "#C64B3C"),
    ):
        points = grouped_mean(logical, ["logical_tokens"], "search_latency_p50_seconds", predicate)
        ax.plot(
            [float(row["logical_tokens"]) for row in points],
            [1000 * row["mean"] for row in points],
            marker="o",
            label=label,
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Measured routing latency")
    ax.set_xlabel("Logical reference tokens")
    ax.set_ylabel("Median search latency (ms)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    _save(fig, figures, "search_latency_vs_logical_memory")

    hbm = grouped_mean(logical, ["logical_tokens"], "peak_device_bytes", exact)
    for row in hbm:
        row["mean"] /= 2**20
        row["ci95"] /= 2**20
    _error_plot(
        hbm,
        x_key="logical_tokens",
        title="Resident routing-index footprint",
        xlabel="Logical reference tokens",
        ylabel="Peak allocated device memory (MiB)",
        directory=figures,
        name="hbm_vs_logical_memory",
        log_x=True,
        color="#6B5B95",
    )

    max_memory = max(int(row["logical_tokens"]) for row in active)
    quality_budget = grouped_mean(
        active,
        ["requested_active_kv_tokens"],
        "evidence_recall",
        lambda row: exact(row) and int(row["logical_tokens"]) == max_memory,
    )
    _error_plot(
        quality_budget,
        x_key="requested_active_kv_tokens",
        title=f"Active-K/V frontier at {max_memory:,} logical tokens",
        xlabel="Active native-K/V tokens",
        ylabel="Evidence recall",
        directory=figures,
        name="quality_vs_active_kv",
        log_x=True,
    )
    layer_quality = [dict(row, layer_tokens=str(int(row["requested_active_kv_tokens"]) * 2)) for row in quality_budget]
    _error_plot(
        layer_quality,
        x_key="layer_tokens",
        title="Layer-token K/V frontier (two consumers)",
        xlabel="Layer-token K/V states",
        ylabel="Evidence recall",
        directory=figures,
        name="quality_vs_layer_token_kv",
        log_x=True,
    )

    measured_models = [row for row in models if row["measured"] == "True" and row["quality"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in measured_models:
        groups[row["training_regime"]].append(row)
    for regime, rows in groups.items():
        ax.scatter(
            [float(row["model_parameters"]) for row in rows],
            [float(row["quality"]) for row in rows],
            label=regime,
            s=38,
        )
    ax.set_xscale("log")
    ax.set_title("Inherited <1M calibration, not a size law")
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("Controlled evidence-only accuracy")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.22)
    _save(fig, figures, "quality_vs_model_parameters")

    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    coverage = np.array([[1, 0, 0, 0]], dtype=float)
    ax.imshow(coverage, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["0.45M", "270M", "1B", "4B"])
    ax.set_yticks([0], ["matched memory ladder"])
    ax.text(0, 0, "pilot", ha="center", va="center")
    for x in (1, 2, 3):
        ax.text(x, 0, "pending", ha="center", va="center")
    ax.set_title("Parameter x logical-memory measurement coverage")
    _save(fig, figures, "parameter_memory_coverage")

    threshold = "0.06"
    adaptive_points = grouped_mean(
        adaptive,
        ["logical_tokens"],
        "expected_active_kv_tokens",
        lambda row: abs(float(row["threshold"]) - float(threshold)) < 1e-9,
    )
    _error_plot(
        adaptive_points,
        x_key="logical_tokens",
        title="Oracle-free adaptive effort (threshold 0.06)",
        xlabel="Logical reference tokens",
        ylabel="Expected active K/V tokens",
        directory=figures,
        name="adaptive_effort_vs_logical_memory",
        log_x=True,
        color="#C64B3C",
    )

    max_serving = max(int(row["logical_tokens"]) for row in serving)
    serving_points = grouped_mean(
        serving,
        ["concurrency"],
        "routing_queries_per_second",
        lambda row: int(row["logical_tokens"]) == max_serving,
    )
    _error_plot(
        serving_points,
        x_key="concurrency",
        title=f"Routing throughput at {max_serving:,} logical tokens",
        xlabel="Query batch size",
        ylabel="Routing queries/s",
        directory=figures,
        name="throughput_vs_concurrency",
        log_x=True,
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Cost per million tokens was not measured.\nNo cost-quality curve is claimed.",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("Cost measurement status")
    _save(fig, figures, "cost_vs_quality_status")

    frontier_source = [
        {
            "backend": row["backend"],
            "budget": int(row["requested_active_kv_tokens"]),
            "quality": float(row["evidence_recall"]),
            "latency": float(row["search_latency_p50_seconds"]),
        }
        for row in active
        if int(row["logical_tokens"]) == max_memory and row["backend"] == "exact_gemm"
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.scatter(
        [row["latency"] * 1000 for row in frontier_source],
        [row["quality"] for row in frontier_source],
        c=[row["budget"] for row in frontier_source],
        cmap="viridis",
    )
    ax.set_title("Measured routing Pareto candidates")
    ax.set_xlabel("Median search latency (ms)")
    ax.set_ylabel("Evidence recall")
    ax.grid(alpha=0.22)
    _save(fig, figures, "pra_routing_pareto_frontier")

    training_models = ("frozen", "consumer_lora", "interface_lora", "broad_lora", "full_weight", "native_scratch")
    model_quality = {
        row["checkpoint"].replace("paper4_tier0_", ""): float(row["quality"])
        for row in measured_models
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(
        range(len(training_models)),
        [model_quality[name] for name in training_models],
        color=["#777777", "#167D9A", "#3B8D6D", "#7A6CA8", "#C64B3C", "#D4A72C"],
    )
    ax.set_xticks(range(len(training_models)), ["Frozen", "Consumer\nLoRA", "Interface\nLoRA", "Broad\nLoRA", "Full", "PRA-native"])
    ax.set_ylabel("Evidence-only accuracy")
    ax.set_title("Inherited Paper 4 plasticity calibration")
    ax.grid(axis="y", alpha=0.22)
    _save(fig, figures, "plasticity_ladder")

    first = [row for row in dispersion if row["first_budget_meeting_target"] == "True"]
    dispersion_points = grouped_mean(first, ["evidence_nodes"], "active_native_kv_tokens")
    _error_plot(
        dispersion_points,
        x_key="evidence_nodes",
        title="Physical state follows evidence working set",
        xlabel="Evidence nodes (regions x depth)",
        ylabel="Minimum active K/V tokens for >=90% recall",
        directory=figures,
        name="evidence_dispersion_vs_required_kv",
        log_x=True,
        color="#C64B3C",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper5_scaling",
    )
    args = parser.parse_args()
    output = args.results_dir.resolve()

    logical = read_csv(output / "logical_memory_scaling.csv")
    active = read_csv(output / "active_kv_scaling.csv")
    search = read_csv(output / "search_index_scaling.csv")
    adaptive = read_csv(output / "adaptive_effort_scaling.csv")
    dispersion = read_csv(output / "evidence_dispersion_scaling.csv")
    serving = read_csv(output / "serving_scaling.csv")
    models = read_csv(output / "model_scaling_runs.csv")
    training = read_csv(output / "training_compute_scaling.csv")

    exact = lambda row: row["backend"] == "exact_gemm"
    coarse = lambda row: row["backend"] == "coarse_to_fine"
    fit_specs = {
        "quality_vs_logical_memory_exact": grouped_mean(logical, ["logical_tokens"], "evidence_recall", exact),
        "active_kv_vs_logical_memory_exact": grouped_mean(logical, ["logical_tokens"], "active_native_kv_tokens", exact),
        "search_latency_vs_logical_memory_exact": grouped_mean(logical, ["logical_tokens"], "search_latency_p50_seconds", exact),
        "search_latency_vs_logical_memory_indexed_p4": grouped_mean(logical, ["logical_tokens"], "search_latency_p50_seconds", coarse),
    }
    fits = {name: _fit_series(name, points) for name, points in fit_specs.items()}
    (output / "scaling_fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    diagnostics = []
    for series, result in fits.items():
        for fit in result["fits"]:
            diagnostics.append(
                {
                    "series": series,
                    "family": fit["family"],
                    "rmse": fit["rmse"],
                    "r_squared": fit["r_squared"],
                    "aic": fit["aic"],
                    "n": fit["n"],
                    "parameter_count": fit["parameter_count"],
                    "parameters_json": json.dumps(fit["parameters"], sort_keys=True),
                    "residuals_json": json.dumps(fit["residuals"]),
                }
            )
    write_csv(output / "scaling_fit_diagnostics.csv", diagnostics)

    max_memory = max(int(row["logical_tokens"]) for row in active)
    frontier_candidates = []
    for row in active:
        if int(row["logical_tokens"]) != max_memory:
            continue
        frontier_candidates.append(
            {
                "seed": row["seed"],
                "backend": row["backend"],
                "probes": row["probes"],
                "active_native_kv_tokens": float(row["active_native_kv_tokens"]),
                "evidence_recall": float(row["evidence_recall"]),
                "search_latency_p50_seconds": float(row["search_latency_p50_seconds"]),
                "measurement_scope": "controlled measured routing frontier",
            }
        )
    frontier = pareto_frontier(
        frontier_candidates,
        maximize=["evidence_recall"],
        minimize=["active_native_kv_tokens", "search_latency_p50_seconds"],
    )
    write_csv(output / "pareto_frontiers.csv", frontier)

    exact_quality = fit_specs["quality_vs_logical_memory_exact"]
    exact_latency = fit_specs["search_latency_vs_logical_memory_exact"]
    indexed_latency = fit_specs["search_latency_vs_logical_memory_indexed_p4"]
    max_ratio_rows = [row for row in logical if int(row["logical_tokens"]) == max_memory and exact(row)]
    max_index_rows = [
        row
        for row in search
        if int(row["logical_tokens"]) == max_memory and int(row["probes"]) == 4
    ]
    quality_means = [row["mean"] for row in exact_quality]
    exact_growth = exact_latency[-1]["mean"] / max(exact_latency[0]["mean"], 1e-12)
    indexed_growth = indexed_latency[-1]["mean"] / max(indexed_latency[0]["mean"], 1e-12)
    indexed_over_exact = indexed_latency[-1]["mean"] / max(exact_latency[-1]["mean"], 1e-12)
    exact_comparisons = statistics.fmean(
        float(row["comparisons"])
        for row in max_ratio_rows
    )
    indexed_comparisons = statistics.fmean(float(row["comparisons"]) for row in max_index_rows)
    exact_overlap = statistics.fmean(float(row["recall_at_k_vs_exact"]) for row in max_index_rows)
    max_index_mib = statistics.fmean(float(row["routing_index_bytes"]) for row in max_index_rows) / 2**20
    findings = {
        "status": "controlled_pilot_complete",
        "seed_count": len({row["seed"] for row in logical}),
        "logical_memory_min_tokens": min(int(row["logical_tokens"]) for row in logical),
        "logical_memory_max_tokens": max_memory,
        "primary_active_kv_tokens": int(float(max_ratio_rows[0]["active_native_kv_tokens"])),
        "logical_to_active_ratio_at_max": statistics.fmean(float(row["logical_to_active_ratio"]) for row in max_ratio_rows),
        "exact_evidence_recall_min": min(quality_means),
        "exact_evidence_recall_max": max(quality_means),
        "exact_search_latency_growth": exact_growth,
        "indexed_search_latency_growth": indexed_growth,
        "indexed_over_exact_latency_at_max": indexed_over_exact,
        "indexed_speedup_over_exact_at_max": 1.0 / indexed_over_exact,
        "indexed_comparison_reduction_at_max": exact_comparisons / indexed_comparisons,
        "indexed_topk_overlap_at_max": exact_overlap,
        "routing_index_mib_at_max": max_index_mib,
        "best_fit_families": {name: result["best_family_by_aic"] for name, result in fits.items()},
        "cost_measured": False,
        "end_to_end_lm_quality_measured": False,
        "gemma_ladder_measured": False,
        "claim": "At fixed synthetic retrieval difficulty, logical memory increased while selected native-K/V stayed fixed and retrieval recall was stable.",
        "limitation": "The result is a controlled routing/systems pilot, not a language-model scaling law or production ANN benchmark.",
    }
    (output / "paper5_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    audit = f"""# Paper 5 Scaling Claim Audit

## Supported by this branch

- **Measured:** five-seed controlled routing from {findings['logical_memory_min_tokens']:,} to {max_memory:,} logical tokens.
- **Measured:** exact and coarse-to-fine retrieval, active native-K/V tokens, layer-token K/V, index bytes, search latency, routing-only concurrency, and CPU/CUDA hardware slices.
- **Measured:** fixed-difficulty evidence recall remained in [{findings['exact_evidence_recall_min']:.3f}, {findings['exact_evidence_recall_max']:.3f}] at {findings['primary_active_kv_tokens']} active tokens.
- **Measured:** evidence-dispersion sweeps vary regions and chain depth separately from logical address-space size.
- **Inherited calibration only:** Paper 4 Frozen, LoRA, full-weight, and PRA-native controlled consumer-learning results.

## Not supported yet

- No claim of infinite context, bounded total runtime, or production ANN efficiency.
- No matched Gemma 270M/1B/4B quality curve and no smaller-PRA-versus-larger-native claim.
- No end-to-end NLL, perplexity, EM/F1, TTFT, TPOT, generation throughput, dollar cost, or production HBM curve.
- No Apple Silicon run and no 4B+ run. These remain preregistered ladder cells.
- Analytical native/RAG/PRA state rows are accounting baselines, not measured quality or latency.

## Interpretation rule

The controlled result tests whether selected physical attention state can track a fixed task working set while the addressable pool grows. GPU-resident routing-index bytes still grow with memory, and the current Python coarse-to-fine path can be slower than exact GPU GEMM. Both are scaling costs, not exceptions to hide.
"""
    (output / "scaling_claim_audit.md").write_text(audit, encoding="utf-8")

    tex = f"""% Generated by summarize_scaling_study.py; do not edit manually.
\\newcommand{{\\PaperFiveSeedCount}}{{{findings['seed_count']}}}
\\newcommand{{\\PaperFiveLogicalMin}}{{{findings['logical_memory_min_tokens']:,}}}
\\newcommand{{\\PaperFiveLogicalMax}}{{{findings['logical_memory_max_tokens']:,}}}
\\newcommand{{\\PaperFiveActiveTokens}}{{{findings['primary_active_kv_tokens']}}}
\\newcommand{{\\PaperFiveLogicalRatio}}{{{findings['logical_to_active_ratio_at_max']:,.0f}}}
\\newcommand{{\\PaperFiveRecallMin}}{{{findings['exact_evidence_recall_min']:.3f}}}
\\newcommand{{\\PaperFiveRecallMax}}{{{findings['exact_evidence_recall_max']:.3f}}}
\\newcommand{{\\PaperFiveExactLatencyGrowth}}{{{findings['exact_search_latency_growth']:.2f}}}
\\newcommand{{\\PaperFiveIndexedLatencyGrowth}}{{{findings['indexed_search_latency_growth']:.2f}}}
\\newcommand{{\\PaperFiveIndexedOverExact}}{{{findings['indexed_over_exact_latency_at_max']:.2f}}}
\\newcommand{{\\PaperFiveIndexedSpeedup}}{{{findings['indexed_speedup_over_exact_at_max']:.2f}}}
\\newcommand{{\\PaperFiveComparisonReduction}}{{{findings['indexed_comparison_reduction_at_max']:.1f}}}
\\newcommand{{\\PaperFiveExactOverlap}}{{{findings['indexed_topk_overlap_at_max']:.3f}}}
\\newcommand{{\\PaperFiveIndexMiB}}{{{findings['routing_index_mib_at_max']:.1f}}}
"""
    (output / "generated_results.tex").write_text(tex, encoding="utf-8")

    logical_table_rows = []
    exact_by_memory = {int(row["logical_tokens"]): row for row in exact_quality}
    exact_latency_by_memory = {int(row["logical_tokens"]): row for row in exact_latency}
    indexed_latency_by_memory = {int(row["logical_tokens"]): row for row in indexed_latency}
    for memory in sorted(exact_by_memory):
        sample = next(row for row in logical if int(row["logical_tokens"]) == memory and exact(row))
        logical_table_rows.append(
            f"{memory:,} & {int(sample['memory_nodes']):,} & "
            f"{exact_by_memory[memory]['mean']:.3f} & {int(float(sample['active_native_kv_tokens']))} & "
            f"{float(sample['logical_to_active_ratio']):,.0f} & "
            f"{1000 * exact_latency_by_memory[memory]['mean']:.3f} & "
            f"{1000 * indexed_latency_by_memory[memory]['mean']:.3f} & "
            f"{float(sample['routing_index_bytes']) / 2**20:.2f} \\\\"
        )
    first_budget = [row for row in dispersion if row["first_budget_meeting_target"] == "True"]
    dispersion_summary = grouped_mean(first_budget, ["evidence_nodes"], "active_native_kv_tokens")
    dispersion_table_rows = [
        f"{int(row['evidence_nodes'])} & {row['mean']:.1f} & {row['ci95']:.1f} & {int(row['n'])} \\\\"
        for row in dispersion_summary
    ]
    tables_tex = (
        "% Generated by summarize_scaling_study.py; do not edit manually.\n"
        "\\begin{tabular}{rrrrrrrr}\n"
        "\\toprule\nLogical tokens & Nodes & Recall & Active K/V & Ratio & Exact ms & Indexed ms & Index MiB \\\\\n"
        "\\midrule\n"
        + "\n".join(logical_table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\begin{tabular}{rrrr}\n\\toprule\nEvidence nodes & Required K/V & 95\\% CI & Cells \\\\\n"
        "\\midrule\n"
        + "\n".join(dispersion_table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (output / "generated_tables.tex").write_text(tables_tex, encoding="utf-8")
    (output / "logical_memory_table.tex").write_text(
        "\\begin{tabular}{rrrrrrrr}\n"
        "\\toprule\nLogical tokens & Nodes & Recall & Active K/V & Ratio & Exact ms & Indexed ms & Index MiB \\\\\n"
        "\\midrule\n"
        + "\n".join(logical_table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    (output / "dispersion_table.tex").write_text(
        "\\begin{tabular}{rrrr}\n\\toprule\nEvidence nodes & Required K/V & 95\\% CI & Cells \\\\\n"
        "\\midrule\n"
        + "\n".join(dispersion_table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    make_plots(output, logical, active, adaptive, dispersion, serving, models, training)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
