"""Summarize Paper 3.1 multi-index rows into publication artifacts."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper3_1_summary_index/multi_index"
DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa", "musique")
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "qasper": "QASPER",
    "2wikimultihopqa": "2Wiki",
    "musique": "MuSiQue",
}
COLORS = {
    "L": "#2F6690",
    "E": "#3A7D44",
    "S": "#C56A1A",
    "QK": "#7A5195",
    "multi_channel": "#4C956C",
    "missed_by_all": "#9A9A9A",
}
PRIMARY_K = 4
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260826


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty artifact: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: Sequence[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate(rows: Iterable[dict[str, str]]) -> list[dict]:
    groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["family"],
            row["policy"],
            row["channels"],
            row["parameter"],
            int(row["k_total"]),
        )
        groups[key].append(row)
    output = []
    for key, local in sorted(groups.items()):
        output.append(
            {
                "dataset": key[0],
                "family": key[1],
                "policy": key[2],
                "channels": key[3],
                "parameter": key[4],
                "k_total": key[5],
                "n": len(local),
                "evidence_recall": statistics.fmean(float(row["evidence_recall"]) for row in local),
                "complete_recovery": statistics.fmean(float(row["complete_recovery"]) for row in local),
                "precision": statistics.fmean(float(row["precision"]) for row in local),
                "reciprocal_rank": statistics.fmean(float(row["reciprocal_rank"]) for row in local),
                "routing_index_bytes": statistics.fmean(float(row["routing_index_bytes"]) for row in local),
                "routing_seconds": statistics.fmean(float(row["routing_seconds"]) for row in local),
                "native_kv_tokens_materialized": statistics.fmean(
                    float(row["native_kv_tokens_materialized"]) for row in local
                ),
            }
        )
    return output


def _ablation_rows(all_rows: Sequence[dict[str, str]]) -> list[dict]:
    primary = [row for row in all_rows if int(row["k_total"]) == PRIMARY_K]
    by_key = {
        (row["dataset"], row["example_id"], row["family"], row["channels"]): row
        for row in primary
    }
    families = ("union_round_robin", "union_agreement", "rrf", "normalized_fusion")
    ablations = {
        "L": "E+S+QK",
        "E": "L+S+QK",
        "S": "L+E+QK",
        "QK": "L+E+S",
    }
    output = []
    for dataset_index, dataset in enumerate(DATASETS):
        identities = sorted({row["example_id"] for row in primary if row["dataset"] == dataset})
        for family_index, family in enumerate(families):
            for removed_index, (removed, ablated_channels) in enumerate(ablations.items()):
                deltas = []
                full_values = []
                ablated_values = []
                for identity in identities:
                    full = by_key.get((dataset, identity, family, "L+E+S+QK"))
                    ablated = by_key.get((dataset, identity, family, ablated_channels))
                    if full is None or ablated is None:
                        continue
                    full_value = float(full["evidence_recall"])
                    ablated_value = float(ablated["evidence_recall"])
                    full_values.append(full_value)
                    ablated_values.append(ablated_value)
                    deltas.append(full_value - ablated_value)
                low, high = _bootstrap(
                    deltas,
                    BOOTSTRAP_SEED + dataset_index * 100 + family_index * 10 + removed_index,
                )
                output.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "k_total": PRIMARY_K,
                        "full_channels": "L+E+S+QK",
                        "removed_channel": removed,
                        "ablated_channels": ablated_channels,
                        "n": len(deltas),
                        "full_recall": statistics.fmean(full_values),
                        "ablated_recall": statistics.fmean(ablated_values),
                        "marginal_delta": statistics.fmean(deltas),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "wins": sum(value > 0 for value in deltas),
                        "ties": sum(value == 0 for value in deltas),
                        "losses": sum(value < 0 for value in deltas),
                        "resolved_positive": low > 0,
                        "unit": "held_out_identity",
                    }
                )
    return output


def _paired_policy_effects(
    combined_rows: Sequence[dict[str, str]],
    union_rows: Sequence[dict[str, str]],
) -> list[dict]:
    singles = {
        (row["dataset"], row["example_id"], int(row["k_total"]), row["channels"]): row
        for row in union_rows
        if row["family"] == "single" and row["channels"] in {"L", "E", "S", "QK"}
    }
    groups: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for row in combined_rows:
        if row["family"] == "single":
            continue
        for reference in row["channels"].split("+"):
            baseline = singles.get(
                (row["dataset"], row["example_id"], int(row["k_total"]), reference)
            )
            if baseline is not None:
                key = (
                    row["dataset"],
                    row["family"],
                    row["policy"],
                    row["channels"],
                    row["parameter"],
                    int(row["k_total"]),
                    reference,
                )
                groups[key].append(
                    (float(row["evidence_recall"]), float(baseline["evidence_recall"]))
                )
    output = []
    for index, (key, values) in enumerate(sorted(groups.items())):
        deltas = [combined - reference for combined, reference in values]
        low, high = _bootstrap(deltas, BOOTSTRAP_SEED + 1000 + index)
        output.append(
            {
                "dataset": key[0],
                "family": key[1],
                "policy": key[2],
                "channels": key[3],
                "parameter": key[4],
                "k_total": key[5],
                "reference_channel": key[6],
                "n": len(values),
                "combined_recall": statistics.fmean(value[0] for value in values),
                "reference_recall": statistics.fmean(value[1] for value in values),
                "paired_delta": statistics.fmean(deltas),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wins": sum(value > 0 for value in deltas),
                "ties": sum(value == 0 for value in deltas),
                "losses": sum(value < 0 for value in deltas),
                "resolved_positive": low > 0,
                "unit": "held_out_identity",
            }
        )
    return output


def _cost_summary(rows: Sequence[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["scope"], row["address_views"])].append(row)
    summary = []
    for key, local in sorted(groups.items()):
        summary.append(
            {
                "dataset": key[0],
                "scope": key[1],
                "address_views": key[2],
                "n": len(local),
                "persistent_bytes": statistics.fmean(float(row["persistent_bytes"]) for row in local),
                "ingestion_seconds": statistics.fmean(float(row["ingestion_seconds"]) for row in local),
                "routing_seconds": statistics.fmean(float(row["routing_seconds"]) for row in local),
                "summary_generation_seconds": statistics.fmean(
                    float(row["summary_generation_seconds"]) for row in local
                ),
                "summary_embedding_seconds": statistics.fmean(
                    float(row["summary_embedding_seconds"]) for row in local
                ),
                "shared_backing_memory_copies": 1,
            }
        )
    amortization = []
    lookup = {(row["dataset"], row["address_views"]): row for row in summary}
    for dataset in DATASETS:
        base = lookup[(dataset, "L+E+QK")]
        full = lookup[(dataset, "L+E+S+QK")]
        for queries in (1, 10, 100, 1000):
            amortization.append(
                {
                    "dataset": dataset,
                    "queries_per_ingestion": queries,
                    "base_stack_bytes": base["persistent_bytes"],
                    "full_stack_bytes": full["persistent_bytes"],
                    "summary_increment_bytes": full["persistent_bytes"] - base["persistent_bytes"],
                    "summary_increment_ingestion_seconds_per_query": (
                        full["ingestion_seconds"] - base["ingestion_seconds"]
                    )
                    / queries,
                    "shared_backing_memory_copies": 1,
                }
            )
    return summary, amortization


def _plot_overlap(hits: Sequence[dict[str, str]]) -> None:
    categories = ("L_only", "E_only", "S_only", "QK_only", "multi_channel", "missed_by_all")
    figure, axes = plt.subplots(1, 4, figsize=(11.5, 3.3), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        local = [row for row in hits if row["dataset"] == dataset and row["is_evidence"] == "True"]
        values = [sum(row["recovery_type"] == category for row in local) for category in categories]
        axis.bar(
            range(len(categories)),
            values,
            color=[COLORS.get(category.removesuffix("_only"), COLORS.get(category, "#777777")) for category in categories],
        )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xticks(range(len(categories)), ("L", "E", "S", "QK", "multi", "miss"), rotation=45)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Evidence parents")
    figure.tight_layout()
    figure.savefig(RESULTS / "multi_index_channel_overlap.png", dpi=190)
    figure.savefig(RESULTS / "multi_index_channel_overlap.pdf")
    plt.close(figure)


def _plot_frontier(summary: Sequence[dict]) -> None:
    conditions = (
        ("single", "L", "L"),
        ("single", "E", "E"),
        ("single", "S", "S"),
        ("single", "QK", "QK"),
        ("rrf", "L+S:RRF", "L+S"),
        ("rrf", "L+QK:RRF", "L+QK"),
        ("rrf", "L+E+QK:RRF", "L+E+QK"),
        ("rrf", "L+E+S+QK:RRF", "all four"),
    )
    styles = (
        ("#1f77b4", "o"),
        ("#ff7f0e", "s"),
        ("#2ca02c", "^"),
        ("#d62728", "D"),
        ("#9467bd", "P"),
        ("#8c564b", "X"),
        ("#17becf", "v"),
        ("#222222", "*"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.7, 7.0))
    for axis, dataset in zip(axes.flat, DATASETS):
        for (family, policy, label), (color, marker) in zip(conditions, styles):
            row = next(
                (
                    row
                    for row in summary
                    if row["dataset"] == dataset
                    and row["family"] == family
                    and row["policy"] == policy
                    and row["k_total"] == PRIMARY_K
                ),
                None,
            )
            if row:
                axis.scatter(
                    row["routing_index_bytes"],
                    row["evidence_recall"],
                    color=color,
                    marker=marker,
                    s=42,
                    label=label if axis is axes.flat[0] else None,
                )
        axis.set_xscale("log")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Persistent address-index bytes/source")
    axes[1, 1].set_xlabel("Persistent address-index bytes/source")
    axes[0, 0].set_ylabel("Evidence recall@4")
    axes[1, 0].set_ylabel("Evidence recall@4")
    figure.legend(loc="upper center", ncol=4, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(RESULTS / "multi_index_recall_cost_frontier.png", dpi=190)
    figure.savefig(RESULTS / "multi_index_recall_cost_frontier.pdf")
    plt.close(figure)


def _plot_summary_marginal(ablation: Sequence[dict]) -> None:
    families = ("union_round_robin", "union_agreement", "rrf", "normalized_fusion")
    labels = ("round-robin", "agreement", "RRF", "score fusion")
    figure, axis = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(DATASETS))
    width = 0.18
    for offset, (family, label) in enumerate(zip(families, labels)):
        values = [
            next(
                row["marginal_delta"]
                for row in ablation
                if row["dataset"] == dataset
                and row["family"] == family
                and row["removed_channel"] == "S"
            )
            for dataset in DATASETS
        ]
        axis.bar(x + (offset - 1.5) * width, values, width, label=label)
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set_xticks(x, ("Hotpot", "QASPER", "2Wiki", "MuSiQue"))
    axis.set_ylabel("Summary marginal recall at fixed K=4")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS / "summary_marginal_contribution.png", dpi=190)
    figure.savefig(RESULTS / "summary_marginal_contribution.pdf")
    plt.close(figure)


def _plot_coverage(rows: Sequence[dict[str, str]]) -> None:
    metrics = ("entity_recall", "number_date_recall", "rare_term_recall", "relation_term_recall", "evidence_key_recall")
    labels = ("entity", "number/date", "rare term", "relation", "evidence key")
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(metrics))
    width = 0.19
    for offset, dataset in enumerate(DATASETS):
        values = []
        for metric in metrics:
            available = [float(row[metric]) for row in rows if row["dataset"] == dataset and row[metric] != ""]
            values.append(statistics.fmean(available) if available else 0.0)
        axis.bar(x + (offset - 1.5) * width, values, width, label=DATASET_LABELS[dataset])
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Summary address-key retention")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS / "summary_address_retention.png", dpi=190)
    figure.savefig(RESULTS / "summary_address_retention.pdf")
    plt.close(figure)


def _plot_correlations(rows: Sequence[dict[str, str]]) -> None:
    local = [
        row
        for row in rows
        if row["dataset"] == "pooled"
        and row["routing_outcome"] == "summary_target_recovered_at4"
        and row["spearman_rho"] != ""
    ]
    figure, axis = plt.subplots(figsize=(7.8, 3.8))
    labels = [row["diagnostic"].replace("_recall", "").replace("_", " ") for row in local]
    values = [float(row["spearman_rho"]) for row in local]
    axis.bar(range(len(local)), values, color="#4C956C")
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set_xticks(range(len(local)), labels, rotation=25, ha="right")
    axis.set_ylabel("Spearman rho with target recovered@4")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS / "summary_quality_retrieval_correlation.png", dpi=190)
    figure.savefig(RESULTS / "summary_quality_retrieval_correlation.pdf")
    plt.close(figure)


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _write_tex(summary: Sequence[dict], ablation: Sequence[dict], costs: Sequence[dict]) -> None:
    primary_policies = (
        ("single", "L", "L"),
        ("single", "E", "E"),
        ("single", "S", "S"),
        ("single", "QK", "QK"),
        ("rrf", "L+S:RRF", "L+S (RRF)"),
        ("rrf", "L+QK:RRF", "L+QK (RRF)"),
        ("rrf", "L+E+QK:RRF", "L+E+QK (RRF)"),
        ("rrf", "L+E+S+QK:RRF", "L+E+S+QK (RRF)"),
    )
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Policy & HotpotQA & QASPER & 2Wiki & MuSiQue \\", r"\midrule"]
    for family, policy, label in primary_policies:
        values = []
        for dataset in DATASETS:
            row = next(
                row
                for row in summary
                if row["dataset"] == dataset
                and row["family"] == family
                and row["policy"] == policy
                and row["k_total"] == PRIMARY_K
            )
            values.append(row["evidence_recall"])
        lines.append(f"{_tex_escape(label)} & " + " & ".join(f"{value:.3f}" for value in values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (RESULTS / "generated_multi_index_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_rows = [row for row in ablation if row["family"] == "rrf" and row["removed_channel"] == "S"]
    marginal_lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Dataset & Base & +Summary & $\Delta_S$ & 95\% CI \\", r"\midrule"]
    for row in summary_rows:
        marginal_lines.append(
            f"{DATASET_LABELS[row['dataset']]} & {row['ablated_recall']:.3f} & {row['full_recall']:.3f} & "
            f"{row['marginal_delta']:+.3f} & [{row['bootstrap_ci_low']:+.3f}, {row['bootstrap_ci_high']:+.3f}] " + r"\\"
        )
    marginal_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (RESULTS / "generated_summary_marginal_table.tex").write_text(
        "\n".join(marginal_lines) + "\n", encoding="utf-8"
    )

    cost_lookup = {(row["dataset"], row["address_views"]): row for row in costs}
    cost_lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Dataset & Base bytes & +Summary bytes & Summary ingestion (s) \\", r"\midrule"]
    for dataset in DATASETS:
        base = cost_lookup[(dataset, "L+E+QK")]
        full = cost_lookup[(dataset, "L+E+S+QK")]
        cost_lines.append(
            f"{DATASET_LABELS[dataset]} & {base['persistent_bytes']:.0f} & {full['persistent_bytes']:.0f} & "
            f"{full['summary_generation_seconds'] + full['summary_embedding_seconds']:.1f} " + r"\\"
        )
    cost_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (RESULTS / "generated_multi_index_cost_table.tex").write_text(
        "\n".join(cost_lines) + "\n", encoding="utf-8"
    )


def _write_macros(
    ablation: Sequence[dict],
    paired_effects: Sequence[dict],
    hits: Sequence[dict[str, str]],
    coverage: Sequence[dict[str, str]],
    correlations: Sequence[dict[str, str]],
) -> None:
    def ablation_row(dataset: str, family: str) -> dict:
        return next(
            row
            for row in ablation
            if row["dataset"] == dataset
            and row["family"] == family
            and row["removed_channel"] == "S"
        )

    def command(name: str, value: str | int | float) -> str:
        return rf"\newcommand{{\{name}}}{{{value}}}"

    unique_summary = {
        dataset: sum(
            row["dataset"] == dataset
            and row["is_evidence"] == "True"
            and row["recovery_type"] == "S_only"
            for row in hits
        )
        for dataset in DATASETS
    }
    evidence_values = [
        float(row["evidence_key_recall"])
        for row in coverage
        if row["evidence_key_recall"] != ""
    ]
    evidence_correlation = next(
        row
        for row in correlations
        if row["dataset"] == "pooled"
        and row["diagnostic"] == "evidence_key_recall"
        and row["routing_outcome"] == "summary_target_recovered_at4"
    )
    musique_reserved = next(
        row
        for row in paired_effects
        if row["dataset"] == "musique"
        and row["policy"] == "L+S:L2_S2"
        and row["reference_channel"] == "L"
    )
    lines = [
        command("PaperThreeOneMultiValidationN", 8),
        command("PaperThreeOneMultiTestN", 24),
        command("PaperThreeOneMultiK", 4),
        command("PaperThreeOneSummaryUniqueQasper", unique_summary["qasper"]),
        command("PaperThreeOneSummaryUniqueMusique", unique_summary["musique"]),
        command("PaperThreeOneEvidenceKeyRetention", f"{statistics.fmean(evidence_values):.3f}"),
        command("PaperThreeOneEvidenceKeyCorrelation", f"{float(evidence_correlation['spearman_rho']):.3f}"),
        command("PaperThreeOneMusiqueReservedRecall", f"{musique_reserved['combined_recall']:.3f}"),
        command("PaperThreeOneMusiqueReservedDelta", f"{musique_reserved['paired_delta']:+.3f}"),
        command("PaperThreeOneMusiqueReservedLow", f"{musique_reserved['bootstrap_ci_low']:+.3f}"),
        command("PaperThreeOneMusiqueReservedHigh", f"{musique_reserved['bootstrap_ci_high']:+.3f}"),
    ]
    macro_dataset = {
        "hotpotqa": "Hotpot",
        "qasper": "Qasper",
        "2wikimultihopqa": "TwoWiki",
        "musique": "Musique",
    }
    macro_family = {
        "rrf": "Rrf",
        "union_round_robin": "RoundRobin",
        "union_agreement": "Agreement",
        "normalized_fusion": "Fusion",
    }
    for dataset in DATASETS:
        for family in macro_family:
            row = ablation_row(dataset, family)
            prefix = f"PaperThreeOne{macro_dataset[dataset]}{macro_family[family]}"
            lines.extend(
                (
                    command(prefix + "Base", f"{row['ablated_recall']:.3f}"),
                    command(prefix + "Full", f"{row['full_recall']:.3f}"),
                    command(prefix + "SummaryDelta", f"{row['marginal_delta']:+.3f}"),
                    command(prefix + "Low", f"{row['bootstrap_ci_low']:+.3f}"),
                    command(prefix + "High", f"{row['bootstrap_ci_high']:+.3f}"),
                    command(prefix + "Wins", row["wins"]),
                    command(prefix + "Ties", row["ties"]),
                    command(prefix + "Losses", row["losses"]),
                )
            )
    (RESULTS / "generated_multi_index_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> dict:
    union = _read_csv(RESULTS / "multi_index_union_results.csv")
    reserved = _read_csv(RESULTS / "multi_index_reserved_slot_results.csv")
    rrf = _read_csv(RESULTS / "multi_index_rrf_results.csv")
    fusion = _read_csv(RESULTS / "multi_index_fusion_results.csv")
    all_rows = [*union, *reserved, *rrf, *fusion]
    summary = _aggregate(all_rows)
    ablation = _ablation_rows(all_rows)
    paired_effects = _paired_policy_effects(all_rows, union)
    costs, amortization = _cost_summary(_read_csv(RESULTS / "multi_index_costs.csv"))
    hits = _read_csv(RESULTS / "multi_index_channel_hits.csv")
    coverage = _read_csv(RESULTS / "summary_address_coverage.csv")
    correlations = _read_csv(RESULTS / "summary_quality_retrieval_correlation.csv")

    _write_csv(RESULTS / "multi_index_summary.csv", summary)
    _write_csv(RESULTS / "multi_index_ablation.csv", ablation)
    _write_csv(RESULTS / "multi_index_paired_effects.csv", paired_effects)
    _write_csv(RESULTS / "multi_index_cost_summary.csv", costs)
    _write_csv(RESULTS / "multi_index_amortization.csv", amortization)
    _plot_overlap(hits)
    _plot_frontier(summary)
    _plot_summary_marginal(ablation)
    _plot_coverage(coverage)
    _plot_correlations(correlations)
    _write_tex(summary, ablation, costs)
    _write_macros(ablation, paired_effects, hits, coverage, correlations)

    summary_marginal = [row for row in ablation if row["family"] == "rrf" and row["removed_channel"] == "S"]
    findings = {
        "primary_method": "validation-frozen reciprocal rank fusion",
        "primary_k": PRIMARY_K,
        "summary_marginal_by_dataset": summary_marginal,
        "summary_positive_resolved_any_dataset": any(row["resolved_positive"] for row in summary_marginal),
        "summary_positive_resolved_any_method": any(
            row["removed_channel"] == "S" and row["resolved_positive"] for row in ablation
        ),
        "lora_gate_open": False,
        "downstream_generation_gate_open": False,
        "gate_reason": (
            "LoRA and downstream native-K/V generation remain closed unless summary marginal recall "
            "is positive with a paired interval above zero after L+E+QK."
        ),
        "rouge_run": False,
        "rouge_reason": "No independent reference summaries exist for these source chunks.",
    }
    (RESULTS / "multi_index_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2, sort_keys=True))
    return findings


if __name__ == "__main__":
    main()
