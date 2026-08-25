"""Summarize the five-seed callable-catalog scaling experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_scaling"
SIZES = (32, 128, 512, 2048, 8192)
QUALITY_POLICIES = (
    "A2_keywords_synonyms",
    "A5_embedding",
    "A6_auto_hybrid",
)
K10_STRATEGIES = (
    "A2_topk",
    "A5_topk",
    "A6_topk",
    "A7_fused_score",
    "A7_raw_union",
    "A7_diversity_union",
)


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(RESULTS / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(RESULTS / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def _macro(name: str, value: float, digits: int = 3) -> str:
    return f"\\newcommand{{\\{name}}}{{{value:.{digits}f}}}"


def _aggregate_quality(rows: pd.DataFrame) -> pd.DataFrame:
    seed = (
        rows.groupby(["catalog_size", "policy", "seed"], as_index=False)
        [["top1", "recall_at_3", "recall_at_5", "recall_at_10"]]
        .mean()
    )
    summary = seed.groupby(["catalog_size", "policy"])[
        ["top1", "recall_at_3", "recall_at_5", "recall_at_10"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _aggregate_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    seed = (
        rows.groupby(["catalog_size", "strategy", "max_candidates", "seed"], as_index=False)
        [["required_recall", "useful_precision", "unsafe_exposure", "context_tokens"]]
        .mean()
    )
    summary = seed.groupby(["catalog_size", "strategy", "max_candidates"])[
        ["required_recall", "useful_precision", "unsafe_exposure", "context_tokens"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _aggregate_costs(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "callable_generation_ms",
        "inspection_and_view_ms",
        "lexical_index_build_ms",
        "embedding_build_ms",
        "bm25_score_mean_ms",
        "a2_score_mean_ms",
        "a4_score_mean_ms",
        "embedding_score_amortized_ms",
        "a7_candidate_build_mean_ms",
        "lexical_index_bytes",
        "automatic_semantic_bytes",
        "embedding_bytes",
        "total_discovery_component_bytes",
        "logical_catalog_schema_tokens",
        "mean_schema_tokens",
    ]
    summary = rows.groupby("catalog_size")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _plot_quality(summary: pd.DataFrame) -> None:
    styles = {
        "A2_keywords_synonyms": ("A2 lexical + synonyms", "#247BA0", "o"),
        "A5_embedding": ("A5 compact embedding", "#6B5CA5", "s"),
        "A6_auto_hybrid": ("A6 frozen hybrid", "#D1495B", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), sharex=True)
    for policy, (label, color, marker) in styles.items():
        selected = summary[summary.policy == policy].sort_values("catalog_size")
        axes[0].plot(selected.catalog_size, selected.top1_mean, marker=marker, color=color, label=label)
        axes[1].plot(selected.catalog_size, selected.recall_at_10_mean, marker=marker, color=color, label=label)
    for axis, ylabel in zip(axes, ("Top-1", "Recall@10")):
        axis.set_xscale("log", base=2)
        axis.set_xticks(SIZES, ["32", "128", "512", "2K", "8K"])
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Callable catalog size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    _save("scaled_auto_quality")


def _plot_k10(summary: pd.DataFrame) -> None:
    labels = {
        "A2_topk": "A2 lexical",
        "A5_topk": "A5 embedding",
        "A6_topk": "A6 hybrid",
        "A7_fused_score": "A7 equal fusion",
        "A7_raw_union": "A7 raw union",
        "A7_diversity_union": "A7 diversity union",
    }
    colors = {
        "A2_topk": "#247BA0",
        "A5_topk": "#6B5CA5",
        "A6_topk": "#D1495B",
        "A7_fused_score": "#F2A541",
        "A7_raw_union": "#3B8C6E",
        "A7_diversity_union": "#444444",
    }
    selected = summary[summary.max_candidates == 10]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
    for strategy in K10_STRATEGIES:
        rows = selected[selected.strategy == strategy].sort_values("catalog_size")
        axes[0].plot(rows.catalog_size, rows.required_recall_mean, marker="o", color=colors[strategy], label=labels[strategy])
        axes[1].plot(rows.catalog_size, rows.unsafe_exposure_mean, marker="o", color=colors[strategy], label=labels[strategy])
    for axis, ylabel in zip(axes, ("Required-tool recall", "Unsafe-candidate exposure")):
        axis.set_xscale("log", base=2)
        axis.set_xticks(SIZES, ["32", "128", "512", "2K", "8K"])
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Callable catalog size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    _save("scaled_union_k10")


def _plot_frontier(summary: pd.DataFrame) -> None:
    selected = summary[summary.catalog_size == 8192]
    styles = {
        "A6_topk": ("A6 frozen hybrid", "#D1495B"),
        "A7_fused_score": ("A7 equal fusion", "#F2A541"),
        "A7_raw_union": ("A7 raw union", "#3B8C6E"),
        "A7_diversity_union": ("A7 diversity union", "#444444"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for strategy, (label, color) in styles.items():
        rows = selected[selected.strategy == strategy].sort_values("max_candidates")
        axes[0].plot(rows.max_candidates, rows.required_recall_mean, marker="o", color=color, label=label)
        axes[1].plot(rows.max_candidates, rows.useful_precision_mean, marker="o", color=color, label=label)
    for axis, ylabel in zip(axes, ("Required-tool recall", "Useful precision")):
        axis.set_xlabel("Maximum candidates K")
        axis.set_ylabel(ylabel)
        axis.set_xticks((2, 4, 6, 8, 10, 20))
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    _save("scaled_union_frontier_8192")


def _plot_costs(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65))
    axes[0].plot(summary.catalog_size, summary.bm25_score_mean_ms_mean, marker="o", label="BM25")
    axes[0].plot(summary.catalog_size, summary.a2_score_mean_ms_mean, marker="s", label="A2 lexical")
    axes[0].plot(summary.catalog_size, summary.a4_score_mean_ms_mean, marker="^", label="A4 inferred metadata")
    axes[0].plot(summary.catalog_size, summary.a7_candidate_build_mean_ms_mean, marker="d", label="A7 candidate build")
    axes[0].set_ylabel("Reference implementation latency (ms/query)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(summary.catalog_size, summary.total_discovery_component_bytes_mean / (1024 ** 2), marker="o", color="#3B8C6E")
    axes[1].set_ylabel("Discovery components (MiB)")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=10)
        axis.set_xticks(SIZES, ["32", "128", "512", "2K", "8K"])
        axis.set_xlabel("Callable catalog size")
        axis.grid(alpha=.25)
    _save("scaled_auto_costs")


def _write_table(quality: pd.DataFrame, candidates: pd.DataFrame) -> None:
    k10 = candidates[candidates.max_candidates == 10]
    lines = [
        "\\begin{tabular}{rcccccccc}",
        "\\toprule",
        "Catalog & A2 Top-1 & A5 Top-1 & A6 Top-1 & A6 R@10 & Union R & Union P & Unsafe & Tokens \\\\",
        "\\midrule",
    ]
    for size in SIZES:
        q = quality[quality.catalog_size == size].set_index("policy")
        u = k10[(k10.catalog_size == size) & (k10.strategy == "A7_raw_union")].iloc[0]
        size_label = f"{size:,}"
        lines.append(
            f"{size_label} & {q.loc['A2_keywords_synonyms', 'top1_mean']:.3f} & "
            f"{q.loc['A5_embedding', 'top1_mean']:.3f} & "
            f"{q.loc['A6_auto_hybrid', 'top1_mean']:.3f} & "
            f"{q.loc['A6_auto_hybrid', 'recall_at_10_mean']:.3f} & "
            f"{u.required_recall_mean:.3f} & {u.useful_precision_mean:.3f} & "
            f"{u.unsafe_exposure_mean:.3f} & {u.context_tokens_mean:.0f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (RESULTS / "generated_scaled_auto_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    quality_rows = pd.read_csv(RESULTS / "scaled_auto_quality_rows.csv")
    candidate_rows = pd.read_csv(RESULTS / "scaled_union_candidate_rows.csv")
    cost_rows = pd.read_csv(RESULTS / "scaled_discovery_costs.csv")

    expected = {
        "quality_rows": 5 * 5 * 144 * len(QUALITY_POLICIES),
        "candidate_rows": 5 * 5 * 144 * len(K10_STRATEGIES) * 8,
        "cost_rows": 5 * 5,
    }
    actual = {
        "quality_rows": len(quality_rows),
        "candidate_rows": len(candidate_rows),
        "cost_rows": len(cost_rows),
    }
    if actual != expected:
        raise ValueError(f"Incomplete scaling artifacts: expected {expected}, observed {actual}")
    if (candidate_rows.candidate_count > candidate_rows.max_candidates).any():
        raise ValueError("A candidate set exceeded its matched K budget")

    quality = _aggregate_quality(quality_rows)
    candidates = _aggregate_candidates(candidate_rows)
    costs = _aggregate_costs(cost_rows)
    quality.to_csv(RESULTS / "scaled_auto_quality_summary.csv", index=False)
    candidates.to_csv(RESULTS / "scaled_union_frontier.csv", index=False)
    costs.to_csv(RESULTS / "scaled_cost_summary.csv", index=False)

    _plot_quality(quality)
    _plot_k10(candidates)
    _plot_frontier(candidates)
    _plot_costs(costs)
    _write_table(quality, candidates)

    q = quality.set_index(["catalog_size", "policy"])
    c = candidates.set_index(["catalog_size", "strategy", "max_candidates"])
    cost = costs.set_index("catalog_size")
    macros = [
        _macro("PaperSixFiveScaleThirtyTwoHybridTopOne", q.loc[(32, "A6_auto_hybrid"), "top1_mean"]),
        _macro("PaperSixFiveScaleEightKHybridTopOne", q.loc[(8192, "A6_auto_hybrid"), "top1_mean"]),
        _macro("PaperSixFiveScaleEightKLexicalTopOne", q.loc[(8192, "A2_keywords_synonyms"), "top1_mean"]),
        _macro("PaperSixFiveScaleEightKHybridRecallTen", c.loc[(8192, "A6_topk", 10), "required_recall_mean"]),
        _macro("PaperSixFiveScaleEightKUnionRecallTen", c.loc[(8192, "A7_raw_union", 10), "required_recall_mean"]),
        _macro("PaperSixFiveScaleEightKUnionPrecisionTen", c.loc[(8192, "A7_raw_union", 10), "useful_precision_mean"]),
        _macro("PaperSixFiveScaleEightKUnionUnsafeTen", c.loc[(8192, "A7_raw_union", 10), "unsafe_exposure_mean"]),
        _macro("PaperSixFiveScaleEightKUnionTokensTen", c.loc[(8192, "A7_raw_union", 10), "context_tokens_mean"], 0),
        _macro("PaperSixFiveScaleEightKBMLatency", cost.loc[8192, "bm25_score_mean_ms_mean"], 1),
        _macro("PaperSixFiveScaleEightKASevenLatency", cost.loc[8192, "a7_candidate_build_mean_ms_mean"], 1),
        _macro("PaperSixFiveScaleEightKMemoryMiB", cost.loc[8192, "total_discovery_component_bytes_mean"] / (1024 ** 2), 1),
        _macro("PaperSixFiveScaleEightKLogicalTokens", cost.loc[8192, "logical_catalog_schema_tokens_mean"], 0),
    ]
    (RESULTS / "generated_scaled_auto_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    findings = {
        "protocol": {
            "catalog_sizes": list(SIZES),
            "catalog_seeds": sorted(int(value) for value in quality_rows.seed.unique()),
            "test_queries": int(quality_rows.query_id.nunique()),
            "candidate_budgets": sorted(int(value) for value in candidate_rows.max_candidates.unique()),
            "rows": actual,
        },
        "eight_k": {
            "a2_top1": q.loc[(8192, "A2_keywords_synonyms"), "top1_mean"],
            "a6_top1": q.loc[(8192, "A6_auto_hybrid"), "top1_mean"],
            "a6_recall_at_10": c.loc[(8192, "A6_topk", 10), "required_recall_mean"],
            "raw_union_recall_at_10": c.loc[(8192, "A7_raw_union", 10), "required_recall_mean"],
            "raw_union_useful_precision_at_10": c.loc[(8192, "A7_raw_union", 10), "useful_precision_mean"],
            "raw_union_unsafe_exposure_at_10": c.loc[(8192, "A7_raw_union", 10), "unsafe_exposure_mean"],
            "raw_union_context_tokens_at_10": c.loc[(8192, "A7_raw_union", 10), "context_tokens_mean"],
        },
        "interpretation": {
            "frozen_hybrid_advantage_ends_before": 2048,
            "union_recovers_large_catalog_recall": True,
            "union_default_gate": "closed",
            "reason": "At 8K and matched K=10, union improves recall but useful precision remains low; candidate exposure is not execution authorization.",
            "speed_claim": "reference_only",
        },
    }
    (RESULTS / "scaled_auto_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
