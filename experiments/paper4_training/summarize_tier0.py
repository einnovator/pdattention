"""Summarize the five-seed Paper 4 controlled gate without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


ORDER = (
    "frozen",
    "consumer_lora",
    "interface_lora",
    "broad_lora",
    "full_weight",
    "native_scratch",
)
LABELS = {
    "frozen": "Frozen",
    "consumer_lora": "Consumer LoRA",
    "interface_lora": "Interface LoRA",
    "broad_lora": "Broad LoRA",
    "full_weight": "Full weight",
    "native_scratch": "Native scratch",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def values(rows: list[dict], model: str, metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row["model"] == model]


def summarize(rows: list[dict]) -> list[dict]:
    metrics = (
        "trainable_fraction",
        "correct_memory_margin_gain_vs_none",
        "correct_memory_margin_gain_vs_distractor",
        "evidence_only_minus_parent_nll",
        "evidence_only_minus_parent_accuracy",
        "evidence_only_minus_parent_margin",
        "whole_parent_evidence_selectivity",
        "useful_memory_residual_divergence",
        "distractor_residual_divergence",
    )
    output = []
    for model in ORDER:
        members = [row for row in rows if row["model"] == model]
        summary = {"model": model, "seeds": len(members)}
        for metric in metrics:
            samples = [float(row[metric]) for row in members]
            summary[metric] = statistics.mean(samples)
            summary[f"{metric}_sd"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(summary)
    return output


def paired_effect(rows: list[dict], model: str, metric: str) -> dict:
    frozen = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["model"] == "frozen"
    }
    paired = [
        float(row[metric]) - frozen[int(row["seed"])]
        for row in rows
        if row["model"] == model
    ]
    return {
        "mean": statistics.mean(paired),
        "sd": statistics.stdev(paired) if len(paired) > 1 else 0.0,
        "same_direction": len(set(value > 0 for value in paired)) == 1,
        "positive_seeds": sum(value > 0 for value in paired),
        "n": len(paired),
    }


def make_plots(output_dir: Path, ladder: list[dict], profiles: list[dict], portability: list[dict]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    summary = summarize(ladder)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(ORDER))
    axes[0].errorbar(
        x,
        [float(row["correct_memory_margin_gain_vs_distractor"]) for row in summary],
        yerr=[float(row["correct_memory_margin_gain_vs_distractor_sd"]) for row in summary],
        marker="o",
        color="#006d77",
        capsize=3,
    )
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_ylabel("Gold-memory margin gain vs distractor")
    axes[1].errorbar(
        x,
        [float(row["evidence_only_minus_parent_nll"]) for row in summary],
        yerr=[float(row["evidence_only_minus_parent_nll_sd"]) for row in summary],
        marker="s",
        color="#b44724",
        capsize=3,
    )
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_ylabel("Evidence-only NLL minus whole-parent NLL")
    for axis in axes:
        axis.set_xticks(list(x), [LABELS[name] for name in ORDER], rotation=35, ha="right")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "causal_memory_and_modularity.png", dpi=180)
    figure.savefig(figures / "causal_memory_and_modularity.pdf")
    plt.close(figure)

    parent = [row for row in profiles if row["condition"] == "whole_parent"]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for model in ORDER:
        members = [row for row in parent if row["model"] == model]
        layer_ids = sorted({int(row["layer"]) for row in members})
        means = [
            statistics.mean(
                float(row["evidence_attention_mass"]) / max(float(row["memory_attention_mass"]), 1e-12)
                for row in members
                if int(row["layer"]) == layer
            )
            for layer in layer_ids
        ]
        axis.plot(layer_ids, means, marker="o", label=LABELS[model])
    axis.set(xlabel="PRA consumer layer (zero indexed)", ylabel="Evidence share of memory attention", ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "consumer_layer_selectivity.png", dpi=180)
    figure.savefig(figures / "consumer_layer_selectivity.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for model in ORDER:
        members = [row for row in portability if row["model"] == model]
        layer_ids = sorted({int(row["layer"]) for row in members})
        means = [
            statistics.mean(
                float(row["key_context_divergence"])
                for row in members
                if int(row["layer"]) == layer
            )
            for layer in layer_ids
        ]
        axis.plot(layer_ids, means, marker="o", label=LABELS[model])
    axis.set(xlabel="PRA-facing layer (zero indexed)", ylabel="Evidence K context divergence")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "representation_portability.png", dpi=180)
    figure.savefig(figures / "representation_portability.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/papers/shared/results/paper4_training/tier0"))
    args = parser.parse_args()
    ladder = read_csv(args.output_dir / "adaptation_ladder_results.csv")
    metrics = read_csv(args.output_dir / "adaptation_ladder_seed_results.csv")
    profiles = read_csv(args.output_dir / "consumer_layer_profiles.csv")
    portability = read_csv(args.output_dir / "representation_portability.csv")
    retention_path = args.output_dir / "retention_and_depth_results.csv"
    retention = read_csv(retention_path) if retention_path.exists() else []
    summary = summarize(ladder)
    write_csv(args.output_dir / "adaptation_ladder_summary.csv", summary)

    margin_effects = {
        model: paired_effect(ladder, model, "correct_memory_margin_gain_vs_distractor")
        for model in ORDER[1:]
    }
    selectivity_effects = {
        model: paired_effect(ladder, model, "whole_parent_evidence_selectivity")
        for model in ORDER[1:]
    }
    best_margin = max(margin_effects, key=lambda model: margin_effects[model]["mean"])
    best_selectivity = max(selectivity_effects, key=lambda model: selectivity_effects[model]["mean"])
    frozen_gap = abs(statistics.mean(values(ladder, "frozen", "evidence_only_minus_parent_nll")))
    modularity = {
        model: frozen_gap - abs(statistics.mean(values(ladder, model, "evidence_only_minus_parent_nll")))
        for model in ORDER[1:]
    }
    best_modularity = max(modularity, key=modularity.get)

    metric_lookup = {(row["model"], int(row["seed"]), row["condition"]): row for row in metrics}
    native_minus_local = [
        float(metric_lookup[("native_scratch", seed, "evidence_only")]["accuracy"])
        - float(metric_lookup[("local_sa", seed, "full_context")]["accuracy"])
        for seed in sorted({int(row["seed"]) for row in metrics})
    ]
    criteria = {
        "adaptation_improves_correct_memory_margin": margin_effects[best_margin]["mean"] > 0,
        "evidence_only_approaches_parent_quality": modularity[best_modularity] > 0,
        "consumer_selectivity_improves": selectivity_effects[best_selectivity]["mean"] > 0,
        "native_pra_beats_local_sa": statistics.mean(native_minus_local) > 0,
    }
    passed = sum(criteria.values())
    decision = "pass_to_tier1" if passed >= 2 else "hold_and_reconsider"
    seed_count = len({int(row["seed"]) for row in ladder})
    complete = seed_count >= 5
    findings = {
        "status": "tier0_controlled_complete" if complete else "tier0_pilot",
        "seed_count": seed_count,
        "gate_0": decision if complete else "pending_five_seed_completion",
        "criteria_passed": passed,
        "criteria": criteria,
        "best_margin_regime": best_margin,
        "best_margin_paired_effect": margin_effects[best_margin],
        "best_modularity_regime": best_modularity,
        "best_modularity_abs_nll_gap_reduction": modularity[best_modularity],
        "best_selectivity_regime": best_selectivity,
        "best_selectivity_paired_effect": selectivity_effects[best_selectivity],
        "native_minus_local_accuracy_mean": statistics.mean(native_minus_local),
        "scope": "consumer learning under fixed oracle memory; routing/materialization held constant",
        "statistical_note": "Five paired seeds are descriptive; the minimum exact two-sided sign-flip p-value is 0.0625.",
    }
    retention_means = {}
    depth_four_means = {}
    if retention:
        for model in ORDER:
            full_values = [
                float(row["accuracy"])
                for row in retention
                if row["model"] == model
                and row["condition"] == "full_context_no_memory"
            ]
            depth_values = [
                float(row["accuracy"])
                for row in retention
                if row["model"] == model
                and row["condition"] == "evidence_only"
                and row["depth"] == "4"
            ]
            retention_means[model] = statistics.mean(full_values)
            depth_four_means[model] = statistics.mean(depth_values)
        findings["full_context_accuracy"] = retention_means
        findings["evidence_only_depth4_accuracy"] = depth_four_means
    (args.output_dir / "paper4_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    summary_by_model = {row["model"]: row for row in summary}
    def condition_mean(model: str, condition: str, metric: str) -> float:
        samples = [
            float(row[metric])
            for row in metrics
            if row["model"] == model and row["condition"] == condition
        ]
        return statistics.mean(samples)

    tex = "\n".join(
        (
            "% Generated by experiments.paper4_training.summarize_tier0; do not edit.",
            r"\newcommand{\PaperFourSeedCount}{%d}" % findings["seed_count"],
            r"\newcommand{\PaperFourGateDecision}{%s}" % decision.replace("_", r"\_"),
            r"\newcommand{\PaperFourCriteriaPassed}{%d}" % passed,
            r"\newcommand{\PaperFourBestMarginRegime}{%s}" % LABELS[best_margin],
            r"\newcommand{\PaperFourBestMarginEffect}{%.3f}" % margin_effects[best_margin]["mean"],
            r"\newcommand{\PaperFourBestModularityRegime}{%s}" % LABELS[best_modularity],
            r"\newcommand{\PaperFourBestModularityEffect}{%.3f}" % modularity[best_modularity],
            r"\newcommand{\PaperFourBestSelectivityRegime}{%s}" % LABELS[best_selectivity],
            r"\newcommand{\PaperFourBestSelectivityEffect}{%.3f}" % selectivity_effects[best_selectivity]["mean"],
            r"\newcommand{\PaperFourNativeMinusLocalAccuracy}{%.3f}" % findings["native_minus_local_accuracy_mean"],
            r"\newcommand{\PaperFourFrozenEvidenceAccuracy}{%.3f}" % condition_mean("frozen", "evidence_only", "accuracy"),
            r"\newcommand{\PaperFourConsumerEvidenceAccuracy}{%.3f}" % condition_mean("consumer_lora", "evidence_only", "accuracy"),
            r"\newcommand{\PaperFourInterfaceEvidenceAccuracy}{%.3f}" % condition_mean("interface_lora", "evidence_only", "accuracy"),
            r"\newcommand{\PaperFourBroadEvidenceAccuracy}{%.3f}" % condition_mean("broad_lora", "evidence_only", "accuracy"),
            r"\newcommand{\PaperFourFullEvidenceAccuracy}{%.3f}" % condition_mean("full_weight", "evidence_only", "accuracy"),
            r"\newcommand{\PaperFourNativeEvidenceAccuracy}{%.3f}" % condition_mean("native_scratch", "evidence_only", "accuracy"),
            r"\newcommand{\PaperFourFrozenParentAccuracy}{%.3f}" % condition_mean("frozen", "whole_parent", "accuracy"),
            r"\newcommand{\PaperFourConsumerParentAccuracy}{%.3f}" % condition_mean("consumer_lora", "whole_parent", "accuracy"),
            r"\newcommand{\PaperFourInterfaceParentAccuracy}{%.3f}" % condition_mean("interface_lora", "whole_parent", "accuracy"),
            r"\newcommand{\PaperFourBroadParentAccuracy}{%.3f}" % condition_mean("broad_lora", "whole_parent", "accuracy"),
            r"\newcommand{\PaperFourFullParentAccuracy}{%.3f}" % condition_mean("full_weight", "whole_parent", "accuracy"),
            r"\newcommand{\PaperFourNativeParentAccuracy}{%.3f}" % condition_mean("native_scratch", "whole_parent", "accuracy"),
            r"\newcommand{\PaperFourFrozenRetention}{%.3f}" % retention_means.get("frozen", float("nan")),
            r"\newcommand{\PaperFourConsumerRetention}{%.3f}" % retention_means.get("consumer_lora", float("nan")),
            r"\newcommand{\PaperFourInterfaceRetention}{%.3f}" % retention_means.get("interface_lora", float("nan")),
            r"\newcommand{\PaperFourBroadRetention}{%.3f}" % retention_means.get("broad_lora", float("nan")),
            r"\newcommand{\PaperFourFullRetention}{%.3f}" % retention_means.get("full_weight", float("nan")),
            r"\newcommand{\PaperFourNativeRetention}{%.3f}" % retention_means.get("native_scratch", float("nan")),
            r"\newcommand{\PaperFourFullDepthFour}{%.3f}" % depth_four_means.get("full_weight", float("nan")),
            r"\newcommand{\PaperFourNativeDepthFour}{%.3f}" % depth_four_means.get("native_scratch", float("nan")),
            "",
        )
    )
    (args.output_dir / "generated_results.tex").write_text(tex, encoding="utf-8")
    table_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Regime & Trainable & Margin gain & $\Delta$NLL & Selectivity \\",
        r"\midrule",
    ]
    for model in ORDER:
        row = summary_by_model[model]
        table_lines.append(
            "%s & %.3f & %.3f $\pm$ %.3f & %.3f & %.3f \\\\" % (
                LABELS[model],
                float(row["trainable_fraction"]),
                float(row["correct_memory_margin_gain_vs_distractor"]),
                float(row["correct_memory_margin_gain_vs_distractor_sd"]),
                float(row["evidence_only_minus_parent_nll"]),
                float(row["whole_parent_evidence_selectivity"]),
            )
        )
    table_lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    (args.output_dir / "generated_ladder_table.tex").write_text(
        "\n".join(table_lines), encoding="utf-8"
    )
    make_plots(args.output_dir, ladder, profiles, portability)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
