"""Summarize query aggregation and frozen-Qwen learned-routing experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


QUERY_DIR = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "routing" / "query_strategies"
LEARNED_DIR = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "routing" / "learned_adapter"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combined(artifact: dict, strategy: str, top_k: int = 3) -> dict:
    rows = [
        row
        for row in artifact["aggregates"]
        if row["query_strategy"] == strategy and int(row["top_k"]) == top_k
    ]
    metrics = (
        "recall_at_3",
        "recall_at_8",
        "recall_at_16",
        "mrr",
        "score_position_correlation",
        "query_cosine_to_last",
    )
    return {
        metric: statistics.fmean(float(row[metric]) for row in rows if row.get(metric) is not None)
        for metric in metrics
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _query_summary() -> dict:
    broad = _load(QUERY_DIR / "query_strategy_sweep.json")
    refine = _load(QUERY_DIR / "query_half_life_sweep.json")
    confirmation = _load(QUERY_DIR / "query_strategy_confirmation.json")
    validation = {
        strategy["name"]: _combined(broad, strategy["name"])
        for strategy in broad["strategies"]
    }
    confirmation_results = {
        strategy["name"]: _combined(confirmation, strategy["name"])
        for strategy in confirmation["strategies"]
    }
    last = confirmation_results["last"]
    candidates = {
        name: {
            **metrics,
            "recall_at_3_delta_vs_last": metrics["recall_at_3"] - last["recall_at_3"],
            "mrr_delta_vs_last": metrics["mrr"] - last["mrr"],
        }
        for name, metrics in confirmation_results.items()
    }
    interaction = []
    rows = [row for row in confirmation["rows"] if int(row["top_k"]) == 3]
    lengths = sorted({int(row["question_tokens"]) for row in rows})
    median = statistics.median(lengths)
    by_key = {
        (row["dataset"], row["example_id"], row["query_strategy"]): row for row in rows
    }
    for bucket, predicate in (
        ("short", lambda value: value <= median),
        ("long", lambda value: value > median),
    ):
        for strategy in ("last", "question_exp_h2.0", "uniform_w32"):
            selected = [
                row
                for row in rows
                if row["query_strategy"] == strategy and predicate(int(row["question_tokens"]))
            ]
            interaction.append(
                {
                    "question_length_bucket": bucket,
                    "strategy": strategy,
                    "examples": len(selected),
                    "recall_at_3": statistics.fmean(row["recall_at_3"] for row in selected),
                    "mrr": statistics.fmean(row["mrr"] for row in selected),
                }
            )
    summary = {
        "source_git_sha": confirmation["runtime"]["git_sha"],
        "validation": validation,
        "confirmation": candidates,
        "question_length_median_tokens": median,
        "question_length_interaction": interaction,
        "gate": {
            "required_combined_recall_at_3_delta": 0.10,
            "no_dataset_recall_at_3_loss": True,
            "max_position_correlation_magnitude_increase": 0.10,
            "passed": False,
        },
        "decision": {
            "best_zero_parameter_query": "last",
            "aggregation_promoted": False,
            "reason": "No aggregate beat last-token Recall@3 without a material dataset loss.",
        },
    }
    (QUERY_DIR / "query_strategy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        QUERY_DIR / "query_dataset_breakdown.csv",
        [row for row in confirmation["aggregates"] if int(row["top_k"]) == 3],
    )
    _write_csv(QUERY_DIR / "query_length_interaction.csv", interaction)

    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    names = list(confirmation_results)
    x = range(len(names))
    axis.bar([value - 0.18 for value in x], [confirmation_results[name]["recall_at_3"] for name in names], 0.36, label="Recall@3")
    axis.bar([value + 0.18 for value in x], [confirmation_results[name]["mrr"] for name in names], 0.36, label="MRR")
    axis.set_xticks(list(x), names, rotation=20, ha="right")
    axis.set_ylim(0.0, 0.8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(QUERY_DIR / f"query_confirmation.{suffix}", dpi=180)
    plt.close(figure)

    half_life_rows = [
        row
        for row in refine["aggregates"]
        if int(row["top_k"]) == 3 and row["query_strategy"].startswith("question_exp_h")
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    for dataset in ("hotpotqa", "qasper"):
        selected = sorted(
            [row for row in half_life_rows if row["dataset"] == dataset],
            key=lambda row: float(row["query_strategy"].split("h")[-1]),
        )
        axis.plot(
            [float(row["query_strategy"].split("h")[-1]) for row in selected],
            [row["recall_at_3"] for row in selected],
            marker="o",
            label=dataset,
        )
    axis.set_xlabel("Question-state half-life (tokens)")
    axis.set_ylabel("Validation Recall@3")
    axis.set_xticks([2, 4, 8, 16])
    axis.set_ylim(0.0, 0.8)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(QUERY_DIR / f"query_half_life_recall.{suffix}", dpi=180)
    plt.close(figure)
    return summary


def _learned_summary() -> dict:
    linear = _load(LEARNED_DIR / "linear_adapter_results.json")
    mlp = _load(LEARNED_DIR / "mlp_results.json")
    shuffled = _load(LEARNED_DIR / "shuffled_label_control_canonical.json")
    transfer = _load(LEARNED_DIR / "cross_domain_results_canonical.json")
    e2e = _load(LEARNED_DIR / "learned_router_e2e.json")

    def validation_score(artifact, aggregate):
        runs = [
            run
            for run in artifact["runs"]
            if run["architecture"] == aggregate["architecture"]
            and int(run["routing_width"]) == int(aggregate["routing_width"])
            and run["query_strategy"] == aggregate["query_strategy"]
            and run["train_domain"] == aggregate["train_domain"]
        ]
        recall = statistics.fmean(
            run["validation"]["aggregates"]["combined"]["recall_at_3"] for run in runs
        )
        mrr = statistics.fmean(
            run["validation"]["aggregates"]["combined"]["mrr"] for run in runs
        )
        return recall, mrr

    candidates = [(row, *validation_score(linear, row)) for row in linear["aggregates"]]
    candidates.extend((row, *validation_score(mlp, row)) for row in mlp["aggregates"])
    selected, selected_validation_recall, selected_validation_mrr = max(
        candidates, key=lambda value: (value[1], value[2])
    )
    exploratory = max(
        [*linear["aggregates"], *mlp["aggregates"]],
        key=lambda row: row["combined_recall_at_3_mean"],
    )
    learned_aggregate = {
        "cosine_last": linear["baselines"]["last"]["aggregates"]["combined"],
        "cosine_question_h2": linear["baselines"]["question_exp_h2.0"]["aggregates"]["combined"],
        "learned_last": {
            "recall_at_3": selected["combined_recall_at_3_mean"],
            "recall_at_8": selected["combined_recall_at_8_mean"],
            "mrr": selected["combined_mrr_mean"],
        },
    }
    learned_question = next(
        row
        for row in linear["aggregates"]
        if row["architecture"] == selected["architecture"]
        and int(row["routing_width"]) == int(selected["routing_width"])
        and row["query_strategy"] == "question_exp_h2.0"
    )
    learned_aggregate["learned_question_h2"] = {
        "recall_at_3": learned_question["combined_recall_at_3_mean"],
        "recall_at_8": learned_question["combined_recall_at_8_mean"],
        "mrr": learned_question["combined_mrr_mean"],
    }
    shuffled_best = shuffled["aggregates"][0]
    transfer_rows = {
        row["train_domain"]: row for row in transfer["aggregates"]
    }
    mlp_best = max(mlp["aggregates"], key=lambda row: row["combined_recall_at_3_mean"])
    summary = {
        "source_git_sha": linear["runtime"]["git_sha"],
        "selection_protocol": "architecture and seed selected on validation only",
        "validation_selected_adapter": selected,
        "validation_selected_recall_at_3": selected_validation_recall,
        "validation_selected_mrr": selected_validation_mrr,
        "exploratory_test_best_adapter": exploratory,
        "adapter_percentage_of_qwen": selected["adapter_parameters"] / 596_049_920 * 100,
        "ablation_2x2": learned_aggregate,
        "mlp_best": mlp_best,
        "shuffled_label_recall_at_3": shuffled_best["combined_recall_at_3_mean"],
        "cross_domain": {
            "hotpot_to_qasper_recall_at_3": transfer_rows["hotpotqa"]["qasper_recall_at_3_mean"],
            "qasper_to_hotpot_recall_at_3": transfer_rows["qasper"]["hotpotqa_recall_at_3_mean"],
        },
        "end_to_end": e2e["aggregates"],
        "promotion_gate": {
            "combined_recall_at_3_at_least_0_70": selected["combined_recall_at_3_mean"] >= 0.70,
            "each_dataset_recall_at_3_at_least_0_50": min(
                selected["hotpotqa_recall_at_3_mean"], selected["qasper_recall_at_3_mean"]
            ) >= 0.50,
            "combined_recall_at_8_at_least_0_80": selected["combined_recall_at_8_mean"] >= 0.80,
            "absolute_position_correlation_at_most_0_20": abs(
                selected["combined_score_position_correlation_mean"]
            ) <= 0.20,
            "routing_width_at_most_128": int(selected["routing_width"]) <= 128,
            "passed": False,
        },
        "decision": {
            "canonical_query": "last",
            "canonical_metric": "asymmetric_linear_d128",
            "qwen_to_llama": "do_not_promote",
            "reason": "Validation-selected capacity missed the held-out gate and transferred poorly; a test-best shared model is exploratory only.",
            "next_problem_if_retrieval_improves": "memory-use alignment",
        },
    }
    (LEARNED_DIR / "learned_router_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    conditions = [
        ("Cosine\nlast", linear["baselines"]["last"]["aggregates"]),
        ("Validation-selected\nasymmetric linear", {
            dataset: {"recall_at_3": selected[f"{dataset}_recall_at_3_mean"]}
            for dataset in ("combined", "hotpotqa", "qasper")
        }),
        ("Exploratory test-best\nshared linear", {
            dataset: {"recall_at_3": exploratory[f"{dataset}_recall_at_3_mean"]}
            for dataset in ("combined", "hotpotqa", "qasper")
        }),
        ("Best MLP\nlast", {
            dataset: {"recall_at_3": mlp_best[f"{dataset}_recall_at_3_mean"]}
            for dataset in ("combined", "hotpotqa", "qasper")
        }),
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    x = range(len(conditions))
    width = 0.25
    for offset, dataset, color in (
        (-width, "combined", "#4472c4"),
        (0.0, "hotpotqa", "#70ad47"),
        (width, "qasper", "#ed7d31"),
    ):
        axis.bar(
            [value + offset for value in x],
            [condition[1][dataset]["recall_at_3"] for condition in conditions],
            width,
            label=dataset,
            color=color,
        )
    axis.axhline(0.70, color="black", linestyle="--", linewidth=1, label="combined gate")
    axis.set_xticks(list(x), [condition[0] for condition in conditions])
    axis.set_ylabel("Held-out Recall@3")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(LEARNED_DIR / f"learned_router_recall.{suffix}", dpi=180)
    plt.close(figure)

    labels = ["Cosine\nlast", "Cosine\nquestion", "Learned\nlast", "Learned\nquestion"]
    recall = [
        learned_aggregate["cosine_last"]["recall_at_3"],
        learned_aggregate["cosine_question_h2"]["recall_at_3"],
        learned_aggregate["learned_last"]["recall_at_3"],
        learned_aggregate["learned_question_h2"]["recall_at_3"],
    ]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.bar(range(4), recall, color=["#a5a5a5", "#a5a5a5", "#4472c4", "#4472c4"])
    axis.set_xticks(range(4), labels)
    axis.set_ylabel("Held-out Recall@3")
    axis.set_ylim(0.0, 0.8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(LEARNED_DIR / f"query_metric_ablation.{suffix}", dpi=180)
    plt.close(figure)
    return summary


def _learned_extension_summary(previous: dict) -> dict:
    """Summarize the capacity, objective, and hard-negative extension protocol."""
    capacity = _load(LEARNED_DIR / "capacity_d256_results.json")
    margin = _load(LEARNED_DIR / "margin_objective_results.json")
    mixed = _load(LEARNED_DIR / "mixed_negative_results.json")
    mined = _load(LEARNED_DIR / "hard_negative_results.json")
    shuffled = _load(LEARNED_DIR / "shuffled_margin_control.json")
    transfer = _load(LEARNED_DIR / "cross_domain_margin_results.json")
    e2e = _load(LEARNED_DIR / "learned_router_margin_e2e.json")

    def validation_score(artifact: dict, aggregate: dict) -> tuple[float, float]:
        runs = [
            run
            for run in artifact["runs"]
            if run["architecture"] == aggregate["architecture"]
            and int(run["routing_width"]) == int(aggregate["routing_width"])
            and run["train_domain"] == aggregate["train_domain"]
            and run["objective"] == aggregate["objective"]
            and run["negative_policy"] == aggregate["negative_policy"]
        ]
        return (
            statistics.fmean(
                run["validation"]["aggregates"]["combined"]["recall_at_3"]
                for run in runs
            ),
            statistics.fmean(
                run["validation"]["aggregates"]["combined"]["mrr"] for run in runs
            ),
        )

    candidates = []
    for artifact in (capacity, margin):
        for aggregate in artifact["aggregates"]:
            recall, mrr = validation_score(artifact, aggregate)
            candidates.append((artifact, aggregate, recall, mrr))
    _, selected, validation_recall, validation_mrr = max(
        candidates, key=lambda item: (item[2], item[3])
    )
    selected_runs = [
        run
        for run in margin["runs"]
        if run["architecture"] == selected["architecture"]
        and int(run["routing_width"]) == int(selected["routing_width"])
    ]
    baseline = margin["baselines"]["last"]["aggregates"]
    seed_deltas = [
        run["test"]["aggregates"]["combined"]["recall_at_3"]
        - baseline["combined"]["recall_at_3"]
        for run in selected_runs
    ]
    delta_std = statistics.stdev(seed_deltas)
    paired = {
        "seeds": len(seed_deltas),
        "recall_at_3_delta_mean": statistics.fmean(seed_deltas),
        "recall_at_3_delta_std": delta_std,
        "recall_at_3_delta_ci95": 2.776 * delta_std / math.sqrt(len(seed_deltas)),
        "per_seed": seed_deltas,
    }
    transfer_rows = {row["train_domain"]: row for row in transfer["aggregates"]}
    capacity_rows = {
        row["architecture"]: row for row in capacity["aggregates"]
    }
    mixed_row = mixed["aggregates"][0]
    mined_row = mined["aggregates"][0]
    shuffled_row = shuffled["aggregates"][0]
    gate = {
        "combined_recall_at_3_at_least_0_70": selected["combined_recall_at_3_mean"] >= 0.70,
        "each_dataset_recall_at_3_at_least_0_50": min(
            selected["hotpotqa_recall_at_3_mean"], selected["qasper_recall_at_3_mean"]
        ) >= 0.50,
        "combined_recall_at_8_at_least_0_80": selected["combined_recall_at_8_mean"] >= 0.80,
        "absolute_position_correlation_at_most_0_20": abs(
            selected["combined_score_position_correlation_mean"]
        ) <= 0.20,
        "routing_width_at_most_128": int(selected["routing_width"]) <= 128,
    }
    gate["passed"] = all(gate.values())
    summary = {
        "source_git_sha": margin["runtime"]["git_sha"],
        "hypothesis": "Frozen hidden states contain semantic knowledge, but require a learned query-memory relevance geometry.",
        "selection_protocol": "Architecture, width, and objective selected by validation Recall@3 then MRR; held-out test remained sealed.",
        "validation_selected_adapter": selected,
        "validation_selected_recall_at_3": validation_recall,
        "validation_selected_mrr": validation_mrr,
        "adapter_cost": {
            "parameters": selected["adapter_parameters"],
            "bytes_fp32": selected["adapter_parameters"] * 4,
            "percentage_of_qwen": selected["adapter_parameters"] / 596_049_920 * 100,
            "routing_vector_dimension": selected["routing_width"],
            "routing_vector_bytes_fp32": selected["routing_width"] * 4,
            "index_fraction_of_detail_kv": selected[
                "routing_index_fraction_of_detail_kv"
            ],
            "adapter_score_seconds_per_example": selected[
                "adapter_seconds_per_example"
            ],
            "full_routing_seconds_per_example": selected[
                "routing_seconds_per_example"
            ],
        },
        "paired_vs_zero_shot": paired,
        "zero_shot_last": baseline,
        "previous_contrastive_selection": previous["validation_selected_adapter"],
        "capacity_d256": capacity_rows,
        "negative_policy": {
            "mixed": mixed_row,
            "mined_round_1": mined_row,
            "interpretation": "Sampling and one learned-false-positive refresh both underperformed the exhaustive in-document denominator.",
        },
        "shuffled_label_recall_at_3": shuffled_row["combined_recall_at_3_mean"],
        "cross_domain": {
            "hotpot_to_qasper_recall_at_3": transfer_rows["hotpotqa"]["qasper_recall_at_3_mean"],
            "qasper_to_hotpot_recall_at_3": transfer_rows["qasper"]["hotpotqa_recall_at_3_mean"],
        },
        "resolution_comparison": {
            "zero_shot_token_level_recall_at_3": 0.375,
            "zero_shot_token_level_index_fraction": 0.5,
            "learned_single_mean_recall_at_3": selected["combined_recall_at_3_mean"],
            "learned_single_mean_index_fraction": selected[
                "routing_index_fraction_of_detail_kv"
            ],
            "cohort_note": "Token-level diagnostic used an earlier 16-example cohort; comparison is descriptive, not paired.",
        },
        "end_to_end": e2e["aggregates"],
        "native_limit_violations": e2e["native_limit_violations"],
        "promotion_gate": gate,
        "decision": {
            "canonical_query": "last",
            "canonical_adapter": "asymmetric_linear_d128_margin_exhaustive",
            "qwen_to_llama": "do_not_promote",
            "reason": "The margin objective improves balance and sparse recall but remains below the predeclared combined Recall@3 gate; cross-domain transfer and causal QA use remain weak.",
            "next_problem_if_retrieval_improves": "memory-use alignment",
        },
    }
    for name in ("learned_router_extended_summary.json", "learned_router_summary.json"):
        (LEARNED_DIR / name).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    conditions = [
        {
            "condition": "Zero-shot cosine",
            "combined_recall_at_3": baseline["combined"]["recall_at_3"],
            "hotpotqa_recall_at_3": baseline["hotpotqa"]["recall_at_3"],
            "qasper_recall_at_3": baseline["qasper"]["recall_at_3"],
            "routing_index_fraction": margin["baselines"]["last"][
                "routing_index_bytes_mean"
            ]
            / margin["baselines"]["last"]["detail_kv_bytes_mean"],
        },
        {
            "condition": "Contrastive asymmetric-128",
            **{
                f"{dataset}_recall_at_3": previous["validation_selected_adapter"][
                    f"{dataset}_recall_at_3_mean"
                ]
                for dataset in ("combined", "hotpotqa", "qasper")
            },
            "routing_index_fraction": 0.003928906317528372,
        },
        {
            "condition": "Margin asymmetric-128",
            **{
                f"{dataset}_recall_at_3": selected[f"{dataset}_recall_at_3_mean"]
                for dataset in ("combined", "hotpotqa", "qasper")
            },
            "routing_index_fraction": selected["routing_index_fraction_of_detail_kv"],
        },
        {
            "condition": "Contrastive shared-256",
            **{
                f"{dataset}_recall_at_3": capacity_rows["shared_linear"][
                    f"{dataset}_recall_at_3_mean"
                ]
                for dataset in ("combined", "hotpotqa", "qasper")
            },
            "routing_index_fraction": capacity_rows["shared_linear"][
                "routing_index_fraction_of_detail_kv"
            ],
        },
        {
            "condition": "Mixed negatives",
            **{
                f"{dataset}_recall_at_3": mixed_row[f"{dataset}_recall_at_3_mean"]
                for dataset in ("combined", "hotpotqa", "qasper")
            },
            "routing_index_fraction": mixed_row["routing_index_fraction_of_detail_kv"],
        },
        {
            "condition": "Mined round 1",
            **{
                f"{dataset}_recall_at_3": mined_row[f"{dataset}_recall_at_3_mean"]
                for dataset in ("combined", "hotpotqa", "qasper")
            },
            "routing_index_fraction": mined_row["routing_index_fraction_of_detail_kv"],
        },
    ]
    _write_csv(LEARNED_DIR / "learned_router_extended_conditions.csv", conditions)
    _write_csv(
        LEARNED_DIR / "learned_router_paired_effects.csv",
        [{"seed": run["seed"], "recall_at_3_delta": delta} for run, delta in zip(selected_runs, seed_deltas)],
    )

    figure, axis = plt.subplots(figsize=(9.0, 4.7))
    x = range(len(conditions))
    width = 0.25
    for offset, dataset, color in (
        (-width, "combined", "#4472c4"),
        (0.0, "hotpotqa", "#70ad47"),
        (width, "qasper", "#ed7d31"),
    ):
        axis.bar(
            [value + offset for value in x],
            [row[f"{dataset}_recall_at_3"] for row in conditions],
            width,
            label=dataset,
            color=color,
        )
    axis.axhline(0.70, color="black", linestyle="--", linewidth=1, label="combined gate")
    axis.set_xticks(list(x), [row["condition"] for row in conditions], rotation=18, ha="right")
    axis.set_ylabel("Held-out Recall@3")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(LEARNED_DIR / f"learned_router_extended_recall.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    cost_rows = [conditions[0], conditions[2], conditions[3]]
    for row, label, color, marker in zip(
        cost_rows,
        ("Raw cosine, 1024D", "Margin, 128D", "Contrastive, 256D"),
        ("#a5a5a5", "#4472c4", "#70ad47"),
        ("o", "s", "^"),
    ):
        axis.scatter(
            row["routing_index_fraction"] * 100,
            row["combined_recall_at_3"],
            s=60,
            color=color,
            marker=marker,
            label=label,
        )
    axis.set_xlabel("Routing index / native detail K/V (%)")
    axis.set_ylabel("Held-out Recall@3")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="center right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(LEARNED_DIR / f"learned_router_cost_quality.{suffix}", dpi=180)
    plt.close(figure)
    return summary


if __name__ == "__main__":
    learned = _learned_summary()
    result = {
        "query": _query_summary(),
        "learned": _learned_extension_summary(learned),
    }
    print(json.dumps({key: value["decision"] for key, value in result.items()}, indent=2))
