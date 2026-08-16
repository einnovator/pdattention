"""Derive the frozen Paper 2.5 final-reviewer tables from existing traces.

This script does not run a model. It joins the canonical 400 controlled
model-example units to already-recorded mechanistic diagnostics, compares the
path-improved minority with all other units, and summarizes the existing
cross-dataset routing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    REPO_ROOT / "docs" / "papers" / "shared" / "results" / "paper2_5_iterative_pra"
)
WINDOWS = ("w16", "w32", "w64", "w128", "global")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float:
    return float(str(value).strip())


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def bootstrap_difference_ci(
    improved: list[float], other: list[float], *, seed: int = 34159, draws: int = 2000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        left = statistics.fmean(rng.choices(improved, k=len(improved)))
        right = statistics.fmean(rng.choices(other, k=len(other)))
        estimates.append(left - right)
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[min(int(0.975 * draws), draws - 1)]


def standardized_mean_difference(improved: list[float], other: list[float]) -> float:
    if len(improved) < 2 or len(other) < 2:
        return math.nan
    variance = (
        (len(improved) - 1) * statistics.variance(improved)
        + (len(other) - 1) * statistics.variance(other)
    ) / (len(improved) + len(other) - 2)
    return (mean(improved) - mean(other)) / math.sqrt(variance) if variance > 0 else 0.0


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Return the two-sided Fisher exact probability for a 2x2 table."""

    row_one = a + b
    row_two = c + d
    col_one = a + c
    total = row_one + row_two

    def probability(x: int) -> float:
        return (
            math.comb(col_one, x)
            * math.comb(total - col_one, row_one - x)
            / math.comb(total, row_one)
        )

    low = max(0, row_one - (total - col_one))
    high = min(row_one, col_one)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-15))


def keyed(rows: list[dict[str, str]], *, condition: str | None = None) -> dict[tuple[str, str, str], dict[str, str]]:
    selected = rows if condition is None else [row for row in rows if row.get("condition") == condition]
    return {(row["window"], row["seed"], row["example_id"]): row for row in selected}


def mean_by_key(rows: list[dict[str, str]], condition: str) -> dict[tuple[str, str, str], dict[str, float]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["condition"] == condition:
            groups.setdefault((row["window"], row["seed"], row["example_id"]), []).append(row)
    metrics = (
        "evidence_attention_mass",
        "distractor_attention_mass",
        "native_attention_mass",
        "attention_entropy",
    )
    return {
        key: {
            **{metric: mean([number(row[metric]) for row in group]) for metric in metrics},
            "intervention_count": float(len(group)),
        }
        for key, group in groups.items()
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def build_iteration_rows(results: Path) -> list[dict]:
    controlled = results / "controlled_local_sa_v6"
    mechanism = controlled / "mechanistic"
    traversal = read_csv(controlled / "traversal_to_use_rows.csv")
    causal = read_csv(mechanism / "causal_memory_ablation.csv")
    attention = read_csv(mechanism / "memory_attention_decomposition.csv")
    erasure = read_csv(controlled / "later_layer_erasure.csv")

    one = keyed(causal, condition="one_shot_selected")
    iterative = keyed(causal, condition="iterative_matched_selected")
    one_attention = mean_by_key(attention, "one_shot_selected")
    iterative_attention = mean_by_key(attention, "iterative_matched_selected")
    selected_erasure = {
        (row["window"], row["seed"], row["example_id"]): row
        for row in erasure
        if row["policy"] == "iterative_matched" and row["evidence_condition"] == "selected"
    }

    output = []
    for source in traversal:
        key = (source["window"], source["seed"], source["example_id"])
        baseline = one[key]
        retry = iterative[key]
        baseline_attention = one_attention[key]
        retry_attention = iterative_attention[key]
        erasure_row = selected_erasure[key]
        path_gain = number(source["path_gain"])
        row = {
            "unit_id": "|".join(key),
            "group": "G+" if path_gain > 0 else "G0",
            "path_improved": int(path_gain > 0),
            "window": source["window"],
            **{f"window_is_{window}": int(source["window"] == window) for window in WINDOWS},
            "seed": int(source["seed"]),
            "example_id": source["example_id"],
            "chain_depth": number(source["depth"]),
            "query_token_count": number(baseline["query_token_count"]),
            "evidence_span_tokens": number(baseline["evidence_span_tokens"]),
            "max_hop_distance_tokens": number(baseline["max_hop_distance_tokens"]),
            "evidence_span_over_window": number(baseline["span_over_window"]),
            "max_hop_distance_over_window": number(baseline["hop_over_window"]),
            "one_shot_path_recovery": number(source["one_shot_path_recovery"]),
            "one_shot_reference_recall": number(baseline["reference_recall"]),
            "one_shot_correct": number(source["one_shot_correct"]),
            "one_shot_margin": number(source["one_shot_margin"]),
            "one_shot_correct_probability": number(baseline["correct_probability"]),
            "one_shot_prediction_entropy": number(baseline["prediction_entropy"]),
            "one_shot_attention_entropy": baseline_attention["attention_entropy"],
            "one_shot_evidence_attention_mass": baseline_attention["evidence_attention_mass"],
            "one_shot_distractor_attention_mass": baseline_attention["distractor_attention_mass"],
            "one_shot_memory_attention_mass": (
                baseline_attention["evidence_attention_mass"]
                + baseline_attention["distractor_attention_mass"]
            ),
            "one_shot_evidence_distractor_ratio": safe_ratio(
                baseline_attention["evidence_attention_mass"],
                baseline_attention["distractor_attention_mass"],
            ),
            "one_shot_native_attention_mass": baseline_attention["native_attention_mass"],
            "iterative_path_recovery": number(source["iterative_path_recovery"]),
            "path_gain": path_gain,
            "iterative_reference_recall": number(retry["reference_recall"]),
            "reference_recall_gain": number(retry["reference_recall"]) - number(baseline["reference_recall"]),
            "iterative_correct": number(source["iterative_correct"]),
            "answer_gain": number(source["answer_gain"]),
            "iterative_margin": number(source["iterative_margin"]),
            "margin_gain": number(source["margin_gain"]),
            "iterative_correct_probability": number(retry["correct_probability"]),
            "iterative_prediction_entropy": number(retry["prediction_entropy"]),
            "iterative_evidence_attention_mass": retry_attention["evidence_attention_mass"],
            "iterative_distractor_attention_mass": retry_attention["distractor_attention_mass"],
            "iterative_evidence_distractor_ratio": safe_ratio(
                retry_attention["evidence_attention_mass"],
                retry_attention["distractor_attention_mass"],
            ),
            "iterative_native_attention_mass": retry_attention["native_attention_mass"],
            "consumer_layer_count": 4.0,
            "intervention_count": retry_attention["intervention_count"],
            "positive_immediate_margin_gain": number(erasure_row["positive_immediate_gain"]),
            "erased_by_final_layer": number(erasure_row["erased_by_final_layer"]),
        }
        output.append(row)

    assert len(output) == 400
    assert sum(row["path_improved"] for row in output) == 59
    return output


PRE_DECISION = (
    "window_is_w16",
    "window_is_w32",
    "window_is_w64",
    "window_is_w128",
    "window_is_global",
    "chain_depth",
    "query_token_count",
    "evidence_span_tokens",
    "max_hop_distance_tokens",
    "evidence_span_over_window",
    "max_hop_distance_over_window",
    "one_shot_path_recovery",
    "one_shot_reference_recall",
    "one_shot_correct",
    "one_shot_margin",
    "one_shot_correct_probability",
    "one_shot_prediction_entropy",
    "one_shot_attention_entropy",
    "one_shot_evidence_attention_mass",
    "one_shot_distractor_attention_mass",
    "one_shot_memory_attention_mass",
    "one_shot_evidence_distractor_ratio",
    "one_shot_native_attention_mass",
)
POST_TREATMENT = (
    "iterative_path_recovery",
    "path_gain",
    "iterative_reference_recall",
    "reference_recall_gain",
    "iterative_correct",
    "answer_gain",
    "iterative_margin",
    "margin_gain",
    "iterative_correct_probability",
    "iterative_prediction_entropy",
    "iterative_evidence_attention_mass",
    "iterative_distractor_attention_mass",
    "iterative_evidence_distractor_ratio",
    "iterative_native_attention_mass",
    "consumer_layer_count",
    "intervention_count",
    "positive_immediate_margin_gain",
    "erased_by_final_layer",
)
BINARY = {
    *(f"window_is_{window}" for window in WINDOWS),
    "one_shot_path_recovery",
    "one_shot_correct",
    "iterative_path_recovery",
    "iterative_correct",
    "positive_immediate_margin_gain",
    "erased_by_final_layer",
}

LABEL_FREE_PREDICTABILITY_FEATURES = (
    "window_is_w16",
    "window_is_w32",
    "window_is_w64",
    "window_is_w128",
    "window_is_global",
    "one_shot_prediction_entropy",
    "one_shot_attention_entropy",
    "one_shot_memory_attention_mass",
    "one_shot_native_attention_mass",
)


def summarize_features(rows: list[dict]) -> list[dict]:
    output = []
    comparisons = (
        (
            "all_400",
            [row for row in rows if row["path_improved"]],
            [row for row in rows if not row["path_improved"]],
            (("pre-decision", PRE_DECISION), ("post-treatment", POST_TREATMENT)),
        ),
        (
            "one_shot_miss_only",
            [row for row in rows if row["path_improved"]],
            [row for row in rows if not row["path_improved"] and row["one_shot_path_recovery"] == 0],
            (("pre-decision", tuple(feature for feature in PRE_DECISION if feature != "one_shot_path_recovery")),),
        ),
    )
    for comparison_scope, improved_rows, other_rows, timing_groups in comparisons:
        for timing, features in timing_groups:
            for feature in features:
                improved = [number(row[feature]) for row in improved_rows]
                other = [number(row[feature]) for row in other_rows]
                low, high = bootstrap_difference_ci(improved, other)
                summary = {
                    "comparison_scope": comparison_scope,
                    "timing": timing,
                    "feature": feature,
                    "status": "available",
                    "path_improved_n": len(improved),
                    "other_n": len(other),
                    "path_improved_mean": mean(improved),
                    "path_improved_median": median(improved),
                    "other_mean": mean(other),
                    "other_median": median(other),
                    "mean_difference": mean(improved) - mean(other),
                    "difference_bootstrap_ci95_low": low,
                    "difference_bootstrap_ci95_high": high,
                }
                if feature in BINARY:
                    a = sum(value > 0.5 for value in improved)
                    b = len(improved) - a
                    c = sum(value > 0.5 for value in other)
                    d = len(other) - c
                    summary.update(
                        {
                            "effect_type": "risk difference",
                            "effect": mean(improved) - mean(other),
                            "association_test": "two-sided Fisher exact",
                            "association_p": fisher_exact_two_sided(a, b, c, d),
                        }
                    )
                else:
                    summary.update(
                        {
                            "effect_type": "standardized mean difference",
                            "effect": standardized_mean_difference(improved, other),
                            "association_test": "descriptive bootstrap; no multiplicity-adjusted hypothesis test",
                            "association_p": "",
                        }
                    )
                output.append(summary)

    unavailable = (
        ("pre-decision", "root_score_gap", "not stored for the controlled 400-unit cohort"),
        ("pre-decision", "root_routing_entropy", "not stored for the controlled 400-unit cohort"),
        ("pre-decision", "facet_disagreement", "faceted root routing was not used in the controlled cohort"),
        ("pre-decision", "successor_rank", "not joined at model-example level in the frozen traces"),
        ("pre-decision", "native_R_at_K", "available only as aggregate topology, not per paired unit"),
        ("pre-decision", "native_MRR", "available only as aggregate topology, not per paired unit"),
        ("pre-decision", "shortcut_or_contraction", "available only as aggregate topology, not per paired unit"),
        ("pre-decision", "branching_or_frontier_competition", "not stored for this controlled policy"),
        ("post-treatment", "single_consumer_layer", "the matched policy uses a fixed four-layer consumer schedule"),
        ("post-treatment", "intervention_state_displacement", "stored for a different 32-example cohort, not all paired units"),
        ("post-treatment", "pra_output_divergence_ratio", "stored for a different 32-example cohort, not all paired units"),
    )
    for timing, feature, note in unavailable:
        output.append(
            {
                "comparison_scope": "all_400",
                "timing": timing,
                "feature": feature,
                "status": "unavailable",
                "note": note,
            }
        )
    return output


def confusion_metrics(truth: list[int], predictions: list[int]) -> dict[str, float | int]:
    true_positive = sum(actual == 1 and predicted == 1 for actual, predicted in zip(truth, predictions))
    false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in zip(truth, predictions))
    true_negative = sum(actual == 0 and predicted == 0 for actual, predicted in zip(truth, predictions))
    false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in zip(truth, predictions))
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / len(truth),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative
            else 0.0
        ),
    }


def threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    return [unique[0] - 1e-12, *[(left + right) / 2 for left, right in zip(unique, unique[1:])], unique[-1] + 1e-12]


def threshold_predictions(rows: list[dict], feature: str, direction: str, threshold: float) -> list[int]:
    if direction == "le":
        return [int(number(row[feature]) <= threshold) for row in rows]
    return [int(number(row[feature]) >= threshold) for row in rows]


def select_stump(rows: list[dict], features: tuple[str, ...]) -> dict[str, float | str]:
    truth = [int(row["path_improved"]) for row in rows]
    best: tuple[tuple[float, float, int, int], dict[str, float | str]] | None = None
    for feature_index, feature in enumerate(features):
        values = [number(row[feature]) for row in rows]
        for direction_index, direction in enumerate(("le", "ge")):
            for threshold in threshold_candidates(values):
                predictions = threshold_predictions(rows, feature, direction, threshold)
                metrics = confusion_metrics(truth, predictions)
                # Prefer balanced accuracy, then fewer retries, then the fixed feature/direction order.
                key = (
                    number(metrics["balanced_accuracy"]),
                    -statistics.fmean(predictions),
                    -feature_index,
                    -direction_index,
                )
                candidate = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "training_balanced_accuracy": metrics["balanced_accuracy"],
                    "training_predicted_retry_rate": statistics.fmean(predictions),
                }
                if best is None or key > best[0]:
                    best = key, candidate
    assert best is not None
    return best[1]


def grouped_bootstrap_interval(
    prediction_rows: list[dict], metric: str, *, seed: int = 2535, draws: int = 10000
) -> list[float]:
    grouped: dict[str, list[dict]] = {}
    for row in prediction_rows:
        grouped.setdefault(str(row["example_id"]), []).append(row)
    identities = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(identities) for _ in identities]
        rows = [row for identity in sampled for row in grouped[identity]]
        metrics = confusion_metrics(
            [int(row["actual"]) for row in rows],
            [int(row["predicted"]) for row in rows],
        )
        estimates.append(number(metrics[metric]))
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[min(int(0.975 * draws), draws - 1)]]


def cross_validate_stump(eligible: list[dict], features: tuple[str, ...]) -> dict:
    identities = sorted({str(row["example_id"]) for row in eligible})
    prediction_rows = []
    folds = []
    for held_out_identity in identities:
        training = [row for row in eligible if row["example_id"] != held_out_identity]
        held_out = [row for row in eligible if row["example_id"] == held_out_identity]
        stump = select_stump(training, features)
        predictions = threshold_predictions(
            held_out,
            str(stump["feature"]),
            str(stump["direction"]),
            number(stump["threshold"]),
        )
        truth = [int(row["path_improved"]) for row in held_out]
        fold_metrics = confusion_metrics(truth, predictions)
        folds.append(
            {
                "held_out_example_id": held_out_identity,
                "held_out_units": len(held_out),
                "held_out_positive_units": sum(truth),
                **stump,
                **{f"held_out_{key}": value for key, value in fold_metrics.items()},
            }
        )
        prediction_rows.extend(
            {
                "example_id": held_out_identity,
                "actual": actual,
                "predicted": predicted,
            }
            for actual, predicted in zip(truth, predictions)
        )

    metrics = confusion_metrics(
        [int(row["actual"]) for row in prediction_rows],
        [int(row["predicted"]) for row in prediction_rows],
    )
    selected_feature_counts = {
        feature: sum(fold["feature"] == feature for fold in folds)
        for feature in features
        if any(fold["feature"] == feature for fold in folds)
    }
    return {
        "method": "one-feature threshold selected inside each leave-one-example-identity-out fold",
        "selection_metric": "training balanced accuracy; retry-rate and fixed-order tie breaks",
        "candidate_features": list(features),
        "post_treatment_features_used": False,
        "pooled_held_out_metrics": metrics,
        "grouped_bootstrap_ci95": {
            "balanced_accuracy": grouped_bootstrap_interval(prediction_rows, "balanced_accuracy"),
            "precision": grouped_bootstrap_interval(prediction_rows, "precision"),
            "recall": grouped_bootstrap_interval(prediction_rows, "recall"),
        },
        "always_no_retry_baseline": {
            "accuracy": sum(not int(row["path_improved"]) for row in eligible) / len(eligible),
            "balanced_accuracy": 0.5,
        },
        "selected_feature_counts": selected_feature_counts,
        "folds": folds,
    }


def analyze_predictability(rows: list[dict]) -> dict:
    eligible = [row for row in rows if number(row["one_shot_path_recovery"]) == 0]
    identities = sorted({str(row["example_id"]) for row in eligible})
    label_free = cross_validate_stump(eligible, LABEL_FREE_PREDICTABILITY_FEATURES)
    query_length = cross_validate_stump(eligible, ("query_token_count",))
    label_free_metrics = label_free["pooled_held_out_metrics"]
    query_length_metrics = query_length["pooled_held_out_metrics"]
    result = {
        "schema_version": "1.1",
        "analysis_scope": "existing controlled traces only; no model or router execution",
        "eligibility": "one_shot_path_recovery == 0",
        "target": "path_gain > 0",
        "eligible_units": len(eligible),
        "positive_units": sum(int(row["path_improved"]) for row in eligible),
        "negative_units": sum(not int(row["path_improved"]) for row in eligible),
        "independent_task_identities": len(identities),
        "primary_label_free_diagnostic": label_free,
        "generator_coupled_query_length_sensitivity": query_length,
        "interpretation": (
            "Label-free one-shot observables excluding generator-coupled query length are weak: "
            f"balanced accuracy is {number(label_free_metrics['balanced_accuracy']):.3f} and "
            f"precision is {number(label_free_metrics['precision']):.3f}. Query length appears "
            f"stronger ({number(query_length_metrics['balanced_accuracy']):.3f} balanced accuracy) "
            "because the synthetic generator couples it to chain depth. Neither result justifies a "
            "deployable retry controller; larger natural independent cohorts are required."
        ),
    }
    assert len(eligible) == 248
    assert sum(int(row["path_improved"]) for row in eligible) == 59
    assert len(label_free["folds"]) == 16
    assert (
        label_free_metrics["true_positive"],
        label_free_metrics["false_positive"],
        label_free_metrics["true_negative"],
        label_free_metrics["false_negative"],
    ) == (35, 60, 129, 24)
    assert round(number(query_length_metrics["balanced_accuracy"]), 3) == 0.839
    return result


def build_dataset_summary(results: Path) -> list[dict]:
    cross_rows = {row["dataset"]: row for row in read_csv(results / "final_metrics" / "cross_dataset_summary.csv")}
    query_rows = read_csv(results / "query_entry_facets" / "query_entry_summary.csv")

    def query(dataset: str, variant: str) -> dict[str, str]:
        return next(
            row
            for row in query_rows
            if row["dataset"] == dataset
            and row["partition"] == "test"
            and row["fraction"] == "0.2"
            and row["variant"] == variant
        )

    labels = {
        "qasper": (
            "low in the held-out direct-retrieval control",
            "low/moderate",
            "not central; no annotated transition graph",
            "no selected-root gain at 20%",
            "direct retrieval and materialization",
        ),
        "hotpotqa": (
            "bridge-ambiguous despite high small-cohort candidate R@4",
            "moderate; coarse recovery selects about half the source",
            "useful after a correct bridge root, but propagation adds little here",
            "clear selected-root inclusion gain from 4-token contextual facets",
            "robust multi-intent root activation and evidence concentration",
        ),
        "2wikimultihopqa": (
            "moderate executable-to-oracle gap",
            "moderate/high at the broad recovery point",
            "strong at moderate K (edge R@6 reported)",
            "high all-offset oracle ceiling",
            "root/facet selection before strong successor topology",
        ),
        "musique": (
            "high executable-to-oracle gap",
            "distributed evidence with substantial selected source",
            "useful but contracted; annotated depth is not preserved",
            "high all-offset oracle ceiling",
            "root selection, dispersion, and contraction",
        ),
    }
    output = []
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa", "musique"):
        source = cross_rows[dataset]
        geometry = labels[dataset]
        row = {
            "dataset": dataset,
            "role": source["role"],
            "routed_root_R_at_4": source["routed_root_R_at_4"],
            "oracle_facet_R_at_4": source["oracle_facet_R_at_4"],
            "edge_R_at_6": source["edge_R_at_6"],
            "complete_recovery": source["complete_recovery"],
            "selected_source_fraction": source["selected_source_fraction"],
            "query_to_root_difficulty": geometry[0],
            "evidence_dispersion": geometry[1],
            "memory_to_memory_topology": geometry[2],
            "facet_benefit": geometry[3],
            "main_bottleneck": geometry[4],
            "canonical_source": "final_metrics/cross_dataset_summary.csv",
        }
        if dataset in {"qasper", "hotpotqa"}:
            global_row = query(dataset, "A_global_semantic")
            facet_row = query(dataset, "B_multi_span_semantic")
            row.update(
                {
                    "global_selected_root_presence_at_20pct": global_row["oracle_root_present"],
                    "facet_selected_root_presence_at_20pct": facet_row["oracle_root_present"],
                    "facet_presence_gain_at_20pct": number(facet_row["oracle_root_present"])
                    - number(global_row["oracle_root_present"]),
                    "global_query_parent_comparisons": global_row["search_comparisons"],
                    "facet_query_parent_comparisons": facet_row["search_comparisons"],
                }
            )
        output.append(row)
    return output


def write_parameter_directionality(path: Path) -> None:
    path.write_text(
        "# PRA Parameter Directionality\n\n"
        "These are dominant directions, not universal monotonic laws. Effects depend on dataset, "
        "layer, granularity, and routing quality.\n\n"
        "| Parameter increase | Root recall | Path recall | Distractor load | Active K/V | Search cost | Typical benefit |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        "| `F` query facets | up | indirectly up | up | indirectly up | up | ambiguous or multi-intent queries |\n"
        "| `R` roots | up | up | up strongly | up | up | uncertain root rank |\n"
        "| `K` neighbors | unchanged | up | up | up | up strongly | noisy successor ranking |\n"
        "| `H` hops | unchanged | up for deep paths | up strongly | up | up strongly | distributed evidence |\n"
        "| `B` final budget | up or unchanged | up | up strongly | up strongly | unchanged or up | incomplete coverage |\n"
        "| `theta` threshold | precision up, recall down | varies | down | down | down | confidence filtering |\n"
        "| PRA consumer layers | unchanged | potentially up | interference possible | up strongly | up | repeated assimilation |\n"
        "| Finer chunks | varies | edge recall can fall | payload down | down per node | node count up | precise disclosure |\n\n"
        "Query-region location and extent are separate adaptive variables: real prompts may place the "
        "active request before, within, or after logs, URLs, and other serialized context. No "
        "query-region controller is evaluated in Paper 2.5.\n\n"
        "Artifact anchors: `query_entry_facets/query_entry_summary.csv`, "
        "`natural_graph_depth/natural_graph_depth_results.json`, "
        "`natural_graph_depth/cross_dataset_granularity.csv`, "
        "`controlled_local_sa_v6/oracle_consumption_ceiling.csv`, and "
        "`controlled_local_sa_v6/consumer_layer_profile.csv`.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    output = args.results / "final_reviewer_patch"
    output.mkdir(parents=True, exist_ok=True)

    rows = build_iteration_rows(args.results)
    summary = summarize_features(rows)
    predictability = analyze_predictability(rows)
    datasets = build_dataset_summary(args.results)
    write_csv(output / "iteration_benefit_59_vs_341.csv", rows)
    write_csv(output / "iteration_benefit_feature_summary.csv", summary)
    (output / "iteration_benefit_predictability.json").write_text(
        json.dumps(predictability, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "dataset_routing_geometry_summary.csv", datasets)
    write_parameter_directionality(output / "pra_parameter_directionality.md")

    margin_values = [number(row["margin_gain"]) for row in rows if row["path_improved"]]
    assert round(mean(margin_values), 3) == 2.164
    print(f"wrote {len(rows)} units: 59 G+, 341 G0")
    print("G+ margin gain 2.164; frozen paired-bootstrap 95% CI [1.336, 2.971]")


if __name__ == "__main__":
    main()
