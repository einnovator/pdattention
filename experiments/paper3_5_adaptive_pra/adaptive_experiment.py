"""Frozen validation/held-out adaptive-compute experiment for Paper 3.5."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pra_hf.adaptive_runtime import (
    AdaptiveRetryAgent,
    AttemptResult,
    ControllerFeatures,
    HandRuleController,
    LinearEffortController,
    StopPolicy,
    calibration_metrics,
    default_effort_profiles,
    risk_coverage_curve,
    save_effort_profiles,
)


PROFILE_FRACTIONS = {"E0_low": 0.1, "E1_medium": 0.2, "E2_high": 0.4}
CONTROLLER_FEATURE_NAMES = (
    "query_length",
    "relation_density",
    "facet_disagreement",
    "top_root_score",
    "root_score_gap",
    "topk_score_gap",
    "routing_entropy",
    "competitive_roots",
    "facet_agreement",
    "frontier_mean",
    "frontier_std",
    "newly_discovered_memory",
    "path_convergence",
    "selected_source_fraction",
    "active_native_kv",
    "attempt",
)


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


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return statistics.fmean(values) if values else default


def _routing_entropy(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    tensor = torch.tensor(scores, dtype=torch.float64)
    probabilities = torch.softmax(tensor, dim=0)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return entropy / math.log(len(scores))


def routing_features(row: dict[str, str], attempt: int = 0) -> ControllerFeatures:
    """Extract only runtime-observable fields from one frozen routing trace."""

    scores = [_finite(value) for value in json.loads(row["root_scores"])]
    ordered = sorted(scores, reverse=True)
    confidence = json.loads(row["root_confidence"])
    transitions = json.loads(row["transition_confidence"])
    overlap = _mean(_finite(value.get("top4_overlap")) for value in transitions)
    same_top1 = _mean(float(bool(value.get("same_top1"))) for value in transitions)
    normalized_transition_entropy = _mean(
        _finite(value.get("normalized_entropy"), 1.0) for value in transitions
    )
    root_gap = (
        ordered[0] - ordered[1]
        if len(ordered) > 1
        else 1.0
    )
    probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0)
    competitive = int((probabilities >= 0.75 / max(len(scores), 1)).sum())
    return ControllerFeatures.from_runtime_mapping(
        {
            "query_length": float(len(scores)),
            "sentence_count": 1.0,
            "entity_density": competitive / max(len(scores), 1),
            "relation_density": _finite(row.get("transition_comparisons")) / max(
                _finite(row.get("root_comparisons"), 1.0), 1.0
            ),
            "facet_disagreement": 1.0 - overlap,
            "top_root_score": ordered[0] if ordered else 0.0,
            "root_score_gap": root_gap,
            "topk_score_gap": math.tanh(
                _finite(confidence.get("top1_topB_spread_z"), root_gap)
            ),
            "routing_entropy": max(_routing_entropy(scores), normalized_transition_entropy),
            "competitive_roots": float(competitive),
            "facet_agreement": overlap,
            "frontier_mean": _mean(scores),
            "frontier_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "newly_discovered_memory": float(len(json.loads(row["propagated_ids"]))),
            "path_convergence": same_top1,
            "selected_source_fraction": _finite(row["active_final_kv_fraction"]),
            "active_native_kv": _finite(row["active_final_kv_tokens"]),
            "attempt": float(attempt),
        }
    )


def _selected_rows(path: Path) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(path)
        if row["transition_geometry"] == "semantic"
        and row["lock_policy"] == "seed_agreement_0.6"
        and row["transition_policy"] == "fixed_k4"
        and round(float(row["fraction"]), 1) in {0.1, 0.2, 0.4}
    ]
    if not rows:
        raise ValueError("No rows match the frozen Paper-2.5 semantic exploration policy.")
    return rows


def build_examples(path: Path) -> list[dict[str, Any]]:
    """Group E0/E1/E2 attempts without using evaluator labels as features."""

    profile_by_fraction = {value: name for name, value in PROFILE_FRACTIONS.items()}
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in _selected_rows(path):
        identity = (
            row["partition"],
            row["dataset"],
            row["example_id"],
            int(row["seed"]),
        )
        profile = profile_by_fraction[round(float(row["fraction"]), 1)]
        grouped[identity][profile] = row
    output = []
    profiles = default_effort_profiles()
    for identity, attempts in sorted(grouped.items()):
        if set(attempts) != set(PROFILE_FRACTIONS):
            continue
        minimum = profiles[-1].name
        for profile in profiles:
            if _finite(attempts[profile.name]["chain_complete"]) >= 1.0:
                minimum = profile.name
                break
        partition, dataset, example_id, seed = identity
        output.append(
            {
                "partition": partition,
                "dataset": dataset,
                "example_id": example_id,
                "seed": seed,
                "features": routing_features(attempts["E0_low"]),
                "minimum_effort": minimum,
                "attempts": attempts,
            }
        )
    if not output:
        raise ValueError("Frozen routing traces did not form complete effort ladders.")
    return output


def _fit_hand_rule(examples: list[dict[str, Any]]) -> HandRuleController:
    entropies = sorted(example["features"].routing_entropy for example in examples)
    gaps = sorted(example["features"].root_score_gap for example in examples)

    def quantile(values: list[float], fraction: float) -> float:
        return values[min(int((len(values) - 1) * fraction), len(values) - 1)]

    return HandRuleController(
        medium_entropy=quantile(entropies, 0.40),
        high_entropy=quantile(entropies, 0.75),
        medium_root_gap=quantile(gaps, 0.60),
        high_root_gap=quantile(gaps, 0.25),
    )


def _attempt_features(example: dict[str, Any], profile_name: str) -> ControllerFeatures:
    level = [profile.name for profile in default_effort_profiles()].index(profile_name)
    return routing_features(example["attempts"][profile_name], attempt=level)


def _fit_failure_controller(examples: list[dict[str, Any]]) -> LinearEffortController:
    features, targets = [], []
    for example in examples:
        for profile in default_effort_profiles():
            row = example["attempts"][profile.name]
            features.append(_attempt_features(example, profile.name))
            targets.append("success" if _finite(row["chain_complete"]) >= 1.0 else "failure")
    return LinearEffortController.fit(
        features,
        targets,
        ("success", "failure"),
        feature_names=CONTROLLER_FEATURE_NAMES,
        ridge=0.05,
    )


def _failure_ablation(
    validation: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    train_features, train_targets = [], []
    for example in validation:
        for profile in default_effort_profiles():
            train_features.append(_attempt_features(example, profile.name))
            train_targets.append(
                "failure"
                if _finite(example["attempts"][profile.name]["chain_complete"]) < 1.0
                else "success"
            )
    controller = LinearEffortController.fit(
        train_features,
        train_targets,
        ("success", "failure"),
        feature_names=feature_names,
        ridge=0.05,
    )
    probabilities, labels = [], []
    for example in heldout:
        for profile in default_effort_profiles():
            probabilities.append(
                controller.probabilities(_attempt_features(example, profile.name))["failure"]
            )
            labels.append(
                int(_finite(example["attempts"][profile.name]["chain_complete"]) < 1.0)
            )
    return {
        **calibration_metrics(probabilities, labels),
        "feature_names": list(feature_names),
        "risk_coverage": risk_coverage_curve(probabilities, labels),
    }


def _profile_index(name: str) -> int:
    return [profile.name for profile in default_effort_profiles()].index(name)


def _direct_outcome(example: dict[str, Any], name: str, method: str) -> dict[str, Any]:
    profile = default_effort_profiles()[_profile_index(name)]
    row = example["attempts"][name]
    return {
        "partition": example["partition"],
        "dataset": example["dataset"],
        "example_id": example["example_id"],
        "seed": example["seed"],
        "method": method,
        "initial_effort": name,
        "final_effort": name,
        "attempts": 1,
        "quality": _finite(row["chain_complete"]),
        "oracle_recall": _finite(row["oracle_recall"]),
        "effort_cost": profile.cost_units,
        "routing_seconds": _finite(row["estimated_full_routing_seconds"]),
        "active_native_kv": int(_finite(row["active_final_kv_tokens"])),
        "initially_wrong_corrected": 0,
        "initially_correct_broken": 0,
        "no_change_retry": 0,
    }


def _retry_outcome(
    example: dict[str, Any],
    initial_effort: str,
    failure_controller: LinearEffortController,
    threshold: float,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = default_effort_profiles()
    initial_index = _profile_index(initial_effort)
    prior_profiles = profiles[initial_index:]
    agent = AdaptiveRetryAgent(
        prior_profiles,
        StopPolicy(
            max_incorrect_probability=threshold,
            max_routing_entropy=1.01,
            min_answer_margin=-math.inf,
            min_retry_consistency=0.0,
        ),
        max_retries=len(prior_profiles) - 1,
    )

    def execute(profile, previous):
        row = example["attempts"][profile.name]
        features = _attempt_features(example, profile.name)
        probability = failure_controller.probabilities(features)["failure"]
        selected = tuple(json.loads(row["final_ids"]))
        success = int(_finite(row["chain_complete"]) >= 1.0)
        return AttemptResult(
            answer="path-complete" if success else "path-incomplete",
            features=features,
            incorrect_probability=probability,
            search_seconds=_finite(row["estimated_full_routing_seconds"]),
            materialization_seconds=0.0,
            generation_seconds=0.001,
            active_native_kv=int(_finite(row["active_final_kv_tokens"])),
            selected_parents=selected,
            reusable_state={"selected": selected},
            reused_search_items=(
                len(set(selected) & set(previous.selected_parents)) if previous else 0
            ),
            reused_kv_tokens=(
                min(int(_finite(row["active_final_kv_tokens"])), previous.active_native_kv)
                if previous
                else 0
            ),
            metadata={
                "evaluation_success": success,
                "oracle_recall": _finite(row["oracle_recall"]),
            },
        )

    result = agent.run(execute, example["features"], mode="manual", effort=prior_profiles[0].name, retry_with_more_effort=True)
    final_success = int(result.result.metadata["evaluation_success"])
    initial_success = int(
        _finite(example["attempts"][initial_effort]["chain_complete"]) >= 1.0
    )
    profile = profiles[_profile_index(result.final_effort)]
    # Search/K/V work is reused monotonically; each regeneration adds a small
    # fixed controlled cost rather than paying the whole previous profile again.
    effort_cost = profile.cost_units + max(0, result.attempts - 1) * 2.0
    traces = []
    for trace in result.traces:
        value = trace.to_dict()
        value.update(
            {
                "partition": example["partition"],
                "dataset": example["dataset"],
                "example_id": example["example_id"],
                "seed": example["seed"],
                "method": method,
            }
        )
        traces.append(value)
    outcome = {
        "partition": example["partition"],
        "dataset": example["dataset"],
        "example_id": example["example_id"],
        "seed": example["seed"],
        "method": method,
        "initial_effort": initial_effort,
        "final_effort": result.final_effort,
        "attempts": result.attempts,
        "quality": final_success,
        "oracle_recall": result.result.metadata["oracle_recall"],
        "effort_cost": effort_cost,
        "routing_seconds": sum(trace.search_seconds for trace in result.traces),
        "active_native_kv": result.result.active_native_kv,
        "initially_wrong_corrected": int(not initial_success and final_success),
        "initially_correct_broken": int(initial_success and not final_success),
        "no_change_retry": int(result.attempts > 1 and initial_success == final_success),
    }
    return outcome, traces


def _aggregate_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)
    output = []
    for (dataset, method), values in sorted(groups.items()):
        costs = sorted(float(row["effort_cost"]) for row in values)
        latencies = sorted(float(row["routing_seconds"]) for row in values)

        def percentile(numbers: list[float], fraction: float) -> float:
            return numbers[min(int((len(numbers) - 1) * fraction), len(numbers) - 1)]

        output.append(
            {
                "dataset": dataset,
                "method": method,
                "examples": len(values),
                "quality": _mean(float(row["quality"]) for row in values),
                "oracle_recall": _mean(float(row["oracle_recall"]) for row in values),
                "mean_effort_cost": _mean(costs),
                "median_effort_cost": statistics.median(costs),
                "p90_effort_cost": percentile(costs, 0.90),
                "p95_effort_cost": percentile(costs, 0.95),
                "worst_effort_cost": max(costs),
                "mean_routing_seconds": _mean(latencies),
                "p95_routing_seconds": percentile(latencies, 0.95),
                "mean_attempts": _mean(float(row["attempts"]) for row in values),
                "escalation_rate": _mean(float(row["attempts"] > 1) for row in values),
                "correction_rate": _mean(float(row["initially_wrong_corrected"]) for row in values),
                "breakage_rate": _mean(float(row["initially_correct_broken"]) for row in values),
            }
        )
    return output


def _select_retry_threshold(
    validation: list[dict[str, Any]],
    effort_controller: LinearEffortController,
    failure_controller: LinearEffortController,
) -> tuple[float, list[dict[str, Any]]]:
    candidates = []
    always_high = _mean(
        _finite(example["attempts"]["E2_high"]["chain_complete"])
        for example in validation
    )
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        outcomes = []
        for example in validation:
            initial = effort_controller.choose(example["features"])
            outcome, _ = _retry_outcome(
                example, initial, failure_controller, threshold, "learned_retry"
            )
            outcomes.append(outcome)
        quality = _mean(row["quality"] for row in outcomes)
        cost = _mean(row["effort_cost"] for row in outcomes)
        candidates.append(
            {
                "threshold": threshold,
                "quality": quality,
                "mean_effort_cost": cost,
                "quality_gap_vs_always_high": quality - always_high,
                "eligible": quality >= always_high - 0.01,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    selected = min(eligible or candidates, key=lambda row: (-row["quality"], row["mean_effort_cost"]))
    return float(selected["threshold"]), candidates


def _output_entropy_calibration(toy_path: Path) -> dict[str, Any]:
    rows = [
        row for row in read_csv(toy_path)
        if row["policy"] == "T0_none" and row["W"] == "w16"
    ]
    grouped = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in ("validation", "heldout")
    }
    def features(row: dict[str, str]) -> ControllerFeatures:
        return ControllerFeatures.from_runtime_mapping(
            {
                "output_entropy_mean": _finite(row["prediction_entropy"]),
                "output_entropy_max": _finite(row["prediction_entropy"]),
                "answer_log_probability": -_finite(row["nll"]),
                "answer_margin": _finite(row["correct_margin"]),
            }
        )

    labels = [int(not int(float(row["correct"]))) for row in grouped["heldout"]]
    configurations = {
        "entropy_only": ("output_entropy_mean",),
        "answer_margin_only": ("answer_margin",),
        "answer_log_probability_only": ("answer_log_probability",),
        "combined_output": (
            "output_entropy_mean",
            "answer_log_probability",
            "answer_margin",
        ),
    }
    ablations = {}
    controllers = {}
    for name, feature_names in configurations.items():
        controller = LinearEffortController.fit(
            [features(row) for row in grouped["validation"]],
            ["incorrect" if not int(float(row["correct"])) else "correct" for row in grouped["validation"]],
            ("correct", "incorrect"),
            feature_names=feature_names,
            ridge=0.05,
        )
        probabilities = [
            controller.probabilities(features(row))["incorrect"]
            for row in grouped["heldout"]
        ]
        confident_wrong = sum(
            probability < 0.2 and label
            for probability, label in zip(probabilities, labels)
        )
        ablations[name] = {
            **calibration_metrics(probabilities, labels),
            "feature_names": list(feature_names),
            "confidently_wrong_count_at_p_lt_0_2": confident_wrong,
            "confidently_wrong_rate": confident_wrong / max(sum(labels), 1),
            "risk_coverage": risk_coverage_curve(probabilities, labels),
        }
        controllers[name] = controller.to_dict()
    return {
        **ablations["combined_output"],
        "test_examples": len(labels),
        "incorrect_rate": _mean(labels),
        "ablations": ablations,
        "controllers": controllers,
    }


def run_adaptive_experiment(
    transition_rows_path: Path,
    toy_rows_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train on validation, freeze, evaluate held-out, and write public artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = default_effort_profiles()
    save_effort_profiles(output_dir / "adaptive_effort_profiles.json", profiles)
    examples = build_examples(transition_rows_path)
    validation = [example for example in examples if example["partition"] == "validation"]
    heldout = [example for example in examples if example["partition"] == "test"]
    effort_controller = LinearEffortController.fit(
        [example["features"] for example in validation],
        [example["minimum_effort"] for example in validation],
        [profile.name for profile in profiles],
        feature_names=CONTROLLER_FEATURE_NAMES,
        ridge=0.05,
    )
    failure_controller = _fit_failure_controller(validation)
    hand = _fit_hand_rule(validation)
    threshold, threshold_audit = _select_retry_threshold(
        validation, effort_controller, failure_controller
    )

    feature_rows, training_rows = [], []
    for example in examples:
        base = {
            "partition": example["partition"],
            "dataset": example["dataset"],
            "example_id": example["example_id"],
            "seed": example["seed"],
            **example["features"].to_dict(),
        }
        feature_rows.append(base)
        if example["partition"] == "validation":
            training_rows.append({**base, "minimum_effort_target": example["minimum_effort"]})
    write_csv(output_dir / "controller_features.csv", feature_rows)
    write_csv(output_dir / "controller_training.csv", training_rows)

    outcomes: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    oracle_rows = []
    regret_rows = []
    failure_probabilities, failure_labels = [], []
    for example in heldout:
        for profile in profiles:
            outcomes.append(_direct_outcome(example, profile.name, f"fixed_{profile.name}"))
            attempt_features = _attempt_features(example, profile.name)
            failure_probabilities.append(
                failure_controller.probabilities(attempt_features)["failure"]
            )
            failure_labels.append(
                int(_finite(example["attempts"][profile.name]["chain_complete"]) < 1.0)
            )
        hand_choice = hand.choose(example["features"], [profile.name for profile in profiles])
        learned_choice = effort_controller.choose(example["features"])
        outcomes.append(_direct_outcome(example, hand_choice, "hand_rule_direct"))
        outcomes.append(_direct_outcome(example, learned_choice, "learned_direct"))
        for method, choice in (("hand_rule_retry", hand_choice), ("learned_retry", learned_choice)):
            outcome, attempt_traces = _retry_outcome(
                example, choice, failure_controller, threshold, method
            )
            outcomes.append(outcome)
            traces.extend(attempt_traces)
        cheap_outcome, cheap_traces = _retry_outcome(
            example, "E0_low", failure_controller, threshold, "cheap_default_retry"
        )
        outcomes.append(cheap_outcome)
        traces.extend(cheap_traces)
        oracle_name = example["minimum_effort"]
        oracle_profile = profiles[_profile_index(oracle_name)]
        oracle_row = {
            "dataset": example["dataset"],
            "example_id": example["example_id"],
            "seed": example["seed"],
            "minimum_effort": oracle_name,
            "minimum_effort_cost": oracle_profile.cost_units,
            "quality": _finite(example["attempts"][oracle_name]["chain_complete"]),
            "analysis_only": True,
        }
        oracle_rows.append(oracle_row)
        for controller_name, choice in (("hand_rule", hand_choice), ("learned", learned_choice)):
            selected_profile = profiles[_profile_index(choice)]
            selected_quality = _finite(example["attempts"][choice]["chain_complete"])
            regret_rows.append(
                {
                    "dataset": example["dataset"],
                    "example_id": example["example_id"],
                    "seed": example["seed"],
                    "controller": controller_name,
                    "selected_effort": choice,
                    "oracle_effort": oracle_name,
                    "cost_regret": selected_profile.cost_units - oracle_profile.cost_units,
                    "quality_matched": int(selected_quality >= oracle_row["quality"]),
                }
            )

    write_csv(output_dir / "retry_outcomes.csv", outcomes)
    with (output_dir / "retry_traces.jsonl").open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, sort_keys=True) + "\n")
    frontier = _aggregate_outcomes(outcomes)
    write_csv(output_dir / "adaptive_compute_frontier.csv", frontier)
    write_csv(output_dir / "oracle_effort_ceiling.csv", oracle_rows)
    write_csv(output_dir / "controller_regret.csv", regret_rows)

    retrieval_calibration = calibration_metrics(failure_probabilities, failure_labels)
    retrieval_calibration["risk_coverage"] = risk_coverage_curve(
        failure_probabilities, failure_labels
    )
    output_calibration = _output_entropy_calibration(toy_rows_path)
    routing_ablations = {
        "routing_entropy_only": _failure_ablation(
            validation, heldout, ("routing_entropy",)
        ),
        "root_gap_only": _failure_ablation(
            validation, heldout, ("root_score_gap", "topk_score_gap")
        ),
        "routing_and_frontier": _failure_ablation(
            validation,
            heldout,
            (
                "routing_entropy",
                "root_score_gap",
                "topk_score_gap",
                "facet_disagreement",
                "facet_agreement",
                "frontier_mean",
                "frontier_std",
                "path_convergence",
            ),
        ),
        "combined_runtime": _failure_ablation(
            validation, heldout, CONTROLLER_FEATURE_NAMES
        ),
    }
    calibration = {
        "schema_version": "1.0",
        "selection_partition": "validation",
        "heldout_partition": "test",
        "selected_retry_failure_threshold": threshold,
        "retry_threshold_selection": threshold_audit,
        "routing_failure_calibration": retrieval_calibration,
        "routing_confidence_ablations": routing_ablations,
        "output_entropy_calibration": output_calibration,
        "learned_effort_controller": effort_controller.to_dict(),
        "learned_failure_controller": failure_controller.to_dict(),
        "hand_rule": hand.__dict__,
        "oracle_fields_in_controller": False,
    }
    (output_dir / "controller_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "profiles": [profile.to_dict() for profile in profiles],
        "validation_examples": len(validation),
        "heldout_examples": len(heldout),
        "selected_retry_threshold": threshold,
        "frontier": frontier,
        "retrieval_calibration": retrieval_calibration,
        "output_entropy_calibration": {
            key: value for key, value in output_calibration.items()
            if key not in {"risk_coverage", "controller"}
        },
        "oracle_distribution": {
            name: sum(row["minimum_effort"] == name for row in oracle_rows)
            / max(len(oracle_rows), 1)
            for name in PROFILE_FRACTIONS
        },
    }
