"""Calibration and confidence diagnostics for PRA discovery policies.

The functions in this module operate on deployment-observable scores. Gold
labels may be supplied as outcomes for evaluation, but never as selector
features. Keeping that boundary here makes the Paper 2.6 analyses reusable by
later adaptive-control work without turning an oracle into a serving policy.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


ROOT_SEARCH_METHODS = ("semantic", "exact", "bm25", "approximate", "hybrid")
SUCCESSOR_SEARCH_METHODS = (
    "native_semantic",
    "exact_new_address",
    "bm25_state",
    "approximate_new_address",
    "hybrid_state",
)

_FORBIDDEN_FEATURE_FRAGMENTS = (
    "answer",
    "dataset",
    "evidence",
    "gold",
    "label",
    "oracle",
    "positive",
    "target",
)


@dataclass(frozen=True)
class CalibrationSummary:
    """Binary ranking and calibration metrics for one confidence signal."""

    examples: int
    positives: int
    auroc: float
    auprc: float
    ece: float
    brier: float


def validate_observable_feature_names(names: Iterable[str]) -> tuple[str, ...]:
    """Reject selector inputs that reveal dataset identity or evaluation truth."""
    normalized = tuple(str(name) for name in names)
    leaked = sorted(
        name
        for name in normalized
        if any(fragment in name.casefold() for fragment in _FORBIDDEN_FEATURE_FRAGMENTS)
    )
    if leaked:
        raise ValueError(f"Non-observable selector features are forbidden: {leaked}")
    return normalized


def binary_auroc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Return tie-aware AUROC, or NaN when only one class is present."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = [float(score) for label, score in zip(labels, scores) if bool(label)]
    negatives = [float(score) for label, score in zip(labels, scores) if not bool(label)]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Return average precision with deterministic score-tie ordering."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(bool(label) for label in labels)
    if not positives:
        return math.nan
    ranked = sorted(
        enumerate(zip(labels, scores)), key=lambda item: (-float(item[1][1]), item[0])
    )
    hits = 0
    total = 0.0
    for rank, (_, (label, _)) in enumerate(ranked, 1):
        if bool(label):
            hits += 1
            total += hits / rank
    return total / positives


def reliability_bins(
    labels: Sequence[int | bool], probabilities: Sequence[float], *, bins: int = 10
) -> list[dict[str, float | int]]:
    """Group probabilities into fixed reliability bins over [0, 1]."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for label, probability in zip(labels, probabilities):
        value = min(1.0, max(0.0, float(probability)))
        grouped[min(int(value * bins), bins - 1)].append((float(bool(label)), value))
    rows = []
    for index, group in enumerate(grouped):
        count = len(group)
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_confidence": sum(value for _, value in group) / count if count else 0.0,
                "accuracy": sum(label for label, _ in group) / count if count else 0.0,
            }
        )
    return rows


def expected_calibration_error(
    labels: Sequence[int | bool], probabilities: Sequence[float], *, bins: int = 10
) -> float:
    """Return fixed-bin expected calibration error."""
    total = len(labels)
    if not total:
        return math.nan
    return sum(
        int(row["count"])
        / total
        * abs(float(row["accuracy"]) - float(row["mean_confidence"]))
        for row in reliability_bins(labels, probabilities, bins=bins)
    )


def summarize_calibration(
    labels: Sequence[int | bool], probabilities: Sequence[float], *, bins: int = 10
) -> CalibrationSummary:
    """Compute ranking and calibration metrics for bounded confidence values."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    clipped = [min(1.0, max(0.0, float(value))) for value in probabilities]
    brier = (
        sum((value - float(bool(label))) ** 2 for label, value in zip(labels, clipped))
        / len(labels)
        if labels
        else math.nan
    )
    return CalibrationSummary(
        examples=len(labels),
        positives=sum(bool(label) for label in labels),
        auroc=binary_auroc(labels, clipped),
        auprc=average_precision(labels, clipped),
        ece=expected_calibration_error(labels, clipped, bins=bins),
        brier=brier,
    )


def percentile_calibrate(reference: Sequence[float], values: Sequence[float]) -> list[float]:
    """Map raw scores to an empirical validation CDF without using test labels."""
    ordered = sorted(float(value) for value in reference)
    if not ordered:
        raise ValueError("reference scores must not be empty")
    return [
        sum(candidate <= float(value) for candidate in ordered) / len(ordered)
        for value in values
    ]


def choose_conservative_threshold(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    minimum_precision: float = 0.8,
) -> float:
    """Select the validation threshold with most retained true positives."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("non-empty labels and probabilities must align")
    candidates = sorted({float(value) for value in probabilities}, reverse=True)
    feasible = []
    fallback = []
    for threshold in candidates:
        retained = [index for index, value in enumerate(probabilities) if value >= threshold]
        true = sum(bool(labels[index]) for index in retained)
        precision = true / max(len(retained), 1)
        row = (true, precision, threshold)
        fallback.append(row)
        if retained and precision >= minimum_precision:
            feasible.append(row)
    selected = max(feasible or fallback, key=lambda row: (row[0], row[1], row[2]))
    return float(selected[2])


def selective_metrics(
    labels: Sequence[int | bool], probabilities: Sequence[float], threshold: float
) -> dict[str, float | int]:
    """Measure precision, recall, and rejection after confidence abstention."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    retained = [index for index, value in enumerate(probabilities) if value >= threshold]
    rejected = [index for index, value in enumerate(probabilities) if value < threshold]
    true_retained = sum(bool(labels[index]) for index in retained)
    positives = sum(bool(label) for label in labels)
    return {
        "threshold": float(threshold),
        "retained": len(retained),
        "rejected": len(rejected),
        "coverage": len(retained) / max(len(labels), 1),
        "precision": true_retained / max(len(retained), 1),
        "recall": true_retained / max(positives, 1),
        "wrong_rejected": sum(not bool(labels[index]) for index in rejected),
        "correct_rejected": sum(bool(labels[index]) for index in rejected),
    }


def paired_bootstrap_interval(
    values: Sequence[float], *, resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Return mean and percentile interval from paired identity-level values."""
    if not values:
        return math.nan, math.nan, math.nan
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    draws = sorted(
        sum(float(values[rng.randrange(len(values))]) for _ in values) / len(values)
        for _ in range(resamples)
    )
    return (
        sum(float(value) for value in values) / len(values),
        draws[int(0.025 * (resamples - 1))],
        draws[int(0.975 * (resamples - 1))],
    )


def bootstrap_best_channel(
    rows: Sequence[Mapping[str, object]],
    *,
    cohort_size: int,
    resamples: int,
    seed: int,
    identity_field: str = "example_id",
    channel_field: str = "channel",
    metric: str = "recall",
    channel_order: Sequence[str],
    tie_metrics: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Resample identities and report the deterministic best channel each time."""
    if cohort_size <= 0 or resamples <= 0:
        raise ValueError("cohort_size and resamples must be positive")
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row[identity_field])][str(row[channel_field])] = {
            name: float(row[name]) for name in (metric, *tie_metrics)
        }
    identities = sorted(grouped)
    if not identities:
        return []
    missing = [
        identity
        for identity in identities
        if not set(channel_order).issubset(grouped[identity])
    ]
    if missing:
        raise ValueError(f"Incomplete channel cross-product for identities: {missing[:3]}")
    rng = random.Random(seed)
    output = []
    for index in range(resamples):
        sample = [rng.choice(identities) for _ in range(cohort_size)]
        means = {
            channel: sum(grouped[identity][channel][metric] for identity in sample) / cohort_size
            for channel in channel_order
        }
        tie_means = {
            channel: tuple(
                sum(grouped[identity][channel][name] for identity in sample) / cohort_size
                for name in tie_metrics
            )
            for channel in channel_order
        }
        best = max(
            channel_order,
            key=lambda channel: (
                means[channel], *tie_means[channel], -channel_order.index(channel)
            ),
        )
        output.append(
            {
                "resample": index,
                "cohort_size": cohort_size,
                "best_channel": best,
                **{f"mean_{channel}": means[channel] for channel in channel_order},
            }
        )
    return output


def validate_search_method_action_spec(spec: Mapping[str, object]) -> None:
    """Validate the Paper 2.6 search-method handoff without importing its runner."""
    if bool(spec.get("materialization_performed", True)):
        raise ValueError("The discovery action spec must not claim K/V materialization")
    groups = (
        ("root_search_methods", ROOT_SEARCH_METHODS, "root"),
        ("successor_search_methods", SUCCESSOR_SEARCH_METHODS, "successor"),
    )
    required = {
        "implementation_id",
        "stage",
        "required_state",
        "required_index",
        "parameters",
        "confidence_outputs",
        "cost_metrics",
        "known_failure_modes",
    }
    for group_name, expected, stage in groups:
        methods = spec.get(group_name)
        if not isinstance(methods, Mapping) or set(methods) != set(expected):
            raise ValueError(f"{group_name} must define exactly {list(expected)}")
        for name, definition in methods.items():
            if not isinstance(definition, Mapping):
                raise ValueError(f"{name} definition must be an object")
            missing = required - set(definition)
            if missing:
                raise ValueError(f"{name} is missing fields: {sorted(missing)}")
            if definition["stage"] != stage:
                raise ValueError(f"{name} has invalid stage {definition['stage']!r}")
