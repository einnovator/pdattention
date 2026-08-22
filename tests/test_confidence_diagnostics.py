import math

import pytest

from pra_hf.confidence_diagnostics import (
    ROOT_SEARCH_METHODS,
    SUCCESSOR_SEARCH_METHODS,
    average_precision,
    binary_auroc,
    bootstrap_best_channel,
    choose_conservative_threshold,
    expected_calibration_error,
    paired_bootstrap_interval,
    selective_metrics,
    validate_observable_feature_names,
    validate_search_method_action_spec,
)


def test_ranking_and_calibration_metrics_are_deterministic():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert binary_auroc(labels, scores) == pytest.approx(0.75)
    assert average_precision(labels, scores) == pytest.approx((1.0 + 2 / 3) / 2)
    assert expected_calibration_error([0, 1], [0.1, 0.9], bins=2) == pytest.approx(0.1)


def test_single_class_auroc_is_explicitly_undefined():
    assert math.isnan(binary_auroc([1, 1], [0.2, 0.8]))


def test_threshold_is_fit_on_supplied_validation_outcomes():
    threshold = choose_conservative_threshold(
        [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], minimum_precision=0.6
    )
    assert threshold == pytest.approx(0.7)
    metrics = selective_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], threshold)
    assert metrics["retained"] == 3
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["wrong_rejected"] == 1


def test_selector_feature_names_reject_truth_and_dataset_identity():
    assert validate_observable_feature_names(["score_gap", "query_length"]) == (
        "score_gap",
        "query_length",
    )
    with pytest.raises(ValueError, match="forbidden"):
        validate_observable_feature_names(["dataset", "gold_recall"])


def test_bootstrap_sampling_is_reproducible_and_identity_paired():
    rows = [
        {"example_id": "a", "channel": "x", "recall": 1.0},
        {"example_id": "a", "channel": "y", "recall": 0.0},
        {"example_id": "b", "channel": "x", "recall": 0.0},
        {"example_id": "b", "channel": "y", "recall": 1.0},
    ]
    first = bootstrap_best_channel(
        rows, cohort_size=2, resamples=10, seed=11, channel_order=("x", "y")
    )
    second = bootstrap_best_channel(
        rows, cohort_size=2, resamples=10, seed=11, channel_order=("x", "y")
    )
    assert first == second
    assert {row["best_channel"] for row in first} <= {"x", "y"}


def test_bootstrap_best_channel_uses_declared_tie_metrics():
    rows = [
        {"example_id": "a", "channel": "x", "recall": 1.0, "precision": 0.2},
        {"example_id": "a", "channel": "y", "recall": 1.0, "precision": 0.8},
    ]
    draws = bootstrap_best_channel(
        rows, cohort_size=1, resamples=3, seed=5, channel_order=("x", "y"),
        tie_metrics=("precision",),
    )
    assert {row["best_channel"] for row in draws} == {"y"}


def test_paired_bootstrap_interval_contains_observed_mean():
    observed, lower, upper = paired_bootstrap_interval([0.0, 1.0, 1.0], seed=7)
    assert observed == pytest.approx(2 / 3)
    assert lower <= observed <= upper


def test_action_spec_requires_independent_root_and_successor_contracts():
    fields = {
        "implementation_id": "module:function",
        "required_state": ["query"],
        "required_index": ["index"],
        "parameters": {},
        "confidence_outputs": ["top_score"],
        "cost_metrics": ["comparisons"],
        "known_failure_modes": ["ambiguity"],
    }
    spec = {
        "materialization_performed": False,
        "root_search_methods": {
            name: {**fields, "stage": "root"} for name in ROOT_SEARCH_METHODS
        },
        "successor_search_methods": {
            name: {**fields, "stage": "successor"}
            for name in SUCCESSOR_SEARCH_METHODS
        },
    }
    validate_search_method_action_spec(spec)
    spec["successor_search_methods"].pop("hybrid_state")
    with pytest.raises(ValueError, match="exactly"):
        validate_search_method_action_spec(spec)
