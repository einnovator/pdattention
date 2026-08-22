from __future__ import annotations

import json

import pytest

from pra_hf.adaptive_search import (
    AdaptiveSearchAction,
    SearchTransition,
    choose_successor_cascade,
    load_search_method_action_spec,
    method_cost_accounting,
    method_retry_action,
    select_method_oracle,
    validate_method_feature_names,
)
from pra_hf.factorized_control import FactorizedEffortAction


def _spec(tmp_path):
    value = {
        "schema_version": "1.0",
        "materialization_performed": False,
        "root_search_methods": {
            name: {"confidence_signals": ["score_gap"], "cost_metrics": ["comparisons"]}
            for name in ("gist", "exact", "bm25", "approx", "hybrid")
        },
        "successor_search_methods": {
            name: {"confidence_signals": ["successor_rank"], "cost_metrics": ["latency_ms"]}
            for name in (
                "native_semantic",
                "exact_new_address",
                "bm25_state",
                "approximate_new_address",
                "hybrid_state",
            )
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_root_and_successor_methods_are_independent() -> None:
    action = AdaptiveSearchAction(
        "semantic", "exact_new_address", FactorizedEffortAction.profile(1)
    )
    assert action.control_vector["S_root"] == "semantic"
    assert action.control_vector["S_succ"] == "exact_new_address"
    assert action.control_vector["B_search"] == 4
    assert action.control_vector["B_KV"] == 4


def test_method_specific_config_validation() -> None:
    effort = FactorizedEffortAction.profile(0)
    with pytest.raises(ValueError, match="root_method"):
        AdaptiveSearchAction("unknown", "native_semantic", effort)
    with pytest.raises(ValueError, match="successor_method"):
        AdaptiveSearchAction("semantic", "unknown", effort)


def test_factorized_method_oracle_is_quality_first_and_deterministic() -> None:
    rows = [
        {"root_method": "exact", "recall": 0.5, "precision": 0.5, "mrr": 0.7, "comparisons": 2},
        {"root_method": "bm25", "recall": 0.5, "precision": 0.6, "mrr": 0.4, "comparisons": 1},
        {"root_method": "semantic", "recall": 0.4, "precision": 1.0, "mrr": 1.0, "comparisons": 0},
    ]
    assert select_method_oracle(rows)["root_method"] == "bm25"


def test_method_features_reject_dataset_and_gold_leakage() -> None:
    assert validate_method_feature_names(("query_tokens", "bm25_gap"))
    with pytest.raises(ValueError, match="Deployment-unsafe"):
        validate_method_feature_names(("dataset", "query_tokens"))
    with pytest.raises(ValueError, match="Deployment-unsafe"):
        validate_method_feature_names(("gold_linked",))


def test_transition_trace_and_targeted_retry() -> None:
    low = FactorizedEffortAction.profile(0)
    before = AdaptiveSearchAction("semantic", "native_semantic", low)
    after = AdaptiveSearchAction("semantic", "exact_new_address", low)
    trace = SearchTransition("semantic", "exact_new_address", 1, "rare address", 0.8)
    assert trace.successor_method == "exact_new_address"
    assert method_retry_action(before, after) == "change_successor_method"


def test_useful_address_gate_is_observable_and_conservative() -> None:
    strong = {"idf": 8, "candidate_count": 1, "successor_rank": 1, "semantic_consistency": 1}
    weak = {"idf": 0, "candidate_count": 8, "successor_rank": 9, "semantic_score_gap": 0.2}
    assert choose_successor_cascade(strong) == "exact_new_address"
    assert choose_successor_cascade(weak) == "native_semantic"


def test_search_and_kv_costs_remain_separate() -> None:
    costs = method_cost_accounting(
        {"comparisons": 10, "index_lookups": 2, "latency_ms": 3},
        {"comparisons": 4, "index_lookups": 1, "latency_ms": 5},
        materialized_kv_tokens=128,
    )
    assert costs["root_comparisons"] == 10
    assert costs["successor_comparisons"] == 4
    assert costs["materialized_kv_tokens"] == 128
    assert "abstract_cost" not in costs


def test_paper2_6_spec_import_is_deterministic(tmp_path) -> None:
    path = _spec(tmp_path)
    first = load_search_method_action_spec(path)
    second = load_search_method_action_spec(path)
    assert first == second
    assert first.root_methods[0] == "semantic"
    assert not first.materialization_performed
