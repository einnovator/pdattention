from __future__ import annotations

import pytest

from pra_hf.factorized_control import (
    FactorizedEffortAction,
    allocation_outcome,
    changed_control,
    cheapest_sufficient,
    evidence_kv_metrics,
    factorized_action_space,
    factorized_cost,
    pareto_frontier,
)
from experiments.paper3_5_adaptive_pra.addon_study import expanded_query_region_fixtures


def test_factorized_action_construction_and_invalid_budget_rejection() -> None:
    action = FactorizedEffortAction(2, 2, 4, 1, 4, 2)
    assert action.identifier == "F2_R2_K4_H1_Bs4_Bkv2"
    assert FactorizedEffortAction.profile(2).hops == 3
    with pytest.raises(ValueError, match="search_budget"):
        FactorizedEffortAction(1, 4, 2, 0, 2, 4)
    with pytest.raises(ValueError, match="Unsupported neighbors"):
        FactorizedEffortAction(1, 1, 3, 0, 2, 2)
    assert len({action.identifier for action in factorized_action_space()}) == len(
        factorized_action_space()
    )


def test_factorized_cost_exposes_exact_components() -> None:
    action = FactorizedEffortAction.profile(1)
    cost = factorized_cost(
        action,
        parent_count=10,
        transition_comparisons=12,
        materialized_kv_tokens=256,
    )
    assert cost["root_comparisons"] == 20
    assert cost["conceptual_parent_budget"] == 4
    assert cost["abstract_cost"] == 22


def test_token_weighted_evidence_precision_and_recall() -> None:
    metrics = evidence_kv_metrics({0, 2}, {0, 1}, [10, 20, 30])
    assert metrics["evidence_kv_precision"] == pytest.approx(0.25)
    assert metrics["evidence_kv_recall"] == pytest.approx(1 / 3)
    assert metrics["selected_kv_tokens"] == 40


def test_pareto_and_factorized_oracle_are_deterministic() -> None:
    rows = [
        {"config_id": "a", "chain_complete": 0, "evidence_kv_recall": 0.5, "evidence_kv_precision": 0.8, "abstract_cost": 2, "selected_kv_tokens": 20},
        {"config_id": "b", "chain_complete": 1, "evidence_kv_recall": 1.0, "evidence_kv_precision": 0.7, "abstract_cost": 4, "selected_kv_tokens": 30},
        {"config_id": "c", "chain_complete": 1, "evidence_kv_recall": 1.0, "evidence_kv_precision": 0.6, "abstract_cost": 5, "selected_kv_tokens": 40},
    ]
    frontier = pareto_frontier(
        rows,
        maximize=("chain_complete", "evidence_kv_recall", "evidence_kv_precision"),
        minimize=("abstract_cost", "selected_kv_tokens"),
    )
    assert [row["config_id"] for row in frontier] == ["a", "b"]
    assert cheapest_sufficient(rows)["config_id"] == "b"


def test_under_over_allocation_and_targeted_action_names() -> None:
    oracle = {"chain_complete": 1, "abstract_cost": 4}
    assert allocation_outcome({"chain_complete": 0, "abstract_cost": 2}, oracle) == "under_allocation"
    assert allocation_outcome({"chain_complete": 1, "abstract_cost": 6}, oracle) == "over_allocation"
    assert allocation_outcome({"chain_complete": 1, "abstract_cost": 4}, oracle) == "matched"
    low = FactorizedEffortAction.profile(0)
    wider = FactorizedEffortAction(2, 1, 2, 0, 2, 2)
    assert changed_control(low, wider) == "widen_facets"


def test_expanded_query_region_fixtures_cover_natural_and_adversarial_roles() -> None:
    natural, adversarial = expanded_query_region_fixtures()
    assert {row["fixture"] for row in natural} >= {"logs", "source_code", "email_thread"}
    assert {row["fixture"] for row in adversarial} >= {
        "quoted_previous_question",
        "question_inside_log",
        "code_comment_question",
    }
    assert all(row["query"] in row["prompt"] for row in [*natural, *adversarial])
