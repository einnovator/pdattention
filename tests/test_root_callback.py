from __future__ import annotations

import pytest

from pra_hf.root_callback import (
    LinearRootCallback,
    NoOpRootCallback,
    RootCallbackExecutor,
    RootDecision,
    RootState,
    ThresholdRootCallback,
)


def _state(**overrides) -> RootState:
    values = {
        "example_id": "example-1",
        "query_features": {"query_tokens": 12.0},
        "facet_mode": "syntactic",
        "facet_count": 2,
        "root_method": "semantic",
        "root_ids": ("a", "b"),
        "root_scores": (0.8, 0.6),
        "root_top1_score": 0.8,
        "root_score_gap": 0.2,
        "candidate_entropy": 0.4,
        "channel_agreement": 0.75,
        "channel_disagreement": 1.0,
        "root_embedding": (1.0, 0.0),
        "new_entities": ("Ada",),
        "new_addresses": ("ada",),
        "address_count": 1,
        "address_rarity": 0.8,
        "facet_agreement": 0.5,
        "root_dispersion": 0.2,
        "evidence_proxy": 0.7,
        "searched_fraction": 0.25,
        "remaining_search_budget": 2,
        "total_search_budget": 4,
        "remaining_kv_budget": 2,
        "total_kv_budget": 4,
    }
    values.update(overrides)
    return RootState(**values)


def test_noop_callback_preserves_preselected_successor() -> None:
    decision = RootDecision(successor_method="bm25_state", successor_k=2)
    assert NoOpRootCallback(decision).on_root_selected(_state()) == decision


def test_threshold_callback_refines_only_uncertain_roots() -> None:
    callback = ThresholdRootCallback(gap_threshold=0.05, entropy_threshold=0.8)
    assert callback.on_root_selected(_state()).action == "continue"
    uncertain = _state(root_score_gap=0.01)
    decision = callback.on_root_selected(uncertain)
    assert decision.action == "graph_refine"
    assert decision.graph_refine


def test_root_state_rejects_evaluator_or_budget_inconsistency() -> None:
    with pytest.raises(ValueError, match="align"):
        _state(root_scores=(0.8,))
    with pytest.raises(ValueError, match="exceeds"):
        _state(remaining_search_budget=5)


def test_linear_callback_uses_post_root_state() -> None:
    continue_decision = RootDecision(action="continue")
    refine_decision = RootDecision(action="graph_refine", graph_refine=True)
    decisions = {
        continue_decision.label: continue_decision,
        refine_decision.label: refine_decision,
    }
    states = [
        _state(example_id=f"clear-{index}", root_score_gap=0.4, candidate_entropy=0.1)
        for index in range(8)
    ] + [
        _state(example_id=f"uncertain-{index}", root_score_gap=0.0, candidate_entropy=0.95)
        for index in range(8)
    ]
    labels = [continue_decision.label] * 8 + [refine_decision.label] * 8
    model = LinearRootCallback.fit(
        states,
        labels,
        decisions,
        query_feature_names=("query_tokens",),
        seed=11,
    )
    assert model.on_root_selected(states[0]).action == "continue"
    assert model.on_root_selected(states[-1]).action == "graph_refine"


def test_root_decision_validates_refinement_flag() -> None:
    with pytest.raises(ValueError, match="must enable"):
        RootDecision(action="graph_refine")


def test_executor_dispatches_exactly_one_root_selected_event() -> None:
    decision = RootDecision(successor_method="bm25_state")
    controller = NoOpRootCallback(decision)
    observed = []

    def root_executor(action):
        observed.append(("root", action))
        return _state()

    def successor_executor(action, state, selected):
        observed.append(("successor", action, state.example_id, selected.label))
        return (state.root_ids, selected.successor_method)

    result = RootCallbackExecutor(controller).run(
        {"query_tokens": 12.0},
        root_executor,
        successor_executor,
        initial_action="global.f1.semantic",
    )
    assert result.callback_events == 1
    assert result.result == (("a", "b"), "bm25_state")
    assert [row[0] for row in observed] == ["root", "successor"]
