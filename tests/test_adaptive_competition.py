from __future__ import annotations

import json

import torch

from pra_hf.adaptive_competition import (
    AdaptiveCompetitionConfig,
    AdaptiveCompetitionRouter,
    RootLockConfig,
    TransitionConfidence,
    TransitionPolicyConfig,
    TransitionScores,
    adaptive_transition_k,
    lock_root_candidates,
    monotonic_final_selection,
    root_seed_agreement,
)
from experiments.paper2_5_iterative_pra.run_monotonic_adaptive_competition import (
    _aggregate,
    evaluate_policy,
    FrozenTrace,
)


def test_root_top_b_is_computed_before_fixed_locking():
    scores = torch.tensor([0.1, 0.9, 0.8, 0.7, 0.6])
    top_b, locked, _ = lock_root_candidates(
        scores, 4, RootLockConfig("fixed", fixed_count=2)
    )
    assert top_b == [1, 2, 3, 4]
    assert locked == [1, 2]


def test_score_drop_is_deterministic_and_uses_the_first_large_drop():
    scores = torch.tensor([0.90, 0.89, 0.50, 0.10])
    config = RootLockConfig("score_drop", threshold=0.20)
    first = lock_root_candidates(scores, 4, config)
    second = lock_root_candidates(scores, 4, config)
    assert first == second
    assert first[1] == [0, 1]


def test_seed_agreement_is_fractional_and_label_free():
    agreement = root_seed_agreement([[0, 1, 2], [1, 0, 3], [0, 2, 1]], 2)
    assert agreement == {0: 1.0, 1: 2 / 3, 2: 1 / 3}
    _, locked, _ = lock_root_candidates(
        torch.tensor([0.9, 0.8, 0.7, 0.6]),
        3,
        RootLockConfig("seed_agreement", threshold=0.8),
        agreement=agreement,
    )
    assert locked == [0]


def test_adaptive_transition_selects_one_two_or_four():
    config = TransitionPolicyConfig(
        "adaptive", moderate_confidence=0.35, high_confidence=0.55
    )
    common = dict(
        top1_top4_spread=1.0,
        normalized_entropy=0.5,
        top1_rank_distance=0,
    )
    assert adaptive_transition_k(
        TransitionConfidence(**common, concentration=0.7, same_top1=True, top4_overlap=1.0),
        config,
    ) == 1
    assert adaptive_transition_k(
        TransitionConfidence(**common, concentration=0.4, same_top1=False, top4_overlap=0.2),
        config,
    ) == 2
    assert adaptive_transition_k(
        TransitionConfidence(**common, concentration=0.2, same_top1=False, top4_overlap=0.0),
        config,
    ) == 4


def test_monotonic_selection_deduplicates_and_never_evicts_locked_roots():
    selected, propagated = monotonic_final_selection(
        [2, 0], [(2, 99.0), (3, 0.8), (3, 0.7), (1, 0.6)], [2, 0, 4, 1], 4
    )
    assert selected == [2, 0, 3, 1]
    assert propagated == [3, 1]
    assert {2, 0} <= set(selected)
    assert len(selected) == 4


def test_router_conserves_budget_and_uses_raw_native_rank():
    root = torch.tensor([0.9, 0.8, 0.7, 0.6])

    def provider(source: int) -> TransitionScores:
        assert source == 0
        return TransitionScores(
            semantic=torch.tensor([-torch.inf, 0.1, 0.9, 0.8]),
            native_raw=torch.tensor([-torch.inf, 10.0, 12.0, 11.0]),
            semantic_comparisons=4,
            native_qk_comparisons=128,
        )

    result = AdaptiveCompetitionRouter().route(
        root,
        provider,
        AdaptiveCompetitionConfig(
            total_budget=3,
            root_lock=RootLockConfig("fixed", fixed_count=1),
            transition=TransitionPolicyConfig("fixed", fixed_k=2),
            transition_geometry="native_rank",
        ),
    )
    assert result.root_top_b == (0, 1, 2)
    assert result.locked_roots == (0,)
    assert result.propagated == (2, 3)
    assert result.selected == (0, 2, 3)
    assert result.native_qk_comparisons == 128


def test_zero_propagation_when_locked_roots_exhaust_budget():
    called = False

    def provider(_: int) -> TransitionScores:
        nonlocal called
        called = True
        raise AssertionError("provider must not run")

    result = AdaptiveCompetitionRouter().route(
        torch.tensor([0.9, 0.8]),
        provider,
        AdaptiveCompetitionConfig(
            total_budget=2,
            root_lock=RootLockConfig("fixed", fixed_count=4),
            transition=TransitionPolicyConfig("fixed", fixed_k=4),
        ),
    )
    assert result.selected == (0, 1)
    assert result.propagation_budget == 0
    assert not called


def test_public_policy_api_has_no_oracle_or_evidence_parameter():
    import inspect

    signatures = (
        inspect.signature(AdaptiveCompetitionRouter.route),
        inspect.signature(lock_root_candidates),
        inspect.signature(adaptive_transition_k),
    )
    names = {name for signature in signatures for name in signature.parameters}
    assert "oracle" not in names
    assert "evidence" not in names


def test_experiment_row_tracks_matched_budget_and_root_state(monkeypatch):
    feature = {
        "dataset": "hotpotqa",
        "example_id": "x",
        "parent_spans": [(0, 10), (10, 20), (20, 30)],
        "source_tokens": 30,
        "parent_positive_mask": torch.tensor([True, False, True]),
        "evidence_spans": [(0, 5), (20, 25)],
    }
    transition = TransitionScores(
        semantic=torch.tensor([-torch.inf, 0.1, 0.9]),
        native_raw=torch.tensor([-torch.inf, 1.0, 2.0]),
    )
    trace = FrozenTrace(
        root_scores=torch.tensor([0.9, 0.8, 0.1]),
        transitions={0: transition},
        transition_traces={0: {"source_parent": 0}},
    )
    monkeypatch.setattr(
        "experiments.paper2_5_iterative_pra.run_monotonic_adaptive_competition.canonical_oracle_parent_indices",
        lambda _: {0, 2},
    )
    monkeypatch.setattr(
        "experiments.paper2_5_iterative_pra.run_monotonic_adaptive_competition.evidence_parent_groups",
        lambda _: [{0}, {2}],
    )
    row = evaluate_policy(
        feature,
        11,
        2 / 3,
        trace,
        root_policy_name="fixed_prefix_1",
        root_policy=RootLockConfig("fixed", fixed_count=1),
        transition_policy_name="fixed_k1",
        transition_policy=TransitionPolicyConfig("fixed", fixed_k=1),
        geometry="native_rank",
        agreement={0: 1.0},
        stage="test",
    )
    assert row["oracle_root_state"] == "locked"
    assert row["chain_complete"] == 1.0
    assert row["unique_parents_selected"] == row["budget"] == 2
    assert set(json.loads(row["locked_root_ids"])) <= set(json.loads(row["final_ids"]))


def test_aggregate_conditions_second_hop_metric_on_available_rows():
    rows = [
        {
            "dataset": "hotpotqa",
            "second_oracle_recovered_given_first_locked": 1.0,
            "oracle_recall": 1.0,
        },
        {
            "dataset": "hotpotqa",
            "second_oracle_recovered_given_first_locked": None,
            "oracle_recall": 0.0,
        },
    ]
    result = _aggregate(rows, ("dataset",))[0]
    assert result["second_oracle_recovered_given_first_locked"] == 1.0
    assert result["second_oracle_recovered_given_first_locked_n"] == 1
