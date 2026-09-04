from __future__ import annotations

import pytest

from pra_hf.rag_materialization import (
    evidence_oracle_plan,
    exact_token_plan,
    score_prefix_plan,
    wrong_memory_plan,
)


def test_score_prefix_plan_keeps_whole_ranked_resources() -> None:
    plan = score_prefix_plan((20, 30, 50), 0.5)
    assert plan.selected_indices == (0, 1)
    assert plan.requested_token_budget == 50
    assert plan.materialized_tokens == 50
    assert plan.materialized_fraction == pytest.approx(0.5)


def test_score_prefix_plan_always_keeps_one_resource() -> None:
    plan = score_prefix_plan((40, 40), 0.1)
    assert plan.selected_indices == (0,)
    assert plan.materialized_tokens == 40
    assert plan.materialized_fraction == pytest.approx(0.5)


def test_exact_token_plan_fills_budget_and_preserves_selection_alignment() -> None:
    plan = exact_token_plan((20, 30, 50), 0.25, priority=(2, 0, 1))
    assert plan.token_counts == (0, 0, 25)
    assert plan.selected_indices == (2,)
    assert plan.requested_token_budget == 25
    assert plan.materialized_fraction == pytest.approx(0.25)


def test_evidence_oracle_prioritizes_gold_without_reordering_output() -> None:
    plan = evidence_oracle_plan((20, 30, 50), (1,), 0.5)
    assert plan.priority == (1, 0, 2)
    assert plan.token_counts == (20, 30, 0)
    assert plan.materialized_tokens == 50


def test_wrong_memory_prioritizes_non_gold_at_the_same_budget() -> None:
    plan = wrong_memory_plan((20, 30, 50), (1,), 0.5)
    assert plan.priority == (0, 2, 1)
    assert plan.token_counts == (20, 0, 30)
    assert plan.materialized_tokens == 50


def test_exact_token_plan_rejects_invalid_priority() -> None:
    with pytest.raises(ValueError):
        exact_token_plan((20, 30), 0.5, priority=(0, 0))


@pytest.mark.parametrize(("counts", "fraction"), (((), 0.5), ((1,), 0.0), ((0,), 1.0)))
def test_score_prefix_plan_rejects_invalid_requests(counts, fraction) -> None:
    with pytest.raises(ValueError):
        score_prefix_plan(counts, fraction)
