from __future__ import annotations

import pytest

from pra_hf.rag_materialization import score_prefix_plan


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


@pytest.mark.parametrize(("counts", "fraction"), (((), 0.5), ((1,), 0.0), ((0,), 1.0)))
def test_score_prefix_plan_rejects_invalid_requests(counts, fraction) -> None:
    with pytest.raises(ValueError):
        score_prefix_plan(counts, fraction)
