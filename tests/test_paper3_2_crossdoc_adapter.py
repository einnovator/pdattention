from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.paper3_2_rag.run_crossdoc_adapter import (
    _exact_sign_flip_p,
    _select_disjoint_cohort,
    summarize_rows,
)


def test_adapter_cohorts_are_deterministic_and_disjoint() -> None:
    questions = tuple(SimpleNamespace(example_id=f"q{index}") for index in range(20))
    used: set[str] = set()
    first = _select_disjoint_cohort(questions, count=5, seed=11, excluded_ids=used)
    second = _select_disjoint_cohort(questions, count=5, seed=23, excluded_ids=used)
    assert len(first) == len(second) == 5
    assert {row.example_id for row in first}.isdisjoint(row.example_id for row in second)
    replay_used: set[str] = set()
    replay = _select_disjoint_cohort(
        questions, count=5, seed=11, excluded_ids=replay_used
    )
    assert [row.example_id for row in replay] == [row.example_id for row in first]


def test_adapter_summary_uses_seed_level_variation() -> None:
    rows = []
    for seed, scores in ((11, (0.0, 0.5)), (23, (0.5, 1.0))):
        for index, score in enumerate(scores):
            rows.append(
                {
                    "condition": "R_TRAINED_RESIDUAL",
                    "seed": seed,
                    "example_id": f"q{seed}-{index}",
                    "exact_match": float(score == 1.0),
                    "token_f1": score,
                    "official_multihop_rag_score": float(score >= 0.5),
                    "gold_answer_mean_nll": 2.0 - score,
                    "first_step_js_divergence": 0.1,
                    "output_matches_packed": index == 0,
                    "kv_rmse": 0.2,
                    "value_rmse": 0.3,
                    "reencoded_tokens": 0,
                    "physical_native_tokens": 256,
                    "request_transform_ms": 4.0,
                    "adapter_parameters": 128,
                }
            )
    result = summarize_rows(rows)["conditions"][0]
    assert result["token_f1"] == pytest.approx(0.5)
    assert result["seed_token_f1_std"] == pytest.approx(0.25)
    assert result["output_match_rate"] == pytest.approx(0.5)
    assert result["official_multihop_rag_score"] == pytest.approx(0.75)
    assert result["physical_native_tokens"] == pytest.approx(256)
    assert result["adapter_parameters"] == 128


def test_five_same_direction_seed_effects_have_minimum_two_sided_p() -> None:
    assert _exact_sign_flip_p((1, 1, 1, 1, 1)) == pytest.approx(0.0625)
