"""Tests for dataset-independent recall-versus-memory metrics."""

import pytest

from common.recall_sparsity import recall_sparsity_curve


def test_variable_candidate_counts_use_per_example_ceiling():
    result = recall_sparsity_curve(
        [list(range(10)), list(range(25))],
        [{0}, {2}],
        fractions=(0.05, 0.10, 0.30, 1.0),
    )
    rows = {row["fraction"]: row for row in result["curve"]}
    assert rows[0.05]["any_evidence_recall"] == 0.5
    assert rows[0.10]["any_evidence_recall"] == 1.0
    assert rows[0.10]["selected_chunk_fraction"] == pytest.approx((0.1 + 0.12) / 2)


def test_multiple_evidence_reports_coverage_any_all_and_inverse_metrics():
    result = recall_sparsity_curve(
        [["a", "x", "b", "c"]],
        [{"a", "b"}],
        fractions=(0.25, 0.30, 0.50, 0.75, 1.0),
    )
    rows = result["curve"]
    assert rows[0]["recall"] == 0.5
    assert rows[0]["any_evidence_recall"] == 1.0
    assert rows[0]["all_evidence_recall"] == 0.0
    assert rows[3]["recall"] == rows[3]["all_evidence_recall"] == 1.0
    assert result["inverse"] == {"f70": 0.75, "f80": 0.75, "f90": 0.75, "f95": 0.75}
    assert result["endpoint_complete"]


def test_exact_kv_fraction_follows_ranked_token_lengths():
    result = recall_sparsity_curve(
        [["short", "long", "tail"]],
        [{"tail"}],
        fractions=(0.30, 1.0),
        candidate_token_lengths=[[1, 8, 1]],
    )
    assert result["kv_fraction_exact"]
    assert result["curve"][0]["selected_chunk_fraction"] == pytest.approx(1 / 3)
    assert result["curve"][0]["selected_kv_token_fraction"] == pytest.approx(0.1)


def test_endpoint_mismatch_is_visible_and_optionally_fatal():
    kwargs = {
        "rankings": [["a", "b"]],
        "evidence_ids": [{"missing"}],
        "fractions": (0.30, 1.0),
    }
    assert not recall_sparsity_curve(**kwargs)["endpoint_complete"]
    with pytest.raises(AssertionError, match="100%"):
        recall_sparsity_curve(**kwargs, require_complete_endpoint=True)


def test_rejects_duplicate_candidates_and_missing_auc_endpoint():
    with pytest.raises(ValueError, match="duplicate"):
        recall_sparsity_curve([["a", "a"]], [{"a"}], fractions=(0.30, 1.0))
    with pytest.raises(ValueError, match="0.30"):
        recall_sparsity_curve([["a"]], [{"a"}], fractions=(0.20, 1.0))
