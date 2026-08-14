from __future__ import annotations

import copy

from experiments.paper2_5_iterative_pra.run_displacement_calibration import (
    calibrated_selection,
    classify_displacement,
    fit_family_calibration,
    native_saturation_summary,
    protected_root_selection,
    score_family,
    score_family_summary,
    validate_prior_convergence,
)


def _node(parent, hop, direct, edge=None, path=None, representation="semantic_gist"):
    return {
        "node_id": f"x#parent={parent}",
        "parent_chunk_id": f"x#parent={parent}",
        "hop": hop,
        "final_selected": True,
        "direct_query_score": direct,
        "edge_score": direct if edge is None else edge,
        "path_score": direct if path is None else path,
        "projection_type": "root_query" if hop == 1 else "query",
        "representation_type": representation,
    }


def test_displacement_is_root_slot_partition_and_preservation_is_explicit():
    one = {"nodes": [_node(0, 1, 0.9), _node(1, 1, 0.8), _node(2, 1, 0.7)]}
    iterative = {
        "nodes": [_node(0, 1, 0.9, path=0.95), _node(3, 2, 0.2, 0.8, 0.7), _node(4, 2, 0.1, 0.7, 0.6)]
    }
    rows = classify_displacement({0, 1}, one, iterative, method="parent_closure", budget=3)
    assert [row["status"] for row in rows] == ["preserved", "displaced"]
    assert rows[1]["displacement_reason"] == "root_slot_partition"
    assert rows[1]["replacing_hops"] == "[2, 2]"


def test_protected_root_counterfactual_respects_budget_and_fills_frozen_order():
    selected = protected_root_selection([0, 1, 2], [0, 3, 4], {1}, 3)
    assert selected == [1, 0, 3]
    assert len(selected) == 3


def test_score_family_distinguishes_native_from_semantic_and_root():
    assert score_family(_node(0, 1, 0.1)) == "root_semantic"
    assert score_family(_node(1, 2, 0.1, 0.2)) == "propagated_semantic"
    native = _node(2, 2, 0.1, 12.0, representation="pre_rope_native_qk")
    native["projection_type"] = "native_query_to_key"
    assert score_family(native) == "native_qk"


def _calibration_rows():
    return [
        {"method": "m", "partition": "validation", "score_family": "root_semantic", "raw_score": 0.0},
        {"method": "m", "partition": "validation", "score_family": "root_semantic", "raw_score": 1.0},
        {"method": "m", "partition": "validation", "score_family": "native_qk", "raw_score": 10.0},
        {"method": "m", "partition": "validation", "score_family": "native_qk", "raw_score": 20.0},
    ]


def test_family_calibration_uses_validation_only_and_is_test_label_blind():
    rows = _calibration_rows()
    fit = fit_family_calibration(rows, "m")
    changed = rows + [
        {"method": "m", "partition": "test", "score_family": "native_qk", "raw_score": 999.0, "is_oracle": 1.0}
    ]
    assert fit == fit_family_calibration(changed, "m")


def test_calibrated_selection_is_deterministic_and_budget_bounded():
    fit = fit_family_calibration(_calibration_rows(), "m")
    candidates = [
        {"candidate_parent": 2, "score_family": "native_qk", "raw_score": 15.0},
        {"candidate_parent": 0, "score_family": "root_semantic", "raw_score": 0.5},
        {"candidate_parent": 1, "score_family": "root_semantic", "raw_score": 0.5},
    ]
    assert calibrated_selection(candidates, fit, "family_zscore", 2) == [0, 1]
    assert len(calibrated_selection(candidates, fit, "family_quantile", 2)) == 2


def test_score_summary_keeps_oracle_identity_out_of_distribution_moments():
    base = {
        "dataset": "hotpotqa",
        "method": "m",
        "score_family": "root_semantic",
        "example_id": "x",
        "seed": 11,
        "source_parent": None,
        "top1_top2_margin": 0.1,
        "top4_spread": 0.4,
        "score_entropy": 1.0,
        "candidate_count": 2,
    }
    rows = [
        {**base, "raw_score": 1.0, "score_quantile": 1.0, "is_oracle": 1.0},
        {**base, "raw_score": 0.0, "score_quantile": 0.0, "is_oracle": 0.0},
    ]
    first = score_family_summary(rows)[0]
    swapped = copy.deepcopy(rows)
    swapped[0]["is_oracle"], swapped[1]["is_oracle"] = 0.0, 1.0
    second = score_family_summary(swapped)[0]
    assert first["mean"] == second["mean"] == 0.5
    assert first["mean_oracle_score_quantile"] != second["mean_oracle_score_quantile"]


def test_prior_convergence_validation_detects_identity_drift(tmp_path):
    path = tmp_path / "prior.csv"
    path.write_text(
        "dataset,example_id,seed,fraction,method,selected_parent_ids,oracle_recall,oracle_precision,oracle_jaccard,complete_oracle\n"
        'hotpotqa,x,11,0.2,m,"[0]",1.0,1.0,1.0,1.0\n',
        encoding="utf-8",
    )
    row = {
        "dataset": "hotpotqa",
        "example_id": "x",
        "seed": 11,
        "fraction": 0.2,
        "method": "m",
        "selected_parent_ids": "[0]",
        "oracle_recall": 1.0,
        "oracle_precision": 1.0,
        "oracle_jaccard": 1.0,
        "complete_oracle": 1.0,
    }
    validate_prior_convergence([row], path)
    row["selected_parent_ids"] = "[1]"
    try:
        validate_prior_convergence([row], path)
    except ValueError as error:
        assert "identity drift" in str(error)
    else:
        raise AssertionError("Expected parity validation to reject identity drift")


def test_native_saturation_summary_exposes_sigmoid_compression():
    rows = [
        {"dataset": "hotpotqa", "method": "native", "score_family": "native_qk", "raw_score": 12.0},
        {"dataset": "hotpotqa", "method": "native", "score_family": "native_qk", "raw_score": 14.0},
    ]
    summary = native_saturation_summary(rows)[0]
    assert summary["raw_std"] > 0.0
    assert summary["fraction_sigmoid_above_0_999"] == 1.0
    assert summary["sigmoid_std"] < 1e-4
