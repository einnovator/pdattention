from __future__ import annotations

import torch

from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    canonical_oracle_parent_indices,
    competition_rank,
    _convergence_row,
    evaluate_adaptive_policy,
    evidence_parent_groups,
    fit_adaptive_threshold,
    _geometry_comparison,
    local_semantic_scores,
    oracle_set_metrics,
    parent_semantic_scores,
    validation_partition,
)


def _feature():
    return {
        "example_id": "example",
        "parent_spans": [(0, 10), (10, 20), (20, 30), (30, 40)],
        "evidence_spans": [(8, 12), (31, 33)],
        "parent_positive_mask": torch.tensor([True, True, False, True]),
    }


def test_canonical_oracle_reuses_all_intersecting_paper2_parents():
    assert canonical_oracle_parent_indices(_feature()) == {0, 1, 3}
    assert evidence_parent_groups(_feature()) == [{0, 1}, {3}]


def test_oracle_metrics_and_feasible_complete_recovery():
    assert oracle_set_metrics({0, 2}, {0, 1}) == {
        "oracle_recall": 0.5,
        "oracle_precision": 0.5,
        "oracle_jaccard": 1 / 3,
        "complete_oracle": 0.0,
    }
    assert oracle_set_metrics({0, 1, 2}, {0, 1})["complete_oracle"] == 1.0


def test_competition_rank_shares_ties_and_excludes_source():
    result = competition_rank(torch.tensor([9.0, 3.0, 3.0, 2.0]), {2}, {0})
    assert result["target_rank"] == 1
    assert result["top_distractor_parent"] == 1
    assert result["oracle_margin"] == 0.0


def test_parent_and_local_scores_use_query_to_memory_direction():
    pm = torch.eye(3)
    pq = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert parent_semantic_scores(pm, pq, {0}).argmax().item() == 1
    local_parent = torch.tensor([0, 0, 1, 2])
    lm = torch.eye(4)
    lq = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], *torch.eye(4)[2:].tolist()])
    scores = local_semantic_scores(lm, lq, local_parent, {0}, 3)
    assert scores.argmax().item() == 2


def test_adaptive_threshold_is_validation_only_and_deterministic():
    rows = [
        {"partition": "validation", "top1_top2_gap": 0.1, "target_rank": 3},
        {"partition": "validation", "top1_top2_gap": 0.8, "target_rank": 1},
        {"partition": "test", "top1_top2_gap": 0.2, "target_rank": 4},
        {"partition": "test", "top1_top2_gap": 0.9, "target_rank": 2},
    ]
    fit = fit_adaptive_threshold(rows)
    assert fit == fit_adaptive_threshold(rows + [{"partition": "test", "top1_top2_gap": 0.0, "target_rank": 99}])
    result = evaluate_adaptive_policy(rows, fit["threshold"])
    assert result["test_edges"] == 2
    assert validation_partition("stable") == validation_partition("stable")


def test_oracle_identity_never_changes_scores_or_validation_partition():
    scores = torch.tensor([0.9, 0.8, 0.1])
    first = competition_rank(scores, {1}, {0})
    second = competition_rank(scores, {2}, {0})
    assert first["top1_top2_gap"] == second["top1_top2_gap"]
    assert validation_partition("x") in {"validation", "test"}


def test_geometry_classification_compares_frozen_ranks_per_seed():
    rows = []
    for geometry, rank in (("parent_semantic", 6), ("local_semantic", 3), ("native_qk", 5)):
        rows.append(
            {
                "example_id": "x",
                "seed": 11,
                "transition": 0,
                "geometry": geometry,
                "target_rank": rank,
            }
        )
    compared, counts = _geometry_comparison(rows)
    assert compared[0]["classification"] == "semantic_wins"
    assert counts == {"semantic_wins": 1}
    assert set(compared[0]) >= {
        "parent_semantic_rank",
        "local_semantic_rank",
        "native_qk_rank",
        "classification",
    }


def test_convergence_row_preserves_budget_payload_and_separate_costs():
    feature = {
        **_feature(),
        "dataset": "hotpotqa",
    }
    row = {
        "seed": 11,
        "condition": "native_qk_max_topk_p20",
        "budget_parents": 3,
        "semantic_gist_comparisons": 7,
        "native_qk_dot_products": 100,
        "materialized_kv_tokens": 20,
        "materialized_kv_fraction": 0.5,
    }
    graph = {
        "nodes": [
            {
                "node_id": "example#parent=0",
                "parent_chunk_id": "example#parent=0",
                "final_selected": True,
            },
            {
                "node_id": "example#parent=2",
                "parent_chunk_id": "example#parent=2",
                "final_selected": True,
            },
        ]
    }
    result = _convergence_row(feature, row, graph, {0, 1, 3}, 0.2)
    assert result["selected_parent_ids"] == "[0, 2]"
    assert result["oracle_feasible"] == 1.0
    assert result["semantic_gist_comparisons"] == 7
    assert result["native_qk_dot_products"] == 100
    assert result["routing_comparisons"] == 107
    assert result["materialized_kv_tokens"] == 20
