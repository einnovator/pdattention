import pytest
import torch

from pra_hf.query_facets import (
    build_contextual_query_facets,
    contextual_window_spans,
    global_query_facet,
    pool_parent_native_keys,
    score_native_query_facets,
    score_semantic_query_facets,
    select_bounded_parents,
    target_rank_metrics,
)


def test_contextual_windows_cover_question_with_overlap_and_final_boundary():
    assert contextual_window_spans((3, 14), 4, 3) == (
        (3, 7),
        (6, 10),
        (9, 13),
        (10, 14),
    )


def test_facets_derive_from_one_contextual_state_tensor_and_keep_global():
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    native = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    facets = build_contextual_query_facets(
        hidden, (1, 5), window=2, stride=2, native_query=native
    )
    assert [item.kind for item in facets.provenance] == ["global", "local", "local"]
    assert torch.equal(facets.hidden[0], hidden[-1])
    assert torch.equal(facets.hidden[1], hidden[1:3].mean(0))
    assert torch.equal(facets.native_query[2], native[3:5].mean(0))


def test_global_facet_reproduces_existing_semantic_root_scores():
    hidden = torch.tensor([[9.0, 9.0], [1.0, 0.0]])
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    facets = global_query_facet(hidden)
    result = score_semantic_query_facets(facets.hidden, memory)
    assert torch.allclose(result.scores, torch.tensor([1.0, 0.0]))
    assert select_bounded_parents(result, 1).parent_indices == (0,)


def test_semantic_facet_scoring_is_deterministic_and_tracks_winner():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    memory = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    result = score_semantic_query_facets(query, memory)
    assert result.winning_facet.tolist() == [1, 0, 0]
    assert select_bounded_parents(result, 2).parent_indices == (0, 1)


def test_parent_native_key_pooling_is_vectorized_and_mask_aware():
    keys = torch.tensor(
        [
            [[[[1.0]]], [[[3.0]]]],
            [[[[5.0]]], [[[99.0]]]],
        ]
    ).reshape(2, 2, 1, 1)
    mask = torch.tensor([[True, True], [True, False]])
    pooled = pool_parent_native_keys(keys, mask, torch.tensor([0, 1]), 2)
    assert pooled[:, 0, 0].tolist() == [2.0, 5.0]


def test_native_scoring_respects_gqa_and_exposes_head_specialization():
    query = torch.zeros(1, 4, 2)
    query[0, 2] = torch.tensor([2.0, 0.0])
    keys = torch.zeros(2, 2, 2)
    keys[0, 1] = torch.tensor([2.0, 0.0])
    keys[1, 0] = torch.tensor([3.0, 0.0])
    result = score_native_query_facets(query, keys, head_reduction="max")
    assert result.scores[0] > result.scores[1]
    assert result.winning_head[0].item() == 2
    with pytest.raises(ValueError, match="divisible"):
        score_native_query_facets(torch.zeros(1, 3, 2), keys)


def test_span_head_reduction_records_joint_winner():
    query = torch.zeros(2, 2, 2)
    query[1, 1] = torch.tensor([3.0, 0.0])
    keys = torch.tensor([[[3.0, 0.0]], [[0.0, 3.0]]])
    result = score_native_query_facets(query, keys, head_reduction="max")
    assert result.scores.argmax().item() == 0
    assert result.winning_facet[0].item() == 1
    assert result.winning_head[0].item() == 1


def test_per_head_nominations_deduplicate_before_one_final_budget():
    query = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    keys = torch.tensor(
        [
            [[2.0, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    result = score_native_query_facets(query, keys, head_reduction="max")
    selection = select_bounded_parents(result, 2, per_head_nomination_k=2)
    assert selection.nominated_parent_indices == (0, 1)
    assert selection.deduplicated_candidates == 2
    assert selection.parent_indices == (0, 1)
    assert len(selection.parent_indices) == selection.final_budget == 2


def test_target_metrics_compute_rank_recall_precision_and_jaccard():
    metrics = target_rank_metrics(
        torch.tensor([0.9, 0.8, 0.7, 0.6]), {1, 3}, {0, 1}
    )
    assert metrics["target_rank"] == 2
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["oracle_recall"] == pytest.approx(0.5)
    assert metrics["oracle_precision"] == pytest.approx(0.5)
    assert metrics["oracle_jaccard"] == pytest.approx(1 / 3)
    assert metrics["false_positive_parent_count"] == 1
