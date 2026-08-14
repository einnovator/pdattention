import pytest
import torch

from pra_hf.query_facets import (
    build_multiscale_query_facets,
    build_span_query_facets,
    build_token_query_facets,
    build_contextual_query_facets,
    clip_query_support,
    contextual_window_spans,
    deterministic_phrase_spans,
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


def test_query_support_clips_active_context_without_changing_end_boundary():
    assert clip_query_support(20, support_span=(3, 18), max_support_tokens=8) == (10, 18)
    assert clip_query_support(20, max_support_tokens=16) == (4, 20)


def test_window_token_multiscale_and_local_only_facet_families():
    hidden = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    windowed = build_contextual_query_facets(
        hidden, (2, 10), window=4, stride=2, include_global=False
    )
    tokens = build_token_query_facets(hidden, (8, 10))
    multiscale = build_multiscale_query_facets(hidden, (2, 10), windows=(2, 4))
    assert all(row.kind == "local" for row in windowed.provenance)
    assert len(tokens.provenance) == 3
    assert {row.family for row in multiscale.provenance[1:]} == {
        "window_2",
        "window_4",
    }


def test_phrase_facets_use_punctuation_and_relation_neighborhoods():
    texts = ["Who", " won", "?", " Before", " that", ",", " where", "?"]
    spans = deterministic_phrase_spans(texts, (0, len(texts)), neighborhood=4)
    assert (0, 3, "clause") in spans
    assert any(label == "relation_neighborhood" for _, _, label in spans)
    facets = build_span_query_facets(
        torch.randn(len(texts), 3), spans, include_global=True, family="phrase"
    )
    assert facets.provenance[0].kind == "global"
    assert len(facets.provenance) > 2


def test_bounded_support_rejects_stale_prompt_facet_in_control():
    hidden = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    full = build_token_query_facets(hidden, (0, 4), include_global=False)
    latest = build_token_query_facets(hidden, (2, 4), include_global=False)
    full_pick = select_bounded_parents(
        score_semantic_query_facets(full.hidden, memory), 1
    ).parent_indices[0]
    latest_pick = select_bounded_parents(
        score_semantic_query_facets(latest.hidden, memory), 1
    ).parent_indices[0]
    assert full_pick == 0
    assert latest_pick == 1
