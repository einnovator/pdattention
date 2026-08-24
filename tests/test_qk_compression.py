import math

import pytest
import torch

from pra_hf.qk_compression import (
    NativeLandmarkSelector,
    farthest_first_indices,
    gather_landmarks,
    gqa_head_map,
    greedy_qk_landmarks,
    landmark_features,
    last_token_indices,
    masked_mean_keys,
    qk_response_scores,
    random_token_indices,
    response_metrics,
    routing_metrics,
    token_query_key_dots,
)


def _fixture():
    queries = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]])
    keys = torch.zeros(2, 4, 2, 2)
    keys[0, 0, 0] = torch.tensor([3.0, 0.0])
    keys[0, 1, 1] = torch.tensor([0.0, 3.0])
    keys[1, :, :, :] = 0.5
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    return queries, keys, mask


def test_gqa_dots_map_query_heads_to_native_key_heads():
    queries, keys, mask = _fixture()
    assert gqa_head_map(4, 2).tolist() == [0, 0, 1, 1]
    dots, returned_mask = token_query_key_dots(queries, keys, mask)
    assert dots.shape == (1, 2, 4, 4)
    assert torch.equal(mask, returned_mask)
    assert dots[0, 0, 0, 0].item() == pytest.approx(6 / math.sqrt(2))
    assert dots[0, 0, 1, 1].item() == 0
    assert dots[0, 0, 1, 3].item() == pytest.approx(6 / math.sqrt(2))


@pytest.mark.parametrize("function", ["max", "top_r_mean", "logsumexp", "attention_mass"])
def test_teacher_functions_return_finite_chunk_scores(function):
    queries, keys, mask = _fixture()
    scores = qk_response_scores(queries, keys, mask, function=function, top_r=2)
    assert scores.shape == (1, 2)
    assert torch.isfinite(scores).all()


def test_mean_and_gathered_landmarks_preserve_expected_shapes():
    queries, keys, mask = _fixture()
    mean = masked_mean_keys(keys, mask)
    assert mean.shape == (2, 1, 2, 2)
    compact, compact_mask = gather_landmarks(keys, [[0, 1], [1]])
    assert compact.shape == (2, 2, 2, 2)
    assert compact_mask.tolist() == [[True, True], [True, False]]
    assert qk_response_scores(queries, compact, compact_mask).shape == (1, 2)


def test_native_subset_controllers_are_valid_and_reproducible():
    _, keys, mask = _fixture()
    assert last_token_indices(mask, 2) == [[0, 1], [2, 3]]
    first = random_token_indices(mask, 2, generator=torch.Generator().manual_seed(7))
    second = random_token_indices(mask, 2, generator=torch.Generator().manual_seed(7))
    assert first == second
    farthest = farthest_first_indices(keys, mask, 2)
    assert all(len(row) == 2 for row in farthest)
    assert all(mask[chunk, row].all() for chunk, row in enumerate(farthest))


def test_greedy_oracle_reduces_teacher_error_over_mean():
    queries, keys, mask = _fixture()
    teacher = qk_response_scores(queries, keys, mask, function="max")[0]
    mean = qk_response_scores(
        queries,
        masked_mean_keys(keys, mask),
        torch.ones(2, 1, dtype=torch.bool),
        function="max",
    )[0]
    indices = greedy_qk_landmarks(queries, keys, mask, 2, function="max")
    compact, compact_mask = gather_landmarks(keys, indices)
    oracle = qk_response_scores(queries, compact, compact_mask, function="max")[0]
    assert torch.mean((teacher - oracle) ** 2) <= torch.mean((teacher - mean) ** 2)
    assert indices[0] == [0, 1]


def test_selector_features_are_query_independent_and_masked():
    _, keys, mask = _fixture()
    features = landmark_features(keys, mask)
    assert features.shape == (2, 4, 8)
    assert torch.equal(features[0, 2:], torch.zeros_like(features[0, 2:]))
    selector = NativeLandmarkSelector(hidden_width=4)
    selected = selector.select(keys, mask, 2)
    assert all(len(row) == 2 for row in selected)
    assert sum(parameter.numel() for parameter in selector.parameters()) < 100_000


def test_response_and_routing_metrics_cover_requested_contract():
    preservation = response_metrics(
        torch.tensor([3.0, 2.0, 1.0]), torch.tensor([2.9, 2.1, 1.0])
    )
    assert preservation.topk_overlap[1] == 1.0
    assert preservation.spearman == pytest.approx(1.0)
    routing = routing_metrics(
        [2, 0, 1, 3], torch.tensor([True, False, True, False]), budget=2
    )
    assert routing["evidence_recall"] == 1.0
    assert routing["evidence_precision"] == 1.0
    assert routing["chain_completion"] == 1.0
    assert routing["mrr"] == 1.0


def test_invalid_shapes_and_empty_chunks_fail_early():
    queries, keys, mask = _fixture()
    with pytest.raises(ValueError):
        qk_response_scores(queries, keys, torch.zeros_like(mask))
    with pytest.raises(ValueError):
        gqa_head_map(3, 2)
