import math

import pytest
import torch

from pra_hf.qk_compression import (
    NativeLandmarkSelector,
    QueryConditionedLandmarkSelector,
    chunk_routing_loss,
    differentiable_landmark_scores,
    farthest_first_indices,
    gather_landmarks,
    gqa_head_map,
    greedy_qk_landmarks,
    kmeans_centroids,
    kmeans_medoid_indices,
    landmark_features,
    landmark_training_loss,
    last_token_indices,
    low_rank_response_scores,
    masked_mean_keys,
    qk_response_scores,
    random_token_indices,
    response_metrics,
    routing_metrics,
    stable_topk_indices,
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


def test_query_conditioned_selector_is_bounded_and_changes_with_query():
    features = torch.randn(2, 3, 4, 8)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, 0, -1] = False
    selector = QueryConditionedLandmarkSelector(6, hidden_width=4, rank=3)
    first = selector(features, torch.zeros(2, 6), mask)
    second = selector(features, torch.ones(2, 6), mask)
    assert first.shape == mask.shape
    assert torch.isneginf(first[:, 0, -1]).all()
    assert not torch.allclose(first[mask], second[mask])
    assert sum(parameter.numel() for parameter in selector.parameters()) < 100_000


@pytest.mark.parametrize(
    ("use_salience", "use_interaction"), [(True, True), (True, False), (False, True)]
)
def test_cached_query_conditioned_scoring_matches_uncached(
    use_salience, use_interaction
):
    features = torch.randn(2, 3, 4, 8)
    query = torch.randn(2, 6)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, 0, -1] = False
    selector = QueryConditionedLandmarkSelector(
        6,
        hidden_width=4,
        rank=3,
        use_salience=use_salience,
        use_interaction=use_interaction,
    )
    expected = selector(features, query, mask)
    cached = selector.cache_features(features)
    actual = selector.score_cached(*cached, query, mask)
    assert torch.equal(torch.isneginf(expected), torch.isneginf(actual))
    assert torch.allclose(expected[mask], actual[mask])


def test_selector_ablation_parameter_counts_include_only_active_terms():
    salience = QueryConditionedLandmarkSelector(
        6, feature_width=8, hidden_width=4, rank=3, use_interaction=False
    )
    bilinear = QueryConditionedLandmarkSelector(
        6, feature_width=8, hidden_width=4, rank=3, use_salience=False
    )
    assert sum(parameter.numel() for parameter in salience.parameters()) == 8 * 4 + 4 + 4 + 1
    assert sum(parameter.numel() for parameter in bilinear.parameters()) == 6 * 3 + 8 * 3


def test_stable_topk_and_centroids_are_deterministic_on_short_chunks():
    tied = torch.tensor([2.0, 2.0, 1.0, 2.0])
    assert stable_topk_indices(tied, 2).tolist() == [0, 1]
    values = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [10.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    centroids, centroid_mask = kmeans_centroids(values, mask, 4)
    repeated, repeated_mask = kmeans_centroids(values, mask, 4)
    assert torch.equal(centroid_mask, repeated_mask)
    assert torch.equal(centroids, repeated)
    assert centroid_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    one, one_mask = kmeans_centroids(values, mask, 1)
    assert torch.allclose(one[0, 0], torch.tensor([1.0, 0.0]))
    assert one_mask.all()
    medoids = kmeans_medoid_indices(values, mask, 2)
    assert medoids == kmeans_medoid_indices(values, mask, 2)
    assert all(mask[chunk, row].all() for chunk, row in enumerate(medoids))


def test_low_rank_scoring_and_direct_losses_are_finite_and_differentiable():
    queries = torch.randn(2, 3, requires_grad=True)
    tokens = torch.randn(2, 5, 3, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    scores = low_rank_response_scores(queries, tokens, mask, top_r=2)
    assert scores.shape == (2, 2)
    positives = torch.tensor([[True, False], [False, True]])
    candidates = torch.ones_like(positives)
    loss, components = chunk_routing_loss(
        "combined",
        scores,
        positives,
        candidates,
        teacher_scores=torch.randn_like(scores),
        budget=1,
    )
    assert torch.isfinite(loss)
    assert set(components) == {"listwise", "response", "boundary"}
    loss.backward()
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(tokens.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_low_rank_scoring_cpu_cuda_parity():
    query = torch.randn(3)
    tokens = torch.randn(2, 5, 3)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    cpu = low_rank_response_scores(query, tokens, mask, top_r=2)
    cuda = low_rank_response_scores(query.cuda(), tokens.cuda(), mask.cuda(), top_r=2)
    assert torch.allclose(cpu, cuda.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "objective", ["oracle_imitation", "listwise", "combined", "decision_aware"]
)
def test_retrieval_aware_losses_are_finite_and_differentiable(objective):
    logits = torch.randn(2, 4, 5, requires_grad=True)
    responses = torch.randn(2, 4, 5)
    mask = torch.ones(2, 4, 5, dtype=torch.bool)
    positives = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    oracle = torch.zeros_like(logits)
    oracle[:, :, :2] = 1
    teacher = torch.randn(2, 4)
    surrogate = differentiable_landmark_scores(logits, responses, mask, 2)
    assert surrogate.shape == positives.shape
    loss, components = landmark_training_loss(
        objective,
        logits,
        responses,
        mask,
        positives,
        m=2,
        teacher_scores=teacher,
        oracle_targets=oracle,
        budget=2,
    )
    assert torch.isfinite(loss)
    assert set(components) == {"oracle", "listwise", "response", "boundary"}
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


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
