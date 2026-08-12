import pytest
import torch

from experiments.paper1_5_rope.learned_routing import materialize_native_payload
from experiments.paper1_5_rope.pooling_geometry import (
    centered_rope_subgists,
    contiguous_subspans,
    native_token_chunk_score,
    pearson_correlation,
    post_rope_mean,
    pre_rope_mean,
    spearman_correlation,
    subspan_centers,
    topk_overlap,
)
from pra_torch.positions import RotaryPositionEncoding


def test_contiguous_subspans_are_balanced_complete_and_non_overlapping():
    spans = contiguous_subspans(11, 4)

    assert spans == ((0, 3), (3, 6), (6, 9), (9, 11))
    assert [index for start, end in spans for index in range(start, end)] == list(range(11))


def test_subspan_centers_preserve_fractional_positions():
    positions = torch.arange(10, 15)

    assert torch.equal(subspan_centers(positions, 2), torch.tensor([11.0, 13.5]))


def test_centered_g1_is_mean_pre_key_rotated_once_at_exact_center():
    torch.manual_seed(3)
    rope = RotaryPositionEncoding(4)
    raw = torch.randn(1, 2, 4, 4)
    positions = torch.arange(20, 24)

    actual, centers = centered_rope_subgists(raw, positions, 1, rope)
    expected = rope.apply_rotary(raw.mean(dim=2, keepdim=True), torch.tensor([21.5]))

    assert torch.equal(centers, torch.tensor([21.5]))
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_requested_resolution_caps_at_one_nonempty_gist_per_token():
    rope = RotaryPositionEncoding(4)
    raw = torch.randn(1, 2, 3, 4)

    gists, centers = centered_rope_subgists(raw, torch.arange(7, 10), 8, rope)

    assert gists.shape == (1, 2, 3, 4)
    assert torch.equal(centers, torch.tensor([7.0, 8.0, 9.0]))


def test_post_pre_and_centered_gists_do_not_alias_native_payload():
    torch.manual_seed(5)
    rope = RotaryPositionEncoding(4)
    raw = torch.randn(1, 2, 8, 4)
    post = rope.apply_rotary(raw, torch.arange(8))
    value = torch.randn_like(post)
    payload = {"chunk": (post.clone(), value.clone())}
    expected_k, expected_v = materialize_native_payload(payload, ["chunk"])

    pre_gist = pre_rope_mean(raw)
    post_gist = post_rope_mean(post)
    centered, _ = centered_rope_subgists(raw, torch.arange(8), 4, rope)
    pre_gist.add_(10)
    post_gist.sub_(10)
    centered.zero_()
    actual_k, actual_v = materialize_native_payload(payload, ["chunk"])

    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)
    assert actual_k.data_ptr() != post_gist.data_ptr()


def test_native_qk_reduction_and_metrics_are_deterministic():
    query = torch.tensor([[[[1.0, 0.0]]]])
    keys = torch.tensor([[[[0.0, 1.0], [2.0, 0.0], [1.0, 0.0]]]])

    score = native_token_chunk_score(query, keys)

    assert score.item() == torch.tensor(2.0 / (2.0**0.5)).item()
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert topk_overlap([3, 2, 1], [3, 1, 2], 2) == 0.5
