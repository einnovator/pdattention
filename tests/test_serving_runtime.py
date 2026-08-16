from __future__ import annotations

import pytest
import torch

from pra_hf.serving_runtime import (
    NativeQKIndex,
    PagedKVCache,
    fused_gather_kv,
    merge_token_intervals,
    pack_ragged,
    unpack_ragged,
)


def test_indexed_gemm_search_matches_brute_force() -> None:
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(64, 16, generator=generator)
    queries = torch.randn(5, 16, generator=generator)
    index = NativeQKIndex(keys, coarse_clusters=8)
    brute = index.search(queries, 6, backend="brute_force")
    gemm = index.search(queries, 6, backend="gemm")
    assert torch.equal(brute.indices, gemm.indices)
    assert torch.allclose(brute.scores, gemm.scores, atol=1e-6)
    exhaustive_coarse = index.search(queries, 6, backend="coarse_to_fine", probes=8)
    assert torch.equal(gemm.indices, exhaustive_coarse.indices)


def test_fused_gather_merges_overlap_and_preserves_kv() -> None:
    key = torch.arange(1 * 2 * 10 * 3).reshape(1, 2, 10, 3).float()
    value = key + 1000
    result = fused_gather_kv(key, value, ((1, 4), (3, 7), (9, 10)))
    expected_indices = torch.tensor([1, 2, 3, 4, 5, 6, 9])
    assert merge_token_intervals(((3, 7), (1, 4), (9, 10)), 10) == ((1, 7), (9, 10))
    assert torch.equal(result.token_indices.cpu(), expected_indices)
    assert torch.equal(result.key, key.index_select(2, expected_indices))
    assert torch.equal(result.value, value.index_select(2, expected_indices))
    assert result.requested_tokens == 8
    assert result.materialized_tokens == 7


def test_paged_kv_cache_gather_and_fragmentation() -> None:
    key = torch.arange(2 * 10 * 3).reshape(2, 10, 3).float()
    value = key + 100
    cache = PagedKVCache(page_size=4, capacity_pages=3, policy="lru")
    cache.put("request", key, value)
    selected_key, selected_value = cache.gather("request", [0, 5, 9])
    assert torch.equal(selected_key, key[:, [0, 5, 9], :])
    assert torch.equal(selected_value, value[:, [0, 5, 9], :])
    assert cache.fragmentation_tokens == 2
    assert cache.hit_rate == 1.0


@pytest.mark.parametrize("policy", ["lru", "lfu", "hybrid"])
def test_paged_kv_cache_enforces_capacity_and_evicts(policy: str) -> None:
    cache = PagedKVCache(page_size=2, capacity_pages=2, policy=policy)
    values = torch.randn(1, 4, 2)
    cache.put("first", values, values)
    cache.put("second", values[:, :2], values[:, :2])
    assert cache.stats()["resident_pages"] == 2
    assert cache.evictions == 1
    with pytest.raises(KeyError):
        cache.gather("first", [0])


def test_ragged_pack_round_trip_has_no_padding() -> None:
    tensors = (torch.randn(2, 3), torch.randn(5, 3), torch.randn(1, 3))
    packed = pack_ragged(tensors)
    assert packed.values.shape == (8, 3)
    assert packed.offsets.tolist() == [0, 2, 7, 8]
    for actual, expected in zip(unpack_ragged(packed), tensors):
        assert torch.equal(actual, expected)
