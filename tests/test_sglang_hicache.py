from __future__ import annotations

import numpy as np

from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory
from pra_sglang.hicache import PRAHiCacheTier, SGLangPRAHiCache


def _memory(value: float, *, tokens: int = 4) -> MLXNativeMemory:
    keys = np.full((1, 2, tokens, 4), value, dtype=np.float32)
    values = np.full((1, 2, tokens, 4), value + 1, dtype=np.float32)
    return MLXNativeMemory((MLXNativeLayerKV(keys, values),), tokens)


def _cache(tmp_path, *, objects_per_tier: int = 2) -> SGLangPRAHiCache:
    size = _memory(1).nbytes
    return SGLangPRAHiCache(
        tmp_path,
        max_l1_bytes=size * objects_per_tier,
        max_l2_bytes=size * objects_per_tier,
        to_host=lambda memory: memory,
        to_device=lambda memory: memory,
    )


def test_hicache_promotes_l3_through_l2_to_l1(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.put("resource-R", _memory(3), tier=PRAHiCacheTier.L3)

    restored = cache.get("resource-R")

    assert cache.placement("resource-R") is PRAHiCacheTier.L1
    assert np.array_equal(restored.layers[0].keys, _memory(3).layers[0].keys)
    metrics = cache.metrics()
    assert metrics.l3_hits == 1
    assert metrics.l3_to_l2_promotions == 1
    assert metrics.l2_to_l1_promotions == 1


def test_hicache_l1_pressure_demotes_to_separate_l2_namespace(tmp_path) -> None:
    cache = _cache(tmp_path, objects_per_tier=1)
    cache.put("resource-A", _memory(1))
    cache.put("resource-B", _memory(2))

    assert cache.placement("resource-A") is PRAHiCacheTier.L2
    assert cache.placement("resource-B") is PRAHiCacheTier.L1
    assert cache.metrics().l1_to_l2_demotions == 1


def test_hicache_l2_pressure_persists_oldest_memory_to_l3(tmp_path) -> None:
    cache = _cache(tmp_path, objects_per_tier=1)
    cache.put("resource-A", _memory(1), tier=PRAHiCacheTier.L2)
    cache.put("resource-B", _memory(2), tier=PRAHiCacheTier.L2)

    assert cache.placement("resource-A") is PRAHiCacheTier.L3
    assert cache.placement("resource-B") is PRAHiCacheTier.L2
    assert cache.metrics().l2_to_l3_demotions == 1


def test_default_host_codec_can_demote_an_already_host_owned_memory(tmp_path) -> None:
    size = _memory(1).nbytes
    cache = SGLangPRAHiCache(
        tmp_path,
        max_l1_bytes=size,
        max_l2_bytes=size,
    )

    cache.put("resource-A", _memory(1), tier=PRAHiCacheTier.L2)
    cache.put("resource-B", _memory(2), tier=PRAHiCacheTier.L2)

    assert cache.placement("resource-A") is PRAHiCacheTier.L3
    assert cache.metrics().l2_to_l3_demotions == 1


def test_hicache_reports_warm_l1_hit_without_another_promotion(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.put("resource-R", _memory(1), tier=PRAHiCacheTier.L2)
    cache.get("resource-R")
    cache.get("resource-R")

    metrics = cache.metrics()
    assert metrics.l2_hits == 1
    assert metrics.l1_hits == 1
    assert metrics.l2_to_l1_promotions == 1
