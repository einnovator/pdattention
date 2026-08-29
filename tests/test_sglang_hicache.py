from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory
from pra_sglang.hicache import PRAHiCacheTier, SGLangPRAHiCache
from pra_sglang.hicache_backend import SGLangHiCacheStorageBackend


class _FakeSGLangStorage:
    """Small ``HiCacheStorage`` stand-in preserving its preallocated-get API."""

    def __init__(self) -> None:
        self.values = {}

    def set(self, key, value) -> bool:
        self.values[key] = value.clone()
        return True

    def get(self, key, target):
        value = self.values.get(key)
        if value is None or value.numel() != target.numel():
            return None
        target.copy_(value)
        return target

    def exists(self, key) -> bool:
        return key in self.values


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


def test_sglang_storage_backend_round_trips_immutable_memory() -> None:
    storage = _FakeSGLangStorage()
    writer = SGLangHiCacheStorageBackend(storage, namespace="test")

    stored_bytes = writer.put("tenant/private-resource", _memory(7))
    reader = SGLangHiCacheStorageBackend(storage, namespace="test")
    restored = reader.get("tenant/private-resource")

    assert stored_bytes == reader.size("tenant/private-resource")
    assert reader.exists("tenant/private-resource")
    assert np.array_equal(restored.layers[0].keys, _memory(7).layers[0].keys)
    assert all("private-resource" not in key for key in storage.values)


def test_hicache_promotes_from_sglang_storage_backend(tmp_path) -> None:
    storage = _FakeSGLangStorage()
    backend = SGLangHiCacheStorageBackend(storage)
    size = _memory(1).nbytes
    cache = SGLangPRAHiCache(
        tmp_path,
        max_l1_bytes=size * 2,
        max_l2_bytes=size * 2,
        to_host=lambda memory: memory,
        to_device=lambda memory: memory,
        l3_backend=backend,
    )
    cache.put("resource-R", _memory(9), tier=PRAHiCacheTier.L3)

    restored = cache.get("resource-R")

    assert cache.placement("resource-R") is PRAHiCacheTier.L1
    assert np.array_equal(restored.layers[0].values, _memory(9).layers[0].values)
    metrics = cache.metrics()
    assert metrics.l3_hits == 1
    assert metrics.l3_to_l2_promotions == 1
    assert metrics.l2_to_l1_promotions == 1


def test_backend_remove_revokes_logical_access_without_global_clear() -> None:
    storage = _FakeSGLangStorage()
    backend = SGLangHiCacheStorageBackend(storage)
    backend.put("resource-R", _memory(1))

    backend.remove("resource-R")

    assert not backend.exists("resource-R")
    assert storage.values  # Physical reclamation belongs to backend eviction.


def test_hicache_prefetch_uses_caller_executor_and_reports_completion(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.put("resource-R", _memory(4), tier=PRAHiCacheTier.L2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        restored = cache.prefetch("resource-R", executor).result()

    assert np.array_equal(restored.layers[0].keys, _memory(4).layers[0].keys)
    metrics = cache.metrics()
    assert metrics.prefetch_requests == 1
    assert metrics.prefetch_completed == 1
    assert metrics.prefetch_failures == 0
    assert metrics.prefetched_bytes == restored.nbytes
