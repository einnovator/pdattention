from __future__ import annotations

from dataclasses import dataclass

from pra_mlx.native import (
    MLXNativeFingerprint,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXPositionedKVCache,
    MLXQuantizedLayerKV,
    MLXQuantizedMemory,
    combine_native_memories,
)


@dataclass(frozen=True)
class _Array:
    nbytes: int
    shape: tuple[int, ...]


def test_native_memory_reports_disjoint_kv_bytes() -> None:
    layer = MLXNativeLayerKV(_Array(32, (1, 2, 4, 8)), _Array(32, (1, 2, 4, 8)))
    memory = MLXNativeMemory((layer, layer), source_tokens=4)
    assert layer.nbytes == 64
    assert memory.nbytes == 128


def test_quantized_memory_counts_payload_and_scale_bytes() -> None:
    layer = MLXQuantizedLayerKV(
        keys=_Array(32, (1, 2, 4, 8)),
        values=_Array(32, (1, 2, 4, 8)),
        key_scale=_Array(4, (1, 2, 1, 1)),
        value_scale=_Array(4, (1, 2, 1, 1)),
        original_dtype="float16",
    )
    memory = MLXQuantizedMemory((layer, layer), source_tokens=4)

    assert layer.nbytes == 72
    assert memory.nbytes == 144
    assert memory.scheme == "symmetric_int8_per_head"


def test_single_memory_combination_preserves_identity() -> None:
    layer = MLXNativeLayerKV(_Array(32, (1, 2, 4, 8)), _Array(32, (1, 2, 4, 8)))
    memory = MLXNativeMemory((layer,), source_tokens=4)
    assert combine_native_memories((memory,)) is memory


def test_selected_nbytes_obeys_consumer_layer_profile() -> None:
    layer = MLXNativeLayerKV(_Array(32, (1, 2, 4, 8)), _Array(32, (1, 2, 4, 8)))
    memory = MLXNativeMemory((layer, layer, layer), source_tokens=4)
    assert memory.selected_nbytes((1, 2)) == 128
    assert memory.selected_nbytes(()) == 0


def test_positioned_cache_separates_source_and_local_offsets() -> None:
    class Local:
        offset = 3
        state = ()

    cache = MLXPositionedKVCache(Local(), position_base=17)
    assert cache.offset == 20
    assert cache.local_offset == 3


def test_native_fingerprint_is_explicit_and_stable() -> None:
    fingerprint = MLXNativeFingerprint(
        model_id="model",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        dtype="float16",
        position_policy="source_local",
        consumer_profile="all",
        resource_version="resource-v1",
    )
    assert fingerprint.to_dict()["model_revision"] == "revision"
