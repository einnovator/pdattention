from __future__ import annotations

from dataclasses import dataclass

from pra_mlx.native import (
    MLXNativeColdCodec,
    MLXNativeFingerprint,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXPositionedKVCache,
    MLXQuantizedLayerKV,
    MLXQuantizedMemory,
    combine_native_memories,
    deserialize_native_memory,
    serialize_native_memory,
)
from pra_mlx.native_storage import MLXNativeSegmentStore


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


def test_native_memory_wire_format_round_trips_and_quantizes() -> None:
    import numpy as np

    keys = np.linspace(-2, 2, 64, dtype=np.float32).reshape(1, 2, 4, 8)
    values = keys * 0.5
    memory = MLXNativeMemory((MLXNativeLayerKV(keys, values),), source_tokens=4)

    lossless = serialize_native_memory(memory)
    restored = deserialize_native_memory(lossless)
    assert np.array_equal(restored.layers[0].keys, keys)

    codec = MLXNativeColdCodec()
    quantized, metadata = codec.encode(lossless, "int8")
    decoded = deserialize_native_memory(codec.decode(quantized, metadata))
    assert len(quantized) < len(lossless)
    assert np.max(np.abs(decoded.layers[0].keys - keys)) < 0.02


def test_segment_store_round_trips_full_memory_and_reads_selected_layer(tmp_path) -> None:
    import numpy as np

    layers = tuple(
        MLXNativeLayerKV(
            np.full((1, 2, 4, 8), index, dtype=np.float32),
            np.full((1, 2, 4, 8), index + 10, dtype=np.float32),
        )
        for index in range(3)
    )
    memory = MLXNativeMemory(layers, source_tokens=4)
    payload = serialize_native_memory(memory)
    store = MLXNativeSegmentStore(tmp_path / "segmented")
    metadata = {"fingerprint": "model-v1"}

    store.put("memory", payload, metadata)
    restored = deserialize_native_memory(store.get("memory", metadata))
    selected = store.get_layer_arrays("memory", (1,), metadata)

    assert np.array_equal(restored.layers[2].values, layers[2].values)
    assert set(selected) == {"layer_0001_k", "layer_0001_v"}
    assert np.array_equal(selected["layer_0001_v"], layers[1].values)
