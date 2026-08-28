from __future__ import annotations

from dataclasses import dataclass

from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory, combine_native_memories


@dataclass(frozen=True)
class _Array:
    nbytes: int
    shape: tuple[int, ...]


def test_native_memory_reports_disjoint_kv_bytes() -> None:
    layer = MLXNativeLayerKV(_Array(32, (1, 2, 4, 8)), _Array(32, (1, 2, 4, 8)))
    memory = MLXNativeMemory((layer, layer), source_tokens=4)
    assert layer.nbytes == 64
    assert memory.nbytes == 128


def test_single_memory_combination_preserves_identity() -> None:
    layer = MLXNativeLayerKV(_Array(32, (1, 2, 4, 8)), _Array(32, (1, 2, 4, 8)))
    memory = MLXNativeMemory((layer,), source_tokens=4)
    assert combine_native_memories((memory,)) is memory
