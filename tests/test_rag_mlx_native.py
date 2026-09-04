from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_mlx_native import MLXNativeLayerKV, MLXNativeMemory


class _Array:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


def test_native_memory_accounting_is_dependency_free() -> None:
    layer = MLXNativeLayerKV(_Array(12), _Array(20))
    memory = MLXNativeMemory((layer, layer), source_tokens=7)
    assert layer.nbytes == 32
    assert memory.nbytes == 64
    assert memory.position_base == 7


def test_native_memory_separates_physical_tokens_from_query_position() -> None:
    layer = MLXNativeLayerKV(_Array(12), _Array(20))
    memory = MLXNativeMemory((layer,), source_tokens=9, query_position_base=5)
    assert memory.source_tokens == 9
    assert memory.position_base == 5


@pytest.mark.parametrize(
    ("source_tokens", "query_position_base"), ((0, None), (1, -1))
)
def test_native_memory_rejects_invalid_geometry(
    source_tokens: int, query_position_base: int | None
) -> None:
    with pytest.raises(ValueError):
        MLXNativeMemory(
            (MLXNativeLayerKV(_Array(1), _Array(1)),),
            source_tokens,
            query_position_base,
        )


def test_native_memory_is_immutable() -> None:
    memory = MLXNativeMemory((MLXNativeLayerKV(_Array(1), _Array(1)),), 1)
    with pytest.raises(Exception):
        memory.source_tokens = 2
    assert replace(memory, source_tokens=2).source_tokens == 2
