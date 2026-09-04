from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_mlx_native import (
    MLXNativeLayerKV,
    MLXNativeMemory,
    repair_token_indices,
)


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


def test_repair_token_indices_supports_mechanistic_policies() -> None:
    assert repair_token_indices(20, 0.25, mode="prefix") == (0, 1, 2, 3, 4)
    boundary = repair_token_indices(
        20, 0.2, mode="boundary", resource_lengths=(10, 10)
    )
    assert boundary == (8, 9, 10, 11)
    later = repair_token_indices(
        20, 0.2, mode="later_prefix", resource_lengths=(8, 6, 6)
    )
    assert later == (8, 9, 14, 15)


def test_repair_token_indices_validates_resource_geometry() -> None:
    with pytest.raises(ValueError, match="do not match"):
        repair_token_indices(20, 0.5, mode="boundary", resource_lengths=(5, 5))
