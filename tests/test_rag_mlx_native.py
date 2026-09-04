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


def test_native_memory_is_immutable() -> None:
    memory = MLXNativeMemory((MLXNativeLayerKV(_Array(1), _Array(1)),), 1)
    with pytest.raises(Exception):
        memory.source_tokens = 2
    assert replace(memory, source_tokens=2).source_tokens == 2

