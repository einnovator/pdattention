from __future__ import annotations

from dataclasses import dataclass

from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory
from pra_sglang.mlx_native import (
    SGLangMLXNativeBridge,
    SGLangNativeRequest,
    SGLangSelectedKVCache,
)


@dataclass(frozen=True)
class _Array:
    shape: tuple[int, ...]


class _LocalCache:
    def __init__(self) -> None:
        self.offset = 7
        self.keys = object()
        self.values = object()
        self.state = ()

    def reset(self) -> None:
        self.offset = 0


@dataclass(frozen=True)
class _Layout:
    attention_layer_indices: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class _Runner:
    _cache_layout: _Layout = _Layout()


def test_sglang_cache_keeps_scheduler_and_rope_offsets_separate() -> None:
    local = _LocalCache()
    memory = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    cache = SGLangSelectedKVCache(local, memory, position_base=31)

    assert cache.offset == 7
    assert cache.rope_offset == 38
    assert cache.memory_tokens == 11
    assert cache.attention_view.offset == 38
    assert cache.keys is local.keys


def test_sglang_cache_reset_preserves_immutable_memory() -> None:
    local = _LocalCache()
    memory = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    cache = SGLangSelectedKVCache(local, memory, position_base=31)

    cache.reset()

    assert cache.offset == 0
    assert cache.rope_offset == 31
    assert cache.memory is memory


def test_sglang_pool_release_strips_selected_memory_wrapper() -> None:
    local = _LocalCache()
    memory = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    wrapped = SGLangSelectedKVCache(local, memory, position_base=31)

    assert SGLangMLXNativeBridge._unwrap_cache([wrapped]) == [local]


def test_sglang_radix_transition_wraps_twice_but_attaches_once() -> None:
    local = _LocalCache()
    layer = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    bridge = object.__new__(SGLangMLXNativeBridge)
    bridge.runner = _Runner()
    bridge._requests = {
        "request": SGLangNativeRequest(
            MLXNativeMemory((layer,), source_tokens=11), ("resource-R",)
        )
    }
    bridge.isolation = EnginePRAIsolationGuard()
    bridge.isolation.open_request("request", ("resource-R",))

    pool_backed = bridge._wrap_cache("request", [local])
    same_stage = bridge._wrap_cache("request", pool_backed)
    contiguous = bridge._wrap_cache("request", [_LocalCache()])

    assert same_stage is pool_backed
    assert isinstance(contiguous[0], SGLangSelectedKVCache)
    assert bridge.isolation.view("request").attached
