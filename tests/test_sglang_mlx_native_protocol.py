from __future__ import annotations

from dataclasses import dataclass

from pra_mlx.native import MLXNativeLayerKV
from pra_sglang.mlx_native import SGLangMLXNativeBridge, SGLangSelectedKVCache


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
