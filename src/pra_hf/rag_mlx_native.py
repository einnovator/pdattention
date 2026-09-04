"""Minimal MLX selected-native-K/V realization used by Paper 3.2.

The product runtime owns persistence, authorization, and lifecycle policy.
This module intentionally contains only the public mlx-lm cache seam needed to
compare a frozen selected-text receipt with the same contiguous native K/V.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MLXNativeLayerKV:
    """One layer's post-position K/V arrays in ``[B, Hkv, T, Dh]`` layout."""

    keys: object
    values: object

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass(frozen=True)
class MLXNativeMemory:
    """Immutable contiguous selected evidence aligned to all model layers."""

    layers: tuple[MLXNativeLayerKV, ...]
    source_tokens: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


class MLXSelectedKVCache:
    """Join immutable selected K/V with request-local sequential K/V.

    ``offset`` places the direct query after selected evidence for RoPE. Mask
    construction uses only the local offset, while every direct query can see
    all selected memory. The host attention still performs one softmax over the
    concatenated selected and local keys.
    """

    def __init__(self, local_cache: object, memory: MLXNativeLayerKV, position_base: int):
        if position_base < 0:
            raise ValueError("MLX native query position base cannot be negative")
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def local_offset(self) -> int:
        return int(self.local_cache.offset)

    @property
    def memory_tokens(self) -> int:
        return int(self.memory.keys.shape[2])

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        return (self.memory.keys, self.memory.values, *local)

    @state.setter
    def state(self, value) -> None:
        raise RuntimeError("selected native memory is immutable")

    @property
    def nbytes(self) -> int:
        return self.memory.nbytes + int(getattr(self.local_cache, "nbytes", 0))

    def empty(self) -> bool:
        return False

    def is_trimmable(self) -> bool:
        operation = getattr(self.local_cache, "is_trimmable", None)
        return bool(operation and operation())

    def trim(self, n: int) -> int:
        return int(self.local_cache.trim(n))

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        local_keys, local_values = self.local_cache.update_and_fetch(keys, values)
        return (
            mx.concatenate((self.memory.keys, local_keys), axis=2),
            mx.concatenate((self.memory.values, local_values), axis=2),
        )

    def make_mask(
        self,
        n: int,
        return_array: bool = False,
        window_size: int | None = None,
        **_: object,
    ):
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        if window_size is not None:
            local_window = min(window_size, self.local_offset + n)
            local = create_causal_mask(n, self.local_offset, window_size=local_window)
        else:
            local = create_causal_mask(n, self.local_offset)
        memory = mx.ones((n, self.memory_tokens), dtype=mx.bool_)
        return mx.concatenate((memory, local), axis=1)


def encode_native_memory(model: object, token_ids: Sequence[int]) -> MLXNativeMemory:
    """Encode one contiguous selection and retain each layer's native K/V."""

    if not token_ids:
        raise ValueError("cannot encode empty evidence as native K/V")
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model)
    model(mx.array(token_ids, dtype=mx.int32)[None], cache=caches)
    states = [cache.state for cache in caches]
    mx.eval(states)
    layers = []
    for state in states:
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError("the MLX model exposed a non-attention cache layer")
        layers.append(MLXNativeLayerKV(state[0], state[1]))
    return MLXNativeMemory(tuple(layers), source_tokens=len(token_ids))


def combine_native_memories(memories: Sequence[MLXNativeMemory]) -> MLXNativeMemory:
    """Concatenate independently encoded K/V while retaining source-local phase."""

    if not memories:
        raise ValueError("at least one native memory is required")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("native memories have incompatible layer counts")
    if len(memories) == 1:
        return memories[0]
    import mlx.core as mx

    layers = tuple(
        MLXNativeLayerKV(
            mx.concatenate(
                tuple(memory.layers[layer].keys for memory in memories), axis=2
            ),
            mx.concatenate(
                tuple(memory.layers[layer].values for memory in memories), axis=2
            ),
        )
        for layer in range(len(memories[0].layers))
    )
    # Independent resources retain overlapping source-local positions. The
    # query therefore starts after the longest source frame, not after an
    # invented packed concatenation.
    return MLXNativeMemory(
        layers, source_tokens=max(memory.source_tokens for memory in memories)
    )


def make_native_prompt_cache(model: object, memory: MLXNativeMemory):
    """Create request-local caches backed by one immutable selected memory."""

    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model)
    if len(local) != len(memory.layers):
        raise ValueError("native memory does not match the model layer count")
    return [
        MLXSelectedKVCache(cache, layer, memory.source_tokens)
        for cache, layer in zip(local, memory.layers)
    ]
