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
    """Immutable selected evidence aligned to all model layers.

    ``source_tokens`` is the number of physically materialized K/V tokens.
    ``query_position_base`` is the request coordinate assigned to the first
    query token.  They differ when independently encoded resources retain
    overlapping source-local coordinates.
    """

    layers: tuple[MLXNativeLayerKV, ...]
    source_tokens: int
    query_position_base: int | None = None

    def __post_init__(self) -> None:
        if self.source_tokens <= 0:
            raise ValueError("native memory must contain at least one token")
        if self.query_position_base is not None and self.query_position_base < 0:
            raise ValueError("native query position base cannot be negative")

    @property
    def position_base(self) -> int:
        return (
            self.source_tokens
            if self.query_position_base is None
            else self.query_position_base
        )

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
        layers,
        source_tokens=sum(memory.source_tokens for memory in memories),
        query_position_base=max(memory.position_base for memory in memories),
    )


def _rotate_keys_by_delta(keys, rope: object, delta: int):
    """Apply one constant RoPE phase delta to post-RoPE keys.

    Independently encoded resource keys already contain their source-local
    phase.  RoPE's group property makes ``R(delta) @ R(source)`` equal to the
    packed target phase when every token in a resource is translated by the
    same offset.  This operation changes keys only; values carry no RoPE phase.
    """

    if delta == 0:
        return keys
    import mlx.core as mx

    dimensions = int(getattr(rope, "dims"))
    if dimensions <= 0 or dimensions > int(keys.shape[-1]) or dimensions % 2:
        raise ValueError("unsupported RoPE dimensions for native key rebinding")
    base = float(getattr(rope, "base", 10_000.0))
    scale = float(getattr(rope, "scale", 1.0))
    frequencies = mx.exp(
        -mx.arange(0, dimensions, 2, dtype=mx.float32)
        * (mx.log(mx.array(base, dtype=mx.float32)) / dimensions)
    )
    angles = frequencies * (float(delta) * scale)
    cosine = mx.cos(angles).astype(keys.dtype)
    sine = mx.sin(angles).astype(keys.dtype)
    rotated = keys[..., :dimensions]
    tail = keys[..., dimensions:]
    if bool(getattr(rope, "traditional", False)):
        even = rotated[..., 0::2]
        odd = rotated[..., 1::2]
        rebound = mx.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), axis=-1
        ).reshape(rotated.shape)
    else:
        half = dimensions // 2
        first = rotated[..., :half]
        second = rotated[..., half:]
        rebound = mx.concatenate(
            (first * cosine - second * sine, first * sine + second * cosine),
            axis=-1,
        )
    return mx.concatenate((rebound, tail), axis=-1) if tail.shape[-1] else rebound


def rebind_native_memories_global_packed(
    model: object,
    memories: Sequence[MLXNativeMemory],
) -> MLXNativeMemory:
    """Compose independent memories at contiguous ``GLOBAL_PACKED`` positions.

    Each resource is assumed to have been encoded from source position zero.
    Layer-specific RoPE parameters are read from the host model, allowing GQA
    key shapes to pass through unchanged.  The result has the same token order
    and query position as a fresh packed encoding, but does not invent the
    cross-resource contextualization that independent encoding omitted.
    """

    if not memories:
        raise ValueError("at least one native memory is required")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("native memories have incompatible layer counts")
    import mlx.core as mx

    model_layers = tuple(getattr(getattr(model, "model", model), "layers"))
    if len(model_layers) != len(memories[0].layers):
        raise ValueError("native memories do not match the model layer count")
    starts: list[int] = []
    cursor = 0
    for memory in memories:
        starts.append(cursor)
        cursor += memory.source_tokens

    layers = []
    for layer_index, model_layer in enumerate(model_layers):
        attention = getattr(model_layer, "self_attn", None)
        rope = getattr(attention, "rope", None)
        if rope is None:
            raise ValueError(f"model layer {layer_index} exposes no RoPE module")
        keys = tuple(
            _rotate_keys_by_delta(memory.layers[layer_index].keys, rope, start)
            for memory, start in zip(memories, starts)
        )
        values = tuple(
            memory.layers[layer_index].values for memory in memories
        )
        layers.append(
            MLXNativeLayerKV(
                mx.concatenate(keys, axis=2),
                mx.concatenate(values, axis=2),
            )
        )
    return MLXNativeMemory(
        tuple(layers), source_tokens=cursor, query_position_base=cursor
    )


def diagnostic_repair_memory(
    rebound: MLXNativeMemory,
    fresh_packed: MLXNativeMemory,
    fraction: float,
) -> MLXNativeMemory:
    """Replace a deterministic K/V token subset with fresh packed states.

    This is a mechanism diagnostic, not an efficient runtime algorithm: the
    fresh packed memory must already exist.  Evenly spaced replacements make a
    repair curve quantify how much packed contextual state is needed before
    parity returns, without pretending that the states were free to compute.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("repair fraction must be in [0, 1]")
    if len(rebound.layers) != len(fresh_packed.layers):
        raise ValueError("repair memories have incompatible layer counts")
    if rebound.source_tokens != fresh_packed.source_tokens:
        raise ValueError("repair memories have incompatible token counts")
    if fraction == 0.0:
        return rebound
    if fraction == 1.0:
        return fresh_packed
    import mlx.core as mx

    token_count = rebound.source_tokens
    repair_count = max(1, round(token_count * fraction))
    indices = {
        min(token_count - 1, (index * token_count) // repair_count)
        for index in range(repair_count)
    }
    mask = mx.array(
        [position in indices for position in range(token_count)], dtype=mx.bool_
    )[None, None, :, None]
    layers = tuple(
        MLXNativeLayerKV(
            mx.where(mask, fresh.keys, source.keys),
            mx.where(mask, fresh.values, source.values),
        )
        for source, fresh in zip(rebound.layers, fresh_packed.layers)
    )
    return MLXNativeMemory(
        layers,
        source_tokens=token_count,
        query_position_base=fresh_packed.position_base,
    )


def make_native_prompt_cache(model: object, memory: MLXNativeMemory):
    """Create request-local caches backed by one immutable selected memory."""

    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model)
    if len(local) != len(memory.layers):
        raise ValueError("native memory does not match the model layer count")
    return [
        MLXSelectedKVCache(cache, layer, memory.position_base)
        for cache, layer in zip(local, memory.layers)
    ]
