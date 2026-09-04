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


def _rotate_keys_by_position_deltas(keys, rope: object, deltas: Sequence[int]):
    """Apply a potentially different RoPE phase translation to every key."""

    if len(deltas) != int(keys.shape[2]):
        raise ValueError("RoPE delta count must match the key token dimension")
    if not any(deltas):
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
    offsets = mx.array(deltas, dtype=mx.float32)[:, None]
    angles = offsets * frequencies[None, :] * scale
    cosine = mx.cos(angles)[None, None, :, :].astype(keys.dtype)
    sine = mx.sin(angles)[None, None, :, :].astype(keys.dtype)
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


def rebind_native_memories_to_receipt(
    model: object,
    memories: Sequence[MLXNativeMemory],
    composition_receipt: object,
) -> MLXNativeMemory:
    """Rebind independent native memories to an auditable position receipt.

    The receipt is duck-typed to keep this low-level MLX module independent of
    the RAG policy module. Resource order, source coordinates, target
    coordinates, and query position must align exactly with ``memories``.
    """

    placements = tuple(getattr(composition_receipt, "placements"))
    if not memories or len(memories) != len(placements):
        raise ValueError("composition receipt must align with native memories")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("native memories have incompatible layer counts")
    for memory, placement in zip(memories, placements):
        if memory.source_tokens != len(placement.source_positions):
            raise ValueError("receipt source positions do not match native memory")
        if len(placement.source_positions) != len(placement.effective_positions):
            raise ValueError("receipt source/effective positions are incompatible")

    import mlx.core as mx

    model_layers = tuple(getattr(getattr(model, "model", model), "layers"))
    if len(model_layers) != len(memories[0].layers):
        raise ValueError("native memories do not match the model layer count")
    layers = []
    for layer_index, model_layer in enumerate(model_layers):
        attention = getattr(model_layer, "self_attn", None)
        rope = getattr(attention, "rope", None)
        if rope is None:
            raise ValueError(f"model layer {layer_index} exposes no RoPE module")
        keys = []
        values = []
        for memory, placement in zip(memories, placements):
            deltas = tuple(
                target - source
                for source, target in zip(
                    placement.source_positions, placement.effective_positions
                )
            )
            keys.append(
                _rotate_keys_by_position_deltas(
                    memory.layers[layer_index].keys, rope, deltas
                )
            )
            values.append(memory.layers[layer_index].values)
        layers.append(
            MLXNativeLayerKV(
                mx.concatenate(tuple(keys), axis=2),
                mx.concatenate(tuple(values), axis=2),
            )
        )
    return MLXNativeMemory(
        tuple(layers),
        source_tokens=sum(memory.source_tokens for memory in memories),
        query_position_base=int(getattr(composition_receipt, "query_position")),
    )


def repair_token_indices(
    token_count: int,
    fraction: float,
    *,
    mode: str = "even",
    resource_lengths: Sequence[int] = (),
) -> tuple[int, ...]:
    """Choose a deterministic token subset for contextual-repair diagnostics.

    ``boundary`` concentrates repair around joins between independently encoded
    resources. ``later_prefix`` repairs the prefixes that, in a fresh causal
    packing, would first absorb preceding-resource context. These policies test
    mechanism hypotheses; they do not make fresh packed states free to obtain.
    """

    if token_count <= 0:
        raise ValueError("repair requires a positive token count")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("repair fraction must be in [0, 1]")
    if mode not in {"even", "prefix", "boundary", "later_prefix"}:
        raise ValueError(f"unsupported repair mode: {mode}")
    if resource_lengths:
        if any(length <= 0 for length in resource_lengths):
            raise ValueError("repair resource lengths must be positive")
        if sum(resource_lengths) != token_count:
            raise ValueError("repair resource lengths do not match token count")
    if fraction == 0.0:
        return ()
    repair_count = (
        token_count
        if fraction == 1.0
        else max(1, round(token_count * fraction))
    )
    if repair_count == token_count:
        return tuple(range(token_count))
    if mode == "prefix":
        return tuple(range(repair_count))
    if mode == "even":
        return tuple(
            sorted(
                {
                    min(token_count - 1, (index * token_count) // repair_count)
                    for index in range(repair_count)
                }
            )
        )

    boundaries: list[int] = []
    cursor = 0
    for length in resource_lengths[:-1]:
        cursor += length
        boundaries.append(cursor)
    if not boundaries:
        return tuple(range(repair_count))

    candidates: list[int] = []
    seen: set[int] = set()
    if mode == "boundary":
        for radius in range(token_count):
            for boundary in boundaries:
                for index in (boundary + radius, boundary - 1 - radius):
                    if 0 <= index < token_count and index not in seen:
                        seen.add(index)
                        candidates.append(index)
                        if len(candidates) == repair_count:
                            return tuple(sorted(candidates))
    else:
        starts = boundaries
        lengths = tuple(resource_lengths[1:])
        for offset in range(max(lengths)):
            for start, length in zip(starts, lengths):
                index = start + offset
                if offset < length and index not in seen:
                    seen.add(index)
                    candidates.append(index)
                    if len(candidates) == repair_count:
                        return tuple(sorted(candidates))

    for index in range(token_count):
        if index not in seen:
            candidates.append(index)
            if len(candidates) == repair_count:
                break
    return tuple(sorted(candidates))


def diagnostic_repair_memory(
    rebound: MLXNativeMemory,
    fresh_packed: MLXNativeMemory,
    fraction: float,
    *,
    mode: str = "even",
    resource_lengths: Sequence[int] = (),
    layer_indices: Sequence[int] | None = None,
) -> MLXNativeMemory:
    """Replace a deterministic K/V token subset with fresh packed states.

    This is a mechanism diagnostic, not an efficient runtime algorithm: the
    fresh packed memory must already exist.  Evenly spaced replacements make a
    repair curve quantify how much packed contextual state is needed before
    parity returns, without pretending that the states were free to compute.
    """

    if len(rebound.layers) != len(fresh_packed.layers):
        raise ValueError("repair memories have incompatible layer counts")
    if rebound.source_tokens != fresh_packed.source_tokens:
        raise ValueError("repair memories have incompatible token counts")
    indices = repair_token_indices(
        rebound.source_tokens,
        fraction,
        mode=mode,
        resource_lengths=resource_lengths,
    )
    if not indices:
        return rebound
    selected_layers = (
        set(range(len(rebound.layers)))
        if layer_indices is None
        else set(layer_indices)
    )
    if any(index < 0 or index >= len(rebound.layers) for index in selected_layers):
        raise ValueError("repair layer index is outside the native memory")
    if len(indices) == rebound.source_tokens and len(selected_layers) == len(rebound.layers):
        return fresh_packed
    import mlx.core as mx

    token_count = rebound.source_tokens
    selected_tokens = set(indices)
    mask = mx.array(
        [position in selected_tokens for position in range(token_count)], dtype=mx.bool_
    )[None, None, :, None]
    layers = tuple(
        MLXNativeLayerKV(
            mx.where(mask, fresh.keys, source.keys)
            if index in selected_layers
            else source.keys,
            mx.where(mask, fresh.values, source.values)
            if index in selected_layers
            else source.values,
        )
        for index, (source, fresh) in enumerate(
            zip(rebound.layers, fresh_packed.layers)
        )
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
