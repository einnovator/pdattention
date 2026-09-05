"""Minimal MLX selected-native-K/V realization used by Paper 3.2.

The product runtime owns persistence, authorization, and lifecycle policy.
This module intentionally contains only the public mlx-lm cache seam needed to
compare a frozen selected-text receipt with the same contiguous native K/V.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .crossdoc_composition import (
    CrossDocumentCompositionConfig,
    CrossDocumentCompositionMode,
    CrossDocumentCompositionReceipt,
    build_gist_attention_mask,
    memory_identity_digest,
)


class PositionBindingMode(str, Enum):
    """Whether stored keys already carry a fixed RoPE phase."""

    POST_ROPE = "POST_ROPE"
    PRE_ROPE = "PRE_ROPE"


@dataclass(frozen=True)
class MLXRopeContract:
    """Model-specific geometry required to materialize pre-RoPE keys safely."""

    model_revision: str
    layer_frequency_digest: str
    scaling_policy: tuple[str, ...]
    rope_dims: tuple[int, ...]
    layout: tuple[str, ...]
    schema_version: str = "paper3.2-mlx-rope-contract-v1"

    @property
    def contract_id(self) -> str:
        value = {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "layer_frequency_digest": self.layer_frequency_digest,
            "scaling_policy": self.scaling_policy,
            "rope_dims": self.rope_dims,
            "layout": self.layout,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class MLXNativeLayerKV:
    """One layer's K/V arrays in ``[B, Hkv, T, Dh]`` layout."""

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
    position_binding_mode: PositionBindingMode = PositionBindingMode.POST_ROPE
    source_positions: tuple[int, ...] = ()
    rope_contract: MLXRopeContract | None = None

    def __post_init__(self) -> None:
        if self.source_tokens <= 0:
            raise ValueError("native memory must contain at least one token")
        if self.query_position_base is not None and self.query_position_base < 0:
            raise ValueError("native query position base cannot be negative")
        if not self.source_positions:
            object.__setattr__(self, "source_positions", tuple(range(self.source_tokens)))
        if len(self.source_positions) != self.source_tokens:
            raise ValueError("native source positions must match the token dimension")
        if any(
            right <= left
            for left, right in zip(self.source_positions, self.source_positions[1:])
        ):
            raise ValueError("native source positions must be strictly increasing")
        if (
            self.position_binding_mode is PositionBindingMode.PRE_ROPE
            and self.rope_contract is None
        ):
            raise ValueError("pre-RoPE native memory requires a RoPE contract")

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

    @property
    def pre_rope_storage(self) -> bool:
        return self.position_binding_mode is PositionBindingMode.PRE_ROPE


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


def _memory_from_post_rope_states(
    model: object,
    states: Sequence[object],
    source_tokens: int,
    *,
    position_binding_mode: PositionBindingMode,
    model_revision: str,
) -> MLXNativeMemory:
    """Normalize host cache states into an explicit position-binding contract."""

    layers = []
    for state in states:
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError("the MLX model exposed a non-attention cache layer")
        layers.append(MLXNativeLayerKV(state[0], state[1]))
    contract = _rope_contract(model, model_revision=model_revision)
    if position_binding_mode is PositionBindingMode.PRE_ROPE:
        positions = tuple(range(source_tokens))
        model_layers = _model_layers(model)
        layers = [
            MLXNativeLayerKV(
                _rotate_keys_by_position_deltas(
                    layer.keys,
                    _layer_rope(model_layer, layer_index),
                    tuple(-position for position in positions),
                ),
                layer.values,
            )
            for layer_index, (layer, model_layer) in enumerate(
                zip(layers, model_layers)
            )
        ]
    return MLXNativeMemory(
        tuple(layers),
        source_tokens=source_tokens,
        position_binding_mode=position_binding_mode,
        source_positions=tuple(range(source_tokens)),
        rope_contract=contract,
    )


def encode_native_memory(
    model: object,
    token_ids: Sequence[int],
    *,
    position_binding_mode: PositionBindingMode = PositionBindingMode.POST_ROPE,
    model_revision: str = "UNKNOWN",
) -> MLXNativeMemory:
    """Encode contiguous evidence and retain post- or pre-RoPE native K/V.

    MLX's public cache exposes post-RoPE keys. ``PRE_ROPE`` applies the exact
    inverse host transform at each source position before persistence. This is
    algebraically the raw projected key and avoids a model-specific hook.
    """

    if not token_ids:
        raise ValueError("cannot encode empty evidence as native K/V")
    if position_binding_mode is PositionBindingMode.PRE_ROPE:
        causal = tuple(
            tuple(column <= row for column in range(len(token_ids)))
            for row in range(len(token_ids))
        )
        return encode_native_memory_with_mask(
            model,
            token_ids,
            causal,
            position_binding_mode=position_binding_mode,
            model_revision=model_revision,
        )
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model)
    model(mx.array(token_ids, dtype=mx.int32)[None], cache=caches)
    states = [cache.state for cache in caches]
    mx.eval(states)
    return _memory_from_post_rope_states(
        model,
        states,
        len(token_ids),
        position_binding_mode=position_binding_mode,
        model_revision=model_revision,
    )


def encode_native_memory_with_mask(
    model: object,
    token_ids: Sequence[int],
    attention_mask: Sequence[Sequence[bool]],
    *,
    position_binding_mode: PositionBindingMode = PositionBindingMode.POST_ROPE,
    model_revision: str = "UNKNOWN",
) -> MLXNativeMemory:
    """Encode a packed document prefix under an explicit document mask.

    The model's public top-level call constructs its own causal mask. Walking
    the unchanged host layers is the narrowest available seam for the B arm;
    projections, RoPE, attention kernels, residuals, and cache objects remain
    the host implementation.
    """

    if not token_ids:
        raise ValueError("cannot encode empty evidence as native K/V")
    size = len(token_ids)
    if len(attention_mask) != size or any(len(row) != size for row in attention_mask):
        raise ValueError("document attention mask must have shape [tokens, tokens]")
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    host = getattr(model, "model", model)
    layers = _model_layers(model)
    hidden = host.embed_tokens(mx.array(token_ids, dtype=mx.int32)[None])
    mask = mx.array(attention_mask, dtype=mx.bool_)
    stored_layers: list[MLXNativeLayerKV] = []
    for layer_index, layer in enumerate(layers):
        layer_mask = mask
        if bool(getattr(layer, "use_sliding", False)):
            window = int(getattr(host, "sliding_window"))
            positions = mx.arange(size)
            layer_mask = layer_mask & (
                positions[:, None] < positions[None, :] + window
            )
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        batch, tokens, _ = normalized.shape
        queries = attention.q_proj(normalized).reshape(
            batch, tokens, attention.n_heads, -1
        ).transpose(0, 2, 1, 3)
        keys_pre = attention.k_proj(normalized).reshape(
            batch, tokens, attention.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        values = attention.v_proj(normalized).reshape(
            batch, tokens, attention.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        if q_norm is not None:
            queries = q_norm(queries.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        if k_norm is not None:
            keys_pre = k_norm(keys_pre.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        queries = attention.rope(queries)
        keys_post = attention.rope(keys_pre)
        stored_layers.append(
            MLXNativeLayerKV(
                keys_pre
                if position_binding_mode is PositionBindingMode.PRE_ROPE
                else keys_post,
                values,
            )
        )
        attended = scaled_dot_product_attention(
            queries,
            keys_post,
            values,
            cache=None,
            scale=attention.scale,
            mask=layer_mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, -1)
        hidden = residual + attention.o_proj(attended)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    hidden = host.norm(hidden)
    mx.eval(hidden, [(layer.keys, layer.values) for layer in stored_layers])
    return MLXNativeMemory(
        tuple(stored_layers),
        source_tokens=size,
        position_binding_mode=position_binding_mode,
        source_positions=tuple(range(size)),
        rope_contract=_rope_contract(model, model_revision=model_revision),
    )


def combine_native_memories(memories: Sequence[MLXNativeMemory]) -> MLXNativeMemory:
    """Concatenate independently encoded K/V while retaining source-local phase."""

    if not memories:
        raise ValueError("at least one native memory is required")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("native memories have incompatible layer counts")
    if any(
        memory.position_binding_mode is not PositionBindingMode.POST_ROPE
        for memory in memories
    ):
        raise ValueError("pre-RoPE memory must be bound before cache construction")
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
        position_binding_mode=PositionBindingMode.POST_ROPE,
    )


def _rope_inverse_frequencies(rope: object, dimensions: int):
    """Return the exact per-pair angular frequencies used by an MLX RoPE."""

    import mlx.core as mx

    configured = getattr(rope, "_freqs", None)
    if configured is not None:
        # Llama-3 RoPE stores wavelength-like divisors after its piecewise
        # long-context scaling. Reusing this tensor avoids approximating the
        # host geometry with a single base value.
        return mx.reciprocal(configured.astype(mx.float32))
    base = float(getattr(rope, "base", 10_000.0))
    return mx.exp(
        -mx.arange(0, dimensions, 2, dtype=mx.float32)
        * (mx.log(mx.array(base, dtype=mx.float32)) / dimensions)
    )


def _model_layers(model: object) -> tuple[object, ...]:
    layers = tuple(getattr(getattr(model, "model", model), "layers"))
    if not layers:
        raise ValueError("model exposes no transformer layers")
    return layers


def _layer_rope(model_layer: object, layer_index: int) -> object:
    attention = getattr(model_layer, "self_attn", None)
    rope = getattr(attention, "rope", None)
    if rope is None:
        raise ValueError(f"model layer {layer_index} exposes no RoPE module")
    return rope


def _rope_contract(model: object, *, model_revision: str) -> MLXRopeContract:
    """Fingerprint the exact host frequency/scaling geometry for every layer."""

    descriptors: list[dict[str, object]] = []
    policies: list[str] = []
    dimensions: list[int] = []
    layouts: list[str] = []
    for layer_index, layer in enumerate(_model_layers(model)):
        rope = _layer_rope(layer, layer_index)
        dims = int(getattr(rope, "dims"))
        dimensions.append(dims)
        layout = "traditional_interleaved" if bool(
            getattr(rope, "traditional", False)
        ) else "half_rotation"
        layouts.append(layout)
        configured = getattr(rope, "_freqs", None)
        if configured is not None:
            frequencies = [float(value) for value in configured.tolist()]
            policy = "host_piecewise_frequency_tensor"
        else:
            frequencies = []
            policy = "host_base_frequency"
        scale = float(getattr(rope, "scale", 1.0))
        policies.append(f"{policy}:scale={scale:g}")
        descriptors.append(
            {
                "layer": layer_index,
                "dims": dims,
                "layout": layout,
                "scale": scale,
                "base": float(getattr(rope, "base", 10_000.0)),
                "configured_frequencies": frequencies,
            }
        )
    frequency_digest = hashlib.sha256(
        json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return MLXRopeContract(
        model_revision=model_revision,
        layer_frequency_digest=frequency_digest,
        scaling_policy=tuple(policies),
        rope_dims=tuple(dimensions),
        layout=tuple(layouts),
    )


def _validate_rope_contract(model: object, memory: MLXNativeMemory) -> MLXRopeContract:
    if memory.rope_contract is None:
        raise ValueError("pre-RoPE memory has no model geometry contract")
    actual = _rope_contract(
        model, model_revision=memory.rope_contract.model_revision
    )
    if actual.contract_id != memory.rope_contract.contract_id:
        raise ValueError("pre-RoPE memory does not match the host RoPE contract")
    return actual


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
    scale = float(getattr(rope, "scale", 1.0))
    frequencies = _rope_inverse_frequencies(rope, dimensions)
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
    scale = float(getattr(rope, "scale", 1.0))
    frequencies = _rope_inverse_frequencies(rope, dimensions)
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


def _bind_pre_rope_keys(keys, rope: object, positions: Sequence[int]):
    """Apply host RoPE directly when request positions form one exact interval."""

    if len(positions) != int(keys.shape[2]):
        raise ValueError("request position count must match pre-RoPE keys")
    if positions and tuple(positions) == tuple(
        range(int(positions[0]), int(positions[0]) + len(positions))
    ):
        return rope(keys, offset=int(positions[0]))
    return _rotate_keys_by_position_deltas(keys, rope, positions)


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

    model_layers = _model_layers(model)
    if len(model_layers) != len(memories[0].layers):
        raise ValueError("native memories do not match the model layer count")
    for memory in memories:
        if memory.position_binding_mode is PositionBindingMode.PRE_ROPE:
            _validate_rope_contract(model, memory)
    starts: list[int] = []
    cursor = 0
    for memory in memories:
        starts.append(cursor)
        cursor += memory.source_tokens

    layers = []
    for layer_index, model_layer in enumerate(model_layers):
        rope = _layer_rope(model_layer, layer_index)
        keys = tuple(
            _bind_pre_rope_keys(
                memory.layers[layer_index].keys,
                rope,
                tuple(start + index for index in range(memory.source_tokens)),
            )
            if memory.position_binding_mode is PositionBindingMode.PRE_ROPE
            else _rotate_keys_by_position_deltas(
                memory.layers[layer_index].keys,
                rope,
                tuple(
                    start + index - source
                    for index, source in enumerate(memory.source_positions)
                ),
            )
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
        tuple(layers),
        source_tokens=cursor,
        query_position_base=cursor,
        position_binding_mode=PositionBindingMode.POST_ROPE,
        source_positions=tuple(range(cursor)),
        rope_contract=_rope_contract(
            model,
            model_revision=next(
                (
                    memory.rope_contract.model_revision
                    for memory in memories
                    if memory.rope_contract is not None
                ),
                "UNKNOWN",
            ),
        ),
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

    model_layers = _model_layers(model)
    if len(model_layers) != len(memories[0].layers):
        raise ValueError("native memories do not match the model layer count")
    for memory in memories:
        if memory.position_binding_mode is PositionBindingMode.PRE_ROPE:
            _validate_rope_contract(model, memory)
    layers = []
    effective_positions = tuple(
        position
        for placement in placements
        for position in placement.effective_positions
    )
    all_pre_rope = all(
        memory.position_binding_mode is PositionBindingMode.PRE_ROPE
        for memory in memories
    )
    one_contiguous_frame = effective_positions == tuple(
        range(effective_positions[0], effective_positions[0] + len(effective_positions))
    )
    for layer_index, model_layer in enumerate(model_layers):
        rope = _layer_rope(model_layer, layer_index)
        if all_pre_rope and one_contiguous_frame:
            raw_keys = mx.concatenate(
                tuple(memory.layers[layer_index].keys for memory in memories), axis=2
            )
            layers.append(
                MLXNativeLayerKV(
                    rope(raw_keys, offset=effective_positions[0]),
                    mx.concatenate(
                        tuple(memory.layers[layer_index].values for memory in memories),
                        axis=2,
                    ),
                )
            )
            continue
        keys = []
        values = []
        for memory, placement in zip(memories, placements):
            deltas = (
                tuple(placement.effective_positions)
                if memory.position_binding_mode is PositionBindingMode.PRE_ROPE
                else tuple(
                    target - source
                    for source, target in zip(
                        placement.source_positions, placement.effective_positions
                    )
                )
            )
            keys.append(
                _bind_pre_rope_keys(
                    memory.layers[layer_index].keys,
                    rope,
                    tuple(placement.effective_positions),
                )
                if memory.position_binding_mode is PositionBindingMode.PRE_ROPE
                else _rotate_keys_by_position_deltas(
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
        position_binding_mode=PositionBindingMode.POST_ROPE,
        source_positions=effective_positions,
        rope_contract=_rope_contract(
            model,
            model_revision=next(
                (
                    memory.rope_contract.model_revision
                    for memory in memories
                    if memory.rope_contract is not None
                ),
                "UNKNOWN",
            ),
        ),
    )


def _contextualize_mlx_gists(keys, values, mask, residual_scale: float):
    """MLX equivalent of parameter-free identity-projection gist attention."""

    import mlx.core as mx

    width = int(keys.shape[-1])
    scores = mx.matmul(
        keys.astype(mx.float32), mx.swapaxes(keys.astype(mx.float32), -1, -2)
    ) / math.sqrt(width)
    visible = mask[None, None, :, :]
    scores = mx.where(visible, scores, mx.array(float("-inf"), dtype=scores.dtype))
    attention = mx.softmax(scores, axis=-1)
    contextual_keys = keys + residual_scale * mx.matmul(
        attention.astype(keys.dtype), keys
    )
    contextual_values = values + residual_scale * mx.matmul(
        attention.astype(values.dtype), values
    )
    return contextual_keys, contextual_values, attention


def compose_cross_document_memory(
    model: object,
    memories: Sequence[MLXNativeMemory],
    composition_receipt: object,
    *,
    record_ids: Sequence[str],
    config: CrossDocumentCompositionConfig,
    document_ids: Sequence[str] = (),
) -> tuple[MLXNativeMemory, CrossDocumentCompositionReceipt]:
    """Append ephemeral contextual gist or boundary K/V beside immutable records.

    Persistent inputs must contain pre-RoPE keys. Their layerwise mean K/V are
    composed bidirectionally by default. New request-local K/V are then bound
    into a compact band immediately before the query. Stored record tensors are
    never replaced or mutated.
    """

    if not memories or len(memories) != len(record_ids):
        raise ValueError("record identities must align with native memories")
    if any(
        memory.position_binding_mode is not PositionBindingMode.PRE_ROPE
        for memory in memories
    ):
        raise ValueError("cross-document composition requires persistent pre-RoPE memory")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("cross-document memories have incompatible layer counts")
    if document_ids and len(document_ids) != len(memories):
        raise ValueError("document identities must align with native memories")

    import mlx.core as mx

    started = time.perf_counter()
    rebound = rebind_native_memories_to_receipt(model, memories, composition_receipt)
    mask_torch = build_gist_attention_mask(
        len(memories), config.attention_mask, document_ids=document_ids
    )
    mask = mx.array(mask_torch.tolist(), dtype=mx.bool_)
    model_layers = _model_layers(model)
    output_layers: list[MLXNativeLayerKV] = []
    local_token_count: int | None = None
    gist_dim: int | None = None
    gist_positions: tuple[int, ...] = ()

    for layer_index, model_layer in enumerate(model_layers):
        key_gists = mx.concatenate(
            tuple(
                mx.mean(memory.layers[layer_index].keys, axis=2, keepdims=True)
                for memory in memories
            ),
            axis=2,
        )
        value_gists = mx.concatenate(
            tuple(
                mx.mean(memory.layers[layer_index].values, axis=2, keepdims=True)
                for memory in memories
            ),
            axis=2,
        )
        contextual_keys, contextual_values, _ = _contextualize_mlx_gists(
            key_gists, value_gists, mask, config.residual_scale
        )
        gist_dim = int(contextual_keys.shape[1] * contextual_keys.shape[-1])

        if config.mode is CrossDocumentCompositionMode.GIST_SA_APPEND:
            local_keys = contextual_keys
            local_values = contextual_values
        else:
            boundary = config.mode.boundary_tokens
            key_parts = []
            value_parts = []
            for record_index, memory in enumerate(memories):
                count = memory.source_tokens
                indices = tuple(
                    sorted(
                        set(range(min(boundary, count)))
                        | set(range(max(0, count - boundary), count))
                    )
                )
                index_array = mx.array(indices, dtype=mx.int32)
                source = memory.layers[layer_index]
                source_key_gist = key_gists[:, :, record_index : record_index + 1, :]
                source_value_gist = value_gists[:, :, record_index : record_index + 1, :]
                delta_key = (
                    contextual_keys[:, :, record_index : record_index + 1, :]
                    - source_key_gist
                )
                delta_value = (
                    contextual_values[:, :, record_index : record_index + 1, :]
                    - source_value_gist
                )
                key_parts.append(mx.take(source.keys, index_array, axis=2) + delta_key)
                value_parts.append(mx.take(source.values, index_array, axis=2) + delta_value)
            local_keys = mx.concatenate(tuple(key_parts), axis=2)
            local_values = mx.concatenate(tuple(value_parts), axis=2)

        current_count = int(local_keys.shape[2])
        if local_token_count is None:
            local_token_count = current_count
            gist_positions = tuple(
                range(rebound.position_base, rebound.position_base + current_count)
            )
        elif current_count != local_token_count:
            raise RuntimeError("composition token count changed across layers")
        bound_keys = _bind_pre_rope_keys(
            local_keys, _layer_rope(model_layer, layer_index), gist_positions
        )
        output_layers.append(
            MLXNativeLayerKV(
                mx.concatenate((rebound.layers[layer_index].keys, bound_keys), axis=2),
                mx.concatenate((rebound.layers[layer_index].values, local_values), axis=2),
            )
        )

    mx.eval([(layer.keys, layer.values) for layer in output_layers])
    local_tokens = int(local_token_count or 0)
    composed = MLXNativeMemory(
        tuple(output_layers),
        source_tokens=rebound.source_tokens + local_tokens,
        query_position_base=rebound.position_base + local_tokens,
        position_binding_mode=PositionBindingMode.POST_ROPE,
        source_positions=tuple(
            (*rebound.source_positions, *range(rebound.position_base, rebound.position_base + local_tokens))
        ),
        rope_contract=rebound.rope_contract,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    boundary_tokens = config.mode.boundary_tokens
    receipt = CrossDocumentCompositionReceipt(
        mode=config.mode.value,
        gist_count=len(memories),
        gist_dim=int(gist_dim or 0),
        gist_attention_mask=config.attention_mask.value,
        gist_attention_edges=int(mask_torch.sum().item()),
        boundary_tokens_per_record=boundary_tokens,
        corrected_token_count=local_tokens if boundary_tokens else 0,
        request_composition_ms=elapsed_ms,
        request_composition_bytes=composed.nbytes - rebound.nbytes,
        persistent_native_tokens=rebound.source_tokens,
        request_local_native_tokens=local_tokens,
        gist_positions=gist_positions,
        record_ids=tuple(record_ids),
        source_memory_digest=memory_identity_digest(
            record_ids=record_ids,
            source_tokens=tuple(memory.source_tokens for memory in memories),
            layer_count=len(model_layers),
        ),
        pooling_method=config.pooling_method,
        normalization_policy=config.normalization_policy,
        position_policy=config.position_policy,
    )
    return composed, receipt


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
        position_binding_mode=PositionBindingMode.POST_ROPE,
        source_positions=fresh_packed.source_positions,
        rope_contract=fresh_packed.rope_contract,
    )


def native_memory_diagnostics(
    reference: MLXNativeMemory, candidate: MLXNativeMemory
) -> dict[str, object]:
    """Return per-layer K/V RMSE and maximum error for matched memories."""

    if len(reference.layers) != len(candidate.layers):
        raise ValueError("diagnostic memories have incompatible layer counts")
    if reference.source_tokens != candidate.source_tokens:
        raise ValueError("diagnostic memories have incompatible token counts")
    import mlx.core as mx
    import numpy as np

    rows = []
    for layer_index, (left, right) in enumerate(
        zip(reference.layers, candidate.layers)
    ):
        left_keys = np.asarray(left.keys.astype(mx.float32))
        right_keys = np.asarray(right.keys.astype(mx.float32))
        left_values = np.asarray(left.values.astype(mx.float32))
        right_values = np.asarray(right.values.astype(mx.float32))
        key_delta = left_keys - right_keys
        value_delta = left_values - right_values
        rows.append(
            {
                "layer": layer_index,
                "key_rmse": float(np.sqrt(np.mean(key_delta * key_delta))),
                "key_max_abs_delta": float(np.max(np.abs(key_delta))),
                "value_rmse": float(np.sqrt(np.mean(value_delta * value_delta))),
                "value_max_abs_delta": float(np.max(np.abs(value_delta))),
            }
        )
    return {
        "layers": rows,
        "max_key_rmse": max(row["key_rmse"] for row in rows),
        "max_value_rmse": max(row["value_rmse"] for row in rows),
        "max_key_abs_delta": max(row["key_max_abs_delta"] for row in rows),
        "max_value_abs_delta": max(row["value_max_abs_delta"] for row in rows),
    }


def make_native_prompt_cache(model: object, memory: MLXNativeMemory):
    """Create request-local caches backed by one immutable selected memory."""

    from mlx_lm.models.cache import make_prompt_cache

    if memory.position_binding_mode is not PositionBindingMode.POST_ROPE:
        raise ValueError("pre-RoPE memory must be bound to request positions first")
    local = make_prompt_cache(model)
    if len(local) != len(memory.layers):
        raise ValueError("native memory does not match the model layer count")
    return [
        MLXSelectedKVCache(cache, layer, memory.position_base)
        for cache, layer in zip(local, memory.layers)
    ]
