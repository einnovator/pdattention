"""In-process MLX executor for query-selected native K/V memory.

The implementation uses the public mlx-lm model and cache protocols.  Selected
resources are encoded into post-RoPE per-layer K/V once.  A request-local cache
then presents ``selected memory + local sequential cache`` to each attention
layer, so both sources participate in one softmax without making selected text
part of the visible prompt.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_hf.engine_memory import LogicalPRABlock, LogicalPRABlockId, LogicalPRABlockStore
from pra_hf.engine_residency import EnginePRAResidencyManager, PRAEvictionPolicy
from pra_hf.live_history import LiveKVSelectionPlan
from pra_hf.storage_lifecycle import (
    PRARetentionClass,
    PRAStorageEntry,
    PRAStorageManager,
    PRAStoragePolicy,
    ResidencyManagerHotBridge,
)


@dataclass(frozen=True)
class MLXNativeLayerKV:
    """One layer's post-position K/V arrays in ``[B,Hkv,T,D]`` layout."""

    keys: object
    values: object

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass(frozen=True)
class MLXNativeMemory:
    """Immutable selected-resource payload aligned to all model layers."""

    layers: tuple[MLXNativeLayerKV, ...]
    source_tokens: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    def selected_nbytes(self, layer_indices: Iterable[int] | None = None) -> int:
        """Return active bytes for a consumer-layer profile."""

        if layer_indices is None:
            return self.nbytes
        selected = set(map(int, layer_indices))
        return sum(layer.nbytes for index, layer in enumerate(self.layers) if index in selected)


@dataclass(frozen=True)
class MLXDisjointLayerKV:
    """Ordered K/V views over disjoint intervals of one canonical layer.

    The segment objects retain their source arrays instead of packing the
    selected intervals into one dense tensor. Whether an engine implements a
    slice as a storage alias is a runtime qualification question; this type
    guarantees only that PRA itself performs no concatenating selection copy.
    """

    segments: tuple[MLXNativeLayerKV, ...]
    source_keys: object | None = None
    source_values: object | None = None
    intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("Disjoint MLX K/V requires at least one interval.")
        supplied = self.source_keys is not None or self.source_values is not None
        if supplied and (
            self.source_keys is None
            or self.source_values is None
            or len(self.intervals) != len(self.segments)
        ):
            raise ValueError(
                "Source-backed disjoint MLX K/V requires both parents and one "
                "interval per segment."
            )

    @property
    def tokens(self) -> int:
        return sum(int(segment.keys.shape[2]) for segment in self.segments)

    @property
    def nbytes(self) -> int:
        return sum(segment.nbytes for segment in self.segments)


@dataclass(frozen=True)
class MLXDisjointNativeMemory:
    """Layer-aligned, non-packed views selected from canonical live K/V."""

    layers: tuple[MLXDisjointLayerKV, ...]
    source_tokens: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    def selected_nbytes(self, layer_indices: Iterable[int] | None = None) -> int:
        if layer_indices is None:
            return self.nbytes
        selected = set(map(int, layer_indices))
        return sum(
            layer.nbytes for index, layer in enumerate(self.layers) if index in selected
        )


@dataclass(frozen=True)
class MLXResidentKVSelection:
    """Native memory selected from a live MLX cache, never from source text."""

    memory: MLXNativeMemory | MLXDisjointNativeMemory
    plan: LiveKVSelectionPlan
    physical_kv_copy: bool | None
    selected_text_reencoded_tokens: int = 0


@dataclass(frozen=True)
class MLXQuantizedLayerKV:
    """Symmetric int8 K/V plus per-head scales for one model layer."""

    keys: object
    values: object
    key_scale: object
    value_scale: object
    original_dtype: str

    @property
    def nbytes(self) -> int:
        return int(
            self.keys.nbytes
            + self.values.nbytes
            + self.key_scale.nbytes
            + self.value_scale.nbytes
        )


@dataclass(frozen=True)
class MLXQuantizedMemory:
    """Compact selected-K/V residency dequantized only for materialization."""

    layers: tuple[MLXQuantizedLayerKV, ...]
    source_tokens: int
    scheme: str = "symmetric_int8_per_head"

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


def quantize_native_memory(memory: MLXNativeMemory) -> MLXQuantizedMemory:
    """Quantize selected K/V to symmetric int8 with ``[B,H,1,1]`` scales."""

    import mlx.core as mx

    layers = []
    for layer in memory.layers:
        arrays = []
        for value in (layer.keys, layer.values):
            maximum = mx.max(mx.abs(value.astype(mx.float32)), axis=(2, 3), keepdims=True)
            scale = mx.maximum(maximum / 127.0, mx.array(1e-8, dtype=mx.float32))
            quantized = mx.clip(mx.round(value.astype(mx.float32) / scale), -127, 127)
            arrays.append((quantized.astype(mx.int8), scale.astype(mx.float16)))
        layers.append(
            MLXQuantizedLayerKV(
                keys=arrays[0][0],
                values=arrays[1][0],
                key_scale=arrays[0][1],
                value_scale=arrays[1][1],
                original_dtype=str(layer.keys.dtype),
            )
        )
    result = MLXQuantizedMemory(tuple(layers), memory.source_tokens)
    mx.eval(
        *(
            array
            for layer in result.layers
            for array in (layer.keys, layer.values, layer.key_scale, layer.value_scale)
        )
    )
    return result


def dequantize_native_memory(memory: MLXQuantizedMemory) -> MLXNativeMemory:
    """Restore model-native arrays for the current MLX attention protocol."""

    import mlx.core as mx

    layers = []
    for layer in memory.layers:
        dtype = getattr(mx, layer.original_dtype, mx.float16)
        layers.append(
            MLXNativeLayerKV(
                (layer.keys.astype(mx.float32) * layer.key_scale).astype(dtype),
                (layer.values.astype(mx.float32) * layer.value_scale).astype(dtype),
            )
        )
    result = MLXNativeMemory(tuple(layers), memory.source_tokens)
    mx.eval(*(array for layer in result.layers for array in (layer.keys, layer.values)))
    return result


@dataclass(frozen=True)
class MLXNativeFingerprint:
    """Compatibility fields required before persisted K/V may be reused."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    position_policy: str
    consumer_profile: str
    resource_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "position_policy": self.position_policy,
            "consumer_profile": self.consumer_profile,
            "resource_version": self.resource_version,
        }


class MLXPositionedKVCache:
    """Local-only cache whose queries retain the source-position frame.

    Non-consumer layers do not receive selected memory, but their query and
    local K/V positions must still match ordinary split-cache execution.
    """

    def __init__(self, local_cache: object, position_base: int) -> None:
        self.local_cache = local_cache
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def local_offset(self) -> int:
        return int(self.local_cache.offset)

    @property
    def state(self):
        return self.local_cache.state

    @property
    def nbytes(self) -> int:
        return int(getattr(self.local_cache, "nbytes", 0))

    def empty(self) -> bool:
        return bool(getattr(self.local_cache, "empty", lambda: self.local_offset == 0)())

    def is_trimmable(self) -> bool:
        operation = getattr(self.local_cache, "is_trimmable", None)
        return bool(operation and operation())

    def trim(self, n: int) -> int:
        return int(self.local_cache.trim(n))

    def update_and_fetch(self, keys, values):
        return self.local_cache.update_and_fetch(keys, values)

    def make_mask(self, n: int, return_array: bool = False, window_size=None, **kwargs):
        return self.local_cache.make_mask(
            n,
            return_array=return_array,
            window_size=window_size,
            **kwargs,
        )


class MLXSelectedKVCache:
    """mlx-lm cache wrapper combining selected memory with local sequential K/V.

    ``offset`` includes ``position_base`` for RoPE while causal mask construction
    uses only the local cache offset.  Every query may see all selected memory;
    local prompt/decode keys remain causal.
    """

    def __init__(self, local_cache: object, memory: MLXNativeLayerKV, position_base: int):
        if position_base < 0:
            raise ValueError("MLX native query position base cannot be negative.")
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
        raise RuntimeError("A PRA selected-memory cache cannot replace its immutable state.")

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


class MLXSegmentedSelectedKVCache(MLXSelectedKVCache):
    """Selected-memory cache that keeps memory and local K/V physically separate.

    Qwen3's ordinary MLX-LM attention asks a cache for one K/V pair.  The
    segmented attention patch instead calls :meth:`update_and_fetch_segments`
    and combines the two attention numerators and denominators without first
    allocating ``concat(memory, local)``.
    """

    def update_and_fetch(self, keys, values):
        raise RuntimeError(
            "Segmented PRA cache requires an installed segmented attention patch."
        )

    def update_and_fetch_segments(self, keys, values):
        """Return immutable selected K/V and updated local K/V as two segments."""

        local_keys, local_values = self.local_cache.update_and_fetch(keys, values)
        return self.memory.keys, self.memory.values, local_keys, local_values


class MLXDisjointSelectedKVCache:
    """Request cache consuming several immutable history K/V segments.

    Unlike :class:`MLXSelectedKVCache`, this wrapper never concatenates the
    selected history intervals. The Qwen3 attention patch combines every
    segment and the local causal cache under one exact softmax normalization.
    """

    def __init__(
        self,
        local_cache: object,
        memory: MLXDisjointLayerKV,
        position_base: int,
        fused_disjoint_attention: bool = True,
    ) -> None:
        if position_base < 0:
            raise ValueError("MLX native query position base cannot be negative.")
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)
        self.fused_disjoint_attention = bool(fused_disjoint_attention)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def local_offset(self) -> int:
        return int(self.local_cache.offset)

    @property
    def memory_tokens(self) -> int:
        return self.memory.tokens

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        selected = tuple(
            value
            for segment in self.memory.segments
            for value in (segment.keys, segment.values)
        )
        return (*selected, *local)

    @state.setter
    def state(self, value) -> None:
        raise RuntimeError("A PRA disjoint selected-memory cache is immutable.")

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
        raise RuntimeError(
            "Disjoint PRA cache requires an installed multi-segment attention patch."
        )

    def update_and_fetch_segments(self, keys, values):
        local_keys, local_values = self.local_cache.update_and_fetch(keys, values)
        return (
            tuple(segment.keys for segment in self.memory.segments),
            tuple(segment.values for segment in self.memory.segments),
            local_keys,
            local_values,
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


def segmented_selected_attention(
    queries: object,
    memory_keys: object,
    memory_values: object,
    local_keys: object,
    local_values: object,
    *,
    scale: float,
    mask: object | None = None,
) -> object:
    """Attend over selected and local K/V with one exact normalization.

    Inputs use ``[batch, heads, tokens, head_dim]`` after any GQA/MQA head
    expansion required by the host model.  The function avoids materializing
    ``concat(memory_kv, local_kv)``: it combines segment maxima, denominators,
    and value numerators instead.  Decode-time callers must ensure that the
    supplied local segment is already causally valid for the query.
    """

    memory_mask = None
    local_mask = None
    if mask is not None:
        memory_tokens = int(memory_keys.shape[2])
        local_tokens = int(local_keys.shape[2])
        if int(mask.shape[-1]) != memory_tokens + local_tokens:
            raise ValueError("Segmented attention mask does not match K/V width.")
        # Split dynamic masks before entering a compiled graph. MLX cannot
        # infer output shapes for token-dependent Slice primitives in a
        # shapeless graph, while each already-split mask remains shapeless.
        memory_mask = mask[..., :memory_tokens]
        local_mask = mask[..., memory_tokens:]
    return _segmented_selected_attention_impl(
        queries,
        memory_keys,
        memory_values,
        local_keys,
        local_values,
        scale=scale,
        memory_mask=memory_mask,
        local_mask=local_mask,
    )


def disjoint_segmented_selected_attention(
    queries: object,
    memory_keys: Sequence[object],
    memory_values: Sequence[object],
    local_keys: object,
    local_values: object,
    *,
    scale: float,
    mask: object | None = None,
    source_keys: object | None = None,
    source_values: object | None = None,
    source_intervals: Sequence[tuple[int, int]] = (),
) -> object:
    """Attend to disjoint resident K/V views without packing selected history.

    This is the N-memory-segment analogue of
    :func:`segmented_selected_attention`. It performs one mathematical softmax
    by combining per-segment maxima, denominators, and numerators. The
    implementation deliberately remains eager: runtime qualification must
    measure view materialization and kernel-launch overhead before a fused
    Metal implementation can be claimed.
    """

    import mlx.core as mx

    key_segments = tuple(memory_keys)
    value_segments = tuple(memory_values)
    if not key_segments or len(key_segments) != len(value_segments):
        raise ValueError("Disjoint attention requires matching non-empty K/V segments.")
    kv_heads = int(key_segments[0].shape[1])
    query_heads = int(queries.shape[1])
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by K/V head count.")
    for keys, values in zip(key_segments, value_segments):
        if int(keys.shape[1]) != kv_heads or int(values.shape[1]) != kv_heads:
            raise ValueError("All disjoint K/V segments must use the same head count.")
        if int(keys.shape[2]) != int(values.shape[2]):
            raise ValueError("Each disjoint key/value segment must have equal width.")
    if int(local_keys.shape[1]) != kv_heads or int(local_values.shape[1]) != kv_heads:
        raise ValueError("Selected and local K/V must use the same head count.")

    if (
        platform.system() == "Darwin"
        and hasattr(mx.fast, "metal_kernel")
        and source_keys is not None
        and source_values is not None
        and source_intervals
    ):
        return _metal_disjoint_selected_attention(
            queries,
            source_keys,
            source_values,
            source_intervals,
            local_keys,
            local_values,
            scale=scale,
            mask=mask,
        )

    groups = query_heads // kv_heads
    grouped_queries = mx.unflatten(queries, 1, (kv_heads, groups))
    scaled_queries = grouped_queries * scale

    def scores(keys):
        expanded = mx.expand_dims(keys, axis=2)
        return scaled_queries @ mx.swapaxes(expanded, -1, -2)

    score_segments = [scores(keys).astype(mx.float32) for keys in key_segments]
    score_segments.append(scores(local_keys).astype(mx.float32))

    if mask is not None:
        widths = [int(keys.shape[2]) for keys in key_segments]
        widths.append(int(local_keys.shape[2]))
        if int(mask.shape[-1]) != sum(widths):
            raise ValueError("Disjoint attention mask does not match total K/V width.")
        start = 0
        masked_scores = []
        for values, width in zip(score_segments, widths):
            segment_mask = mask[..., start : start + width]
            start += width
            for _ in range(values.ndim - segment_mask.ndim):
                segment_mask = mx.expand_dims(segment_mask, axis=0)
            if segment_mask.dtype == mx.bool_:
                values = mx.where(segment_mask, values, mx.array(-1e9, values.dtype))
            else:
                values = values + segment_mask
            masked_scores.append(values)
        score_segments = masked_scores

    # Scores are small compared with K/V. Concatenating only the score rows
    # lets MLX apply one native softmax reduction across the complete selected
    # axis without ever packing resident keys or values. This also avoids the
    # segment-count-dependent denominator rounding of independent exp/sum
    # reductions.
    normalized = mx.softmax(mx.concatenate(score_segments, axis=-1), axis=-1)

    numerator = None
    all_values = (*value_segments, local_values)
    cursor = 0
    for score, values in zip(score_segments, all_values):
        # Normalize globally before narrowing weights to the model dtype.  This
        # avoids casting the resident value segment to fp32 (which materialized
        # a selected-K/V-sized temporary) while following the native attention
        # contract: high-precision softmax, model-dtype value product, and a
        # small fp32 output accumulation across intervals.
        width = int(score.shape[-1])
        weights = normalized[..., cursor : cursor + width].astype(values.dtype)
        cursor += width
        expanded_values = mx.expand_dims(values, axis=2)
        partial_numerator = (weights @ expanded_values).astype(mx.float32)
        numerator = (
            partial_numerator if numerator is None else numerator + partial_numerator
        )
    return mx.flatten(numerator, 1, 2).astype(queries.dtype)


_DISJOINT_METAL_KERNELS: dict[float, object] = {}


def _metal_disjoint_selected_attention(
    queries: object,
    source_keys: object,
    source_values: object,
    source_intervals: Sequence[tuple[int, int]],
    local_keys: object,
    local_values: object,
    *,
    scale: float,
    mask: object | None,
) -> object:
    """Fused Metal attention over interval-addressed source K/V and local K/V."""

    import mlx.core as mx

    if int(queries.shape[0]) != 1:
        raise ValueError("Metal disjoint attention currently supports batch size one.")
    head_dim = int(queries.shape[-1])
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("Metal disjoint attention supports head dimensions 1..256.")
    query_heads = int(queries.shape[1])
    query_tokens = int(queries.shape[2])
    kv_heads = int(source_keys.shape[1])
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by K/V head count.")
    arrays = (source_keys, source_values, local_keys, local_values)
    if any(array.dtype != queries.dtype for array in arrays):
        raise ValueError("Metal disjoint attention requires one model-native dtype.")
    intervals = tuple((int(start), int(end)) for start, end in source_intervals)
    if not intervals or any(start < 0 or end <= start for start, end in intervals):
        raise ValueError("Metal disjoint attention requires non-empty ordered intervals.")
    selected_tokens = sum(end - start for start, end in intervals)
    total_tokens = selected_tokens + int(local_keys.shape[2])
    if mask is None:
        mask = mx.ones((query_tokens, total_tokens), dtype=mx.bool_)
    if mask.ndim != 2 or mask.dtype != mx.bool_ or int(mask.shape[-1]) != total_tokens:
        raise ValueError("Metal disjoint attention requires a matching rank-two bool mask.")
    interval_array = mx.array(intervals, dtype=mx.uint32)

    kernel_key = float(scale)
    kernel = _DISJOINT_METAL_KERNELS.get(kernel_key)
    if kernel is None:
        source = f"""
    uint d = thread_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x;
    uint query_tokens = q_shape[2];
    uint qhead = row / query_tokens;
    uint qi = row - qhead * query_tokens;
    uint head_dim = q_shape[3];
    uint groups = q_shape[1] / source_k_shape[1];
    uint kvhead = qhead / groups;
    uint simdgroups = (head_dim + 31) / 32;
    threadgroup float partials[8];
    threadgroup float shared_score;
    float running_max = -INFINITY;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    uint compact_k = 0;
    float scale_value = {float(scale):.17g}f;

    for (uint interval = 0; interval < intervals_shape[0]; ++interval) {{
        uint begin = intervals[interval * 2];
        uint end = intervals[interval * 2 + 1];
        for (uint kt = begin; kt < end; ++kt, ++compact_k) {{
            bool visible = mask[qi * mask_strides[0] + compact_k * mask_strides[1]];
            if (!visible) {{ continue; }}
            float product = 0.0f;
            if (d < head_dim) {{
                uint q_index = qhead * q_strides[1] + qi * q_strides[2]
                    + d * q_strides[3];
                uint k_index = kvhead * source_k_strides[1]
                    + kt * source_k_strides[2] + d * source_k_strides[3];
                product = float(q[q_index]) * float(source_k[k_index]);
            }}
            float lane_sum = simd_sum(product);
            if (thread_index_in_simdgroup == 0) {{
                partials[simdgroup_index_in_threadgroup] = lane_sum;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (d == 0) {{
                float dot = 0.0f;
                for (uint sg = 0; sg < simdgroups; ++sg) {{ dot += partials[sg]; }}
                shared_score = dot * scale_value;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float score = shared_score;
            float next_max = metal::max(running_max, score);
            float prior_scale = isinf(running_max)
                ? 0.0f : metal::precise::exp(running_max - next_max);
            float weight = metal::precise::exp(score - next_max);
            if (d < head_dim) {{
                uint v_index = kvhead * source_v_strides[1]
                    + kt * source_v_strides[2] + d * source_v_strides[3];
                accumulator = accumulator * prior_scale
                    + weight * float(source_v[v_index]);
            }}
            denominator = denominator * prior_scale + weight;
            running_max = next_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}

    for (uint kt = 0; kt < local_k_shape[2]; ++kt, ++compact_k) {{
        bool visible = mask[qi * mask_strides[0] + compact_k * mask_strides[1]];
        if (!visible) {{ continue; }}
        float product = 0.0f;
        if (d < head_dim) {{
            uint q_index = qhead * q_strides[1] + qi * q_strides[2]
                + d * q_strides[3];
            uint k_index = kvhead * local_k_strides[1]
                + kt * local_k_strides[2] + d * local_k_strides[3];
            product = float(q[q_index]) * float(local_k[k_index]);
        }}
        float lane_sum = simd_sum(product);
        if (thread_index_in_simdgroup == 0) {{
            partials[simdgroup_index_in_threadgroup] = lane_sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d == 0) {{
            float dot = 0.0f;
            for (uint sg = 0; sg < simdgroups; ++sg) {{ dot += partials[sg]; }}
            shared_score = dot * scale_value;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = shared_score;
        float next_max = metal::max(running_max, score);
        float prior_scale = isinf(running_max)
            ? 0.0f : metal::precise::exp(running_max - next_max);
        float weight = metal::precise::exp(score - next_max);
        if (d < head_dim) {{
            uint v_index = kvhead * local_v_strides[1]
                + kt * local_v_strides[2] + d * local_v_strides[3];
            accumulator = accumulator * prior_scale + weight * float(local_v[v_index]);
        }}
        denominator = denominator * prior_scale + weight;
        running_max = next_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (d < head_dim) {{
        uint out_index = (qhead * query_tokens + qi) * head_dim + d;
        out[out_index] = accumulator / denominator;
    }}
"""
        kernel = mx.fast.metal_kernel(
            name="pra_interval_addressed_attention",
            input_names=(
                "q", "source_k", "source_v", "local_k", "local_v",
                "intervals", "mask",
            ),
            output_names=("out",),
            source=source,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
        _DISJOINT_METAL_KERNELS[kernel_key] = kernel

    threadgroup_width = 32 * ((head_dim + 31) // 32)
    return kernel(
        inputs=(
            queries, source_keys, source_values, local_keys, local_values,
            interval_array, mask,
        ),
        grid=(query_heads * query_tokens * threadgroup_width, 1, 1),
        threadgroup=(threadgroup_width, 1, 1),
        output_shapes=(queries.shape,),
        output_dtypes=(mx.float32,),
    )[0].astype(queries.dtype)


def _segmented_selected_attention_impl(
    queries: object,
    memory_keys: object,
    memory_values: object,
    local_keys: object,
    local_values: object,
    *,
    scale: float,
    memory_mask: object | None,
    local_mask: object | None,
) -> object:
    """Compute segmented attention from masks already split by segment."""

    import mlx.core as mx

    query_heads = int(queries.shape[1])
    kv_heads = int(memory_keys.shape[1])
    if int(local_keys.shape[1]) != kv_heads:
        raise ValueError("Selected and local K/V must use the same head count.")
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by K/V head count.")
    groups = query_heads // kv_heads
    # Preserve dynamic prefill/decode widths inside a shapeless compiled graph.
    # ``unflatten`` and ``flatten`` infer untouched dimensions from the input;
    # constructing Python reshape tuples would freeze the first query length.
    grouped_queries = mx.unflatten(queries, 1, (kv_heads, groups))
    scaled_queries = grouped_queries * scale

    def scores(keys):
        expanded = mx.expand_dims(keys, axis=2)
        return scaled_queries @ mx.swapaxes(expanded, -1, -2)

    # Accumulate the split normalization in fp32. MLX's fused SDPA also uses
    # higher-precision reduction internally; retaining fp16 exponentials here
    # creates a small per-layer error that compounds through deep decoders.
    memory_scores = scores(memory_keys).astype(mx.float32)
    local_scores = scores(local_keys).astype(mx.float32)
    if memory_mask is not None:
        if local_mask is None:
            raise ValueError("Selected and local masks must be supplied together.")
        def apply_mask(values, segment_mask):
            # Add broadcast axes by rank, not by a Python shape tuple. The
            # latter freezes the first decode width inside a shapeless graph.
            for _ in range(values.ndim - segment_mask.ndim):
                segment_mask = mx.expand_dims(segment_mask, axis=0)
            if segment_mask.dtype == mx.bool_:
                return mx.where(segment_mask, values, mx.array(-1e9, values.dtype))
            return values + segment_mask

        memory_scores = apply_mask(memory_scores, memory_mask)
        local_scores = apply_mask(local_scores, local_mask)
    maximum = mx.maximum(
        mx.max(memory_scores, axis=-1, keepdims=True),
        mx.max(local_scores, axis=-1, keepdims=True),
    )
    memory_exp = mx.exp(memory_scores - maximum)
    local_exp = mx.exp(local_scores - maximum)
    denominator = mx.sum(memory_exp, axis=-1, keepdims=True) + mx.sum(
        local_exp, axis=-1, keepdims=True
    )
    memory_weights = (memory_exp / denominator).astype(memory_values.dtype)
    local_weights = (local_exp / denominator).astype(local_values.dtype)
    expanded_memory_values = mx.expand_dims(memory_values, axis=2)
    expanded_local_values = mx.expand_dims(local_values, axis=2)
    numerator = (
        (memory_weights @ expanded_memory_values).astype(mx.float32)
        + (local_weights @ expanded_local_values).astype(mx.float32)
    )
    return mx.flatten(numerator, 1, 2).astype(queries.dtype)


_COMPILED_SEGMENTED_ATTENTION: dict[bool, object] = {}


def _segmented_selected_attention_unmasked(
    queries, memory_keys, memory_values, local_keys, local_values, *, scale
):
    return _segmented_selected_attention_impl(
        queries,
        memory_keys,
        memory_values,
        local_keys,
        local_values,
        scale=scale,
        memory_mask=None,
        local_mask=None,
    )


def _segmented_selected_attention_masked(
    queries,
    memory_keys,
    memory_values,
    local_keys,
    local_values,
    memory_mask,
    local_mask,
    *,
    scale,
):
    return _segmented_selected_attention_impl(
        queries,
        memory_keys,
        memory_values,
        local_keys,
        local_values,
        scale=scale,
        memory_mask=memory_mask,
        local_mask=local_mask,
    )


def compiled_segmented_selected_attention(
    queries: object,
    memory_keys: object,
    memory_values: object,
    local_keys: object,
    local_values: object,
    *,
    scale: float,
    mask: object | None = None,
) -> object:
    """Run segmented attention through MLX's shapeless graph compiler.

    Separate masked and unmasked graphs avoid retracing when decode switches
    between the microbenchmark and model paths. ``shapeless=True`` permits the
    local-cache token dimension to grow during autoregressive decoding.
    """

    import mlx.core as mx

    masked = mask is not None
    compiled = _COMPILED_SEGMENTED_ATTENTION.get(masked)
    if compiled is None:
        target = (
            _segmented_selected_attention_masked
            if masked
            else _segmented_selected_attention_unmasked
        )
        compiled = mx.compile(target, shapeless=True)
        _COMPILED_SEGMENTED_ATTENTION[masked] = compiled
    if mask is None:
        return compiled(
            queries,
            memory_keys,
            memory_values,
            local_keys,
            local_values,
            scale=scale,
        )
    memory_tokens = int(memory_keys.shape[2])
    local_tokens = int(local_keys.shape[2])
    if int(mask.shape[-1]) != memory_tokens + local_tokens:
        raise ValueError("Segmented attention mask does not match K/V width.")
    return compiled(
        queries,
        memory_keys,
        memory_values,
        local_keys,
        local_values,
        mask[..., :memory_tokens],
        mask[..., memory_tokens:],
        scale=scale,
    )


def encode_native_memory(model: object, token_ids: Sequence[int]) -> MLXNativeMemory:
    """Encode one complete resource and retain post-RoPE K/V for every layer."""

    if not token_ids:
        raise ValueError("Cannot encode an empty PRA resource as native K/V.")
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model)
    model(mx.array(token_ids, dtype=mx.int32)[None], cache=caches)
    states = [cache.state for cache in caches]
    mx.eval(states)
    layers = []
    for state in states:
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError("The MLX model exposed a non-attention cache layer.")
        # Cache states may be views into reusable model-runner buffers. Force
        # an owned result so encoding the next resource cannot mutate a HOT
        # logical memory that the lifecycle manager already retained.
        zero_k = mx.array(0, dtype=state[0].dtype)
        zero_v = mx.array(0, dtype=state[1].dtype)
        layers.append(
            MLXNativeLayerKV(state[0] + zero_k, state[1] + zero_v)
        )
    result = MLXNativeMemory(tuple(layers), source_tokens=len(token_ids))
    mx.eval(
        *(array for layer in result.layers for array in (layer.keys, layer.values))
    )
    return result


def capture_live_native_memory(
    caches: Sequence[object], plan: LiveKVSelectionPlan
) -> MLXResidentKVSelection:
    """Expose selected cells from an already evaluated mlx-lm prompt cache.

    A contiguous selection stays a slice view over the live cache.  The
    portable MLX attention protocol currently packs disjoint intervals with
    ``concatenate``; this copies K/V but still evaluates zero selected text.
    Segmented zero-copy consumption remains a separate materialization policy.
    """

    import mlx.core as mx

    layers = []
    packed = len(plan.intervals) > 1
    for cache in caches:
        state = cache.state
        if not isinstance(state, tuple) or len(state) < 2:
            raise TypeError("Live MLX K/V selection requires attention cache layers.")
        keys, values = state[:2]
        if int(keys.shape[2]) < plan.source_tokens:
            raise ValueError("MLX source cache is shorter than the selection plan.")
        key_parts = tuple(keys[:, :, row.start : row.end, :] for row in plan.intervals)
        value_parts = tuple(values[:, :, row.start : row.end, :] for row in plan.intervals)
        if not key_parts:
            chosen_keys = keys[:, :, :0, :]
            chosen_values = values[:, :, :0, :]
        elif len(key_parts) == 1:
            chosen_keys = key_parts[0]
            chosen_values = value_parts[0]
        else:
            chosen_keys = mx.concatenate(key_parts, axis=2)
            chosen_values = mx.concatenate(value_parts, axis=2)
        layers.append(MLXNativeLayerKV(chosen_keys, chosen_values))
    memory = MLXNativeMemory(tuple(layers), plan.selected_tokens)
    mx.eval(*(array for layer in memory.layers for array in (layer.keys, layer.values)))
    return MLXResidentKVSelection(memory, plan, packed)


def select_live_native_memory(
    source: MLXNativeMemory, plan: LiveKVSelectionPlan
) -> MLXResidentKVSelection:
    """Select request K/V from one canonical, already evaluated MLX memory.

    The canonical source remains independently owned by the session runtime.
    A contiguous interval is exposed as an MLX slice; disjoint intervals are
    packed with ``concatenate`` because the portable mlx-lm cache protocol
    accepts one dense K/V array per layer.  In both cases no selected token is
    sent through the model again, and the query position remains the source's
    original logical extent through :class:`LiveKVSelectionPlan`.
    """

    if source.source_tokens != plan.source_tokens:
        raise ValueError(
            "MLX canonical source length does not match the selection plan."
        )
    if (
        len(plan.intervals) == 1
        and plan.intervals[0].start == 0
        and plan.intervals[0].end == plan.source_tokens
    ):
        # The 100% semantic control borrows the canonical object itself. This
        # is stronger than relying on an MLX full-range slice being a view.
        return MLXResidentKVSelection(source, plan, False)
    import mlx.core as mx

    layers = []
    packed = len(plan.intervals) > 1
    for layer in source.layers:
        keys, values = layer.keys, layer.values
        if int(keys.shape[2]) < plan.source_tokens:
            raise ValueError("MLX canonical source K/V is shorter than its manifest.")
        key_parts = tuple(keys[:, :, row.start : row.end, :] for row in plan.intervals)
        value_parts = tuple(values[:, :, row.start : row.end, :] for row in plan.intervals)
        if not key_parts:
            chosen_keys = keys[:, :, :0, :]
            chosen_values = values[:, :, :0, :]
        elif len(key_parts) == 1:
            chosen_keys = key_parts[0]
            chosen_values = value_parts[0]
        else:
            chosen_keys = mx.concatenate(key_parts, axis=2)
            chosen_values = mx.concatenate(value_parts, axis=2)
        layers.append(MLXNativeLayerKV(chosen_keys, chosen_values))
    selected = MLXNativeMemory(tuple(layers), plan.selected_tokens)
    mx.eval(*(array for layer in selected.layers for array in (layer.keys, layer.values)))
    return MLXResidentKVSelection(selected, plan, packed)


def select_live_native_memory_disjoint(
    source: MLXNativeMemory, plan: LiveKVSelectionPlan
) -> MLXResidentKVSelection:
    """Retain disjoint live-history K/V as independent source-array views.

    PRA performs no interval concatenation in this path. It is suitable only
    for a consumer such as :class:`MLXDisjointSelectedKVCache` that can combine
    several memory segments under one attention normalization.
    """

    if source.source_tokens != plan.source_tokens:
        raise ValueError(
            "MLX canonical source length does not match the selection plan."
        )
    if not plan.intervals:
        raise ValueError("Disjoint MLX selection requires at least one interval.")

    layers = []
    for layer in source.layers:
        if int(layer.keys.shape[2]) < plan.source_tokens:
            raise ValueError("MLX canonical source K/V is shorter than its manifest.")
        layers.append(
            MLXDisjointLayerKV(
                tuple(
                    MLXNativeLayerKV(
                        layer.keys[:, :, interval.start : interval.end, :],
                        layer.values[:, :, interval.start : interval.end, :],
                    )
                    for interval in plan.intervals
                ),
                source_keys=layer.keys,
                source_values=layer.values,
                intervals=tuple(
                    (interval.start, interval.end) for interval in plan.intervals
                ),
            )
        )
    selected = MLXDisjointNativeMemory(tuple(layers), plan.source_tokens)
    # PRA performs no explicit interval-pack copy, but MLX slice storage and
    # transient attention allocation remain runtime-measured properties.
    return MLXResidentKVSelection(selected, plan, None)


def combine_native_memories(memories: Sequence[MLXNativeMemory]) -> MLXNativeMemory:
    """Concatenate immutable resource K/V without changing source-local positions."""

    if not memories:
        raise ValueError("At least one native PRA memory is required.")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("PRA memories have incompatible layer counts.")
    if len(memories) == 1:
        return memories[0]
    import mlx.core as mx

    layers = []
    for layer_idx in range(len(memories[0].layers)):
        layers.append(
            MLXNativeLayerKV(
                mx.concatenate(
                    tuple(memory.layers[layer_idx].keys for memory in memories), axis=2
                ),
                mx.concatenate(
                    tuple(memory.layers[layer_idx].values for memory in memories), axis=2
                ),
            )
        )
    return MLXNativeMemory(
        tuple(layers), source_tokens=max(memory.source_tokens for memory in memories)
    )


def serialize_native_memory(
    memory: MLXNativeMemory,
    *,
    quantization: str = "none",
    quantized_layers: Iterable[int] | None = None,
    quantize_keys: bool = True,
    quantize_values: bool = True,
) -> bytes:
    """Serialize native K/V with independently selectable int8 arrays.

    The default remains all-layer K/V quantization.  Calibration experiments
    can retain exact keys, values, or layer bands to localize quality loss.
    """

    import numpy as np

    if quantization not in {"none", "int8"}:
        raise ValueError("MLX native serialization supports none or int8 quantization.")
    layer_filter = (
        None if quantized_layers is None else frozenset(map(int, quantized_layers))
    )
    arrays: dict[str, object] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for index, layer in enumerate(memory.layers):
        for suffix, value in (("k", layer.keys), ("v", layer.values)):
            name = f"layer_{index:04d}_{suffix}"
            logical_dtype = str(value.dtype)
            try:
                host = np.asarray(value).copy()
            except (TypeError, ValueError, RuntimeError):
                import mlx.core as mx

                # MLX bfloat16 does not expose a NumPy-compatible PEP 3118
                # buffer. Float32 is an exact value-preserving carrier for
                # bfloat16 and is cast back to the logical dtype on restore.
                host = np.asarray(value.astype(mx.float32)).copy()
            component_enabled = quantize_keys if suffix == "k" else quantize_values
            selected_for_quantization = (
                quantization == "int8"
                and component_enabled
                and (layer_filter is None or index in layer_filter)
            )
            if selected_for_quantization:
                floating = host.astype(np.float32)
                maximum = float(np.max(np.abs(floating))) if floating.size else 0.0
                scale = maximum / 127.0 if maximum else 1.0
                arrays[name] = np.rint(floating / scale).clip(-127, 127).astype(np.int8)
                descriptors[name] = {
                    "logical_dtype": logical_dtype,
                    "quantization": "int8",
                    "scale": scale,
                }
            else:
                arrays[name] = host
                descriptors[name] = {
                    "logical_dtype": logical_dtype,
                    "quantization": "none",
                }
    metadata = json.dumps(
        {
            "schema": "pra-native-memory-v2",
            "source_tokens": memory.source_tokens,
            "layer_count": len(memory.layers),
            "quantization": quantization,
            "quantized_layers": (
                "all" if layer_filter is None else sorted(layer_filter)
            ),
            "quantize_keys": bool(quantize_keys),
            "quantize_values": bool(quantize_values),
            "arrays": descriptors,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    arrays["__pra_metadata__"] = np.frombuffer(metadata, dtype=np.uint8)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def deserialize_native_memory(payload: bytes) -> MLXNativeMemory:
    """Restore a serialized memory to MLX arrays when MLX is available."""

    import numpy as np

    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["__pra_metadata__"]).decode("utf-8"))
        if metadata.get("schema") != "pra-native-memory-v2":
            raise ValueError("Unsupported PRA native-memory payload schema.")
        restored: dict[str, object] = {}
        for name, descriptor in metadata["arrays"].items():
            array = archive[name].copy()
            if descriptor.get("quantization") == "int8":
                array = array.astype(np.float32) * float(descriptor["scale"])
            restored[name] = array
    try:
        import mlx.core as mx
    except ImportError:
        convert = lambda value, _dtype: value
    else:
        def convert(value, logical_dtype):
            array = mx.array(value)
            if "bfloat16" in logical_dtype:
                return array.astype(mx.bfloat16)
            dtype = getattr(mx, logical_dtype, None)
            return array if dtype is None else array.astype(dtype)

    layers = tuple(
        MLXNativeLayerKV(
            convert(
                restored[f"layer_{index:04d}_k"],
                metadata["arrays"][f"layer_{index:04d}_k"]["logical_dtype"],
            ),
            convert(
                restored[f"layer_{index:04d}_v"],
                metadata["arrays"][f"layer_{index:04d}_v"]["logical_dtype"],
            ),
        )
        for index in range(int(metadata["layer_count"]))
    )
    memory = MLXNativeMemory(layers, int(metadata["source_tokens"]))
    if "mx" in locals():
        mx.eval(*(value for layer in layers for value in (layer.keys, layer.values)))
    return memory


class MLXNativeColdCodec:
    """Transform lossless native-memory blobs to and from quantized COLD blobs."""

    def __init__(
        self,
        *,
        quantized_layers: Iterable[int] | None = None,
        quantize_keys: bool = True,
        quantize_values: bool = True,
    ) -> None:
        self.quantized_layers = (
            None
            if quantized_layers is None
            else tuple(sorted(set(map(int, quantized_layers))))
        )
        self.quantize_keys = bool(quantize_keys)
        self.quantize_values = bool(quantize_values)

    def encode(
        self, payload: bytes, quantization: str
    ) -> tuple[bytes, Mapping[str, object]]:
        memory = deserialize_native_memory(payload)
        encoded = serialize_native_memory(
            memory,
            quantization=quantization,
            quantized_layers=self.quantized_layers,
            quantize_keys=self.quantize_keys,
            quantize_values=self.quantize_values,
        )
        return encoded, {
            "schema": "pra-mlx-cold-codec-v1",
            "quantization": quantization,
            "quantized_layers": (
                "all" if self.quantized_layers is None else list(self.quantized_layers)
            ),
            "quantize_keys": self.quantize_keys,
            "quantize_values": self.quantize_values,
            "lossless_bytes": len(payload),
            "encoded_bytes": len(encoded),
        }

    def decode(self, payload: bytes, metadata: Mapping[str, object]) -> bytes:
        if metadata.get("schema") != "pra-mlx-cold-codec-v1":
            raise ValueError("Unsupported PRA MLX COLD codec metadata.")
        memory = deserialize_native_memory(payload)
        return serialize_native_memory(memory, quantization="none")


def make_native_prompt_cache(
    model: object,
    memory: MLXNativeMemory | MLXDisjointNativeMemory,
    *,
    max_kv_size: int | None = None,
    selected_layers: Iterable[int] | None = None,
    segmented: bool = False,
    fused_disjoint_attention: bool = True,
    query_position_base: int | None = None,
):
    """Create request-local sequential caches backed by immutable selected K/V.

    ``query_position_base`` normally equals the encoded resource length.  The
    explicit override exists for positional-policy controls and adapters that
    have already resolved a different logical source frame.  Production
    callers should leave it unset unless stored post-RoPE K/V and the query use
    the same coordinate policy.
    """

    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model, max_kv_size=max_kv_size)
    if len(local) != len(memory.layers):
        raise ValueError("MLX native memory does not match the model layer count.")
    selected = (
        set(range(len(memory.layers)))
        if selected_layers is None
        else set(map(int, selected_layers))
    )
    invalid = selected - set(range(len(memory.layers)))
    if invalid:
        raise ValueError(f"MLX consumer layers are out of range: {sorted(invalid)}")
    if isinstance(memory, MLXDisjointNativeMemory):
        if not segmented:
            raise ValueError("Disjoint MLX memory requires segmented=True.")
        selected_cache_type = MLXDisjointSelectedKVCache
    else:
        selected_cache_type = (
            MLXSegmentedSelectedKVCache if segmented else MLXSelectedKVCache
        )
    position_base = resolve_query_position_base(
        memory.source_tokens, query_position_base
    )
    result = []
    for index, (cache, layer) in enumerate(zip(local, memory.layers)):
        if index not in selected:
            result.append(MLXPositionedKVCache(cache, position_base))
        elif selected_cache_type is MLXDisjointSelectedKVCache:
            result.append(
                selected_cache_type(
                    cache,
                    layer,
                    position_base,
                    fused_disjoint_attention=fused_disjoint_attention,
                )
            )
        else:
            result.append(selected_cache_type(cache, layer, position_base))
    return result


def resolve_query_position_base(
    source_tokens: int, query_position_base: int | None
) -> int:
    """Resolve and validate the RoPE offset used by a native-memory query."""

    value = int(source_tokens if query_position_base is None else query_position_base)
    if value < 0:
        raise ValueError("MLX native query position base cannot be negative.")
    return value


def save_native_memory(
    path: str | Path,
    memory: MLXNativeMemory,
    fingerprint: MLXNativeFingerprint,
) -> tuple[Path, Path]:
    """Persist immutable selected K/V and a strict compatibility manifest."""

    import mlx.core as mx

    target = Path(path)
    arrays_path = target.with_suffix(".npz")
    manifest_path = target.with_suffix(".json")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, layer in enumerate(memory.layers):
        arrays[f"layer_{index:04d}_k"] = layer.keys
        arrays[f"layer_{index:04d}_v"] = layer.values
    mx.savez(str(arrays_path), **arrays)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_tokens": memory.source_tokens,
                "layer_count": len(memory.layers),
                "fingerprint": fingerprint.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return arrays_path, manifest_path


def load_native_memory(
    path: str | Path,
    *,
    expected_fingerprint: MLXNativeFingerprint,
) -> MLXNativeMemory:
    """Load persisted K/V only when every compatibility field agrees."""

    import mlx.core as mx

    target = Path(path)
    arrays_path = target.with_suffix(".npz")
    manifest = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != expected_fingerprint.to_dict():
        raise ValueError("Persisted MLX PRA cache fingerprint does not match runtime.")
    arrays = mx.load(str(arrays_path))
    layers = tuple(
        MLXNativeLayerKV(
            arrays[f"layer_{index:04d}_k"],
            arrays[f"layer_{index:04d}_v"],
        )
        for index in range(int(manifest["layer_count"]))
    )
    memory = MLXNativeMemory(layers, source_tokens=int(manifest["source_tokens"]))
    return memory


class MLXInProcessNativeExecutor:
    """Capability-honest E2 executor using an in-process mlx-lm model.

    Resources are immutable and shareable only when their complete logical key
    agrees.  Selection is supplied by ``pra_policy.selected_resource_ids``;
    absent an explicit list, resources are consumed in request order up to the
    typed budget.
    """

    integration_level = "E2"

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        model_id: str,
        model_revision: str,
        block_store: LogicalPRABlockStore,
        max_resident_bytes: int = 512 * 1024 * 1024,
        eviction_policy: PRAEvictionPolicy | str = PRAEvictionPolicy.LRU,
        storage_policy: PRAStoragePolicy | None = None,
        storage_manager: PRAStorageManager | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self.block_store = block_store
        self.residency = EnginePRAResidencyManager(
            block_store,
            max_resident_bytes=max_resident_bytes,
            policy=eviction_policy,
        )
        self.storage = storage_manager or PRAStorageManager(
            storage_policy or PRAStoragePolicy.named("balanced"),
            hot=ResidencyManagerHotBridge(self.residency, deserialize_native_memory),
            cold_codec=MLXNativeColdCodec(),
        )
        self.storage.start_maintenance()
        self._session_keys: dict[str, set[str]] = {}
        self.isolation = EnginePRAIsolationGuard()
        # MLX-LM model objects are not reentrant. The HTTP gateway accepts
        # concurrent arrivals, but source encoding and decode must enter one
        # engine-owned queue or requests can observe another request's state.
        self._model_runner_lock = threading.RLock()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        model_revision: str,
        block_store: LogicalPRABlockStore,
        **kwargs: object,
    ) -> "MLXInProcessNativeExecutor":
        from mlx_lm import load

        model, tokenizer = load(model_id, revision=model_revision)
        return cls(
            model,
            tokenizer,
            model_id=model_id,
            model_revision=model_revision,
            block_store=block_store,
            **kwargs,
        )

    def _selected_resources(self, request: PRAWireRequest) -> tuple[PRAWireResource, ...]:
        selected = tuple(
            map(str, request.pra_policy.get("selected_resource_ids", ()))
        )
        resources = {resource.resource_id: resource for resource in request.resources}
        if selected:
            missing = [resource_id for resource_id in selected if resource_id not in resources]
            if missing:
                raise KeyError(f"Selected PRA resources are absent from the request: {missing}")
            return tuple(resources[resource_id] for resource_id in selected)
        return request.resources[: request.budget.max_resources]

    def _resource_tokens(self, resource: PRAWireResource) -> tuple[int, ...]:
        if resource.text is None:
            raise ValueError(f"Resource {resource.resource_id!r} has no encodable text.")
        values = self.tokenizer.encode(resource.text, add_special_tokens=False)
        return tuple(map(int, values[:]))

    def _register(
        self, request: PRAWireRequest, resource: PRAWireResource, token_count: int
    ) -> str:
        version = str(
            resource.metadata.get("version")
            or hashlib.sha256((resource.text or "").encode("utf-8")).hexdigest()
        )
        shareable = bool(resource.metadata.get("shareable", False))
        identity = LogicalPRABlockId(
            tenant_id=request.tenant_id,
            session_id=None if shareable else request.session_id,
            resource_id=resource.resource_id,
            resource_version=version,
            record_type=resource.record_type,
            token_start=0,
            token_end=token_count,
            layer=0,
            model_revision=self.model_revision,
            dtype="mlx-model-native",
            layout="per-layer-bhld",
            materialization_profile=str(
                request.pra_policy.get("materialization_profile", "full_selected_resource")
            ),
            position_policy=str(request.pra_policy.get("position_policy", "source_local")),
            security_scope=resource.authorization_scope,
        )
        key = self.block_store.register(
            LogicalPRABlock(identity, address_bytes=32, detail_bytes=0)
        )
        if request.session_id and identity.session_id is not None:
            self._session_keys.setdefault(request.session_id, set()).add(key)
        return key

    def _storage_entry(
        self,
        request: PRAWireRequest,
        resource: PRAWireResource,
        key: str,
        memory: MLXNativeMemory,
    ) -> PRAStorageEntry:
        return PRAStorageEntry(
            logical_key=key,
            record_type=resource.record_type,
            retention_class=PRARetentionClass.RECONSTRUCTABLE,
            tenant_id=request.tenant_id,
            session_id=(
                None if resource.metadata.get("shareable", False) else request.session_id
            ),
            task_id=(
                None
                if resource.metadata.get("task_id") is None
                else str(resource.metadata["task_id"])
            ),
            task_status=(
                None
                if resource.metadata.get("task_status") is None
                else str(resource.metadata["task_status"])
            ),
            resource_version=str(resource.metadata.get("version", "source-hash")),
            detail_bytes=memory.nbytes,
            security_scope=resource.authorization_scope,
            source_reconstructable=resource.text is not None,
            reconstruction_cost_ms=float(
                resource.metadata.get("reconstruction_cost_ms", 0.0)
            ),
            shared_reference_count=int(bool(resource.metadata.get("shareable", False))),
        )

    def _storage_fingerprint(self, resource: PRAWireResource) -> str:
        payload = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "layout": "per-layer-bhld",
            "position_policy": "source-local",
            "resource_id": resource.resource_id,
            "resource_version": resource.metadata.get("version"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _resolve_memory(
        self, request: PRAWireRequest
    ) -> tuple[MLXNativeMemory, tuple[str, ...], int]:
        resources = self._selected_resources(request)
        if not resources:
            raise ValueError("Native PRA execution requires at least one selected resource.")
        tokenized = [(resource, self._resource_tokens(resource)) for resource in resources]
        total = sum(len(tokens) for _, tokens in tokenized)
        if total > request.budget.max_selected_tokens:
            raise ValueError("Selected PRA resources exceed max_selected_tokens.")
        keys = tuple(
            self._register(request, resource, len(tokens))
            for resource, tokens in tokenized
        )
        scopes = tuple(map(str, request.metadata.get("authorization_scopes", ())))
        self.block_store.select(keys, tenant_id=request.tenant_id, authorization_scopes=scopes)

        memories = []
        for key, (resource, tokens) in zip(keys, tokenized):
            def encode(tokens=tokens):
                with self._model_runner_lock:
                    memory = encode_native_memory(self.model, tokens)
                return memory, serialize_native_memory(memory)

            if key not in self.storage.entries:
                memory, payload = encode()
                self.storage.register(
                    self._storage_entry(request, resource, key, memory),
                    payload,
                    hot_value=memory,
                    source_loader=lambda payload=payload: payload,
                    fingerprint=self._storage_fingerprint(resource),
                )
            else:
                self.storage.attach_source_loader(
                    key, lambda encode=encode: encode()[1]
                )
            memory = self.storage.promote(
                key,
                tenant_id=request.tenant_id,
                authorization_scopes=scopes,
            )
            if not isinstance(memory, MLXNativeMemory):
                raise TypeError("MLX lifecycle promotion returned non-native memory.")
            memories.append(memory)
            self.storage.record_access(key, selected=True)
        return combine_native_memories(memories), keys, total

    def _prompt(self, request: PRAWireRequest) -> str:
        return self.tokenizer.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(request.engine_hints.get("enable_thinking", False)),
        )

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]:
        if block_store is not self.block_store:
            raise ValueError("MLX executor and adapter must share one logical block store.")
        with self._model_runner_lock:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            # Promotion may evaluate MLX arrays on the same global command
            # stream as decode, so it belongs to the model-runner critical
            # section together with cache construction and generation.
            memory, keys, selected_tokens = self._resolve_memory(request)
            self.isolation.open_request(
                request.request_id,
                keys,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
            )
            self.isolation.assert_request_scope(
                request.request_id,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
            )
            try:
                self.isolation.attach_once(request.request_id, keys)
                detail_layers = request.pra_policy.get("detail_kv_layers")
                cache = make_native_prompt_cache(
                    self.model,
                    memory,
                    selected_layers=(
                        None if detail_layers is None else tuple(detail_layers)
                    ),
                )
                prompt = self._prompt(request)
                started = time.perf_counter()
                with self.storage.pin_request(request.request_id, keys):
                    for response in stream_generate(
                        self.model,
                        self.tokenizer,
                        prompt,
                        max_tokens=request.resolved_max_new_tokens,
                        prompt_cache=cache,
                        sampler=make_sampler(
                            temp=float(request.engine_hints.get("temperature", 0))
                        ),
                    ):
                        yield {
                            "text": response.text,
                            "finish_reason": getattr(response, "finish_reason", None),
                            "prompt_tokens": int(response.prompt_tokens),
                            "generation_tokens": int(response.generation_tokens),
                            "selected_native_tokens": selected_tokens,
                            "active_native_kv_bytes": memory.selected_nbytes(detail_layers),
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                            "native_kv_used": True,
                        }
            finally:
                self.isolation.close_request(
                    request.request_id, require_attached=False
                )

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult:
        rows = tuple(self.stream(request, block_store))
        text = "".join(str(row.get("text", "")) for row in rows)
        final = dict(rows[-1]) if rows else {}
        final["residency"] = self.residency.metrics().to_dict()
        return PRAEngineResult(text=text, raw=final, trace=self.residency.events())

    def close_session(self, session_id: str) -> None:
        keys = tuple(self._session_keys.pop(session_id, ()))
        self.storage.close_session(session_id)
        for key in keys:
            if key in self.storage.entries and self.storage.entries[key].current_tier.value != "source":
                continue
            identity = self.block_store.get(key).identity
            self.block_store.invalidate_resource(
                identity.tenant_id,
                identity.resource_id,
                resource_version=identity.resource_version,
            )

    def close(self) -> None:
        self.isolation.close()
        self.storage.close()
        self.residency.close()
