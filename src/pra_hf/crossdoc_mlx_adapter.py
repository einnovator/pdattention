"""MLX mechanisms for repairing independently encoded cross-document K/V.

Both mechanisms are request-local. Persistent pre-RoPE records remain
immutable: the learned path creates residual-corrected views, while the
parameter-free path re-encodes only prefixes immediately after document joins.
"""

from __future__ import annotations

import time
from typing import Sequence

from .crossdoc_composition import (
    CrossDocumentResidualAdapterConfig,
    SelectiveBoundaryReencodeReceipt,
    boundary_reencode_spans,
    memory_identity_digest,
)
from .rag_mlx_native import (
    MLXNativeLayerKV,
    MLXNativeMemory,
    PositionBindingMode,
    make_native_prompt_cache,
    rebind_native_memories_to_receipt,
)


def _flatten_heads(array):
    """Convert MLX K/V from ``[B,H,T,Dh]`` to ``[B,T,H*Dh]``."""

    return array.transpose(0, 2, 1, 3).reshape(
        int(array.shape[0]), int(array.shape[2]), -1
    )


def _restore_heads(array, reference):
    """Restore a flattened K/V residual to the reference head layout."""

    return array.reshape(
        int(reference.shape[0]),
        int(reference.shape[2]),
        int(reference.shape[1]),
        int(reference.shape[3]),
    ).transpose(0, 2, 1, 3)


def create_mlx_crossdoc_residual_adapter(
    layer_widths: Sequence[int],
    config: CrossDocumentResidualAdapterConfig,
    *,
    seed: int = 11,
):
    """Create a zero-output MLX low-rank adapter for every K/V layer.

    The down projections are randomly initialized. Every up projection is
    exactly zero, making the complete module an identity correction at step
    zero while still allowing gradients to reach the output path.
    """

    import mlx.core as mx
    import mlx.nn as nn

    widths = tuple(int(width) for width in layer_widths)
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("adapter layer widths must be positive")
    mx.random.seed(seed)

    class _ResidualProjection(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.down = nn.Linear(2 * width, config.rank, bias=False)
            self.up = nn.Linear(config.rank, width, bias=False)
            self.up.weight = mx.zeros_like(self.up.weight)

        def __call__(self, token, previous):
            previous = mx.broadcast_to(previous, token.shape)
            hidden = self.down(mx.concatenate((token, previous), axis=-1))
            hidden = (
                mx.tanh(hidden)
                if config.activation == "tanh"
                else hidden * mx.sigmoid(hidden)
            )
            return self.up(hidden)

    class _LayerAdapter(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.key = _ResidualProjection(width)
            self.value = _ResidualProjection(width)

        def __call__(self, keys, values, previous_keys, previous_values):
            flat_keys = _flatten_heads(keys)
            flat_values = _flatten_heads(values)
            key_delta = self.key(flat_keys, _flatten_heads(previous_keys))
            value_delta = self.value(flat_values, _flatten_heads(previous_values))
            return (
                keys + _restore_heads(key_delta, keys),
                values + _restore_heads(value_delta, values),
            )

    class _CrossDocumentResidualAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = {
                str(index): _LayerAdapter(width)
                for index, width in enumerate(widths)
            }

        def __call__(
            self, layer_index, keys, values, previous_keys, previous_values
        ):
            return self.layers[str(layer_index)](
                keys, values, previous_keys, previous_values
            )

    adapter = _CrossDocumentResidualAdapter()
    mx.eval(adapter.parameters())
    return adapter


def mlx_adapter_parameter_count(adapter: object) -> int:
    """Count trainable scalar parameters without coupling callers to MLX trees."""

    from mlx.utils import tree_flatten

    return sum(
        int(array.size)
        for _, array in tree_flatten(adapter.trainable_parameters())
    )


def apply_mlx_crossdoc_residual_adapter(
    memories: Sequence[MLXNativeMemory], adapter: object
) -> tuple[MLXNativeMemory, ...]:
    """Return corrected pre-RoPE record views; never mutate stored memories.

    For record ``j > 0``, each layer receives a token-weighted gist of all
    corrected preceding records. The first record is unchanged because packed
    causal RAG gives it no earlier document to consume.
    """

    if len(memories) < 2:
        raise ValueError("cross-document adaptation requires at least two records")
    if any(
        memory.position_binding_mode is not PositionBindingMode.PRE_ROPE
        for memory in memories
    ):
        raise ValueError("cross-document adaptation requires pre-RoPE memories")
    layer_count = len(memories[0].layers)
    if not layer_count or any(len(memory.layers) != layer_count for memory in memories):
        raise ValueError("cross-document memories have incompatible layer counts")

    import mlx.core as mx

    corrected: list[MLXNativeMemory] = []
    for record_index, memory in enumerate(memories):
        if record_index == 0:
            corrected.append(memory)
            continue
        layers: list[MLXNativeLayerKV] = []
        for layer_index, source in enumerate(memory.layers):
            prior_keys = mx.concatenate(
                tuple(row.layers[layer_index].keys for row in corrected), axis=2
            )
            prior_values = mx.concatenate(
                tuple(row.layers[layer_index].values for row in corrected), axis=2
            )
            previous_keys = mx.mean(prior_keys, axis=2, keepdims=True)
            previous_values = mx.mean(prior_values, axis=2, keepdims=True)
            keys, values = adapter(
                layer_index,
                source.keys,
                source.values,
                previous_keys,
                previous_values,
            )
            layers.append(MLXNativeLayerKV(keys, values))
        corrected.append(
            MLXNativeMemory(
                tuple(layers),
                source_tokens=memory.source_tokens,
                query_position_base=memory.query_position_base,
                position_binding_mode=memory.position_binding_mode,
                source_positions=memory.source_positions,
                rope_contract=memory.rope_contract,
            )
        )
    return tuple(corrected)


def adapted_crossdoc_memory(
    model: object,
    memories: Sequence[MLXNativeMemory],
    composition_receipt: object,
    adapter: object,
) -> MLXNativeMemory:
    """Apply the residual adapter and bind the result to packed positions."""

    corrected = apply_mlx_crossdoc_residual_adapter(memories, adapter)
    return rebind_native_memories_to_receipt(model, corrected, composition_receipt)


def normalized_kv_distillation_loss(
    candidate_memories: Sequence[MLXNativeMemory],
    packed_teacher: MLXNativeMemory,
):
    """Return scale-normalized MSE against packed causal pre-RoPE K/V."""

    if not candidate_memories:
        raise ValueError("K/V distillation requires candidate memories")
    if packed_teacher.position_binding_mode is not PositionBindingMode.PRE_ROPE:
        raise ValueError("packed K/V teacher must retain pre-RoPE keys")
    if sum(memory.source_tokens for memory in candidate_memories) != packed_teacher.source_tokens:
        raise ValueError("candidate and teacher token counts differ")
    if any(len(memory.layers) != len(packed_teacher.layers) for memory in candidate_memories):
        raise ValueError("candidate and teacher layer counts differ")

    import mlx.core as mx

    losses = []
    epsilon = mx.array(1e-6, dtype=mx.float32)
    for layer_index, teacher in enumerate(packed_teacher.layers):
        keys = mx.concatenate(
            tuple(memory.layers[layer_index].keys for memory in candidate_memories), axis=2
        ).astype(mx.float32)
        values = mx.concatenate(
            tuple(memory.layers[layer_index].values for memory in candidate_memories), axis=2
        ).astype(mx.float32)
        teacher_keys = teacher.keys.astype(mx.float32)
        teacher_values = teacher.values.astype(mx.float32)
        key_scale = mx.mean(mx.square(teacher_keys)) + epsilon
        value_scale = mx.mean(mx.square(teacher_values)) + epsilon
        losses.append(
            0.5
            * (
                mx.mean(mx.square(keys - teacher_keys)) / key_scale
                + mx.mean(mx.square(values - teacher_values)) / value_scale
            )
        )
    return mx.mean(mx.stack(losses))


def selective_boundary_reencode_memory(
    model: object,
    token_segments: Sequence[Sequence[int]],
    memories: Sequence[MLXNativeMemory],
    composition_receipt: object,
    *,
    record_ids: Sequence[str],
    boundary_tokens: int,
) -> tuple[MLXNativeMemory, SelectiveBoundaryReencodeReceipt]:
    """Re-encode later-document prefixes against preceding native tails.

    This is the nonparametric composition control. At each join, the previous
    document's final window is exposed as native context and the next
    document's initial window is passed through the frozen model. The resulting
    request-local K/V replaces only that later prefix in the composed view.
    """

    segments = tuple(tuple(int(token) for token in segment) for segment in token_segments)
    if len(segments) != len(memories) or len(record_ids) != len(memories):
        raise ValueError("tokens, records, and memories must align")
    if any(len(segment) != memory.source_tokens for segment, memory in zip(segments, memories)):
        raise ValueError("token segments do not match native memory lengths")
    spans = boundary_reencode_spans(
        tuple(memory.source_tokens for memory in memories), boundary_tokens
    )

    import mlx.core as mx

    started = time.perf_counter()
    rebound = rebind_native_memories_to_receipt(model, memories, composition_receipt)
    output_layers = list(rebound.layers)
    for span in spans:
        tail = MLXNativeMemory(
            tuple(
                MLXNativeLayerKV(
                    layer.keys[:, :, span.left_start : span.left_end, :],
                    layer.values[:, :, span.left_start : span.left_end, :],
                )
                for layer in output_layers
            ),
            source_tokens=span.context_tokens,
            query_position_base=span.right_start,
            position_binding_mode=PositionBindingMode.POST_ROPE,
            source_positions=tuple(range(span.left_start, span.left_end)),
            rope_contract=rebound.rope_contract,
        )
        caches = make_native_prompt_cache(model, tail)
        right_tokens = segments[span.boundary_index + 1][: span.reencoded_tokens]
        model(mx.array(right_tokens, dtype=mx.int32)[None], cache=caches)
        replacements = []
        for cache in caches:
            state = cache.local_cache.state
            if not isinstance(state, tuple) or len(state) < 2:
                raise RuntimeError("MLX boundary re-encoding exposed no local K/V")
            replacements.append(MLXNativeLayerKV(state[0], state[1]))
        mx.eval([(layer.keys, layer.values) for layer in replacements])
        output_layers = [
            MLXNativeLayerKV(
                mx.concatenate(
                    (
                        source.keys[:, :, : span.right_start, :],
                        replacement.keys,
                        source.keys[:, :, span.right_end :, :],
                    ),
                    axis=2,
                ),
                mx.concatenate(
                    (
                        source.values[:, :, : span.right_start, :],
                        replacement.values,
                        source.values[:, :, span.right_end :, :],
                    ),
                    axis=2,
                ),
            )
            for source, replacement in zip(output_layers, replacements)
        ]

    mx.eval([(layer.keys, layer.values) for layer in output_layers])
    result = MLXNativeMemory(
        tuple(output_layers),
        source_tokens=rebound.source_tokens,
        query_position_base=rebound.query_position_base,
        position_binding_mode=PositionBindingMode.POST_ROPE,
        source_positions=rebound.source_positions,
        rope_contract=rebound.rope_contract,
    )
    receipt = SelectiveBoundaryReencodeReceipt(
        boundary_tokens=boundary_tokens,
        boundary_count=len(spans),
        context_native_tokens=sum(span.context_tokens for span in spans),
        reencoded_tokens=sum(span.reencoded_tokens for span in spans),
        request_reencode_ms=(time.perf_counter() - started) * 1000.0,
        persistent_native_tokens=rebound.source_tokens,
        record_ids=tuple(record_ids),
        spans=spans,
        source_memory_digest=memory_identity_digest(
            record_ids=record_ids,
            source_tokens=tuple(memory.source_tokens for memory in memories),
            layer_count=len(rebound.layers),
        ),
    )
    return result, receipt
