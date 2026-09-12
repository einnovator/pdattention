"""Live Qwen2/Qwen3 MLX-LM patch for exact segmented PRA attention."""

from __future__ import annotations

from typing import Any

from .native import (
    MLXDisjointSelectedKVCache,
    MLXPositionedKVCache,
    MLXSegmentedSelectedKVCache,
    compiled_segmented_selected_attention,
    disjoint_segmented_selected_attention,
    segmented_selected_attention,
)


def install_qwen3_segmented_attention(model: object, *, compiled: bool = True) -> int:
    """Patch Qwen2/Qwen3 attention layers to consume separate PRA K/V.

    The wrapper delegates ordinary and concatenated-cache requests to MLX-LM's
    original module.  Only :class:`MLXSegmentedSelectedKVCache` takes the new
    path, which preserves Qwen3's projections, normalization, RoPE, and output
    projection while replacing K/V concatenation with one exact segmented
    softmax.  The return value is the number of patched decoder layers.
    """

    import mlx.core as mx
    import mlx.nn as nn

    model_type = str(getattr(getattr(model, "args", None), "model_type", ""))
    if model_type not in {"qwen2", "qwen3", "qwen3_moe"}:
        raise ValueError(
            "Live segmented attention currently supports qwen2, qwen3 and qwen3_moe, "
            f"not {model_type!r}."
        )
    if getattr(model, "_pra_segmented_attention_installed", False):
        return 0

    class _SegmentedQwen3Attention(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def __call__(self, x, mask=None, cache=None):
            if not isinstance(
                cache,
                (
                    MLXPositionedKVCache,
                    MLXSegmentedSelectedKVCache,
                    MLXDisjointSelectedKVCache,
                ),
            ):
                return self.inner(x, mask, cache)

            batch, length, _ = x.shape
            attention = self.inner
            queries = attention.q_proj(x)
            keys = attention.k_proj(x)
            values = attention.v_proj(x)
            queries = queries.reshape(
                batch, length, attention.n_heads, -1
            )
            keys = keys.reshape(
                batch, length, attention.n_kv_heads, -1
            )
            # Qwen3 applies per-head Q/K RMSNorm; Qwen2 does not. Keep the
            # model-family projection path exact instead of fabricating norms.
            if hasattr(attention, "q_norm"):
                queries = attention.q_norm(queries)
            if hasattr(attention, "k_norm"):
                keys = attention.k_norm(keys)
            queries = queries.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)
            values = values.reshape(
                batch, length, attention.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)
            queries = attention.rope(queries, offset=cache.offset)
            keys = attention.rope(keys, offset=cache.offset)
            if isinstance(cache, MLXPositionedKVCache):
                # A non-consumer layer still evaluates the query at its
                # source-relative RoPE coordinates, but it has no source K/V.
                # Qwen's model-level mask is derived from cache[0].offset and
                # therefore incorrectly assumes those skipped source tokens
                # are physically present. Build the local causal mask here.
                layer_mask = cache.make_mask(length, return_array=True)
                keys, values = cache.update_and_fetch(keys, values)
                output = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=attention.scale,
                    mask=layer_mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(
                    batch, length, -1
                )
                return attention.o_proj(output)
            # Build the mask before update_and_fetch_segments advances the
            # local cache offset. Otherwise the current query is counted twice.
            layer_mask = cache.make_mask(length, return_array=True)
            memory_k, memory_v, local_k, local_v = (
                cache.update_and_fetch_segments(keys, values)
            )
            # Qwen3 builds one mask from cache[0]. Consumer profiles may start
            # later in the stack, so every selected layer derives its own mask.
            if isinstance(cache, MLXDisjointSelectedKVCache):
                output = disjoint_segmented_selected_attention(
                    queries,
                    memory_k,
                    memory_v,
                    local_k,
                    local_v,
                    scale=attention.scale,
                    mask=layer_mask,
                    source_keys=(
                        cache.memory.source_keys
                        if cache.fused_disjoint_attention
                        else None
                    ),
                    source_values=(
                        cache.memory.source_values
                        if cache.fused_disjoint_attention
                        else None
                    ),
                    source_intervals=(
                        cache.memory.intervals
                        if cache.fused_disjoint_attention
                        else ()
                    ),
                )
            else:
                attention_fn = (
                    compiled_segmented_selected_attention
                    if compiled
                    else segmented_selected_attention
                )
                output = attention_fn(
                    queries,
                    memory_k,
                    memory_v,
                    local_k,
                    local_v,
                    scale=attention.scale,
                    mask=layer_mask,
                )
            output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
            return attention.o_proj(output)

    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise ValueError("Qwen model exposes no decoder layers to patch.")
    for layer in layers:
        layer.self_attn = _SegmentedQwen3Attention(layer.self_attn)
    model._pra_segmented_attention_installed = True
    mx.eval(model.parameters())
    return len(layers)
