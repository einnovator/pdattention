"""Live Qwen3 MLX-LM patch for exact segmented PRA attention."""

from __future__ import annotations

from typing import Any

from .native import MLXSegmentedSelectedKVCache, segmented_selected_attention


def install_qwen3_segmented_attention(model: object) -> int:
    """Patch Qwen3 attention layers to consume physically separate PRA K/V.

    The wrapper delegates ordinary and concatenated-cache requests to MLX-LM's
    original module.  Only :class:`MLXSegmentedSelectedKVCache` takes the new
    path, which preserves Qwen3's projections, normalization, RoPE, and output
    projection while replacing K/V concatenation with one exact segmented
    softmax.  The return value is the number of patched decoder layers.
    """

    import mlx.core as mx
    import mlx.nn as nn

    model_type = str(getattr(getattr(model, "args", None), "model_type", ""))
    if model_type != "qwen3":
        raise ValueError(
            f"Live segmented attention currently supports qwen3, not {model_type!r}."
        )
    if getattr(model, "_pra_segmented_attention_installed", False):
        return 0

    class _SegmentedQwen3Attention(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def __call__(self, x, mask=None, cache=None):
            if not isinstance(cache, MLXSegmentedSelectedKVCache):
                return self.inner(x, mask, cache)

            batch, length, _ = x.shape
            attention = self.inner
            queries = attention.q_proj(x)
            keys = attention.k_proj(x)
            values = attention.v_proj(x)
            queries = attention.q_norm(
                queries.reshape(batch, length, attention.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            keys = attention.k_norm(
                keys.reshape(batch, length, attention.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                batch, length, attention.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)
            queries = attention.rope(queries, offset=cache.offset)
            keys = attention.rope(keys, offset=cache.offset)
            memory_k, memory_v, local_k, local_v = (
                cache.update_and_fetch_segments(keys, values)
            )
            # Qwen3 builds one mask from cache[0]. Consumer profiles may start
            # later in the stack, so every selected layer derives its own mask.
            layer_mask = cache.make_mask(length, return_array=True)
            output = segmented_selected_attention(
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
        raise ValueError("Qwen3 model exposes no decoder layers to patch.")
    for layer in layers:
        layer.self_attn = _SegmentedQwen3Attention(layer.self_attn)
    model._pra_segmented_attention_installed = True
    mx.eval(model.parameters())
    return len(layers)
