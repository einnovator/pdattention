"""Thin native-K/V PRA adapter for Hugging Face Gemma 3 text models."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


def gemma3_global_layer_ids(config) -> tuple[int, ...]:
    """Return decoder layers whose native scope is full causal attention."""
    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if not layer_types:
        raise ValueError("Gemma 3 config does not expose its local/global layer schedule.")
    return tuple(
        layer_id
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )


def _gemma3_symbols(attention):
    """Resolve Gemma 3's installed RoPE and eager-attention functions."""
    if ".gemma3." not in attention.__class__.__module__:
        raise TypeError(
            f"Unsupported Gemma 3 attention class: {attention.__class__.__qualname__}"
        )
    from transformers.models.gemma3.modeling_gemma3 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return apply_rotary_pos_emb, eager_attention_forward, "gemma3"


class Gemma3PRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Expose PRA only through Gemma 3's native global-attention layers.

    Gemma 3 already alternates local sliding-window and global layers. The
    Paper 2 integration treats that schedule as an architectural contract:
    selected reference K/V may extend a global layer, while local layers remain
    untouched and retain their native mask, local RoPE base, and hybrid cache.
    """

    family = "gemma3"

    def __init__(self, original_attention, *args, **kwargs) -> None:
        if bool(getattr(original_attention, "is_sliding", False)):
            raise ValueError(
                "PRA-HF preserves Gemma 3 sliding-attention layers unchanged; "
                f"layer {original_attention.layer_idx} is local."
            )
        super().__init__(original_attention, *args, **kwargs)
        # Gemma3DecoderLayer chooses local versus global RoPE from this field
        # before invoking the attention module. Preserve the replaced module's
        # public metadata in addition to retaining it inside the delegate.
        self.is_sliding = False
        self.sliding_window = None

    @staticmethod
    def resolve_native_symbols(original_attention):
        """Return Gemma 3's installed native attention helpers."""
        return _gemma3_symbols(original_attention)

    def normalize_qkv_layout(self, query, key, value):
        """Validate Gemma 3 query heads and native, unexpanded MQA/GQA K/V."""
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            raise ValueError("Gemma 3 Q and K/V must be canonical rank-four tensors.")
        if query.shape[1] != self.original_attention.config.num_attention_heads:
            raise ValueError("Gemma 3 query-head layout differs from its configuration.")
        if key.shape[1] != self.original_attention.config.num_key_value_heads:
            raise ValueError("Gemma 3 native K/V-head layout differs from its configuration.")
        return query, key, value

    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
        """Call Gemma 3's global eager kernel with selected native-head K/V."""
        attention = self.original_attention
        return self.eager_attention_forward(
            attention,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else attention.attention_dropout,
            scaling=attention.scaling,
            softcap=getattr(attention.config, "attn_logit_softcapping", None),
            sliding_window=None,
            **kwargs,
        )
