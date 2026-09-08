"""Attention-only PRA adapter for Hugging Face GPT-OSS models."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


def gpt_oss_full_layer_ids(config) -> tuple[int, ...]:
    """Return GPT-OSS layers with full rather than sliding attention."""

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if not layer_types:
        raise ValueError("GPT-OSS config does not expose its attention schedule.")
    return tuple(
        layer_id
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )


def _gpt_oss_symbols(attention):
    if ".gpt_oss." not in attention.__class__.__module__:
        raise TypeError(
            f"Unsupported GPT-OSS attention class: {attention.__class__.__qualname__}"
        )
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return apply_rotary_pos_emb, eager_attention_forward, "gpt_oss"


class GptOssPRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Extend GPT-OSS full-attention layers without touching sparse experts.

    Sliding layers are intentionally excluded: prepending nonlocal memory there
    would silently alter the model's local-attention contract. Full layers keep
    GPT-OSS's learned sink logits, native GQA layout, RoPE, and output mapping.
    """

    family = "gpt_oss"

    def __init__(self, original_attention, *args, **kwargs) -> None:
        if getattr(original_attention, "sliding_window", None) is not None:
            raise ValueError(
                "PRA-HF preserves GPT-OSS sliding-attention layers unchanged; "
                f"layer {original_attention.layer_idx} is local."
            )
        super().__init__(original_attention, *args, **kwargs)
        self.sliding_window = None

    @staticmethod
    def resolve_native_symbols(original_attention):
        return _gpt_oss_symbols(original_attention)

    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
        """Call GPT-OSS eager attention with its learned attention sinks."""

        attention = self.original_attention
        return self.eager_attention_forward(
            attention,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else attention.attention_dropout,
            scaling=attention.scaling,
            sliding_window=None,
            s_aux=attention.sinks,
            **kwargs,
        )
