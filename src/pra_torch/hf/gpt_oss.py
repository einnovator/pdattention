"""Attention-only PRA adapter for Hugging Face GPT-OSS models."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


class GptOssPRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Extend full-attention layers while preserving sinks and sparse experts."""

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
        if ".gpt_oss." not in original_attention.__class__.__module__:
            raise TypeError(
                f"Unsupported GPT-OSS attention: {original_attention.__class__.__qualname__}"
            )
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        return apply_rotary_pos_emb, eager_attention_forward, "gpt_oss"

    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
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
