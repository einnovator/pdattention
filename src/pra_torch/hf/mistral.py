"""Native-K/V PRA adapter for Mistral and Mistral 3 text decoders."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


class MistralPRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Bind shared PRA execution to the nested Mistral text decoder."""

    family = "mistral3"

    @staticmethod
    def resolve_native_symbols(original_attention):
        if ".mistral." not in original_attention.__class__.__module__:
            raise TypeError(
                f"Unsupported Mistral attention: {original_attention.__class__.__qualname__}"
            )
        from transformers.models.mistral.modeling_mistral import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        return apply_rotary_pos_emb, eager_attention_forward, "mistral3_text"

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
            sliding_window=getattr(attention, "sliding_window", None),
            **kwargs,
        )
