"""Native-K/V PRA adapter for Mistral and Mistral 3 text decoders."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


def _mistral_symbols(attention):
    """Resolve Mistral's installed RoPE and eager-attention functions."""

    if ".mistral." not in attention.__class__.__module__:
        raise TypeError(
            f"Unsupported Mistral attention class: {attention.__class__.__qualname__}"
        )
    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return apply_rotary_pos_emb, eager_attention_forward, "mistral3_text"


class MistralPRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Reuse the Mistral text decoder inside causal and Mistral 3 wrappers.

    PRA owns only selected attention execution. The multimodal projector,
    vision tower, MLPs, tokenizer contract, and all pretrained projections stay
    with Hugging Face. Mistral 3's current text configuration uses full causal
    attention; a future sliding configuration retains its native mask argument.
    """

    family = "mistral3"

    @staticmethod
    def resolve_native_symbols(original_attention):
        return _mistral_symbols(original_attention)

    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
        """Call Mistral's eager kernel after bounded native K/V composition."""

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
