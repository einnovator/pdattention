"""Thin native-K/V PRA adapter for Hugging Face Llama-family eager attention."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


def _llama_symbols(attention):
    """Resolve Llama's native RoPE and eager-attention functions."""
    if ".llama." not in attention.__class__.__module__:
        raise TypeError(
            f"Unsupported Llama attention class: {attention.__class__.__qualname__}"
        )
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return apply_rotary_pos_emb, eager_attention_forward, "llama"


class LlamaPRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Bind the shared native-RoPE/GQA adapter path to Llama primitives.

    Current Hugging Face Llama and Qwen eager attention expose the same Q/K/V,
    cache, RoPE, and output contracts. Subclassing keeps all PRA behavior in one
    implementation while this class selects only the family-native functions.
    """

    family = "llama"

    @staticmethod
    def resolve_native_symbols(original_attention):
        """Return Llama's installed native attention helpers."""
        return _llama_symbols(original_attention)

    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
        """Call Llama's native eager kernel with selected native-head K/V."""
        attention = self.original_attention
        return self.eager_attention_forward(
            attention,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else attention.attention_dropout,
            scaling=attention.scaling,
            **kwargs,
        )
