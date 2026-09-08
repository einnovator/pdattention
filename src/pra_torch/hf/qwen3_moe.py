"""Thin attention-only PRA adapter for Hugging Face Qwen3 MoE models."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


def _qwen3_moe_symbols(attention):
    """Resolve Qwen3 MoE's installed RoPE and eager-attention functions."""

    if ".qwen3_moe." not in attention.__class__.__module__:
        raise TypeError(
            f"Unsupported Qwen3 MoE attention class: {attention.__class__.__qualname__}"
        )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return apply_rotary_pos_emb, eager_attention_forward, "qwen3_moe"


class Qwen3MoePRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Reuse Qwen3 MoE attention while leaving sparse experts host-owned.

    The attention tensor contract matches dense Qwen3. The important boundary
    is architectural: PRA wraps only ``self_attn``. Expert selection, routing
    logits, auxiliary losses, capacity policy, and expert execution continue
    through the original Hugging Face MLP modules unchanged.
    """

    family = "qwen3_moe"

    @staticmethod
    def resolve_native_symbols(original_attention):
        return _qwen3_moe_symbols(original_attention)
