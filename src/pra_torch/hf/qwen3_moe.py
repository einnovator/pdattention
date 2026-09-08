"""Thin attention-only PRA adapter for Hugging Face Qwen3 MoE models."""

from __future__ import annotations

from .qwen import QwenPRAAttentionAdapter


class Qwen3MoePRAAttentionAdapter(QwenPRAAttentionAdapter):
    """Reuse Qwen3 MoE attention while sparse experts remain host-owned."""

    family = "qwen3_moe"

    @staticmethod
    def resolve_native_symbols(original_attention):
        if ".qwen3_moe." not in original_attention.__class__.__module__:
            raise TypeError(
                "Unsupported Qwen3 MoE attention class: "
                f"{original_attention.__class__.__qualname__}"
            )
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        return apply_rotary_pos_emb, eager_attention_forward, "qwen3_moe"
