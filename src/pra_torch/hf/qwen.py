"""Thin native-K/V PRA adapter for Qwen2/Qwen2.5/Qwen3 eager attention."""

from __future__ import annotations

import torch

from .adapter_base import PRAHFAttentionAdapter
from ..memory import LayerKV
from ..memory_batching import MemoryBatchingStats


def _qwen_symbols(attention):
    """Resolve the matching installed Qwen module without copying model code."""
    module_name = attention.__class__.__module__
    if ".qwen3." in module_name:
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        return apply_rotary_pos_emb, eager_attention_forward, "qwen3"
    if ".qwen2." in module_name:
        from transformers.models.qwen2.modeling_qwen2 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        return apply_rotary_pos_emb, eager_attention_forward, "qwen2"
    raise TypeError(f"Unsupported Qwen attention class: {attention.__class__.__qualname__}")


class QwenPRAAttentionAdapter(PRAHFAttentionAdapter):
    """Reuse Qwen projections, RoPE, GQA, masks, cache update, and output projection."""

    family = "qwen"

    def __init__(self, original_attention, cache, config) -> None:
        super().__init__(original_attention, cache, config)
        self.apply_rotary_pos_emb, self.eager_attention_forward, self.variant = _qwen_symbols(
            original_attention
        )

    def project_qkv(self, hidden_states: torch.Tensor):
        """Project ``[B,T,D]`` using native Qwen Q/K/V and optional QK norms."""
        attention = self.original_attention
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_proj(hidden_states).view(hidden_shape)
        key = attention.k_proj(hidden_states).view(hidden_shape)
        value = attention.v_proj(hidden_states).view(hidden_shape)
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
        if hasattr(attention, "k_norm"):
            key = attention.k_norm(key)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def apply_native_position_encoding(self, query, key, position_embeddings):
        """Apply the exact Qwen RoPE helper supplied by the installed family."""
        cos, sin = position_embeddings
        return self.apply_rotary_pos_emb(query, key, cos, sin)

    def build_native_mask(self, local_key, local_value, prepared, attention_mask, query_tokens):
        """Prepend selected K/V and an all-visible, row-isolated memory mask."""
        return self.pra_core.combine_local_and_memory_kv(
            local_key,
            local_value,
            prepared,
            attention_mask,
            query_tokens=query_tokens,
        )

    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
        """Call Qwen's own eager kernel after shared PRA has prepared native K/V."""
        attention = self.original_attention
        return self.eager_attention_forward(
            attention,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else attention.attention_dropout,
            scaling=attention.scaling,
            sliding_window=attention.sliding_window,
            **kwargs,
        )

    def project_output(self, attention_output: torch.Tensor, input_shape) -> torch.Tensor:
        """Restore ``[B,T,D]`` and apply the unchanged Qwen ``o_proj``."""
        output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.original_attention.o_proj(output)

    def _record_capture(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Save detached post-RoPE K and native V without expanding GQA heads."""
        positions = self.capture_position_ids
        if positions is None:
            positions = torch.arange(key.shape[2], device=key.device).unsqueeze(0)
        self.captured_kv = LayerKV(
            k=key.detach(),
            v=value.detach(),
            position_ids=positions.detach().clone(),
            position_state="post_position",
        )

    def _stats(self, prepared) -> MemoryBatchingStats:
        """Describe HF rectangular packing while retaining physical-token counts."""
        lengths = prepared.selected_lengths
        maximum = max(lengths, default=0)
        valid = sum(lengths)
        allocated = len(lengths) * maximum
        return MemoryBatchingStats(
            selected_lengths=lengths,
            requested_bucket_count=1,
            actual_bucket_count=1 if maximum else 0,
            bucket_membership=(tuple(range(len(lengths))),) if maximum else (),
            bucket_max_lengths=(maximum,) if maximum else (),
            valid_positions=valid,
            allocated_positions=allocated,
            padding_positions=allocated - valid,
            padding_fraction=(allocated - valid) / max(allocated, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Use exact delegation for parity and native Qwen eager attention for PRA."""
        if self.capture_enabled:
            query, key, value = self.project_qkv(hidden_states)
            _query, key = self.apply_native_position_encoding(query, key, position_embeddings)
            self._record_capture(key, value)
        if not self.memory_enabled or self.cache.is_empty():
            self.last_selected_chunks = [[] for _ in range(hidden_states.shape[0])]
            self.last_routing_rankings = [[] for _ in range(hidden_states.shape[0])]
            self.last_diagnostics = {}
            return self.original_attention(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        query, key, value = self.project_qkv(hidden_states)
        query, key = self.apply_native_position_encoding(query, key, position_embeddings)
        if past_key_value is not None:
            cos, sin = position_embeddings
            key, value = past_key_value.update(
                key,
                value,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        prepared = self.pra_core.prepare_memory(query, direct_tokens=int(key.shape[2]))
        self.last_selected_chunks = prepared.selections
        self.last_routing_rankings = prepared.rankings
        if not prepared.has_memory:
            # The cache may already have been updated, so complete this invocation
            # directly instead of delegating and appending the same token twice.
            attention_output, attention_weights = self.invoke_pra(
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )
            return self.project_output(attention_output, input_shape), attention_weights
        combined = self.build_native_mask(
            key,
            value,
            prepared,
            attention_mask,
            query_tokens=int(query.shape[2]),
        )
        started = torch.cuda.Event(enable_timing=True) if query.is_cuda and self.pra_config.collect_detailed_timing else None
        ended = torch.cuda.Event(enable_timing=True) if started is not None else None
        if started is not None:
            started.record()
        attention_output, attention_weights = self.invoke_pra(
            query,
            combined.key,
            combined.value,
            combined.attention_mask,
            **kwargs,
        )
        duration = 0.0
        if ended is not None:
            ended.record()
            torch.cuda.synchronize(query.device)
            duration = started.elapsed_time(ended) / 1000.0
        output = self.project_output(attention_output, input_shape)
        stats = self._stats(prepared)
        self.last_diagnostics = self.pra_core.collect_pra_metrics(
            prepared,
            stats,
            direct_tokens=int(key.shape[2]) * int(key.shape[0]),
            output=output,
            local_output=None,
            memory_attention_duration_seconds=duration,
        )
        self.last_diagnostics.update(
            {
                "hf_native_kv_heads": float(key.shape[1]),
                "hf_query_heads": float(query.shape[1]),
                "hf_memory_width": float(combined.memory_width),
                "hf_cache_tokens": float(key.shape[2]),
            }
        )
        return output, attention_weights
