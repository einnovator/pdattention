"""Thin native-K/V PRA adapter for Qwen2/Qwen2.5/Qwen3 eager attention."""

from __future__ import annotations

import torch

from .adapter_base import HFRoutingCapture, PRAHFAttentionAdapter
from .config import (
    ATTENTION_INPUT_HIDDEN_STATE,
    CENTERED_ROPE_KEY,
    canonical_routing_representation,
)
from .query import aggregate_query_states
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

    @staticmethod
    def resolve_native_symbols(original_attention):
        """Return the installed family's RoPE helper, eager kernel, and variant."""
        return _qwen_symbols(original_attention)

    def __init__(
        self,
        original_attention,
        cache,
        config,
        rotary_embedding,
        routing_representation: str = ATTENTION_INPUT_HIDDEN_STATE,
        query_strategy: str = "last",
        query_window: int = 16,
        query_half_life: float = 4.0,
        routing_projection=None,
        memory_gate=None,
        residual_adapter=None,
        late_band_lora=None,
    ) -> None:
        super().__init__(
            original_attention,
            cache,
            config,
            memory_gate=memory_gate,
            residual_adapter=residual_adapter,
            late_band_lora=late_band_lora,
        )
        self.routing_representation = canonical_routing_representation(
            routing_representation
        )
        self.query_strategy = query_strategy
        self.query_window = int(query_window)
        self.query_half_life = float(query_half_life)
        # The operational handle owns this optional frozen module. A non-owning
        # reference avoids registering the same projection below multiple layers.
        self.__dict__["routing_projection"] = routing_projection
        # The owning Qwen model already registers this module. Keep a non-owning
        # reference so the adapter can request native fractional-position phases.
        self.__dict__["native_rotary_embedding"] = rotary_embedding
        (
            self.apply_rotary_pos_emb,
            self.eager_attention_forward,
            self.variant,
        ) = self.resolve_native_symbols(original_attention)

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

    def rotate_routing_keys(
        self,
        flattened_keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply native Qwen RoPE to ``[G,H_kv*Dh]`` keys at exact positions."""
        if flattened_keys.ndim != 2 or positions.ndim != 1:
            raise ValueError("Centered routing expects keys [G,D] and positions [G].")
        if flattened_keys.shape[0] != positions.shape[0]:
            raise ValueError("Every centered routing gist requires one position.")
        heads = int(self.config.num_key_value_heads)
        head_dim = int(self.original_attention.head_dim)
        if flattened_keys.shape[1] != heads * head_dim:
            raise ValueError("Centered routing gist width does not match native Qwen K heads.")
        keys = (
            flattened_keys.reshape(flattened_keys.shape[0], heads, head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
        )
        position_ids = positions.to(device=keys.device, dtype=torch.float32).unsqueeze(0)
        cos, sin = self.native_rotary_embedding(keys, position_ids)
        _, rotated = self.apply_rotary_pos_emb(keys, keys, cos, sin)
        return rotated.transpose(1, 2).contiguous().view(flattened_keys.shape)

    def normalize_qkv_layout(self, query, key, value):
        """Validate Qwen's canonical query-head and native K/V-head layout."""
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            raise ValueError("Qwen Q and K/V must be canonical rank-four tensors.")
        if query.shape[1] != self.original_attention.config.num_attention_heads:
            raise ValueError("Qwen query-head layout differs from model configuration.")
        if key.shape[1] != self.original_attention.config.num_key_value_heads:
            raise ValueError("Qwen native K/V-head layout differs from model configuration.")
        return query, key, value

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
        return self.original_attention.o_proj(
            self.flatten_attention_output(attention_output, input_shape)
        )

    @staticmethod
    def flatten_attention_output(
        attention_output: torch.Tensor,
        input_shape,
    ) -> torch.Tensor:
        """Restore native eager output to token-major ``[B,T,D]`` features."""
        return attention_output.reshape(*input_shape, -1).contiguous()

    def _record_capture(
        self,
        pre_query: torch.Tensor,
        post_query: torch.Tensor,
        pre_key: torch.Tensor,
        post_key: torch.Tensor,
        value: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Capture matched routing features plus unchanged post-RoPE detail K/V."""
        positions = self.capture_position_ids
        if positions is None:
            positions = torch.arange(post_key.shape[2], device=post_key.device).unsqueeze(0)
        self.captured_routing = HFRoutingCapture(
            pre_query=pre_query.detach(),
            post_query=post_query.detach(),
            pre_key=pre_key.detach(),
            hidden_states=hidden_states.detach(),
            detail_kv=LayerKV(
                k=post_key.detach(),
                v=value.detach(),
                position_ids=positions.detach().clone(),
                position_state="post_position",
            ),
        )

    def _routing_query_states(
        self,
        hidden_states: torch.Tensor,
        pre_query: torch.Tensor,
        post_query: torch.Tensor,
    ) -> torch.Tensor:
        """Choose a query matched to the configured chunk-gist representation."""
        if self.routing_representation in {"post_rope_key", CENTERED_ROPE_KEY}:
            return post_query
        if self.routing_representation == "pre_rope_key":
            return pre_query
        if self.routing_representation == ATTENTION_INPUT_HIDDEN_STATE:
            query = aggregate_query_states(
                hidden_states,
                self.query_strategy,
                window=self.query_window,
                half_life=self.query_half_life,
            )
            if self.routing_projection is not None:
                query = self.routing_projection.project_query(query)
            return query
        raise ValueError(f"Unsupported routing representation: {self.routing_representation}")

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
            pre_query, pre_key, value = self.project_qkv(hidden_states)
            pre_query, pre_key, value = self.normalize_qkv_layout(pre_query, pre_key, value)
            _post_query, post_key = self.apply_native_position_encoding(
                pre_query, pre_key, position_embeddings
            )
            self._record_capture(
                pre_query, _post_query, pre_key, post_key, value, hidden_states
            )
        if not self.memory_enabled or self.cache.is_empty():
            self.last_selected_chunks = [[] for _ in range(hidden_states.shape[0])]
            self.last_routing_rankings = [[] for _ in range(hidden_states.shape[0])]
            self.last_diagnostics = {}
            result = self.original_attention(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
            self.last_attention_weights = (
                result[1].detach()
                if self.collect_attention_diagnostics and result[1] is not None
                else None
            )
            return result

        input_shape = hidden_states.shape[:-1]
        pre_query, pre_key, value = self.project_qkv(hidden_states)
        pre_query, pre_key, value = self.normalize_qkv_layout(pre_query, pre_key, value)
        query, key = self.apply_native_position_encoding(pre_query, pre_key, position_embeddings)
        if past_key_value is not None:
            cos, sin = position_embeddings
            key, value = past_key_value.update(
                key,
                value,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        routing_query_states = self._routing_query_states(hidden_states, pre_query, query)
        if self.fixed_selected_chunks is None:
            prepared = self.pra_core.prepare_memory(
                query,
                direct_tokens=int(key.shape[2]),
                routing_query_states=routing_query_states,
            )
        else:
            prepared = self.pra_core.prepare_selected_memory(
                query,
                self.fixed_selected_chunks,
                direct_tokens=int(key.shape[2]),
                rankings=[
                    [{"selection_source": "fixed_oracle"} for _ in row]
                    for row in self.fixed_selected_chunks
                ],
            )
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
            self.last_attention_weights = (
                attention_weights.detach()
                if self.collect_attention_diagnostics and attention_weights is not None
                else None
            )
            return self.project_output(attention_output, input_shape), attention_weights
        combined = self.build_native_mask(
            key,
            value,
            prepared,
            attention_mask,
            query_tokens=int(query.shape[2]),
        )
        started = (
            torch.cuda.Event(enable_timing=True)
            if query.is_cuda and self.pra_config.collect_detailed_timing
            else None
        )
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
        self.last_attention_weights = (
            attention_weights.detach()
            if self.collect_attention_diagnostics and attention_weights is not None
            else None
        )
        duration = 0.0
        if ended is not None:
            ended.record()
            torch.cuda.synchronize(query.device)
            duration = started.elapsed_time(ended) / 1000.0
        memory_features = self.flatten_attention_output(attention_output, input_shape)
        memory_output = self.original_attention.o_proj(memory_features)
        lora_delta = None
        lora_delta_norm = 0.0
        if self.late_band_lora is not None and self.late_band_lora.enabled:
            lora_delta = self.late_band_lora.transform(
                self.layer_idx,
                memory_features,
            )
            lora_delta_norm = float(lora_delta.detach().float().norm().cpu())
        local_output = None
        gate_value = 1.0
        memory_residual_norm = 0.0
        needs_delta = (
            self.memory_gate is not None and self.memory_gate.requires_delta_path
        ) or (
            self.residual_adapter is not None and self.residual_adapter.enabled
        )
        if needs_delta:
            local_attention_output, _ = self.invoke_pra(
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )
            local_output = self.project_output(local_attention_output, input_shape)
            memory_residual = memory_output - local_output
            gate = self.memory_gate.value(self.layer_idx, memory_residual)
            gated_residual = gate * memory_residual.float()
            if self.residual_adapter is not None:
                gated_residual = self.residual_adapter.transform(
                    self.layer_idx,
                    hidden_states,
                    gated_residual,
                )
            output = local_output.float() + gated_residual
            if lora_delta is not None:
                output = output + lora_delta
            output = output.to(memory_output.dtype)
            gate_value = float(gate.detach().float().cpu())
            memory_residual_norm = float(
                memory_residual.detach().float().norm().cpu()
            )
        else:
            output = memory_output
            if lora_delta is not None:
                output = (output.float() + lora_delta).to(memory_output.dtype)
        stats = self._stats(prepared)
        self.last_diagnostics = self.pra_core.collect_pra_metrics(
            prepared,
            stats,
            direct_tokens=int(key.shape[2]) * int(key.shape[0]),
            output=output,
            local_output=local_output,
            memory_attention_duration_seconds=duration,
        )
        self.last_diagnostics.update(
            {
                "hf_native_kv_heads": float(key.shape[1]),
                "hf_query_heads": float(query.shape[1]),
                "hf_memory_width": float(combined.memory_width),
                "hf_cache_tokens": float(key.shape[2]),
                "hf_memory_gate": gate_value,
                "hf_memory_residual_norm": memory_residual_norm,
                "hf_late_band_lora_delta_norm": lora_delta_norm,
            }
        )
        return output, attention_weights
