"""Progressive Retrieval Attention with chunk-aware, batch-isolated routing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import PRAExecutionCore
from .memory import LayerKV, PRAMemoryCache, SelectedChunk
from .memory_batching import MemoryBatchingStats, dynamic_memory_attention
from .masks import causal_attention_mask
from .positions import build_position_encoding


class PRAttention(nn.Module):
    """Controlled-model attention using the model-family-independent PRA core.

    Input/output hidden states are ``[B,T,d_model]``. Native transport places
    routed token K/V before local K/V under one softmax and output projection.
    The legacy cross-attention mode remains for historical experiments but uses
    the same router, budgeter, and materializer.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        max_seq_len,
        layer_id,
        pra_cache: PRAMemoryCache,
        *,
        config,
    ):
        """Create local projections and bind this layer to shared PRA services."""
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.layer_id = layer_id
        self.config = config
        self.trigger_threshold = config.trigger_threshold
        self.memory_transport = config.memory_transport
        self.memory_alpha = config.memory_alpha
        self.pra_cache = pra_cache
        self.position_encoding = build_position_encoding(
            config.position_encoding,
            head_dim=self.head_dim,
            rope_theta=config.rope_theta,
        )
        self.last_selected_chunks: list[list[SelectedChunk]] = []
        self.last_routing_rankings: list[list[dict]] = []
        self.last_memory_batching_stats: MemoryBatchingStats | None = None
        self.last_diagnostics: dict[str, float] = {}

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.mem_o_proj = (
            nn.Linear(d_model, d_model)
            if self.memory_transport == "cross_attention"
            else None
        )
        self.pra_core = PRAExecutionCore(
            cache=pra_cache,
            config=config,
            layer_id=layer_id,
            num_query_heads=n_heads,
            num_key_value_heads=n_heads,
            head_dim=self.head_dim,
        )

        hidden = causal_attention_mask(
            max_seq_len,
            device="cpu",
            window=config.self_attention_window,
        )
        self.register_buffer("causal_mask", hidden.view(1, 1, max_seq_len, max_seq_len))

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Retarget both the public attention handle and shared execution core."""
        self.pra_cache = pra_cache
        self.pra_core.cache = pra_cache

    def split_heads(self, x):
        """Reshape ``[B,T,D]`` hidden states to ``[B,H,T,Dh]``."""
        b, t, _d = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        """Restore ``[B,H,T,Dh]`` head output to ``[B,T,D]``."""
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def project_kv(
        self,
        hidden_states,
        *,
        detach: bool = True,
        position_ids: torch.Tensor | None = None,
    ) -> LayerKV:
        """Project ``[1,M,D]`` into cache-native ``[1,H,M,Dh]`` K/V."""
        k = self.split_heads(self.k_proj(hidden_states))
        v = self.split_heads(self.v_proj(hidden_states))
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        _, k = self.position_encoding.transform_qk(k, k, position_ids)
        if detach:
            k = k.detach()
            v = v.detach()
        return LayerKV(
            k=k,
            v=v,
            position_ids=position_ids.detach().clone(),
            position_state="post_position",
        )

    def forward_native_kv(
        self,
        x: torch.Tensor,
        memory_k: list[torch.Tensor],
        memory_v: list[torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bypass routing and replay caller-supplied historical native K/V."""
        if self.memory_transport != "native_kv":
            raise RuntimeError("forward_native_kv requires memory_transport='native_kv'.")
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)
        q, k = self.position_encoding.transform_qk(q, k, position_ids)
        from .core import PreparedPRAMemory

        prepared = PreparedPRAMemory(
            keys=memory_k,
            values=memory_v,
            selections=[[] for _ in memory_k],
            rankings=[[] for _ in memory_k],
            budget_stats=[],
        )
        heads, stats, duration = self.pra_core.apply_pra_attention(
            q, k, v, prepared, attention_mask=attention_mask
        )
        output = self.o_proj(self.merge_heads(heads))
        self.last_memory_batching_stats = stats
        self.last_diagnostics = {
            **stats.as_metrics(),
            "direct_context_tokens": float(x.shape[1]),
            "memory_budget_tokens": float(stats.valid_positions),
            "memory_tokens_requested": float(stats.valid_positions),
            "memory_tokens_materialized": float(stats.valid_positions),
            "retrieved_token_kv": float(stats.valid_positions),
            "retrieved_physical_kv_tokens": float(stats.valid_positions),
            "memory_transport_native_kv": 1.0,
            "attention_output_norm": float(output.detach().norm().cpu()),
            "model_max_context_tokens": float(
                self.config.effective_model_max_context_tokens
            ),
        }
        if self.config.collect_detailed_timing:
            self.last_diagnostics["memory_attention_duration_seconds"] = duration
        return output

    # Compatibility entry points retained for notebooks/tests; implementation is shared.
    def _expand_full_references(self, selected: list[SelectedChunk]) -> list[SelectedChunk]:
        return self.pra_core._expand_full_references(selected)

    def _budget_selection(self, selected, *, direct_tokens, routing_candidates):
        return self.pra_core.budget_selected_memory(
            selected,
            direct_tokens=direct_tokens,
            routing_candidates=routing_candidates,
        )

    def _materialize(self, selected, q, *, direct_tokens):
        self.pra_core.trigger_threshold = float(self.trigger_threshold)
        return self.pra_core.materialize_selected_kv(
            selected,
            q,
            direct_tokens=direct_tokens,
        )

    def _local_attention(self, q, k, v, attention_mask):
        """Run the controlled decoder's causal local attention."""
        b, _h, t, _d = q.shape
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :t, :t], float("-inf"))
        valid = None
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(f"Expected attention_mask {(b, t)}, got {tuple(attention_mask.shape)}.")
            valid = attention_mask.to(device=q.device, dtype=torch.bool)
            if not bool(valid.any(dim=1).all()):
                raise ValueError("Every prompt row must contain at least one direct token.")
            scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
            scores = scores.masked_fill(~valid[:, None, :, None], 0.0)
        output = F.softmax(scores, dim=-1) @ v
        return output if valid is None else output * valid[:, None, :, None]

    def _empty_selection_metrics(self, prepared) -> dict[str, float]:
        """Report route/budget activity when no K/V survives materialization."""
        keys = (
            "direct_context_tokens",
            "memory_budget_tokens",
            "routing_candidates",
            "routing_topk_candidates",
            "chunks_materialized",
            "chunks_budget_rejected",
            "memory_tokens_requested",
            "memory_tokens_materialized",
        )
        metrics = {
            key: sum(float(row[key]) for row in prepared.budget_stats)
            for key in keys
        }
        metrics["model_max_context_tokens"] = float(
            self.config.effective_model_max_context_tokens
        )
        return metrics

    def forward(
        self,
        x,
        use_pra_memory: bool = True,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """Compute local attention and optionally invoke shared routed memory."""
        batch, tokens, _ = x.shape
        self.last_selected_chunks = [[] for _ in range(batch)]
        self.last_routing_rankings = [[] for _ in range(batch)]
        self.last_memory_batching_stats = None
        self.last_diagnostics = {}

        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        if position_ids is None:
            position_ids = torch.arange(tokens, device=x.device)
        q, k = self.position_encoding.transform_qk(q, k, position_ids)
        local_heads = self._local_attention(q, k, v, attention_mask)
        local_out = self.o_proj(self.merge_heads(local_heads))
        if not use_pra_memory or self.pra_cache.is_empty():
            return local_out

        self.pra_core.trigger_threshold = float(self.trigger_threshold)
        prepared = self.pra_core.prepare_memory(
            q,
            direct_tokens=tokens,
            routing_attention_mask=attention_mask,
        )
        self.last_selected_chunks = prepared.selections
        self.last_routing_rankings = prepared.rankings
        if not prepared.has_memory:
            self.last_diagnostics = self._empty_selection_metrics(prepared)
            return local_out

        if self.memory_transport == "native_kv":
            native_heads, stats, attention_duration = self.pra_core.apply_pra_attention(
                q,
                k,
                v,
                prepared,
                attention_mask=attention_mask,
            )
            self.last_memory_batching_stats = stats
            output = self.o_proj(self.merge_heads(native_heads))
            local_token_count = int(attention_mask.sum().item()) if attention_mask is not None else batch * tokens
            retrieved = int(stats.valid_positions)
            unique = retrieved if self.config.overlap_materialization == "deduplicate" else max(retrieved - prepared.duplicate_tokens, 0)
            selected_stored = retrieved + prepared.duplicate_tokens if self.config.overlap_materialization == "deduplicate" else retrieved
            self.last_diagnostics = self.pra_core.collect_pra_metrics(
                prepared,
                stats,
                direct_tokens=batch * tokens,
                output=output,
                local_output=local_out,
                memory_attention_duration_seconds=attention_duration,
            )
            self.last_diagnostics.update(
                {
                    "active_local_tokens": float(local_token_count),
                    "retrieved_unique_source_tokens": float(unique),
                    "selected_stored_kv_tokens": float(selected_stored),
                    "memory_overlap_tokens_detected": float(prepared.duplicate_tokens),
                    "memory_overlap_tokens_removed": float(prepared.duplicate_tokens if self.config.overlap_materialization == "deduplicate" else 0),
                    "accessible_kv_tokens": float(local_token_count + retrieved),
                    "active_memory_fraction": retrieved / max(local_token_count + retrieved, 1),
                    "retrieved_kv_storage_bytes": float(2 * retrieved * self.n_heads * self.head_dim * q.element_size()),
                    # Native-KV PRA has one joint softmax, so output-local
                    # divergence is the meaningful residual-change diagnostic.
                    "pra_output_divergence_ratio": float(
                        (output - local_out).detach().norm().cpu()
                    )
                    / max(float(local_out.detach().norm().cpu()), 1e-12),
                }
            )
            return output

        memory_heads, stats = dynamic_memory_attention(
            q,
            prepared.keys,
            prepared.values,
            bucket_count=self.config.memory_bucket_count,
            bucket_strategy=self.config.memory_bucket_strategy,
        )
        self.last_memory_batching_stats = stats
        assert self.mem_o_proj is not None
        memory_out = self.mem_o_proj(self.merge_heads(memory_heads))
        has_memory = torch.tensor(
            [length > 0 for length in stats.selected_lengths],
            device=memory_out.device,
            dtype=memory_out.dtype,
        ).view(batch, 1, 1)
        memory_out = memory_out * has_memory
        self.last_diagnostics = {
            **stats.as_metrics(),
            **self._empty_selection_metrics(prepared),
            "memory_duplicate_chunk_tokens": float(prepared.duplicate_tokens),
            "retrieved_kv_transfer_bytes": float(prepared.transferred_kv_bytes),
            "memory_output_norm": float(memory_out.detach().norm().cpu()),
            "local_output_norm": float(local_out.detach().norm().cpu()),
        }
        self.last_diagnostics["memory_to_local_output_norm_ratio"] = self.last_diagnostics["memory_output_norm"] / max(self.last_diagnostics["local_output_norm"], 1e-12)
        if self.config.collect_detailed_timing:
            self.last_diagnostics.update(
                {
                    "routing_duration_seconds": prepared.routing_duration_seconds,
                    "materialization_duration_seconds": prepared.materialization_duration_seconds,
                    "selected_kv_transfer_duration_seconds": prepared.transfer_duration_seconds,
                }
            )
        return local_out + self.memory_alpha * memory_out
