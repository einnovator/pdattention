"""Progressive Retrieval Attention with chunk-aware, batch-isolated routing."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import LayerKV, PRAMemoryCache, SelectedChunk
from .memory_batching import (
    MemoryBatchingStats,
    dynamic_memory_attention,
    native_kv_attention,
)
from .positions import build_position_encoding


def _synchronize_detailed_timing(tensor: torch.Tensor, enabled: bool) -> None:
    """Fence CUDA only for research phase timing; normal inference stays asynchronous."""
    if enabled and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class PRAttention(nn.Module):
    """Causal attention with routed native-K/V or adapted cross-attention memory.

    Input/output hidden states are ``[B,T,d_model]``. The final local query token
    routes each batch item to layer-specific chunks. Canonical ``native_kv``
    transport inserts their original token K/V before local K/V under one shared
    softmax and the ordinary output projection. Legacy ``cross_attention`` keeps
    the separately normalized, scaled memory branch used by earlier experiments.
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
        """Create local attention projections and bind this layer to a PRA cache."""
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model  # Full hidden/routing width D.
        self.n_heads = n_heads  # Number of parallel attention heads H.
        self.head_dim = d_model // n_heads  # Per-head width Dh.
        self.max_seq_len = max_seq_len  # Size of the precomputed causal mask.
        self.layer_id = layer_id  # Selects matching layer-specific reference K/V.
        self.config = config  # Routing, materialization, batching, and metrics modes.
        self.trigger_threshold = config.trigger_threshold  # Post-search chunk score floor.
        self.memory_transport = config.memory_transport  # Native or adapted transport policy.
        self.memory_alpha = config.memory_alpha  # Legacy cross-attention contribution scale.
        self.pra_cache = pra_cache  # Shared URI cache; selection remains batch-isolated.
        self.position_encoding = build_position_encoding(
            config.position_encoding,
            head_dim=self.head_dim,
            rope_theta=config.rope_theta,
        )  # Applies the same positional semantics to direct and cached native keys.
        self.last_selected_chunks: list[list[SelectedChunk]] = []  # Latest trace by batch row.
        self.last_routing_rankings: list[list[dict]] = []  # Complete candidate ranks by row.
        self.last_memory_batching_stats: MemoryBatchingStats | None = None
        self.last_diagnostics: dict[str, float] = {}  # Latest aggregate attention metrics.

        self.q_proj = nn.Linear(d_model, d_model)  # Shared local and routing queries.
        self.k_proj = nn.Linear(d_model, d_model)  # Local keys and reference-cache keys.
        self.v_proj = nn.Linear(d_model, d_model)  # Local values and reference-cache values.
        self.o_proj = nn.Linear(d_model, d_model)  # Shared native/local output projection.
        if self.memory_transport == "cross_attention":
            # This parameter exists only for backward-compatible adapted transport.
            self.mem_o_proj = nn.Linear(d_model, d_model)
        else:
            self.mem_o_proj = None

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def split_heads(self, x):
        """Reshape ``[B,T,D]`` hidden states to ``[B,H,T,Dh]``."""
        b, t, d = x.shape
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
        """Project an independently encoded chunk into this layer's cache space.

        ``hidden_states`` is ``[1,M,D]`` and returned K/V is ``[1,H,M,Dh]``.
        Detached mode creates ordinary reusable inference memory; trainable-gist
        mode preserves the graph so routing representations can receive gradients.
        """
        k = self.split_heads(self.k_proj(hidden_states))
        v = self.split_heads(self.v_proj(hidden_states))
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        # RoPE positions K at publication time; V is position-independent and remains native.
        _, k = self.position_encoding.transform_qk(k, k, position_ids)
        if detach:
            k = k.detach()
            v = v.detach()
        cached_positions = position_ids.detach().clone()
        return LayerKV(
            k=k,
            v=v,
            position_ids=cached_positions,
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
        """Apply native attention with caller-supplied historical K/V.

        This bypasses gist routing so transport can be tested independently.
        ``x`` is the current normalized tail ``[B,T,D]`` and each memory item is
        historical native K/V ``[1,H,M_i,Dh]`` from the same layer projections.
        """
        if self.memory_transport != "native_kv":
            raise RuntimeError("forward_native_kv requires memory_transport='native_kv'.")
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)
        q, k = self.position_encoding.transform_qk(q, k, position_ids)
        heads, _stats = native_kv_attention(
            q,
            k,
            v,
            memory_k,
            memory_v,
            attention_mask=attention_mask,
            max_context_tokens=self.config.effective_model_max_context_tokens,
        )
        return self.o_proj(self.merge_heads(heads))

    def _expand_full_references(self, selected: list[SelectedChunk]) -> list[SelectedChunk]:
        """Replace routed chunks with every chunk belonging to each selected URI."""
        expanded = []
        seen_references = set()
        for hit in selected:
            if hit.reference_uri in seen_references:
                continue
            seen_references.add(hit.reference_uri)
            memory = hit.entry.layer_memory.get(self.layer_id)
            if memory is None:
                continue
            for chunk_rank, chunk in enumerate(memory.chunks, start=1):
                expanded.append(
                    replace(
                        hit,
                        chunk=chunk,
                        rank_within_reference=chunk_rank,
                        metadata={**hit.metadata, "materialization": "full_reference"},
                    )
                )
        return expanded

    def _budget_selection(
        self,
        selected: list[SelectedChunk],
        *,
        direct_tokens: int,
        routing_candidates: int,
    ) -> tuple[list[SelectedChunk], dict[str, float]]:
        """Choose highest-scoring whole chunks under one native-context budget."""
        hard_remaining = (
            self.config.effective_model_max_context_tokens
            - int(direct_tokens)
            - self.config.context_safety_reserve_tokens
        )
        if hard_remaining < 0:
            raise ValueError(
                "Direct context and safety reserve exceed model_max_context_tokens."
            )
        memory_budget = hard_remaining
        if self.config.max_materialized_memory_tokens is not None:
            memory_budget = min(
                memory_budget, self.config.max_materialized_memory_tokens
            )
        ranked = sorted(
            selected,
            key=lambda hit: (-hit.chunk_score, hit.reference_uri, hit.chunk_id),
        )
        requested = sum(
            1
            if self.config.detail_materialization == "gist_only"
            else hit.selected_token_count
            for hit in ranked
        )
        retained = []
        rejected = []
        remaining = memory_budget
        materialized = 0
        for hit in ranked:
            cost = (
                1
                if self.config.detail_materialization == "gist_only"
                else hit.selected_token_count
            )
            if cost <= remaining:
                retained.append(hit)
                remaining -= cost
                materialized += cost
            else:
                rejected.append(hit)
        return retained, {
            "model_max_context_tokens": float(
                self.config.effective_model_max_context_tokens
            ),
            "direct_context_tokens": float(direct_tokens),
            "memory_budget_tokens": float(memory_budget),
            "routing_candidates": float(routing_candidates),
            "routing_topk_candidates": float(len(selected)),
            "chunks_materialized": float(len(retained)),
            "chunks_budget_rejected": float(len(rejected)),
            "memory_tokens_requested": float(requested),
            "memory_tokens_materialized": float(materialized),
            "materialization_budget_utilization": materialized
            / max(memory_budget, 1),
            "lowest_materialized_score": (
                min(hit.chunk_score for hit in retained) if retained else float("nan")
            ),
            "highest_budget_rejected_score": (
                max(hit.chunk_score for hit in rejected) if rejected else float("nan")
            ),
        }

    def _materialize(
        self,
        selected: list[SelectedChunk],
        q: torch.Tensor,
        *,
        direct_tokens: int,
    ):
        """Turn routed hits into one item's rectangular memory K/V.

        Returns K/V shaped ``[1,H,M,Dh]``, retained trace records, and the number
        of overlapping token positions removed. ``M`` varies by batch item and is
        one per chunk in ``gist_only`` mode, selected detail in the default mode,
        or all chunks of each routed URI in ``full_reference`` mode.
        """
        routing_candidates = len(selected)
        # Full-reference mode preserves URI routing but widens detail after selection.
        if self.config.detail_materialization == "full_reference":
            selected = self._expand_full_references(selected)
        # Thresholding and deduplication happen before moving large K/V to the query device.
        selected = [hit for hit in selected if hit.chunk_score >= self.trigger_threshold]
        deduplicated = []
        seen = set()
        for hit in selected:
            identity = (hit.reference_uri, hit.chunk_id)
            if identity not in seen:
                seen.add(identity)
                deduplicated.append(hit)
        selected = deduplicated
        selected, budget_stats = self._budget_selection(
            selected,
            direct_tokens=direct_tokens,
            routing_candidates=routing_candidates,
        )
        if not selected:
            empty = q.new_empty((1, self.n_heads, 0, self.head_dim))
            return empty, empty, selected, 0, 0, 0.0, budget_stats

        keys = []
        values = []
        duplicate_tokens = 0
        covered_end_by_uri: dict[str, int] = {}
        # Sort source spans so overlapping fixed windows can drop repeated prefix K/V.
        for hit in sorted(selected, key=lambda item: (item.reference_uri, item.token_start, item.chunk_id)):
            if self.config.detail_materialization == "gist_only":
                gist = hit.chunk.routing_gist
                winner = hit.winning_gist_index if hit.winning_gist_index is not None else 0
                winner = min(max(int(winner), 0), max(int(gist.k.shape[0]) - 1, 0))
                key_vector = gist.k[winner]
                value_vector = gist.v[winner] if gist.v is not None else key_vector
                key = key_vector.view(1, self.n_heads, 1, self.head_dim)
                value = value_vector.view(1, self.n_heads, 1, self.head_dim)
            else:
                key = hit.chunk.token_kv.k
                value = hit.chunk.token_kv.v
                covered_end = covered_end_by_uri.get(hit.reference_uri, hit.token_start)
                overlap = max(covered_end - hit.token_start, 0)
                if overlap:
                    overlap = min(overlap, int(key.shape[2]))
                    duplicate_tokens += overlap
                    if self.config.overlap_materialization == "deduplicate":
                        key = key[:, :, overlap:, :]
                        value = value[:, :, overlap:, :]
                covered_end_by_uri[hit.reference_uri] = max(covered_end, hit.token_end)
            if key.shape[2]:
                keys.append(key)
                values.append(value)
        if not keys:
            empty = q.new_empty((1, self.n_heads, 0, self.head_dim))
            return empty, empty, selected, duplicate_tokens, 0, 0.0, budget_stats

        transfer_bytes = sum(
            (key.numel() * key.element_size()) + (value.numel() * value.element_size())
            for key, value in zip(keys, values)
            if key.device != q.device or value.device != q.device
        )
        collect_timing = self.config.collect_detailed_timing
        _synchronize_detailed_timing(q, collect_timing and transfer_bytes > 0)
        transfer_start = time.perf_counter()
        keys = [
            key.to(
                q.device,
                q.dtype,
                non_blocking=self.config.kv_cache_non_blocking,
            )
            for key in keys
        ]
        values = [
            value.to(
                q.device,
                q.dtype,
                non_blocking=self.config.kv_cache_non_blocking,
            )
            for value in values
        ]
        _synchronize_detailed_timing(q, collect_timing and transfer_bytes > 0)
        transfer_duration = time.perf_counter() - transfer_start if transfer_bytes else 0.0
        return (
            torch.cat(keys, dim=2),
            torch.cat(values, dim=2),
            selected,
            duplicate_tokens,
            transfer_bytes,
            transfer_duration,
            budget_stats,
        )

    def forward(
        self,
        x,
        use_pra_memory: bool = True,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """Compute local attention and, when enabled, routed reference attention.

        ``use_pra_memory=False`` is the causal ablation path: it skips search and
        memory fusion while leaving the same PRA block parameters/local attention.
        An empty cache follows the same fast path.
        """
        # Reset observable state so traces always describe this forward call.
        self.last_selected_chunks = [[] for _ in range(x.shape[0])]
        self.last_routing_rankings = [[] for _ in range(x.shape[0])]
        self.last_memory_batching_stats = None
        self.last_diagnostics = {}
        b, t, _ = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        if position_ids is None:
            position_ids = torch.arange(t, device=x.device)
        q, k = self.position_encoding.transform_qk(q, k, position_ids)

        # Local causal self-attention: [B,H,T,Dh] x [B,H,Dh,T] -> [B,H,T,T].
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :t, :t]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"Expected attention_mask {(b, t)}, got {tuple(attention_mask.shape)}."
                )
            valid_tokens = attention_mask.to(device=x.device, dtype=torch.bool)
            if not bool(valid_tokens.any(dim=1).all()):
                raise ValueError("Every prompt row must contain at least one direct token.")
            scores = scores.masked_fill(~valid_tokens[:, None, None, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        local_out = self.o_proj(self.merge_heads(weights @ v))

        if not use_pra_memory or self.pra_cache.is_empty():
            return local_out

        # The newest token decides what memory is relevant for each item/layer.
        # Flattening [B,H,Dh] produces the [B,D] key space used by routing gists.
        if attention_mask is None:
            routing_query = q[:, :, -1, :]
        else:
            positions = torch.arange(t, device=x.device).expand(b, -1)
            last_indices = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            gather_index = last_indices.view(b, 1, 1, 1).expand(-1, self.n_heads, 1, self.head_dim)
            routing_query = q.gather(2, gather_index).squeeze(2)
        routing_query = routing_query.contiguous().view(b, self.d_model)
        collect_timing = self.config.collect_detailed_timing
        _synchronize_detailed_timing(q, collect_timing)
        routing_start = time.perf_counter()
        selected_by_batch = self.pra_cache.search(routing_query, self.layer_id, self.config)
        if hasattr(self.pra_cache, "last_rankings"):
            self.last_routing_rankings = self.pra_cache.last_rankings(self.layer_id)
        _synchronize_detailed_timing(q, collect_timing)
        routing_duration = time.perf_counter() - routing_start

        # Materialize variable-length K/V independently to prevent cross-item leakage.
        _synchronize_detailed_timing(q, collect_timing)
        materialization_start = time.perf_counter()
        memory_k = []
        memory_v = []
        duplicate_tokens = 0
        transferred_kv_bytes = 0
        transfer_duration = 0.0
        budget_stats_by_row = []
        for batch_index, selected in enumerate(selected_by_batch):
            (
                key,
                value,
                retained,
                duplicates,
                row_transfer_bytes,
                row_transfer_duration,
                row_budget_stats,
            ) = (
                self._materialize(
                    selected,
                    q[batch_index : batch_index + 1],
                    direct_tokens=t,
                )
            )
            memory_k.append(key)
            memory_v.append(value)
            self.last_selected_chunks[batch_index] = retained
            duplicate_tokens += duplicates
            transferred_kv_bytes += row_transfer_bytes
            transfer_duration += row_transfer_duration
            budget_stats_by_row.append(row_budget_stats)
        _synchronize_detailed_timing(q, collect_timing)
        materialization_duration = time.perf_counter() - materialization_start

        if self.memory_transport == "native_kv":
            selected_lengths = [int(key.shape[2]) for key in memory_k]
            if not any(selected_lengths):
                self.last_diagnostics = {
                    key: sum(float(row[key]) for row in budget_stats_by_row)
                    for key in (
                        "direct_context_tokens",
                        "memory_budget_tokens",
                        "routing_candidates",
                        "routing_topk_candidates",
                        "chunks_materialized",
                        "chunks_budget_rejected",
                        "memory_tokens_requested",
                        "memory_tokens_materialized",
                    )
                }
                self.last_diagnostics["model_max_context_tokens"] = float(
                    self.config.effective_model_max_context_tokens
                )
                return local_out
            _synchronize_detailed_timing(q, collect_timing)
            memory_start = time.perf_counter()
            native_heads, stats = native_kv_attention(
                q,
                k,
                v,
                memory_k,
                memory_v,
                attention_mask=attention_mask,
                max_context_tokens=self.config.effective_model_max_context_tokens,
            )
            _synchronize_detailed_timing(q, collect_timing)
            memory_attention_duration = time.perf_counter() - memory_start
            self.last_memory_batching_stats = stats
            output = self.o_proj(self.merge_heads(native_heads))
            local_tokens = (
                int(attention_mask.sum().item()) if attention_mask is not None else b * t
            )
            retrieved_tokens = int(stats.valid_positions)
            retrieved_unique_tokens = (
                retrieved_tokens
                if self.config.overlap_materialization == "deduplicate"
                else max(retrieved_tokens - duplicate_tokens, 0)
            )
            selected_stored_tokens = (
                retrieved_tokens + duplicate_tokens
                if self.config.overlap_materialization == "deduplicate"
                else retrieved_tokens
            )
            accessible_tokens = local_tokens + retrieved_tokens
            element_size = q.element_size()
            kv_bytes = 2 * retrieved_tokens * self.n_heads * self.head_dim * element_size
            output_norm = float(output.detach().norm().cpu())
            local_norm = float(local_out.detach().norm().cpu())
            self.last_diagnostics = {
                **stats.as_metrics(),
                "model_max_context_tokens": float(
                    self.config.effective_model_max_context_tokens
                ),
                "direct_context_tokens": float(t * b),
                "memory_budget_tokens": sum(
                    row["memory_budget_tokens"] for row in budget_stats_by_row
                ),
                "routing_candidates": sum(
                    row["routing_candidates"] for row in budget_stats_by_row
                ),
                "routing_topk_candidates": sum(
                    row["routing_topk_candidates"] for row in budget_stats_by_row
                ),
                "chunks_materialized": sum(
                    row["chunks_materialized"] for row in budget_stats_by_row
                ),
                "chunks_budget_rejected": sum(
                    row["chunks_budget_rejected"] for row in budget_stats_by_row
                ),
                "memory_tokens_requested": sum(
                    row["memory_tokens_requested"] for row in budget_stats_by_row
                ),
                "memory_tokens_materialized": sum(
                    row["memory_tokens_materialized"] for row in budget_stats_by_row
                ),
                "materialization_budget_utilization": sum(
                    row["materialization_budget_utilization"]
                    for row in budget_stats_by_row
                )
                / max(len(budget_stats_by_row), 1),
                "lowest_materialized_score": min(
                    (
                        row["lowest_materialized_score"]
                        for row in budget_stats_by_row
                        if math.isfinite(row["lowest_materialized_score"])
                    ),
                    default=float("nan"),
                ),
                "highest_budget_rejected_score": max(
                    (
                        row["highest_budget_rejected_score"]
                        for row in budget_stats_by_row
                        if math.isfinite(row["highest_budget_rejected_score"])
                    ),
                    default=float("nan"),
                ),
                "memory_duplicate_chunk_tokens": float(duplicate_tokens),
                "memory_transport_native_kv": 1.0,
                "active_local_tokens": float(local_tokens),
                "retrieved_token_kv": float(retrieved_tokens),
                "retrieved_physical_kv_tokens": float(retrieved_tokens),
                "retrieved_unique_source_tokens": float(retrieved_unique_tokens),
                "selected_stored_kv_tokens": float(selected_stored_tokens),
                "memory_overlap_tokens_detected": float(duplicate_tokens),
                "memory_overlap_tokens_removed": float(
                    duplicate_tokens
                    if self.config.overlap_materialization == "deduplicate"
                    else 0
                ),
                "accessible_kv_tokens": float(accessible_tokens),
                "active_memory_fraction": retrieved_tokens / max(accessible_tokens, 1),
                "retrieved_kv_storage_bytes": float(kv_bytes),
                "retrieved_kv_transfer_bytes": float(transferred_kv_bytes),
                "attention_output_norm": output_norm,
                "local_output_norm": local_norm,
                "attention_to_local_output_norm_ratio": output_norm / max(local_norm, 1e-12),
            }
            if self.config.collect_detailed_timing:
                self.last_diagnostics.update(
                    {
                        "routing_duration_seconds": routing_duration,
                        "materialization_duration_seconds": materialization_duration,
                        "selected_kv_transfer_duration_seconds": transfer_duration,
                        "memory_attention_duration_seconds": memory_attention_duration,
                    }
                )
            return output

        # Legacy adapted transport: normalize memory separately, then add its projection.
        # Bucket only compatible batch rows, pad within each bucket, and restore order.
        _synchronize_detailed_timing(q, collect_timing)
        memory_start = time.perf_counter()
        memory_heads, stats = dynamic_memory_attention(
            q,
            memory_k,
            memory_v,
            bucket_count=self.config.memory_bucket_count,
            bucket_strategy=self.config.memory_bucket_strategy,
        )
        _synchronize_detailed_timing(q, collect_timing)
        memory_attention_duration = time.perf_counter() - memory_start
        self.last_memory_batching_stats = stats
        assert self.mem_o_proj is not None
        memory_out = self.mem_o_proj(self.merge_heads(memory_heads))
        # Empty rows receive exactly zero memory output despite rectangular batching.
        has_memory = torch.tensor(
            [length > 0 for length in stats.selected_lengths],
            device=memory_out.device,
            dtype=memory_out.dtype,
        ).view(b, 1, 1)
        memory_out = memory_out * has_memory
        memory_norm = float(memory_out.detach().norm().cpu())
        local_norm = float(local_out.detach().norm().cpu())
        self.last_diagnostics = {
            **stats.as_metrics(),
            "model_max_context_tokens": float(
                self.config.effective_model_max_context_tokens
            ),
            "direct_context_tokens": float(t * b),
            "memory_budget_tokens": sum(
                row["memory_budget_tokens"] for row in budget_stats_by_row
            ),
            "routing_candidates": sum(
                row["routing_candidates"] for row in budget_stats_by_row
            ),
            "routing_topk_candidates": sum(
                row["routing_topk_candidates"] for row in budget_stats_by_row
            ),
            "chunks_materialized": sum(
                row["chunks_materialized"] for row in budget_stats_by_row
            ),
            "chunks_budget_rejected": sum(
                row["chunks_budget_rejected"] for row in budget_stats_by_row
            ),
            "memory_tokens_requested": sum(
                row["memory_tokens_requested"] for row in budget_stats_by_row
            ),
            "memory_tokens_materialized": sum(
                row["memory_tokens_materialized"] for row in budget_stats_by_row
            ),
            "materialization_budget_utilization": sum(
                row["materialization_budget_utilization"]
                for row in budget_stats_by_row
            )
            / max(len(budget_stats_by_row), 1),
            "memory_duplicate_chunk_tokens": float(duplicate_tokens),
            "retrieved_kv_transfer_bytes": float(transferred_kv_bytes),
            "memory_output_norm": memory_norm,
            "local_output_norm": local_norm,
            "memory_to_local_output_norm_ratio": memory_norm / max(local_norm, 1e-12),
        }
        if self.config.collect_detailed_timing:
            self.last_diagnostics.update(
                {
                    "routing_duration_seconds": routing_duration,
                    "materialization_duration_seconds": materialization_duration,
                    "selected_kv_transfer_duration_seconds": transfer_duration,
                    "memory_attention_duration_seconds": memory_attention_duration,
                }
            )
        # The enclosing block adds this combined branch through its residual connection.
        return local_out + self.memory_alpha * memory_out
