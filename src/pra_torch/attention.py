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
        self.last_selected_chunks: list[list[SelectedChunk]] = []  # Latest trace by batch row.
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

    def project_kv(self, hidden_states, *, detach: bool = True) -> LayerKV:
        """Project an independently encoded chunk into this layer's cache space.

        ``hidden_states`` is ``[1,M,D]`` and returned K/V is ``[1,H,M,Dh]``.
        Detached mode creates ordinary reusable inference memory; trainable-gist
        mode preserves the graph so routing representations can receive gradients.
        """
        k = self.split_heads(self.k_proj(hidden_states))
        v = self.split_heads(self.v_proj(hidden_states))
        if detach:
            k = k.detach()
            v = v.detach()
        return LayerKV(k=k, v=v)

    def forward_native_kv(
        self,
        x: torch.Tensor,
        memory_k: list[torch.Tensor],
        memory_v: list[torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
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
        heads, _stats = native_kv_attention(
            q,
            k,
            v,
            memory_k,
            memory_v,
            attention_mask=attention_mask,
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

    def _materialize(self, selected: list[SelectedChunk], q: torch.Tensor):
        """Turn routed hits into one item's rectangular memory K/V.

        Returns K/V shaped ``[1,H,M,Dh]``, retained trace records, and the number
        of overlapping token positions removed. ``M`` varies by batch item and is
        one per chunk in ``gist_only`` mode, selected detail in the default mode,
        or all chunks of each routed URI in ``full_reference`` mode.
        """
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
        if not selected:
            empty = q.new_empty((1, self.n_heads, 0, self.head_dim))
            return empty, empty, selected, 0

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
                    key = key[:, :, overlap:, :]
                    value = value[:, :, overlap:, :]
                covered_end_by_uri[hit.reference_uri] = max(covered_end, hit.token_end)
            if key.shape[2]:
                keys.append(key.to(q.device, q.dtype))
                values.append(value.to(q.device, q.dtype))
        if not keys:
            empty = q.new_empty((1, self.n_heads, 0, self.head_dim))
            return empty, empty, selected, duplicate_tokens
        return torch.cat(keys, dim=2), torch.cat(values, dim=2), selected, duplicate_tokens

    def forward(self, x, use_pra_memory: bool = True, attention_mask: torch.Tensor | None = None):
        """Compute local attention and, when enabled, routed reference attention.

        ``use_pra_memory=False`` is the causal ablation path: it skips search and
        memory fusion while leaving the same PRA block parameters/local attention.
        An empty cache follows the same fast path.
        """
        # Reset observable state so traces always describe this forward call.
        self.last_selected_chunks = [[] for _ in range(x.shape[0])]
        self.last_memory_batching_stats = None
        self.last_diagnostics = {}
        b, t, _ = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

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
        routing_start = time.perf_counter()
        selected_by_batch = self.pra_cache.search(routing_query, self.layer_id, self.config)
        routing_duration = time.perf_counter() - routing_start

        # Materialize variable-length K/V independently to prevent cross-item leakage.
        materialization_start = time.perf_counter()
        memory_k = []
        memory_v = []
        duplicate_tokens = 0
        for batch_index, selected in enumerate(selected_by_batch):
            key, value, retained, duplicates = self._materialize(selected, q[batch_index : batch_index + 1])
            memory_k.append(key)
            memory_v.append(value)
            self.last_selected_chunks[batch_index] = retained
            duplicate_tokens += duplicates
        materialization_duration = time.perf_counter() - materialization_start

        if self.memory_transport == "native_kv":
            selected_lengths = [int(key.shape[2]) for key in memory_k]
            if not any(selected_lengths):
                return local_out
            memory_start = time.perf_counter()
            native_heads, stats = native_kv_attention(
                q,
                k,
                v,
                memory_k,
                memory_v,
                attention_mask=attention_mask,
            )
            memory_attention_duration = time.perf_counter() - memory_start
            self.last_memory_batching_stats = stats
            output = self.o_proj(self.merge_heads(native_heads))
            local_tokens = (
                int(attention_mask.sum().item()) if attention_mask is not None else b * t
            )
            retrieved_tokens = int(stats.valid_positions)
            accessible_tokens = local_tokens + retrieved_tokens
            element_size = q.element_size()
            kv_bytes = 2 * retrieved_tokens * self.n_heads * self.head_dim * element_size
            output_norm = float(output.detach().norm().cpu())
            local_norm = float(local_out.detach().norm().cpu())
            self.last_diagnostics = {
                **stats.as_metrics(),
                "memory_duplicate_chunk_tokens": float(duplicate_tokens),
                "memory_transport_native_kv": 1.0,
                "active_local_tokens": float(local_tokens),
                "retrieved_token_kv": float(retrieved_tokens),
                "accessible_kv_tokens": float(accessible_tokens),
                "active_memory_fraction": retrieved_tokens / max(accessible_tokens, 1),
                "retrieved_kv_storage_bytes": float(kv_bytes),
                "retrieved_kv_transfer_bytes": float(kv_bytes),
                "attention_output_norm": output_norm,
                "local_output_norm": local_norm,
                "attention_to_local_output_norm_ratio": output_norm / max(local_norm, 1e-12),
            }
            if self.config.collect_detailed_timing:
                self.last_diagnostics.update(
                    {
                        "routing_duration_seconds": routing_duration,
                        "materialization_duration_seconds": materialization_duration,
                        "memory_attention_duration_seconds": memory_attention_duration,
                    }
                )
            return output

        # Legacy adapted transport: normalize memory separately, then add its projection.
        # Bucket only compatible batch rows, pad within each bucket, and restore order.
        memory_start = time.perf_counter()
        memory_heads, stats = dynamic_memory_attention(
            q,
            memory_k,
            memory_v,
            bucket_count=self.config.memory_bucket_count,
            bucket_strategy=self.config.memory_bucket_strategy,
        )
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
            "memory_duplicate_chunk_tokens": float(duplicate_tokens),
            "memory_output_norm": memory_norm,
            "local_output_norm": local_norm,
            "memory_to_local_output_norm_ratio": memory_norm / max(local_norm, 1e-12),
        }
        if self.config.collect_detailed_timing:
            self.last_diagnostics.update(
                {
                    "routing_duration_seconds": routing_duration,
                    "materialization_duration_seconds": materialization_duration,
                    "memory_attention_duration_seconds": memory_attention_duration,
                }
            )
        # The enclosing block adds this combined branch through its residual connection.
        return local_out + self.memory_alpha * memory_out
