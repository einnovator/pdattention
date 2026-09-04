"""Model-family-independent PRA routing, budgeting, and native-K/V execution."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .memory import PRAMemoryCache, SelectedChunk
from .memory_batching import MemoryBatchingStats, native_kv_attention
from .materialization import (
    LogicalInterval,
    gather_logical_kv,
    logical_domain,
    shards_from_chunks,
)


def synchronize_detailed_timing(tensor: torch.Tensor, enabled: bool) -> None:
    """Fence CUDA only when phase-level research timing is explicitly enabled."""
    if enabled and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


@dataclass
class PreparedPRAMemory:
    """Variable-length native K/V selected for one attention invocation."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    selections: list[list[SelectedChunk]]
    rankings: list[list[dict]]
    budget_stats: list[dict[str, float]]
    duplicate_tokens: int = 0
    transferred_kv_bytes: int = 0
    routing_duration_seconds: float = 0.0
    materialization_duration_seconds: float = 0.0
    transfer_duration_seconds: float = 0.0

    @property
    def selected_lengths(self) -> tuple[int, ...]:
        """Return physical native-K/V tokens materialized for each batch row."""
        return tuple(int(key.shape[2]) for key in self.keys)

    @property
    def has_memory(self) -> bool:
        """Report whether any row selected at least one native-K/V token."""
        return any(self.selected_lengths)


@dataclass(frozen=True)
class CombinedNativeKV:
    """Rectangular native K/V and additive mask ready for an HF attention kernel."""

    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    memory_width: int


class PRAExecutionCore:
    """Shared PRA mechanism beneath controlled and Hugging Face attention modules.

    Family adapters provide native Q/K/V and positional semantics. This class
    owns query preparation, exact cache search, whole-chunk budgeting,
    provenance-preserving materialization, variable-row packing, and metrics.
    K/V remain in native KV-head form ``[B,H_kv,T,Dh]`` throughout storage.
    """

    def __init__(
        self,
        *,
        cache: PRAMemoryCache,
        config,
        layer_id: int,
        num_query_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ) -> None:
        if num_query_heads % num_key_value_heads:
            raise ValueError("Query heads must be divisible by K/V heads.")
        self.cache = cache
        self.config = config
        self.layer_id = int(layer_id)
        self.num_query_heads = int(num_query_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.trigger_threshold = float(config.trigger_threshold)

    @property
    def routing_width(self) -> int:
        """Width of flattened native-KV routing gists."""
        return self.num_key_value_heads * self.head_dim

    def prepare_pra_query(
        self,
        query_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce the newest ``[B,H_q,T,Dh]`` query to ``[B,H_kv*Dh]``.

        GQA query heads that share one K/V head are averaged only for routing.
        The original query heads remain unchanged for native attention.
        """
        if query_states.ndim != 4:
            raise ValueError("PRA queries must have shape [batch,query_heads,tokens,head_dim].")
        batch, heads, tokens, width = query_states.shape
        if heads != self.num_query_heads or width != self.head_dim:
            raise ValueError("PRA query shape does not match the adapter head layout.")
        if attention_mask is None or attention_mask.ndim != 2:
            newest = query_states[:, :, -1, :]
        else:
            if attention_mask.shape != (batch, tokens):
                raise ValueError("A routing attention mask must match [batch,current_tokens].")
            valid = attention_mask.to(device=query_states.device, dtype=torch.bool)
            if not bool(valid.any(dim=1).all()):
                raise ValueError("Every batch row needs at least one routing token.")
            positions = torch.arange(tokens, device=query_states.device).expand(batch, -1)
            last = positions.masked_fill(~valid, -1).max(dim=1).values
            index = last.view(batch, 1, 1, 1).expand(-1, heads, 1, width)
            newest = query_states.gather(2, index).squeeze(2)
        groups = self.num_query_heads // self.num_key_value_heads
        newest = newest.view(batch, self.num_key_value_heads, groups, width).mean(dim=2)
        return newest.contiguous().view(batch, self.routing_width)

    def route_memory(self, routing_query: torch.Tensor) -> tuple[list[list[SelectedChunk]], list[list[dict]]]:
        """Run the configured exact router for this layer and return full rankings."""
        selected = self.cache.search(routing_query, self.layer_id, self.config)
        rankings = (
            self.cache.last_rankings(self.layer_id)
            if hasattr(self.cache, "last_rankings")
            else [[] for _ in range(routing_query.shape[0])]
        )
        return selected, rankings

    def _expand_full_references(self, selected: list[SelectedChunk]) -> list[SelectedChunk]:
        """Expand one routed hit into every chunk from each selected URI."""
        expanded: list[SelectedChunk] = []
        seen_references: set[str] = set()
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

    def budget_selected_memory(
        self,
        selected: list[SelectedChunk],
        *,
        direct_tokens: int,
        routing_candidates: int,
    ) -> tuple[list[SelectedChunk], dict[str, float]]:
        """Retain highest-scoring whole chunks under the hard native-context cap."""
        hard_remaining = (
            self.config.effective_model_max_context_tokens
            - int(direct_tokens)
            - self.config.context_safety_reserve_tokens
        )
        if hard_remaining < 0:
            raise ValueError("Direct context and safety reserve exceed model_max_context_tokens.")
        memory_budget = hard_remaining
        if self.config.max_materialized_memory_tokens is not None:
            memory_budget = min(memory_budget, self.config.max_materialized_memory_tokens)
        ranked = sorted(selected, key=lambda hit: (-hit.chunk_score, hit.reference_uri, hit.chunk_id))
        requested = sum(
            1 if self.config.detail_materialization == "gist_only" else hit.selected_token_count
            for hit in ranked
        )
        retained: list[SelectedChunk] = []
        rejected: list[SelectedChunk] = []
        remaining = int(memory_budget)
        materialized = 0
        for hit in ranked:
            cost = 1 if self.config.detail_materialization == "gist_only" else hit.selected_token_count
            if cost <= remaining:
                retained.append(hit)
                remaining -= cost
                materialized += cost
            else:
                rejected.append(hit)
        return retained, {
            "model_max_context_tokens": float(self.config.effective_model_max_context_tokens),
            "direct_context_tokens": float(direct_tokens),
            "memory_budget_tokens": float(memory_budget),
            "routing_candidates": float(routing_candidates),
            "routing_topk_candidates": float(len(selected)),
            "chunks_materialized": float(len(retained)),
            "chunks_budget_rejected": float(len(rejected)),
            "memory_tokens_requested": float(requested),
            "memory_tokens_materialized": float(materialized),
            "materialization_budget_utilization": materialized / max(memory_budget, 1),
            "lowest_materialized_score": min((hit.chunk_score for hit in retained), default=float("nan")),
            "highest_budget_rejected_score": max((hit.chunk_score for hit in rejected), default=float("nan")),
        }

    def _logical_intervals(self, selected: list[SelectedChunk]) -> list[LogicalInterval]:
        """Parse explicit source-relative disclosure requests from fixed selections."""
        intervals: list[LogicalInterval] = []
        for hit in selected:
            for value in hit.metadata.get("materialization_intervals", ()):
                if isinstance(value, LogicalInterval):
                    intervals.append(value)
                    continue
                if not isinstance(value, dict):
                    raise TypeError("Materialization intervals must be mappings or LogicalInterval objects.")
                domain = logical_domain(
                    str(value.get("domain", value.get("source_uri", hit.source_uri)))
                )
                intervals.append(
                    LogicalInterval(
                        domain=domain,
                        start=int(value["start"]),
                        end=int(value["end"]),
                        evidence_start=(
                            int(value["evidence_start"])
                            if value.get("evidence_start") is not None
                            else None
                        ),
                        evidence_end=(
                            int(value["evidence_end"])
                            if value.get("evidence_end") is not None
                            else None
                        ),
                        score=float(value.get("score", hit.chunk_score)),
                    )
                )
        if not intervals:
            raise ValueError(
                "Logical interval materialization requires explicit "
                "materialization_intervals on at least one selected hit."
            )
        return intervals

    def _materialize_logical_intervals(
        self,
        selected: list[SelectedChunk],
        query: torch.Tensor,
        *,
        direct_tokens: int,
        routing_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[SelectedChunk], int, int, float, dict[str, float]]:
        """Gather exact logical windows from canonical contextual K/V shards."""
        intervals = self._logical_intervals(selected)
        chunks = []
        seen_shards: set[tuple[str, str]] = set()
        for hit in selected:
            layer_memory = hit.entry.layer_memory.get(self.layer_id)
            if layer_memory is None:
                continue
            for chunk in layer_memory.chunks:
                identity = (chunk.source_uri, chunk.chunk_id)
                if identity not in seen_shards:
                    seen_shards.add(identity)
                    chunks.append(chunk)
        gathered = gather_logical_kv(
            shards_from_chunks(chunks),
            intervals,
            device=query.device,
            dtype=query.dtype,
        )

        key = gathered.key
        value = gathered.value
        gist_tokens = 0
        gist_transfer_bytes = 0
        gist_transfer_seconds = 0.0
        if self.config.detail_materialization == "gist_plus_logical_intervals":
            # These are physical native-K/V means, not routing-space gists. That
            # distinction matters for GQA models whose routing width is hidden_size.
            gist_keys = [hit.chunk.token_kv.k.mean(dim=2, keepdim=True) for hit in selected]
            gist_values = [hit.chunk.token_kv.v.mean(dim=2, keepdim=True) for hit in selected]
            gist_tokens = len(gist_keys)
            gist_transfer_bytes = sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in zip(gist_keys, gist_values)
                if k.device != query.device or v.device != query.device
            )
            synchronize_detailed_timing(query, self.config.collect_detailed_timing and gist_transfer_bytes > 0)
            transfer_started = time.perf_counter()
            gist_keys = [k.to(query.device, query.dtype) for k in gist_keys]
            gist_values = [v.to(query.device, query.dtype) for v in gist_values]
            synchronize_detailed_timing(query, self.config.collect_detailed_timing and gist_transfer_bytes > 0)
            if gist_transfer_bytes:
                gist_transfer_seconds = time.perf_counter() - transfer_started
            key = torch.cat((*gist_keys, key), dim=2)
            value = torch.cat((*gist_values, value), dim=2)

        hard_remaining = (
            self.config.effective_model_max_context_tokens
            - int(direct_tokens)
            - self.config.context_safety_reserve_tokens
        )
        if hard_remaining < 0:
            raise ValueError("Direct context and safety reserve exceed model_max_context_tokens.")
        memory_budget = hard_remaining
        if self.config.max_materialized_memory_tokens is not None:
            memory_budget = min(memory_budget, self.config.max_materialized_memory_tokens)
        materialized = int(key.shape[2])
        if materialized > memory_budget:
            raise ValueError(
                f"Logical materialization requested {materialized} native K/V tokens "
                f"but only {memory_budget} are available. Allocate intervals before gather."
            )
        stats = gathered.stats
        budget_stats = {
            "model_max_context_tokens": float(self.config.effective_model_max_context_tokens),
            "direct_context_tokens": float(direct_tokens),
            "memory_budget_tokens": float(memory_budget),
            "routing_candidates": float(routing_candidates),
            "routing_topk_candidates": float(len(selected)),
            "chunks_materialized": float(stats.storage_shards_touched),
            "chunks_budget_rejected": 0.0,
            "memory_tokens_requested": float(stats.requested_tokens_pre_dedup + gist_tokens),
            "memory_tokens_materialized": float(materialized),
            "materialization_budget_utilization": materialized / max(memory_budget, 1),
            "lowest_materialized_score": min((hit.chunk_score for hit in selected), default=float("nan")),
            "highest_budget_rejected_score": float("nan"),
            "requested_tokens_pre_dedup": float(stats.requested_tokens_pre_dedup),
            "deduplicated_materialization_tokens": float(stats.deduplicated_tokens),
            "materialized_native_kv_tokens": float(materialized),
            "materialized_native_kv_bytes": float(
                key.numel() * key.element_size() + value.numel() * value.element_size()
            ),
            "materialized_gist_kv_tokens": float(gist_tokens),
            "evidence_kv_tokens": float(stats.evidence_tokens),
            "non_evidence_kv_tokens": float(stats.non_evidence_tokens + gist_tokens),
            "evidence_density": stats.evidence_tokens / max(materialized, 1),
            "storage_shards_touched": float(stats.storage_shards_touched),
            "cross_shard_interval_count": float(stats.cross_shard_interval_count),
            "interval_resolution_seconds": stats.interval_resolution_seconds,
            "logical_gather_seconds": stats.gather_seconds,
            "logical_h2d_seconds": stats.h2d_seconds + gist_transfer_seconds,
        }
        return (
            key,
            value,
            selected,
            stats.requested_tokens_pre_dedup - stats.deduplicated_tokens,
            stats.transferred_kv_bytes + gist_transfer_bytes,
            stats.h2d_seconds + gist_transfer_seconds,
            budget_stats,
        )

    def _materialize_native_gists(
        self,
        selected: list[SelectedChunk],
        query: torch.Tensor,
        *,
        direct_tokens: int,
        routing_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[SelectedChunk], int, int, float, dict[str, float]]:
        """Expose one native-head K/V mean for each selected conceptual chunk."""
        empty = query.new_empty((1, self.num_key_value_heads, 0, self.head_dim))
        if not selected:
            return empty, empty, selected, 0, 0, 0.0, {
                "model_max_context_tokens": float(self.config.effective_model_max_context_tokens),
                "direct_context_tokens": float(direct_tokens),
                "memory_budget_tokens": 0.0,
                "routing_candidates": float(routing_candidates),
                "routing_topk_candidates": 0.0,
                "chunks_materialized": 0.0,
                "chunks_budget_rejected": 0.0,
                "memory_tokens_requested": 0.0,
                "memory_tokens_materialized": 0.0,
                "materialization_budget_utilization": 0.0,
                "lowest_materialized_score": float("nan"),
                "highest_budget_rejected_score": float("nan"),
            }
        hard_remaining = (
            self.config.effective_model_max_context_tokens
            - int(direct_tokens)
            - self.config.context_safety_reserve_tokens
        )
        memory_budget = max(hard_remaining, 0)
        if self.config.max_materialized_memory_tokens is not None:
            memory_budget = min(memory_budget, self.config.max_materialized_memory_tokens)
        if len(selected) > memory_budget:
            raise ValueError("Native gist materialization exceeds the available K/V budget.")
        keys = [hit.chunk.token_kv.k.mean(dim=2, keepdim=True) for hit in selected]
        values = [hit.chunk.token_kv.v.mean(dim=2, keepdim=True) for hit in selected]
        transfer_bytes = sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in zip(keys, values)
            if key.device != query.device or value.device != query.device
        )
        synchronize_detailed_timing(query, self.config.collect_detailed_timing and transfer_bytes > 0)
        started = time.perf_counter()
        keys = [key.to(query.device, query.dtype) for key in keys]
        values = [value.to(query.device, query.dtype) for value in values]
        synchronize_detailed_timing(query, self.config.collect_detailed_timing and transfer_bytes > 0)
        duration = time.perf_counter() - started if transfer_bytes else 0.0
        tokens = len(selected)
        bytes_used = sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in zip(keys, values)
        )
        stats = {
            "model_max_context_tokens": float(self.config.effective_model_max_context_tokens),
            "direct_context_tokens": float(direct_tokens),
            "memory_budget_tokens": float(memory_budget),
            "routing_candidates": float(routing_candidates),
            "routing_topk_candidates": float(tokens),
            "chunks_materialized": float(tokens),
            "chunks_budget_rejected": 0.0,
            "memory_tokens_requested": float(tokens),
            "memory_tokens_materialized": float(tokens),
            "materialization_budget_utilization": tokens / max(memory_budget, 1),
            "lowest_materialized_score": min(hit.chunk_score for hit in selected),
            "highest_budget_rejected_score": float("nan"),
            "requested_tokens_pre_dedup": float(tokens),
            "deduplicated_materialization_tokens": float(tokens),
            "materialized_native_kv_tokens": float(tokens),
            "materialized_native_kv_bytes": float(bytes_used),
            "materialized_gist_kv_tokens": float(tokens),
            "evidence_kv_tokens": 0.0,
            "non_evidence_kv_tokens": float(tokens),
            "evidence_density": 0.0,
            "storage_shards_touched": float(tokens),
            "cross_shard_interval_count": 0.0,
            "interval_resolution_seconds": 0.0,
            "logical_gather_seconds": 0.0,
            "logical_h2d_seconds": duration,
        }
        return (
            torch.cat(keys, dim=2),
            torch.cat(values, dim=2),
            selected,
            0,
            transfer_bytes,
            duration,
            stats,
        )

    def materialize_selected_kv(
        self,
        selected: list[SelectedChunk],
        query: torch.Tensor,
        *,
        direct_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[SelectedChunk], int, int, float, dict[str, float]]:
        """Materialize one row's selected K/V as ``[1,H_kv,M,Dh]`` tensors."""
        routing_candidates = len(selected)
        if self.config.detail_materialization == "full_reference":
            selected = self._expand_full_references(selected)
        selected = [hit for hit in selected if hit.chunk_score >= self.trigger_threshold]
        deduplicated: list[SelectedChunk] = []
        seen: set[tuple[str, str]] = set()
        for hit in selected:
            identity = (hit.reference_uri, hit.chunk_id)
            if identity not in seen:
                seen.add(identity)
                deduplicated.append(hit)
        if not deduplicated:
            empty = query.new_empty((1, self.num_key_value_heads, 0, self.head_dim))
            _retained, budget_stats = self.budget_selected_memory(
                [],
                direct_tokens=direct_tokens,
                routing_candidates=routing_candidates,
            )
            return empty, empty, [], 0, 0, 0.0, budget_stats
        if self.config.detail_materialization == "native_gist_only":
            return self._materialize_native_gists(
                deduplicated,
                query,
                direct_tokens=direct_tokens,
                routing_candidates=routing_candidates,
            )
        if self.config.detail_materialization in {
            "logical_intervals",
            "gist_plus_logical_intervals",
        }:
            return self._materialize_logical_intervals(
                deduplicated,
                query,
                direct_tokens=direct_tokens,
                routing_candidates=routing_candidates,
            )
        selected, budget_stats = self.budget_selected_memory(
            deduplicated,
            direct_tokens=direct_tokens,
            routing_candidates=routing_candidates,
        )
        empty = query.new_empty((1, self.num_key_value_heads, 0, self.head_dim))
        if not selected:
            return empty, empty, selected, 0, 0, 0.0, budget_stats

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        duplicate_tokens = 0
        covered_end_by_uri: dict[str, int] = {}
        for hit in sorted(selected, key=lambda item: (item.reference_uri, item.token_start, item.chunk_id)):
            if self.config.detail_materialization == "gist_only":
                gist = hit.chunk.routing_gist
                winner = min(max(int(hit.winning_gist_index or 0), 0), max(int(gist.k.shape[0]) - 1, 0))
                key_vector = gist.k[winner]
                value_vector = gist.v[winner] if gist.v is not None else key_vector
                key = key_vector.view(1, self.num_key_value_heads, 1, self.head_dim)
                value = value_vector.view(1, self.num_key_value_heads, 1, self.head_dim)
            else:
                key = hit.chunk.token_kv.k
                value = hit.chunk.token_kv.v
                covered_end = covered_end_by_uri.get(hit.reference_uri, hit.token_start)
                overlap = min(max(covered_end - hit.token_start, 0), int(key.shape[2]))
                if overlap:
                    duplicate_tokens += overlap
                    if self.config.overlap_materialization == "deduplicate":
                        key = key[:, :, overlap:, :]
                        value = value[:, :, overlap:, :]
                covered_end_by_uri[hit.reference_uri] = max(covered_end, hit.token_end)
            if key.shape[1] != self.num_key_value_heads or key.shape[3] != self.head_dim:
                raise ValueError("Cached memory does not match this layer's native K/V layout.")
            if key.shape[2]:
                keys.append(key)
                values.append(value)
        if not keys:
            return empty, empty, selected, duplicate_tokens, 0, 0.0, budget_stats

        transfer_bytes = sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in zip(keys, values)
            if key.device != query.device or value.device != query.device
        )
        detailed = self.config.collect_detailed_timing
        synchronize_detailed_timing(query, detailed and transfer_bytes > 0)
        started = time.perf_counter()
        keys = [key.to(query.device, query.dtype, non_blocking=self.config.kv_cache_non_blocking) for key in keys]
        values = [value.to(query.device, query.dtype, non_blocking=self.config.kv_cache_non_blocking) for value in values]
        synchronize_detailed_timing(query, detailed and transfer_bytes > 0)
        duration = time.perf_counter() - started if transfer_bytes else 0.0
        return (
            torch.cat(keys, dim=2),
            torch.cat(values, dim=2),
            selected,
            duplicate_tokens,
            transfer_bytes,
            duration,
            budget_stats,
        )

    def prepare_memory(
        self,
        query_states: torch.Tensor,
        *,
        direct_tokens: int,
        routing_attention_mask: torch.Tensor | None = None,
        routing_query_states: torch.Tensor | None = None,
    ) -> PreparedPRAMemory:
        """Route and materialize one batch while preserving row isolation.

        Family adapters may supply matched routing features independently of
        the post-position query used by native attention. Rank-four features
        retain the standard GQA reduction; rank-two features are already
        pooled routing queries such as a final-token hidden state.
        """
        batch = int(query_states.shape[0])
        if self.cache.is_empty():
            empty = query_states.new_empty((1, self.num_key_value_heads, 0, self.head_dim))
            return PreparedPRAMemory(
                keys=[empty for _ in range(batch)],
                values=[empty for _ in range(batch)],
                selections=[[] for _ in range(batch)],
                rankings=[[] for _ in range(batch)],
                budget_stats=[],
            )
        detailed = self.config.collect_detailed_timing
        synchronize_detailed_timing(query_states, detailed)
        started = time.perf_counter()
        routing_source = query_states if routing_query_states is None else routing_query_states
        if routing_source.ndim == 4:
            routing_query = self.prepare_pra_query(routing_source, routing_attention_mask)
        elif routing_source.ndim == 2:
            if routing_source.shape[0] != batch:
                raise ValueError("Prepared routing queries must match the attention batch.")
            routing_query = routing_source
        else:
            raise ValueError("Routing query states must be [B,H,T,Dh] or prepared [B,D].")
        selected_by_batch, rankings = self.route_memory(routing_query)
        synchronize_detailed_timing(query_states, detailed)
        routing_duration = time.perf_counter() - started

        return self.prepare_selected_memory(
            query_states,
            selected_by_batch,
            direct_tokens=direct_tokens,
            rankings=rankings,
            routing_duration_seconds=routing_duration,
        )

    def prepare_selected_memory(
        self,
        query_states: torch.Tensor,
        selected_by_batch: list[list[SelectedChunk]],
        *,
        direct_tokens: int,
        rankings: list[list[dict]] | None = None,
        routing_duration_seconds: float = 0.0,
    ) -> PreparedPRAMemory:
        """Materialize fixed identities independently of how they were routed.

        This method is the boundary between semantic selection and native
        attention payload replay. Different routers and oracle controls can
        supply the same ``SelectedChunk`` set; budgeting and post-RoPE K/V
        handling from this point onward are then exactly shared.
        """
        batch = int(query_states.shape[0])
        if len(selected_by_batch) != batch:
            raise ValueError("Fixed PRA selections must match the attention batch.")
        if rankings is None:
            rankings = [[] for _ in range(batch)]
        if len(rankings) != batch:
            raise ValueError("Fixed PRA rankings must match the attention batch.")
        detailed = self.config.collect_detailed_timing
        synchronize_detailed_timing(query_states, detailed)
        started = time.perf_counter()
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        retained_by_batch: list[list[SelectedChunk]] = []
        budget_stats: list[dict[str, float]] = []
        duplicate_tokens = transferred_bytes = 0
        transfer_duration = 0.0
        for row, selected in enumerate(selected_by_batch):
            key, value, retained, duplicates, moved, moved_seconds, row_budget = self.materialize_selected_kv(
                selected,
                query_states[row : row + 1],
                direct_tokens=direct_tokens,
            )
            keys.append(key)
            values.append(value)
            retained_by_batch.append(retained)
            budget_stats.append(row_budget)
            duplicate_tokens += duplicates
            transferred_bytes += moved
            transfer_duration += moved_seconds
        synchronize_detailed_timing(query_states, detailed)
        return PreparedPRAMemory(
            keys=keys,
            values=values,
            selections=retained_by_batch,
            rankings=rankings,
            budget_stats=budget_stats,
            duplicate_tokens=duplicate_tokens,
            transferred_kv_bytes=transferred_bytes,
            routing_duration_seconds=float(routing_duration_seconds),
            materialization_duration_seconds=time.perf_counter() - started,
            transfer_duration_seconds=transfer_duration,
        )

    def combine_local_and_memory_kv(
        self,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        prepared: PreparedPRAMemory,
        attention_mask: torch.Tensor | None,
        *,
        query_tokens: int,
    ) -> CombinedNativeKV:
        """Prepend variable selected memory to native local K/V for an HF kernel."""
        if local_key.ndim != 4 or local_value.shape != local_key.shape:
            raise ValueError("HF local K/V must have matching [B,H_kv,T,Dh] shapes.")
        batch, kv_heads, local_tokens, width = local_key.shape
        if kv_heads != self.num_key_value_heads or width != self.head_dim:
            raise ValueError("HF local K/V does not match the registered adapter layout.")
        if len(prepared.keys) != batch:
            raise ValueError("Prepared PRA rows must match the local K/V batch.")
        max_memory = max(prepared.selected_lengths, default=0)
        if max_memory == 0:
            return CombinedNativeKV(local_key, local_value, attention_mask, 0)
        if local_tokens + max_memory > self.config.effective_model_max_context_tokens:
            raise ValueError("Combined native K/V exceeds model_max_context_tokens.")
        padded_keys: list[torch.Tensor] = []
        padded_values: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        for key, value, length in zip(prepared.keys, prepared.values, prepared.selected_lengths):
            pad = max_memory - length
            padded_keys.append(F.pad(key.to(local_key.device, local_key.dtype), (0, 0, 0, pad)))
            padded_values.append(F.pad(value.to(local_value.device, local_value.dtype), (0, 0, 0, pad)))
            valid_rows.append(torch.arange(max_memory, device=local_key.device) < length)
        memory_key = torch.cat(padded_keys, dim=0)
        memory_value = torch.cat(padded_values, dim=0)
        combined_key = torch.cat((memory_key, local_key), dim=2)
        combined_value = torch.cat((memory_value, local_value), dim=2)
        memory_valid = torch.stack(valid_rows, dim=0)[:, None, None, :]
        mask_dtype = attention_mask.dtype if attention_mask is not None else local_key.dtype
        prefix = torch.zeros((batch, 1, query_tokens, max_memory), device=local_key.device, dtype=mask_dtype)
        prefix = prefix.masked_fill(~memory_valid, torch.finfo(mask_dtype).min)
        if attention_mask is None:
            local_mask = torch.zeros((batch, 1, query_tokens, local_tokens), device=local_key.device, dtype=mask_dtype)
        else:
            local_mask = attention_mask[..., :local_tokens].to(local_key.device)
        return CombinedNativeKV(combined_key, combined_value, torch.cat((prefix, local_mask), dim=-1), max_memory)

    def apply_pra_attention(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        prepared: PreparedPRAMemory,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryBatchingStats, float]:
        """Apply shared native attention for the controlled-model execution path."""
        synchronize_detailed_timing(query, self.config.collect_detailed_timing)
        started = time.perf_counter()
        output, stats = native_kv_attention(
            query,
            local_key,
            local_value,
            prepared.keys,
            prepared.values,
            attention_mask=attention_mask,
            max_context_tokens=self.config.effective_model_max_context_tokens,
            collect_final_token_memory_weights=self.config.collect_per_head_metrics,
        )
        synchronize_detailed_timing(query, self.config.collect_detailed_timing)
        return output, stats, time.perf_counter() - started

    def collect_pra_metrics(
        self,
        prepared: PreparedPRAMemory,
        stats: MemoryBatchingStats,
        *,
        direct_tokens: int,
        output: torch.Tensor,
        local_output: torch.Tensor | None,
        memory_attention_duration_seconds: float,
    ) -> dict[str, float]:
        """Collect transport, budget, timing, and output diagnostics consistently."""
        aggregate_keys = (
            "memory_budget_tokens",
            "routing_candidates",
            "routing_topk_candidates",
            "chunks_materialized",
            "chunks_budget_rejected",
            "memory_tokens_requested",
            "memory_tokens_materialized",
        )
        metrics = {key: sum(row[key] for row in prepared.budget_stats) for key in aggregate_keys}
        paper3_sum_keys = (
            "requested_tokens_pre_dedup",
            "deduplicated_materialization_tokens",
            "materialized_native_kv_tokens",
            "materialized_native_kv_bytes",
            "materialized_gist_kv_tokens",
            "evidence_kv_tokens",
            "non_evidence_kv_tokens",
            "storage_shards_touched",
            "cross_shard_interval_count",
            "interval_resolution_seconds",
            "logical_gather_seconds",
            "logical_h2d_seconds",
        )
        for key in paper3_sum_keys:
            if any(key in row for row in prepared.budget_stats):
                metrics[key] = sum(float(row.get(key, 0.0)) for row in prepared.budget_stats)
        if "materialized_native_kv_tokens" in metrics:
            metrics["evidence_density"] = metrics.get("evidence_kv_tokens", 0.0) / max(
                metrics["materialized_native_kv_tokens"], 1.0
            )
        utilization = [row["materialization_budget_utilization"] for row in prepared.budget_stats]
        metrics.update(
            {
                **stats.as_metrics(),
                "model_max_context_tokens": float(self.config.effective_model_max_context_tokens),
                "direct_context_tokens": float(direct_tokens),
                "materialization_budget_utilization": sum(utilization) / max(len(utilization), 1),
                "lowest_materialized_score": min(
                    (row["lowest_materialized_score"] for row in prepared.budget_stats if math.isfinite(row["lowest_materialized_score"])),
                    default=float("nan"),
                ),
                "highest_budget_rejected_score": max(
                    (row["highest_budget_rejected_score"] for row in prepared.budget_stats if math.isfinite(row["highest_budget_rejected_score"])),
                    default=float("nan"),
                ),
                "memory_duplicate_chunk_tokens": float(prepared.duplicate_tokens),
                "memory_transport_native_kv": 1.0,
                "retrieved_token_kv": float(stats.valid_positions),
                "retrieved_physical_kv_tokens": float(stats.valid_positions),
                "retrieved_kv_transfer_bytes": float(prepared.transferred_kv_bytes),
                "attention_output_norm": float(output.detach().norm().cpu()),
            }
        )
        if local_output is not None:
            metrics["local_output_norm"] = float(local_output.detach().norm().cpu())
            metrics["attention_to_local_output_norm_ratio"] = metrics["attention_output_norm"] / max(metrics["local_output_norm"], 1e-12)
        if self.config.collect_detailed_timing:
            metrics.update(
                {
                    "routing_duration_seconds": prepared.routing_duration_seconds,
                    "materialization_duration_seconds": prepared.materialization_duration_seconds,
                    "selected_kv_transfer_duration_seconds": prepared.transfer_duration_seconds,
                    "memory_attention_duration_seconds": memory_attention_duration_seconds,
                }
            )
        return metrics
