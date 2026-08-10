"""Chunk-aware, layer-specific memory cache and routing for PRA."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .gists import score_gist_set

if TYPE_CHECKING:
    from .config import PRAConfig


@dataclass
class LayerKV:
    """Projected attention memory for one chunk at one decoder layer.

    Both tensors have shape ``[1, heads, memory_tokens, head_dim]``. Reference
    chunks are encoded independently, hence the singleton batch dimension.
    """

    k: torch.Tensor  # Keys used for routing-gist construction and cross-attention.
    v: torch.Tensor  # Values returned when a query attends to this memory.
    position_ids: torch.Tensor | None = None  # [M] or [1,M] positions used to publish K.
    position_state: str = "post_position"  # post_position or experimental pre_position.


ChunkKV = LayerKV


@dataclass
class ChunkRoutingGist:
    """Layer-specific chunk gist sets searched before loading detailed token K/V."""

    k: torch.Tensor  # Content routing keys shaped [gists, d_model].
    v: torch.Tensor | None = None  # Paired gist-only values shaped [gists, d_model].
    method: str = "mean"  # Pooling rule that produced k/v.
    summary_k: torch.Tensor | None = None  # Separately encoded summary keys [gists, d_model].
    metadata: dict = field(default_factory=dict)  # Pooling/source diagnostics.

    def __post_init__(self) -> None:
        """Normalize legacy ``[D]`` constructors into the invariant ``[1,D]`` form."""
        if self.k.ndim == 1:
            self.k = self.k.unsqueeze(0)
        if self.v is not None and self.v.ndim == 1:
            self.v = self.v.unsqueeze(0)
        if self.summary_k is not None and self.summary_k.ndim == 1:
            self.summary_k = self.summary_k.unsqueeze(0)
        if self.k.ndim != 2:
            raise ValueError(f"Chunk routing keys must be [gists,model], got {self.k.shape}.")
        if self.v is not None and self.v.shape != self.k.shape:
            raise ValueError("Chunk routing value gists must match key-gist shape.")


@dataclass
class ReferenceRoutingGists:
    """Cached URI-level routing representation for one decoder layer."""

    k: torch.Tensor  # URI routing keys shaped [gists, d_model].
    v: torch.Tensor | None = None  # Optional paired URI values shaped [gists, d_model].
    mode: str = "mean"  # Strategy that compressed this URI's chunk gists.
    metadata: dict = field(default_factory=dict)  # Requested/actual counts and occupancy.

    def __post_init__(self) -> None:
        if self.k.ndim == 1:
            self.k = self.k.unsqueeze(0)
        if self.v is not None and self.v.ndim == 1:
            self.v = self.v.unsqueeze(0)
        if self.k.ndim != 2:
            raise ValueError(f"Reference routing keys must be [gists,model], got {self.k.shape}.")
        if self.v is not None and self.v.shape != self.k.shape:
            raise ValueError("Reference routing value gists must match key-gist shape.")


@dataclass
class ReferenceChunkMemory:
    """Cached detail and routing state for one provenance-preserving text span."""

    chunk_id: str  # Stable URI-qualified identity, for example ``uri#chunk=2``.
    source_uri: str  # Canonical document from which the span was encoded.
    token_start: int  # Inclusive source-token offset.
    token_end: int  # Exclusive source-token offset after truncation.
    token_kv: ChunkKV  # Full detail K/V, each shaped [1, H, chunk_tokens, Dh].
    routing_gist: ChunkRoutingGist  # Cheap [d_model] vectors searched first.
    char_start: int | None = None  # Optional inclusive source-character offset.
    char_end: int | None = None  # Optional exclusive source-character offset.
    metadata: dict = field(default_factory=dict)  # Chunking and truncation provenance.
    logical_start: int = -1  # Inclusive coordinate in the owning continuous source.
    logical_end: int | None = None  # Exclusive coordinate reconstructed from start + length.

    def __post_init__(self) -> None:
        """Normalize compact logical provenance without storing per-token coordinates."""
        logical_start = self.token_start if self.logical_start < 0 else int(self.logical_start)
        logical_end = (
            logical_start + self.token_count
            if self.logical_end is None
            else int(self.logical_end)
        )
        if logical_start < 0 or logical_end - logical_start != self.token_count:
            raise ValueError("Logical memory offsets must match the contiguous K/V token count.")
        self.logical_start = logical_start
        self.logical_end = logical_end

    @property
    def token_count(self) -> int:
        """Return the retained K/V sequence length for this layer/chunk."""
        return int(self.token_kv.k.shape[2])


@dataclass
class LayerReferenceMemory:
    """All independently routable chunks for one URI at one model layer."""

    chunks: list[ReferenceChunkMemory] = field(default_factory=list)


@dataclass
class PRACacheEntry:
    """Complete, versioned cache object for one resolved reference URI."""

    uri: str  # Stable lookup identity used by routing, recursion, and traces.
    text: str  # Resolved source text used to construct this cache entry.
    layer_memory: dict[int, LayerReferenceMemory] = field(default_factory=dict)
    reference_gists_by_layer: dict[int, ReferenceRoutingGists] = field(default_factory=dict)
    child_uris: list[str] = field(default_factory=list)  # Outgoing recursive references.
    metadata: dict = field(default_factory=dict)  # Fingerprints, version, and build provenance.


@dataclass(frozen=True)
class SelectedChunk:
    """One routed chunk plus scores/ranks needed for materialization and tracing."""

    entry: PRACacheEntry  # Owning URI-level cache entry.
    chunk: ReferenceChunkMemory  # Selected layer-specific chunk payload.
    reference_score: float  # URI score after configured chunk aggregation.
    chunk_score: float  # Cosine score between query and routing gist.
    layer_id: int  # Decoder layer whose K/V and gist were searched.
    reference_rank: int  # One-based URI rank for this query.
    rank_within_reference: int  # One-based chunk rank within the selected URI.
    winning_gist_index: int | None = None  # Chunk gist that best matched the query.
    winning_gist_score: float | None = None  # Score of the winning chunk gist.
    gist_count: int = 1  # Number of chunk gists considered for this hit.
    winning_reference_gist_index: int | None = None  # Winning cached URI gist when used.
    winning_reference_gist_score: float | None = None  # Score of that URI gist.
    reference_gist_count: int = 0  # Number of cached URI gists considered.
    metadata: dict = field(default_factory=dict)  # Routing source/materialization trace.

    @property
    def reference_uri(self) -> str:
        """Return the stable URI of the owning cache entry."""
        return self.entry.uri

    @property
    def chunk_id(self) -> str:
        """Return the selected chunk's stable identity."""
        return self.chunk.chunk_id

    @property
    def source_uri(self) -> str:
        """Return the document URI represented by the selected chunk."""
        return self.chunk.source_uri

    @property
    def token_start(self) -> int:
        """Return the inclusive source-token offset."""
        return self.chunk.token_start

    @property
    def token_end(self) -> int:
        """Return the exclusive source-token offset."""
        return self.chunk.token_end

    @property
    def logical_start(self) -> int:
        """Return the source-relative coordinate used to reconstruct token positions."""
        return self.chunk.logical_start

    @property
    def logical_end(self) -> int:
        """Return the exclusive source-relative coordinate."""
        assert self.chunk.logical_end is not None
        return self.chunk.logical_end

    @property
    def selected_token_count(self) -> int:
        """Return how many token K/V positions this selection can materialize."""
        return self.chunk.token_count

    def as_trace_dict(self) -> dict:
        """Serialize routing identity and scores without copying large K/V tensors."""
        return {
            "reference_uri": self.reference_uri,
            "reference_score": self.reference_score,
            "chunk_id": self.chunk_id,
            "chunk_score": self.chunk_score,
            "layer_id": self.layer_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "logical_start": self.logical_start,
            "logical_end": self.logical_end,
            "selected_token_count": self.selected_token_count,
            "reference_rank": self.reference_rank,
            "rank_within_reference": self.rank_within_reference,
            "winning_gist_index": self.winning_gist_index,
            "winning_gist_score": self.winning_gist_score,
            "gist_count": self.gist_count,
            "winning_reference_gist_index": self.winning_reference_gist_index,
            "winning_reference_gist_score": self.winning_reference_gist_score,
            "reference_gist_count": self.reference_gist_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class _ChunkHit:
    """Internal scored chunk record retained until routing assigns stable ranks."""

    entry: PRACacheEntry
    chunk: ReferenceChunkMemory
    score: float
    routing_source: str
    winning_gist_index: int | None
    winning_gist_score: float | None
    gist_count: int


@dataclass(frozen=True)
class _ReferenceHit:
    """Internal URI score produced before reference-first chunk scoring."""

    entry: PRACacheEntry
    score: float
    winning_gist_index: int | None
    winning_gist_score: float | None
    gist_count: int


@dataclass(frozen=True)
class _TensorizedChunkIndex:
    """Packed, normalized routing gists and ownership maps for one layer/device.

    ``gists`` is ``[C,G,D]`` for C chunks, at most G gists per chunk, and
    routing width D. ``chunk_indices_by_reference`` maps ``[R,C_r]`` padded
    reference slots back to the C chunk rows. The tensors let one query batch
    score every candidate and apply exact top-k selection without per-candidate
    Python/CUDA synchronization.
    """

    records: tuple[tuple[PRACacheEntry, ReferenceChunkMemory], ...]
    gists: torch.Tensor
    gist_mask: torch.Tensor
    reference_entries: tuple[PRACacheEntry, ...]
    chunk_indices_by_reference: torch.Tensor
    chunk_mask_by_reference: torch.Tensor
    chunk_indices_by_reference_cpu: tuple[tuple[int, ...], ...]


class CacheBuildState(str, Enum):
    """Visibility state for atomically constructed recursive cache entries."""

    MISSING = "missing"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


def _aggregate(scores: list[float], mode: str) -> float:
    """Reduce chunk scores to one URI score for hierarchical routing."""
    if not scores:
        return float("-inf")
    if mode == "max":
        return max(scores)
    if mode == "mean":
        return sum(scores) / len(scores)
    if mode == "logsumexp":
        maximum = max(scores)
        return maximum + math.log(sum(math.exp(score - maximum) for score in scores))
    raise ValueError(f"Unsupported reference score aggregation: {mode}")


class PRAMemoryCache(ABC):
    """Storage and layer-aware routing contract consumed by :class:`PRAttention`.

    Backends may store entries differently, but ``search`` must preserve batch
    isolation and return one ordered ``SelectedChunk`` list per query row.
    """

    @abstractmethod
    def put(self, entry: PRACacheEntry) -> None:
        """Publish one completely built URI entry."""
        raise NotImplementedError

    @abstractmethod
    def get(self, uri: str) -> PRACacheEntry | None:
        """Return a ready entry by URI, or ``None`` when unavailable."""
        raise NotImplementedError

    @abstractmethod
    def has(self, uri: str) -> bool:
        """Report whether a URI has a ready, visible cache entry."""
        raise NotImplementedError

    @abstractmethod
    def all_entries(self) -> list[PRACacheEntry]:
        """Return every ready entry visible to routing."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries and backend lifecycle state."""
        raise NotImplementedError

    @abstractmethod
    def is_empty(self) -> bool:
        """Return whether no memory is visible to routing."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: torch.Tensor,
        layer_id: int,
        config: PRAConfig,
    ) -> list[list[SelectedChunk]]:
        """Route ``[B,d_model]`` queries against gists from one decoder layer."""
        raise NotImplementedError

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        """Expose ready entries as a URI-keyed compatibility view."""
        return {entry.uri: entry for entry in self.all_entries()}

    def __len__(self) -> int:
        """Return the number of ready URI entries."""
        return len(self.all_entries())


class PRASimpleMemoryCache(PRAMemoryCache):
    """In-memory cache implementing the three experimental routing strategies.

    Entries become searchable only in ``READY`` state. This matters for recursive
    construction: a parent can never observe a partially encoded child.
    """

    def __init__(self):
        """Create empty payload, lifecycle-state, and failure registries."""
        self._entries: dict[str, PRACacheEntry] = {}
        self._states: dict[str, CacheBuildState] = {}
        self._failures: dict[str, str] = {}
        self._last_rankings_by_layer: dict[int, list[list[dict]]] = {}
        self._tensorized_indexes: dict[tuple[int, str, torch.dtype], _TensorizedChunkIndex] = {}

    def invalidate_routing_indexes(self) -> None:
        """Drop packed gist tensors after cache content or pooling state changes."""
        self._tensorized_indexes.clear()

    def prepare_routing_index(
        self,
        layer_id: int,
        query: torch.Tensor,
        *,
        force_rebuild: bool = False,
    ) -> dict[str, int | bool]:
        """Build or reuse the exact packed index and report its physical size.

        This narrow public hook exists for cache warmup and systems benchmarks.
        It does not score a query or change routing semantics.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        key = (int(layer_id), str(query.device), query.dtype)
        reused = key in self._tensorized_indexes and not force_rebuild
        if force_rebuild:
            self._tensorized_indexes.pop(key, None)
        index = self._tensorized_chunk_index(layer_id, query, reuse=True)
        if index is None:
            return {
                "reused": reused,
                "candidate_chunks": 0,
                "candidate_gists": 0,
                "index_bytes": 0,
            }
        tensors = (
            index.gists,
            index.gist_mask,
            index.chunk_indices_by_reference,
            index.chunk_mask_by_reference,
        )
        return {
            "reused": reused,
            "candidate_chunks": len(index.records),
            "candidate_gists": int(index.gist_mask.sum().item()),
            "index_bytes": sum(t.numel() * t.element_size() for t in tensors),
        }

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        """Return the internal URI map; published entries are normally ``READY``."""
        return self._entries

    def begin_build(self, uri: str) -> None:
        """Mark a URI as under construction to detect recursive re-entry."""
        self.invalidate_routing_indexes()
        self._states[uri] = CacheBuildState.BUILDING

    def mark_failed(self, uri: str, error: Exception | str) -> None:
        """Hide a failed payload while retaining a diagnostic message."""
        self._entries.pop(uri, None)
        self.invalidate_routing_indexes()
        self._states[uri] = CacheBuildState.FAILED
        self._failures[uri] = str(error)

    def state(self, uri: str) -> CacheBuildState:
        """Return the current lifecycle state, defaulting to ``MISSING``."""
        return self._states.get(uri, CacheBuildState.MISSING)

    def failure(self, uri: str) -> str | None:
        """Return the last cache-build error recorded for a URI."""
        return self._failures.get(uri)

    def put(self, entry: PRACacheEntry) -> None:
        """Atomically publish a completed entry as routable memory."""
        self._entries[entry.uri] = entry
        self.invalidate_routing_indexes()
        self._states[entry.uri] = CacheBuildState.READY
        self._failures.pop(entry.uri, None)

    def get(self, uri: str) -> PRACacheEntry | None:
        """Return a URI payload only after it reaches ``READY`` state."""
        return self._entries.get(uri) if self.state(uri) == CacheBuildState.READY else None

    def has(self, uri: str) -> bool:
        """Report whether a ready URI payload is available."""
        return self.get(uri) is not None

    def all_entries(self) -> list[PRACacheEntry]:
        """Return only entries safe for attention to consume."""
        return [entry for uri, entry in self._entries.items() if self.state(uri) == CacheBuildState.READY]

    def clear(self) -> None:
        """Reset payloads, lifecycle states, and failure diagnostics."""
        self._entries.clear()
        self._states.clear()
        self._failures.clear()
        self._last_rankings_by_layer.clear()
        self.invalidate_routing_indexes()

    def is_empty(self) -> bool:
        """Return whether this row-local namespace has no ready entries."""
        return not any(self.state(uri) == CacheBuildState.READY for uri in self._entries)

    def invalidate(self, uri: str) -> None:
        """Remove a stale URI before rebuilding it with new fingerprints."""
        self._entries.pop(uri, None)
        self.invalidate_routing_indexes()
        self._states.pop(uri, None)
        self._failures.pop(uri, None)

    def layer_counts(self, layer_id: int) -> dict[str, int]:
        """Count searchable URIs, chunks, token positions, and routing gists."""
        entries = [entry for entry in self.all_entries() if layer_id in entry.layer_memory]
        chunks = [chunk for entry in entries for chunk in entry.layer_memory[layer_id].chunks]
        reference_gists = [
            entry.reference_gists_by_layer[layer_id]
            for entry in entries
            if layer_id in entry.reference_gists_by_layer
        ]
        return {
            "references": len(entries),
            "chunks": len(chunks),
            "tokens": sum(chunk.token_count for chunk in chunks),
            "chunk_gists": sum(int(chunk.routing_gist.k.shape[0]) for chunk in chunks),
            "reference_gists": sum(int(gists.k.shape[0]) for gists in reference_gists),
        }

    def last_rankings(self, layer_id: int) -> list[list[dict]]:
        """Return complete candidate rankings from the latest search at one layer."""
        return self._last_rankings_by_layer.get(layer_id, [])

    def _tensorized_chunk_index(
        self,
        layer_id: int,
        query: torch.Tensor,
        *,
        reuse: bool,
    ) -> _TensorizedChunkIndex | None:
        """Pack one layer's variable gist sets for exact batched cosine search."""
        key = (layer_id, str(query.device), query.dtype)
        if reuse and key in self._tensorized_indexes:
            return self._tensorized_indexes[key]

        records: list[tuple[PRACacheEntry, ReferenceChunkMemory]] = []
        for entry in self.all_entries():
            memory = entry.layer_memory.get(layer_id)
            if memory is not None:
                records.extend((entry, chunk) for chunk in memory.chunks)
        if not records:
            return None

        max_gists = max(int(chunk.routing_gist.k.shape[0]) for _, chunk in records)
        gist_sets = [
            chunk.routing_gist.k.to(query.device, query.dtype) for _, chunk in records
        ]
        counts = [int(gists.shape[0]) for gists in gist_sets]
        if max_gists == 1:
            # The common path needs one stack, rather than one padding kernel per chunk.
            packed_gists = torch.stack([gists[0] for gists in gist_sets]).unsqueeze(1)
            gist_mask = torch.ones(
                (len(records), 1), dtype=torch.bool, device=query.device
            )
        else:
            # Pack variable gist sets with one concatenation and one indexed copy.
            flat = torch.cat(gist_sets, dim=0)
            owner = torch.repeat_interleave(
                torch.arange(len(records), device=query.device),
                torch.tensor(counts, device=query.device),
            )
            slot = torch.cat(
                [torch.arange(count, device=query.device) for count in counts]
            )
            packed_gists = flat.new_zeros((len(records), max_gists, flat.shape[-1]))
            packed_gists[owner, slot] = flat
            gist_mask = (
                torch.arange(max_gists, device=query.device).unsqueeze(0)
                < torch.tensor(counts, device=query.device).unsqueeze(1)
            )
        packed_gists = F.normalize(packed_gists, dim=-1, eps=1e-12)

        entry_by_uri = {entry.uri: entry for entry, _ in records}
        reference_entries = tuple(entry_by_uri[uri] for uri in sorted(entry_by_uri))
        indices_by_uri: dict[str, list[int]] = defaultdict(list)
        for chunk_index, (entry, _) in enumerate(records):
            indices_by_uri[entry.uri].append(chunk_index)
        max_chunks = max(len(indices_by_uri[entry.uri]) for entry in reference_entries)
        chunk_rows = []
        chunk_masks = []
        for entry in reference_entries:
            indices = indices_by_uri[entry.uri]
            padding = max_chunks - len(indices)
            chunk_rows.append(indices + [0] * padding)
            chunk_masks.append([True] * len(indices) + [False] * padding)

        index = _TensorizedChunkIndex(
            records=tuple(records),
            gists=packed_gists,
            gist_mask=gist_mask,
            reference_entries=reference_entries,
            chunk_indices_by_reference=torch.tensor(
                chunk_rows, dtype=torch.long, device=query.device
            ),
            chunk_mask_by_reference=torch.tensor(
                chunk_masks, dtype=torch.bool, device=query.device
            ),
            chunk_indices_by_reference_cpu=tuple(
                tuple(indices_by_uri[entry.uri]) for entry in reference_entries
            ),
        )
        if reuse:
            self._tensorized_indexes[key] = index
        return index

    @staticmethod
    def _reduce_gist_scores(
        query: torch.Tensor,
        index: _TensorizedChunkIndex,
        aggregation: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return aggregate, winning-index, and winning-score tensors for all chunks."""
        normalized_query = F.normalize(query, dim=-1, eps=1e-12)
        if index.gists.shape[1] == 1:
            scores = (normalized_query @ index.gists[:, 0, :].transpose(0, 1)).unsqueeze(-1)
        else:
            scores = torch.einsum("bd,cgd->bcg", normalized_query, index.gists)
        valid = index.gist_mask.unsqueeze(0)
        masked = scores.masked_fill(~valid, float("-inf"))
        winning_scores, winning_indices = masked.max(dim=-1)
        if aggregation == "max":
            aggregate = winning_scores
        elif aggregation == "mean":
            aggregate = scores.masked_fill(~valid, 0.0).sum(dim=-1)
            aggregate = aggregate / valid.sum(dim=-1).clamp_min(1)
        elif aggregation == "logsumexp":
            aggregate = torch.logsumexp(masked, dim=-1)
        else:
            raise ValueError(f"Unsupported gist score aggregation: {aggregation}")
        return aggregate, winning_indices, winning_scores

    @staticmethod
    def _reference_score_tensor(
        chunk_scores: torch.Tensor,
        index: _TensorizedChunkIndex,
        aggregation: str,
    ) -> torch.Tensor:
        """Aggregate ``[B,C]`` chunk scores into URI scores ``[B,R]``."""
        grouped = chunk_scores[:, index.chunk_indices_by_reference]
        valid = index.chunk_mask_by_reference.unsqueeze(0)
        if aggregation == "max":
            return grouped.masked_fill(~valid, float("-inf")).max(dim=-1).values
        if aggregation == "mean":
            total = grouped.masked_fill(~valid, 0.0).sum(dim=-1)
            return total / valid.sum(dim=-1).clamp_min(1)
        if aggregation == "logsumexp":
            return torch.logsumexp(grouped.masked_fill(~valid, float("-inf")), dim=-1)
        raise ValueError(f"Unsupported reference score aggregation: {aggregation}")

    def _score_chunks_tensorized(
        self,
        query: torch.Tensor,
        layer_id: int,
        config,
        *,
        retain_full_hits: bool,
    ) -> tuple[
        list[list[_ChunkHit]] | None,
        _TensorizedChunkIndex | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Score all gists while copying every candidate only for full diagnostics."""
        index = self._tensorized_chunk_index(
            layer_id,
            query,
            reuse=config.cache_build_mode == "detached",
        )
        if index is None:
            return ([[] for _ in range(query.shape[0])], None, None, None, None, None)
        aggregate, winners, winning_scores = self._reduce_gist_scores(
            query, index, config.gist_score_aggregation
        )
        rows = None
        if retain_full_hits:
            diagnostics = torch.stack(
                (aggregate, winning_scores, winners.to(aggregate.dtype)), dim=-1
            ).detach().cpu()
            rows = []
            for row in diagnostics:
                hits = []
                for chunk_index, (entry, chunk) in enumerate(index.records):
                    hits.append(
                        _ChunkHit(
                            entry=entry,
                            chunk=chunk,
                            score=float(row[chunk_index, 0]),
                            routing_source="content",
                            winning_gist_index=int(row[chunk_index, 2]),
                            winning_gist_score=float(row[chunk_index, 1]),
                            gist_count=int(chunk.routing_gist.k.shape[0]),
                        )
                    )
                rows.append(hits)
        return (
            rows,
            index,
            aggregate,
            self._reference_score_tensor(
                aggregate, index, config.reference_score_aggregation
            ),
            winners,
            winning_scores,
        )

    def _hierarchical_tensorized(
        self,
        hits_by_batch: list[list[_ChunkHit]] | None,
        index: _TensorizedChunkIndex,
        chunk_scores: torch.Tensor,
        reference_scores: torch.Tensor,
        winning_indices: torch.Tensor,
        winning_scores: torch.Tensor,
        layer_id: int,
        config,
    ) -> list[list[SelectedChunk]]:
        """Apply exact top-k and serialize only selected records in normal inference."""
        reference_k = min(config.top_k_references, len(index.reference_entries))
        if reference_k == 0:
            return [[] for _ in range(chunk_scores.shape[0])]
        reference_values, reference_indices = torch.topk(
            reference_scores, k=reference_k, dim=-1, largest=True, sorted=True
        )

        grouped_scores = chunk_scores[:, index.chunk_indices_by_reference]
        grouped_scores = grouped_scores.masked_fill(
            ~index.chunk_mask_by_reference.unsqueeze(0), float("-inf")
        )
        selected_grouped_scores = torch.gather(
            grouped_scores,
            1,
            reference_indices.unsqueeze(-1).expand(-1, -1, grouped_scores.shape[-1]),
        )
        chunk_k = min(
            config.top_k_chunks_per_reference,
            int(index.chunk_indices_by_reference.shape[1]),
        )
        _, chunk_slots = torch.topk(
            selected_grouped_scores, k=chunk_k, dim=-1, largest=True, sorted=True
        )

        selected_index_rows = index.chunk_indices_by_reference[reference_indices]
        selected_chunk_indices = torch.gather(selected_index_rows, 2, chunk_slots)
        flat_chunk_indices = selected_chunk_indices.flatten(1)
        selected_chunk_scores = torch.gather(
            chunk_scores, 1, flat_chunk_indices
        ).reshape_as(selected_chunk_indices)
        selected_winning_indices = torch.gather(
            winning_indices, 1, flat_chunk_indices
        ).reshape_as(selected_chunk_indices)
        selected_winning_scores = torch.gather(
            winning_scores, 1, flat_chunk_indices
        ).reshape_as(selected_chunk_indices)

        reference_indices_cpu = reference_indices.detach().cpu().tolist()
        reference_values_cpu = reference_values.detach().cpu().tolist()
        selected_chunk_indices_cpu = selected_chunk_indices.detach().cpu().tolist()
        selected_chunk_scores_cpu = selected_chunk_scores.detach().cpu().tolist()
        selected_winning_indices_cpu = selected_winning_indices.detach().cpu().tolist()
        selected_winning_scores_cpu = selected_winning_scores.detach().cpu().tolist()
        selected_by_batch = []
        for batch_index, selected_reference_indices in enumerate(reference_indices_cpu):
            reference_rows = list(
                zip(
                    selected_reference_indices,
                    reference_values_cpu[batch_index],
                    range(len(selected_reference_indices)),
                )
            )
            reference_rows.sort(
                key=lambda item: (-item[1], index.reference_entries[item[0]].uri)
            )
            selected = []
            for reference_rank, (ref_index, reference_score, ref_slot) in enumerate(
                reference_rows, start=1
            ):
                source_indices = index.chunk_indices_by_reference_cpu[ref_index]
                valid_count = len(source_indices)
                chunk_rows = []
                for chunk_slot_index in range(min(chunk_k, valid_count)):
                    source_index = selected_chunk_indices_cpu[batch_index][ref_slot][
                        chunk_slot_index
                    ]
                    chunk_rows.append(
                        (
                            source_index,
                            selected_chunk_scores_cpu[batch_index][ref_slot][chunk_slot_index],
                            selected_winning_indices_cpu[batch_index][ref_slot][chunk_slot_index],
                            selected_winning_scores_cpu[batch_index][ref_slot][chunk_slot_index],
                        )
                    )
                chunk_rows.sort(
                    key=lambda item: (-item[1], index.records[item[0]][1].chunk_id)
                )
                for chunk_rank, (
                    chunk_index,
                    chunk_score,
                    winning_index,
                    winning_score,
                ) in enumerate(chunk_rows, start=1):
                    entry, chunk = index.records[chunk_index]
                    hit = (
                        hits_by_batch[batch_index][chunk_index]
                        if hits_by_batch is not None
                        else _ChunkHit(
                            entry=entry,
                            chunk=chunk,
                            score=float(chunk_score),
                            routing_source="content",
                            winning_gist_index=int(winning_index),
                            winning_gist_score=float(winning_score),
                            gist_count=int(chunk.routing_gist.k.shape[0]),
                        )
                    )
                    selected.append(
                        self._selected(
                            hit,
                            float(reference_score),
                            layer_id,
                            reference_rank,
                            chunk_rank,
                        )
                    )
            selected_by_batch.append(selected)
        return selected_by_batch

    @staticmethod
    def _serialize_rankings(
        hits: list[_ChunkHit],
        config,
        *,
        reference_scores: dict[str, float] | None = None,
    ) -> list[dict]:
        """Preserve every reference/chunk score before top-k truncation."""
        grouped = defaultdict(list)
        for hit in hits:
            grouped[hit.entry.uri].append(hit)
        if reference_scores is None:
            reference_scores = {
                uri: _aggregate(
                    [candidate.score for candidate in candidates],
                    config.reference_score_aggregation,
                )
                for uri, candidates in grouped.items()
            }
        ranked_uris = sorted(reference_scores, key=lambda uri: (-reference_scores[uri], uri))
        rankings = []
        for reference_rank, uri in enumerate(ranked_uris, start=1):
            ranked_chunks = sorted(
                grouped.get(uri, []),
                key=lambda hit: (-hit.score, hit.chunk.chunk_id),
            )
            rankings.append(
                {
                    "reference_uri": uri,
                    "reference_rank": reference_rank,
                    "reference_score": float(reference_scores[uri]),
                    "chunks": [
                        {
                            "chunk_id": hit.chunk.chunk_id,
                            "chunk_rank": chunk_rank,
                            "chunk_score": float(hit.score),
                            "token_start": hit.chunk.token_start,
                            "token_end": hit.chunk.token_end,
                            "gist_count": hit.gist_count,
                            "winning_gist_index": hit.winning_gist_index,
                            "winning_gist_score": hit.winning_gist_score,
                        }
                        for chunk_rank, hit in enumerate(ranked_chunks, start=1)
                    ],
                }
            )
        return rankings

    @staticmethod
    def _score_chunk(query: torch.Tensor, entry, chunk, config) -> _ChunkHit:
        """Score one chunk's gist set while retaining its best content-gist index."""
        gist = chunk.routing_gist
        content = score_gist_set(query, gist.k, config.gist_score_aggregation)
        aggregate = content.aggregate_score
        routing_source = "content"
        if config.use_summary and gist.summary_k is not None:
            summary_gists = gist.summary_k
            if summary_gists.ndim == 1:
                summary_gists = summary_gists.unsqueeze(0)
            summary = score_gist_set(query, summary_gists, config.gist_score_aggregation)
            if config.summary_mode == "replace":
                aggregate = summary.aggregate_score
                routing_source = "summary"
            elif config.summary_mode == "hybrid":
                aggregate = max(aggregate, summary.aggregate_score)
                routing_source = "hybrid"
            elif config.summary_mode == "augment":
                summary_vector = F.normalize(summary_gists.mean(dim=0), dim=-1)
                combined = F.normalize(gist.k, dim=-1) + summary_vector.unsqueeze(0)
                aggregate = score_gist_set(
                    query,
                    F.normalize(combined, dim=-1),
                    config.gist_score_aggregation,
                ).aggregate_score
                routing_source = "augment"
            else:
                raise ValueError(f"Unsupported summary mode: {config.summary_mode}")
        return _ChunkHit(
            entry=entry,
            chunk=chunk,
            score=aggregate,
            routing_source=routing_source,
            winning_gist_index=content.winning_index,
            winning_gist_score=(
                float(content.per_gist_scores[content.winning_index].cpu())
                if content.per_gist_scores is not None and content.winning_index is not None
                else None
            ),
            gist_count=int(gist.k.shape[0]),
        )

    def _score_chunks(
        self,
        query: torch.Tensor,
        layer_id: int,
        config,
        *,
        entries: list[PRACacheEntry] | None = None,
    ) -> list[list[_ChunkHit]]:
        """Score chunk gist sets for each query, optionally inside selected URIs only."""
        records = []
        for entry in entries if entries is not None else self.all_entries():
            memory = entry.layer_memory.get(layer_id)
            if memory is None:
                continue
            for chunk in memory.chunks:
                records.append((entry, chunk))
        if not records:
            return [[] for _ in range(query.shape[0])]
        return [
            [self._score_chunk(row_query, entry, chunk, config) for entry, chunk in records]
            for row_query in query
        ]

    @staticmethod
    def _fallback_reference_gists(entry: PRACacheEntry, layer_id: int, config):
        """Build and cache simple URI gists for manually assembled legacy entries."""
        memory = entry.layer_memory.get(layer_id)
        if memory is None or not memory.chunks:
            return None
        mode = config.reference_level_gist_mode
        if mode is None:
            raise ValueError(
                "search_strategy='reference_first' requires reference_level_gist_mode."
            )
        if mode not in {"mean", "last"}:
            raise RuntimeError(
                f"Reference gists for mode {mode!r} must be created during cache construction."
            )
        keys = torch.cat([chunk.routing_gist.k for chunk in memory.chunks], dim=0)
        values = (
            torch.cat([chunk.routing_gist.v for chunk in memory.chunks], dim=0)
            if all(chunk.routing_gist.v is not None for chunk in memory.chunks)
            else None
        )
        if mode == "mean":
            gist_k = keys.mean(dim=0, keepdim=True)
            gist_v = values.mean(dim=0, keepdim=True) if values is not None else None
        else:
            gist_k = keys[-1:].clone()
            gist_v = values[-1:].clone() if values is not None else None
        reference_gists = ReferenceRoutingGists(
            k=gist_k,
            v=gist_v,
            mode=mode,
            metadata={
                "requested_gists": int(config.reference_gists_per_reference),
                "actual_gists": 1,
                "legacy_fallback": True,
            },
        )
        entry.reference_gists_by_layer[layer_id] = reference_gists
        return reference_gists

    def _score_references(self, query: torch.Tensor, layer_id: int, config) -> list[_ReferenceHit]:
        """Score cached URI gist sets without touching detailed or chunk routing K/V."""
        if config.reference_level_gist_mode is None:
            raise ValueError(
                "search_strategy='reference_first' requires reference_level_gist_mode."
            )
        hits = []
        for entry in self.all_entries():
            gists = entry.reference_gists_by_layer.get(layer_id)
            if gists is None:
                gists = self._fallback_reference_gists(entry, layer_id, config)
            if gists is None:
                continue
            score = score_gist_set(query, gists.k, config.reference_gist_score_aggregation)
            winning_score = (
                float(score.per_gist_scores[score.winning_index].cpu())
                if score.per_gist_scores is not None and score.winning_index is not None
                else None
            )
            hits.append(
                _ReferenceHit(
                    entry=entry,
                    score=score.aggregate_score,
                    winning_gist_index=score.winning_index,
                    winning_gist_score=winning_score,
                    gist_count=int(gists.k.shape[0]),
                )
            )
        return hits

    @staticmethod
    def _selected(
        hit: _ChunkHit,
        reference_score,
        layer_id,
        ref_rank,
        chunk_rank,
        reference_hit: _ReferenceHit | None = None,
    ):
        """Attach stable ranks and routing provenance to a selected payload."""
        return SelectedChunk(
            entry=hit.entry,
            chunk=hit.chunk,
            reference_score=reference_score,
            chunk_score=hit.score,
            layer_id=layer_id,
            reference_rank=ref_rank,
            rank_within_reference=chunk_rank,
            winning_gist_index=hit.winning_gist_index,
            winning_gist_score=hit.winning_gist_score,
            gist_count=hit.gist_count,
            winning_reference_gist_index=(
                reference_hit.winning_gist_index if reference_hit is not None else None
            ),
            winning_reference_gist_score=(
                reference_hit.winning_gist_score if reference_hit is not None else None
            ),
            reference_gist_count=reference_hit.gist_count if reference_hit is not None else 0,
            metadata={
                "routing_source": hit.routing_source,
                "reference_source": hit.entry.metadata.get("source", "reference"),
                "implicit_reference": bool(hit.entry.metadata.get("implicit", False)),
                "reference_display_name": hit.entry.metadata.get("display_name", hit.entry.uri),
            },
        )

    def _hierarchical(self, hits, layer_id, config):
        """Aggregate chunk scores per URI, select URIs, then select local chunks."""
        grouped = defaultdict(list)
        for hit in hits:
            grouped[hit.entry.uri].append(hit)
        reference_scores = {
            uri: _aggregate([hit.score for hit in grouped_hits], config.reference_score_aggregation)
            for uri, grouped_hits in grouped.items()
        }
        selected_uris = sorted(reference_scores, key=lambda uri: (-reference_scores[uri], uri))[
            : config.top_k_references
        ]
        selected = []
        for ref_rank, uri in enumerate(selected_uris, start=1):
            ranked = sorted(grouped[uri], key=lambda hit: (-hit.score, hit.chunk.chunk_id))[
                : config.top_k_chunks_per_reference
            ]
            selected.extend(
                self._selected(hit, reference_scores[uri], layer_id, ref_rank, rank)
                for rank, hit in enumerate(ranked, start=1)
            )
        return selected

    def _reference_first(self, query, layer_id, config):
        """Select URIs from cached gists before scoring chunks only inside those URIs."""
        reference_hits = sorted(
            self._score_references(query, layer_id, config),
            key=lambda hit: (-hit.score, hit.entry.uri),
        )[: config.top_k_references]
        if not reference_hits:
            return []
        chunk_hits = self._score_chunks(
            query.unsqueeze(0),
            layer_id,
            config,
            entries=[hit.entry for hit in reference_hits],
        )[0]
        grouped = defaultdict(list)
        for hit in chunk_hits:
            grouped[hit.entry.uri].append(hit)
        selected = []
        for ref_rank, reference_hit in enumerate(reference_hits, start=1):
            ranked = sorted(
                grouped[reference_hit.entry.uri],
                key=lambda hit: (-hit.score, hit.chunk.chunk_id),
            )[
                : config.top_k_chunks_per_reference
            ]
            selected.extend(
                self._selected(
                    hit,
                    reference_hit.score,
                    layer_id,
                    ref_rank,
                    rank,
                    reference_hit,
                )
                for rank, hit in enumerate(ranked, start=1)
            )
        return selected

    def _global_chunks(self, hits, layer_id, config):
        """Rank chunks globally while enforcing distinct-URI and per-URI limits."""
        ranked = sorted(hits, key=lambda hit: (-hit.score, hit.entry.uri, hit.chunk.chunk_id))
        selected = []
        selected_uris = []
        count_by_uri = defaultdict(int)
        for hit in ranked:
            uri = hit.entry.uri
            if count_by_uri[uri] >= config.top_k_chunks_per_reference:
                continue
            if uri not in selected_uris and len(selected_uris) >= config.top_k_references:
                continue
            if uri not in selected_uris:
                selected_uris.append(uri)
            count_by_uri[uri] += 1
            reference_score = _aggregate(
                [candidate.score for candidate in hits if candidate.entry.uri == uri],
                config.reference_score_aggregation,
            )
            selected.append(
                self._selected(
                    hit,
                    reference_score,
                    layer_id,
                    selected_uris.index(uri) + 1,
                    count_by_uri[uri],
                )
            )
        return selected

    def search(
        self,
        query: torch.Tensor,
        layer_id: int,
        config: PRAConfig,
    ) -> list[list[SelectedChunk]]:
        """Route each query row with the configured strategy and independent budgets.

        Args:
            query: Last-token routing keys shaped ``[B,d_model]`` or ``[d_model]``.
            layer_id: Layer whose gists and eventual token K/V must be selected.
            config: ``PRAConfig`` carrying strategy, summary, and top-k modes.

        Returns:
            A batch-length list of ranked chunk lists. No row can see another
            row's selection, even though all rows search the same ready cache.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2:
            raise ValueError(f"Expected query [batch,model] or [model], got {tuple(query.shape)}.")
        if config.top_k_references == 0 or config.top_k_chunks_per_reference == 0:
            self._last_rankings_by_layer[layer_id] = [[] for _ in range(query.shape[0])]
            return [[] for _ in range(query.shape[0])]
        retain_rankings = bool(
            config.collect_routing_metrics or config.collect_rank_diagnostics
        )
        selected_by_batch = []
        if config.search_strategy == "reference_first":
            rankings_by_batch = []
            for row_query in query:
                selected_by_batch.append(self._reference_first(row_query, layer_id, config))
                if retain_rankings:
                    hits = self._score_chunks(row_query.unsqueeze(0), layer_id, config)[0]
                    reference_hits = self._score_references(row_query, layer_id, config)
                    rankings_by_batch.append(
                        self._serialize_rankings(
                            hits,
                            config,
                            reference_scores={
                                hit.entry.uri: hit.score for hit in reference_hits
                            },
                        )
                    )
                else:
                    rankings_by_batch.append([])
            self._last_rankings_by_layer[layer_id] = rankings_by_batch
            return selected_by_batch

        # Summary-combination modes retain the scalar compatibility path for now;
        # ordinary and multi-gist content routing uses the exact packed implementation.
        use_tensorized = config.routing_backend == "tensorized" and not config.use_summary
        tensorized_index = None
        chunk_scores = None
        reference_scores = None
        winning_indices = None
        winning_scores = None
        if use_tensorized:
            (
                hits_by_batch,
                tensorized_index,
                chunk_scores,
                reference_scores,
                winning_indices,
                winning_scores,
            ) = self._score_chunks_tensorized(
                query,
                layer_id,
                config,
                retain_full_hits=(
                    retain_rankings or config.search_strategy != "hierarchical"
                ),
            )
        else:
            hits_by_batch = self._score_chunks(query, layer_id, config)

        if (
            config.search_strategy == "hierarchical"
            and tensorized_index is not None
            and chunk_scores is not None
            and reference_scores is not None
            and winning_indices is not None
            and winning_scores is not None
        ):
            selected_by_batch = self._hierarchical_tensorized(
                hits_by_batch,
                tensorized_index,
                chunk_scores,
                reference_scores,
                winning_indices,
                winning_scores,
                layer_id,
                config,
            )
            self._last_rankings_by_layer[layer_id] = (
                [self._serialize_rankings(hits, config) for hits in hits_by_batch]
                if retain_rankings and hits_by_batch is not None
                else [[] for _ in range(query.shape[0])]
            )
            return selected_by_batch

        # Global routing shares vectorized scores but keeps its constrained policy loop.
        rankings_by_batch = []
        for hits in hits_by_batch:
            rankings = self._serialize_rankings(hits, config) if retain_rankings else []
            if config.search_strategy == "hierarchical":
                selected = self._hierarchical(hits, layer_id, config)
            elif config.search_strategy == "global_chunks":
                selected = self._global_chunks(hits, layer_id, config)
            else:
                raise ValueError(f"Unsupported search_strategy: {config.search_strategy}")
            selected_by_batch.append(selected)
            rankings_by_batch.append(rankings)
        self._last_rankings_by_layer[layer_id] = rankings_by_batch
        return selected_by_batch

    def search_by_routing_key(
        self,
        query: torch.Tensor,
        layer_id: int,
        top_k: int = 2,
    ) -> list[list[tuple[PRACacheEntry, float]]]:
        """Return URI entries/scores for legacy one-chunk-per-reference callers."""
        from types import SimpleNamespace

        search_config = SimpleNamespace(
            top_k_references=top_k,
            top_k_chunks_per_reference=1,
            search_strategy="hierarchical",
            reference_score_aggregation="max",
            reference_level_gist_mode=None,
            reference_gists_per_reference=1,
            reference_gist_score_aggregation="max",
            gist_score_aggregation="max",
            use_summary=False,
            summary_mode="replace",
        )
        selected = self.search(query, layer_id, search_config)
        return [
            [(hit.entry, hit.reference_score) for hit in batch_hits]
            for batch_hits in selected
        ]


class PRABatchedMemoryCache(PRAMemoryCache):
    """Route each logical batch row through its own completed cache namespace.

    The wrapper is attached only for the batched prompt forward. URI lookup is
    intentionally unavailable because two rows may bind the same URI to different
    content. Routing remains row-local while ``PRAttention`` and
    ``dynamic_memory_attention`` execute one query batch.
    """

    def __init__(self, row_caches: list[PRAMemoryCache]):
        """Keep row caches in the same order as ``input_ids`` batch rows."""
        self.row_caches = list(row_caches)

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        """Reject ambiguous flattening of duplicate URIs across row namespaces."""
        raise RuntimeError(
            "PRABatchedMemoryCache has no flat entries view; inspect row_caches explicitly."
        )

    def put(self, entry: PRACacheEntry) -> None:
        """Reject writes because row caches must be complete before wrapping."""
        del entry
        raise RuntimeError("Cannot put entries directly into PRABatchedMemoryCache.")

    def get(self, uri: str) -> PRACacheEntry | None:
        """Reject URI-only lookup because URI identity is scoped by batch row."""
        del uri
        raise RuntimeError("Batched cache lookup requires an explicit row cache.")

    def has(self, uri: str) -> bool:
        """Reject URI-only membership tests because namespaces are row-local."""
        del uri
        raise RuntimeError("Batched cache membership requires an explicit row cache.")

    def all_entries(self) -> list[PRACacheEntry]:
        """Return a diagnostic list that preserves duplicate entries as objects."""
        return [entry for cache in self.row_caches for entry in cache.all_entries()]

    def clear(self) -> None:
        """Clear every owned row namespace."""
        for cache in self.row_caches:
            cache.clear()

    def invalidate_routing_indexes(self) -> None:
        """Invalidate packed routing tensors in every row-local namespace."""
        for cache in self.row_caches:
            invalidate = getattr(cache, "invalidate_routing_indexes", None)
            if invalidate is not None:
                invalidate()

    def is_empty(self) -> bool:
        """Return true only when every row-local namespace is empty."""
        return all(cache.is_empty() for cache in self.row_caches)

    def search(
        self,
        query: torch.Tensor,
        layer_id: int,
        config: PRAConfig,
    ) -> list[list[SelectedChunk]]:
        """Route ``query[i]`` only against ``row_caches[i]``.

        ``query`` is ``[B,d_model]`` and the result has one selected-chunk list
        per row. The row loop is deliberate: the performance fix targets the
        expensive Transformer prompt forward while leaving routing vectorization
        as an independent optimization.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2:
            raise ValueError(f"Expected query [batch,model], got {tuple(query.shape)}.")
        if query.shape[0] != len(self.row_caches):
            raise ValueError(
                "Routing-query batch size must match row-local cache count: "
                f"{query.shape[0]} != {len(self.row_caches)}."
            )
        return [
            cache.search(query[row_index : row_index + 1], layer_id, config)[0]
            for row_index, cache in enumerate(self.row_caches)
        ]

    def last_rankings(self, layer_id: int) -> list[list[dict]]:
        """Return one latest candidate ranking from each row-local cache."""
        rankings = []
        for cache in self.row_caches:
            rows = cache.last_rankings(layer_id) if hasattr(cache, "last_rankings") else []
            rankings.append(rows[0] if rows else [])
        return rankings

    def __len__(self) -> int:
        """Count ready entries without collapsing duplicate URI strings."""
        return sum(len(cache) for cache in self.row_caches)
