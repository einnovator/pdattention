"""Chunk-aware, layer-specific memory cache and routing for PRA."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum

import torch
import torch.nn.functional as F


@dataclass
class LayerKV:
    """Projected attention memory for one chunk at one decoder layer.

    Both tensors have shape ``[1, heads, memory_tokens, head_dim]``. Reference
    chunks are encoded independently, hence the singleton batch dimension.
    """

    k: torch.Tensor  # Keys used for routing-gist construction and cross-attention.
    v: torch.Tensor  # Values returned when a query attends to this memory.


ChunkKV = LayerKV


@dataclass
class ChunkRoutingGist:
    """Small layer-specific representation used to route before loading token K/V."""

    k: torch.Tensor  # Content routing key with shape [d_model].
    v: torch.Tensor | None = None  # Optional gist-only value with shape [d_model].
    method: str = "mean"  # Pooling rule that produced k/v.
    summary_k: torch.Tensor | None = None  # Separately encoded summary key [d_model].
    metadata: dict = field(default_factory=dict)  # Pooling/source diagnostics.


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
            "selected_token_count": self.selected_token_count,
            "reference_rank": self.reference_rank,
            "rank_within_reference": self.rank_within_reference,
            "metadata": self.metadata,
        }


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


def _gist_vectors(gist: ChunkRoutingGist, *, use_summary: bool, summary_mode: str):
    """Return candidate routing vectors and a label describing their source.

    ``hybrid`` keeps content and summary as separate candidates and uses their
    best score. ``augment`` combines their normalized directions into one key.
    """
    content = gist.k
    summary = gist.summary_k
    if not use_summary or summary is None:
        return [content], "content"
    if summary_mode == "replace":
        return [summary], "summary"
    if summary_mode == "hybrid":
        return [content, summary], "hybrid"
    if summary_mode == "augment":
        combined = F.normalize(content, dim=-1) + F.normalize(summary, dim=-1)
        return [F.normalize(combined, dim=-1)], "augment"
    raise ValueError(f"Unsupported summary mode: {summary_mode}")


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
    def search(self, query: torch.Tensor, layer_id: int, config) -> list[list[SelectedChunk]]:
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

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        """Return the internal URI map; published entries are normally ``READY``."""
        return self._entries

    def begin_build(self, uri: str) -> None:
        """Mark a URI as under construction to detect recursive re-entry."""
        self._states[uri] = CacheBuildState.BUILDING

    def mark_failed(self, uri: str, error: Exception | str) -> None:
        """Hide a failed payload while retaining a diagnostic message."""
        self._entries.pop(uri, None)
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

    def is_empty(self) -> bool:
        """Return whether this row-local namespace has no ready entries."""
        return not any(self.state(uri) == CacheBuildState.READY for uri in self._entries)

    def invalidate(self, uri: str) -> None:
        """Remove a stale URI before rebuilding it with new fingerprints."""
        self._entries.pop(uri, None)
        self._states.pop(uri, None)
        self._failures.pop(uri, None)

    def layer_counts(self, layer_id: int) -> dict[str, int]:
        """Count searchable URIs, chunks, and token positions for one layer."""
        entries = [entry for entry in self.all_entries() if layer_id in entry.layer_memory]
        chunks = [chunk for entry in entries for chunk in entry.layer_memory[layer_id].chunks]
        return {
            "references": len(entries),
            "chunks": len(chunks),
            "tokens": sum(chunk.token_count for chunk in chunks),
        }

    def _score_chunks(self, query: torch.Tensor, layer_id: int, config):
        """Cosine-score every layer gist independently for each query row.

        ``query`` is ``[B,d_model]``. Each result tuple retains the entry/chunk
        object so later stages can rank cheaply without moving full token K/V.
        """
        records = []
        for entry in self.all_entries():
            memory = entry.layer_memory.get(layer_id)
            if memory is None:
                continue
            for chunk in memory.chunks:
                vectors, routing_source = _gist_vectors(
                    chunk.routing_gist,
                    use_summary=config.use_summary,
                    summary_mode=config.summary_mode,
                )
                records.append((entry, chunk, vectors, routing_source))
        if not records:
            return [[] for _ in range(query.shape[0])]

        # Normalize once per query; cache gists may live on CPU or another dtype.
        result = []
        query_norm = F.normalize(query, dim=-1)
        for batch_index in range(query.shape[0]):
            hits = []
            for entry, chunk, vectors, routing_source in records:
                scores = [
                    float(
                        torch.dot(
                            query_norm[batch_index],
                            F.normalize(vector.to(query.device, query.dtype), dim=-1),
                        )
                        .detach()
                        .cpu()
                    )
                    for vector in vectors
                ]
                hits.append((entry, chunk, max(scores), routing_source))
            result.append(hits)
        return result

    def _reference_first_scores(self, query, hits, config):
        """Build one vector per URI, then score URIs before ranking their chunks."""
        grouped = defaultdict(list)
        for entry, chunk, chunk_score, routing_source in hits:
            grouped[entry.uri].append((entry, chunk, chunk_score, routing_source))
        mode = config.reference_level_gist_mode
        if mode is None:
            raise ValueError(
                "search_strategy='reference_first' requires reference_level_gist_mode."
            )
        reference_scores = {}
        for uri, reference_hits in grouped.items():
            vectors = [hit[1].routing_gist.k.to(query.device, query.dtype) for hit in reference_hits]
            if mode == "mean":
                vector = torch.stack(vectors).mean(dim=0)
            elif mode == "last":
                vector = vectors[-1]
            else:
                raise NotImplementedError(
                    "reference_level_gist_mode='gru' requires an explicit registered reference aggregator."
                )
            reference_scores[uri] = float(
                torch.dot(F.normalize(query, dim=-1), F.normalize(vector, dim=-1)).detach().cpu()
            )
        return grouped, reference_scores

    @staticmethod
    def _selected(entry, chunk, reference_score, chunk_score, layer_id, ref_rank, chunk_rank, source):
        """Attach stable ranks and routing provenance to a selected payload."""
        return SelectedChunk(
            entry=entry,
            chunk=chunk,
            reference_score=reference_score,
            chunk_score=chunk_score,
            layer_id=layer_id,
            reference_rank=ref_rank,
            rank_within_reference=chunk_rank,
            metadata={"routing_source": source},
        )

    def _hierarchical(self, hits, layer_id, config):
        """Aggregate chunk scores per URI, select URIs, then select local chunks."""
        grouped = defaultdict(list)
        for hit in hits:
            grouped[hit[0].uri].append(hit)
        reference_scores = {
            uri: _aggregate([hit[2] for hit in grouped_hits], config.reference_score_aggregation)
            for uri, grouped_hits in grouped.items()
        }
        selected_uris = sorted(reference_scores, key=lambda uri: (-reference_scores[uri], uri))[
            : config.top_k_references
        ]
        selected = []
        for ref_rank, uri in enumerate(selected_uris, start=1):
            ranked = sorted(grouped[uri], key=lambda hit: (-hit[2], hit[1].chunk_id))[
                : config.top_k_chunks_per_reference
            ]
            selected.extend(
                self._selected(*hit[:2], reference_scores[uri], hit[2], layer_id, ref_rank, rank, hit[3])
                for rank, hit in enumerate(ranked, start=1)
            )
        return selected

    def _reference_first(self, query, hits, layer_id, config):
        """Select URIs from explicit URI gists, then use chunk scores within them."""
        grouped, reference_scores = self._reference_first_scores(query, hits, config)
        selected_uris = sorted(reference_scores, key=lambda uri: (-reference_scores[uri], uri))[
            : config.top_k_references
        ]
        selected = []
        for ref_rank, uri in enumerate(selected_uris, start=1):
            ranked = sorted(grouped[uri], key=lambda hit: (-hit[2], hit[1].chunk_id))[
                : config.top_k_chunks_per_reference
            ]
            selected.extend(
                self._selected(*hit[:2], reference_scores[uri], hit[2], layer_id, ref_rank, rank, hit[3])
                for rank, hit in enumerate(ranked, start=1)
            )
        return selected

    def _global_chunks(self, hits, layer_id, config):
        """Rank chunks globally while enforcing distinct-URI and per-URI limits."""
        ranked = sorted(hits, key=lambda hit: (-hit[2], hit[0].uri, hit[1].chunk_id))
        selected = []
        selected_uris = []
        count_by_uri = defaultdict(int)
        for hit in ranked:
            uri = hit[0].uri
            if count_by_uri[uri] >= config.top_k_chunks_per_reference:
                continue
            if uri not in selected_uris and len(selected_uris) >= config.top_k_references:
                continue
            if uri not in selected_uris:
                selected_uris.append(uri)
            count_by_uri[uri] += 1
            reference_score = _aggregate(
                [candidate[2] for candidate in hits if candidate[0].uri == uri],
                config.reference_score_aggregation,
            )
            selected.append(
                self._selected(
                    hit[0],
                    hit[1],
                    reference_score,
                    hit[2],
                    layer_id,
                    selected_uris.index(uri) + 1,
                    count_by_uri[uri],
                    hit[3],
                )
            )
        return selected

    def search(self, query: torch.Tensor, layer_id: int, config) -> list[list[SelectedChunk]]:
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
            return [[] for _ in range(query.shape[0])]
        # Stage 1 scores lightweight gists; stage 2 applies a policy and budgets.
        hits_by_batch = self._score_chunks(query, layer_id, config)
        selected_by_batch = []
        for batch_index, hits in enumerate(hits_by_batch):
            if config.search_strategy == "hierarchical":
                selected = self._hierarchical(hits, layer_id, config)
            elif config.search_strategy == "reference_first":
                selected = self._reference_first(query[batch_index], hits, layer_id, config)
            elif config.search_strategy == "global_chunks":
                selected = self._global_chunks(hits, layer_id, config)
            else:
                raise ValueError(f"Unsupported search_strategy: {config.search_strategy}")
            selected_by_batch.append(selected)
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

    def is_empty(self) -> bool:
        """Return true only when every row-local namespace is empty."""
        return all(cache.is_empty() for cache in self.row_caches)

    def search(
        self,
        query: torch.Tensor,
        layer_id: int,
        config,
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

    def __len__(self) -> int:
        """Count ready entries without collapsing duplicate URI strings."""
        return sum(len(cache) for cache in self.row_caches)
