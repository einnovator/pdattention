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
    k: torch.Tensor
    v: torch.Tensor


ChunkKV = LayerKV


@dataclass
class ChunkRoutingGist:
    k: torch.Tensor
    v: torch.Tensor | None = None
    method: str = "mean"
    summary_k: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ReferenceChunkMemory:
    chunk_id: str
    source_uri: str
    token_start: int
    token_end: int
    token_kv: ChunkKV
    routing_gist: ChunkRoutingGist
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return int(self.token_kv.k.shape[2])


@dataclass
class LayerReferenceMemory:
    chunks: list[ReferenceChunkMemory] = field(default_factory=list)


@dataclass
class PRACacheEntry:
    uri: str
    text: str
    layer_memory: dict[int, LayerReferenceMemory] = field(default_factory=dict)
    child_uris: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SelectedChunk:
    entry: PRACacheEntry
    chunk: ReferenceChunkMemory
    reference_score: float
    chunk_score: float
    layer_id: int
    reference_rank: int
    rank_within_reference: int
    metadata: dict = field(default_factory=dict)

    @property
    def reference_uri(self) -> str:
        return self.entry.uri

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_uri(self) -> str:
        return self.chunk.source_uri

    @property
    def token_start(self) -> int:
        return self.chunk.token_start

    @property
    def token_end(self) -> int:
        return self.chunk.token_end

    @property
    def selected_token_count(self) -> int:
        return self.chunk.token_count

    def as_trace_dict(self) -> dict:
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
    MISSING = "missing"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


def _aggregate(scores: list[float], mode: str) -> float:
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
    @abstractmethod
    def put(self, entry: PRACacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, uri: str) -> PRACacheEntry | None:
        raise NotImplementedError

    @abstractmethod
    def has(self, uri: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def all_entries(self) -> list[PRACacheEntry]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: torch.Tensor, layer_id: int, config) -> list[list[SelectedChunk]]:
        raise NotImplementedError

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        return {entry.uri: entry for entry in self.all_entries()}

    def __len__(self) -> int:
        return len(self.all_entries())


class PRASimpleMemoryCache(PRAMemoryCache):
    """In-memory cache with independent, hierarchical routing per batch item."""

    def __init__(self):
        self._entries: dict[str, PRACacheEntry] = {}
        self._states: dict[str, CacheBuildState] = {}
        self._failures: dict[str, str] = {}

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        return self._entries

    def begin_build(self, uri: str) -> None:
        self._states[uri] = CacheBuildState.BUILDING

    def mark_failed(self, uri: str, error: Exception | str) -> None:
        self._entries.pop(uri, None)
        self._states[uri] = CacheBuildState.FAILED
        self._failures[uri] = str(error)

    def state(self, uri: str) -> CacheBuildState:
        return self._states.get(uri, CacheBuildState.MISSING)

    def failure(self, uri: str) -> str | None:
        return self._failures.get(uri)

    def put(self, entry: PRACacheEntry) -> None:
        self._entries[entry.uri] = entry
        self._states[entry.uri] = CacheBuildState.READY
        self._failures.pop(entry.uri, None)

    def get(self, uri: str) -> PRACacheEntry | None:
        return self._entries.get(uri) if self.state(uri) == CacheBuildState.READY else None

    def has(self, uri: str) -> bool:
        return self.get(uri) is not None

    def all_entries(self) -> list[PRACacheEntry]:
        return [entry for uri, entry in self._entries.items() if self.state(uri) == CacheBuildState.READY]

    def clear(self) -> None:
        self._entries.clear()
        self._states.clear()
        self._failures.clear()

    def invalidate(self, uri: str) -> None:
        self._entries.pop(uri, None)
        self._states.pop(uri, None)
        self._failures.pop(uri, None)

    def layer_counts(self, layer_id: int) -> dict[str, int]:
        entries = [entry for entry in self.all_entries() if layer_id in entry.layer_memory]
        chunks = [chunk for entry in entries for chunk in entry.layer_memory[layer_id].chunks]
        return {
            "references": len(entries),
            "chunks": len(chunks),
            "tokens": sum(chunk.token_count for chunk in chunks),
        }

    def _score_chunks(self, query: torch.Tensor, layer_id: int, config):
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
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2:
            raise ValueError(f"Expected query [batch,model] or [model], got {tuple(query.shape)}.")
        if config.top_k_references == 0 or config.top_k_chunks_per_reference == 0:
            return [[] for _ in range(query.shape[0])]
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
        """Compatibility API for one-chunk-per-reference callers."""
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
