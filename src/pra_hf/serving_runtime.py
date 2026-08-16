"""Prototype serving primitives for adaptive PRA.

These implementations establish correctness, accounting, and benchmarkable
interfaces.  They are not presented as production CUDA or vLLM kernels.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SearchResult:
    """Top-K identities, scores, latency, and exact comparison accounting."""

    indices: torch.Tensor
    scores: torch.Tensor
    seconds: float
    comparisons: int
    backend: str


class NativeQKIndex:
    """Search native key vectors with exact and coarse-to-fine backends."""

    def __init__(self, keys: torch.Tensor, *, coarse_clusters: int = 32) -> None:
        if keys.ndim != 2 or keys.shape[0] == 0:
            raise ValueError("NativeQKIndex keys must have shape [memory, dimension].")
        self.keys = keys.contiguous()
        self.coarse_clusters = min(max(1, coarse_clusters), len(keys))
        self.assignments, self.centroids = self._build_coarse_index()

    def _build_coarse_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic short k-means used only by the prototype ANN backend."""

        count = len(self.keys)
        seeds = torch.linspace(0, count - 1, self.coarse_clusters, device=self.keys.device).long()
        centroids = self.keys[seeds].float().clone()
        values = self.keys.float()
        assignments = torch.zeros(count, dtype=torch.long, device=self.keys.device)
        for _ in range(4):
            assignments = torch.argmax(values @ centroids.T, dim=1)
            updated = []
            for cluster in range(self.coarse_clusters):
                members = values[assignments == cluster]
                updated.append(members.mean(dim=0) if len(members) else centroids[cluster])
            centroids = torch.stack(updated)
            centroids = torch.nn.functional.normalize(centroids, dim=1)
        return assignments, centroids.to(self.keys.dtype)

    @staticmethod
    def _validate_queries(queries: torch.Tensor, dimension: int, k: int) -> None:
        if queries.ndim != 2 or queries.shape[1] != dimension:
            raise ValueError("Queries must have shape [queries, key_dimension].")
        if k <= 0:
            raise ValueError("k must be positive.")

    def search(self, queries: torch.Tensor, k: int, *, backend: str = "gemm", probes: int = 4) -> SearchResult:
        self._validate_queries(queries, self.keys.shape[1], k)
        k = min(k, len(self.keys))
        started = time.perf_counter()
        if backend == "brute_force":
            row_scores = []
            for query in queries:
                row_scores.append(torch.stack([torch.dot(query, key) for key in self.keys]))
            scores = torch.stack(row_scores)
            top_scores, indices = torch.topk(scores, k, dim=1, sorted=True)
            comparisons = len(queries) * len(self.keys)
        elif backend == "gemm":
            scores = queries @ self.keys.T
            top_scores, indices = torch.topk(scores, k, dim=1, sorted=True)
            comparisons = len(queries) * len(self.keys)
        elif backend == "coarse_to_fine":
            probes = min(max(1, probes), self.coarse_clusters)
            coarse = queries @ self.centroids.T
            coarse_ids = torch.topk(coarse, probes, dim=1).indices
            rows, row_values = [], []
            comparisons = len(queries) * self.coarse_clusters
            for query, selected_clusters in zip(queries, coarse_ids):
                mask = torch.isin(self.assignments, selected_clusters)
                candidates = torch.nonzero(mask, as_tuple=False).flatten()
                if len(candidates) < k:
                    candidates = torch.arange(len(self.keys), device=self.keys.device)
                candidate_scores = query @ self.keys[candidates].T
                values, local = torch.topk(candidate_scores, k, sorted=True)
                rows.append(candidates[local])
                row_values.append(values)
                comparisons += len(candidates)
            indices, top_scores = torch.stack(rows), torch.stack(row_values)
        else:
            raise ValueError(f"Unsupported native-Q/K search backend: {backend}")
        if queries.device.type == "cuda":
            torch.cuda.synchronize(queries.device)
        return SearchResult(
            indices=indices,
            scores=top_scores,
            seconds=time.perf_counter() - started,
            comparisons=comparisons,
            backend=backend,
        )


def merge_token_intervals(intervals: Sequence[tuple[int, int]], token_count: int) -> tuple[tuple[int, int], ...]:
    """Validate, order, merge, and deduplicate half-open token intervals."""

    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    values = sorted((int(start), int(end)) for start, end in intervals)
    if any(start < 0 or end <= start or end > token_count for start, end in values):
        raise ValueError("Intervals must satisfy 0 <= start < end <= token_count.")
    merged: list[list[int]] = []
    for start, end in values:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


@dataclass(frozen=True)
class GatherResult:
    key: torch.Tensor
    value: torch.Tensor
    token_indices: torch.Tensor
    seconds: float
    requested_tokens: int
    materialized_tokens: int


def fused_gather_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    intervals: Sequence[tuple[int, int]],
) -> GatherResult:
    """Resolve intervals once and gather K/V with one device index operation.

    K/V use shape ``[batch, heads, tokens, head_dim]``.  Overlapping intervals
    are deduplicated before ``index_select`` so active memory has exact physical
    accounting.
    """

    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("K/V must have matching [B,H,T,Dh] shapes.")
    started = time.perf_counter()
    merged = merge_token_intervals(intervals, key.shape[2])
    indices = torch.cat(
        [torch.arange(start, end, device=key.device) for start, end in merged]
    )
    gathered_key = torch.index_select(key, 2, indices)
    gathered_value = torch.index_select(value, 2, indices)
    if key.device.type == "cuda":
        torch.cuda.synchronize(key.device)
    return GatherResult(
        gathered_key,
        gathered_value,
        indices,
        time.perf_counter() - started,
        sum(end - start for start, end in intervals),
        len(indices),
    )


@dataclass
class _Page:
    request_id: str
    page_index: int
    key: torch.Tensor
    value: torch.Tensor
    valid_tokens: int
    touched_at: int
    frequency: int = 1

    @property
    def bytes(self) -> int:
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


class PagedKVCache:
    """Fixed-page native-K/V store with LRU, LFU, and hybrid eviction."""

    def __init__(self, page_size: int, capacity_pages: int, *, policy: str = "lru") -> None:
        if page_size <= 0 or capacity_pages <= 0:
            raise ValueError("Page size and capacity must be positive.")
        if policy not in {"lru", "lfu", "hybrid"}:
            raise ValueError("Cache policy must be lru, lfu, or hybrid.")
        self.page_size = page_size
        self.capacity_pages = capacity_pages
        self.policy = policy
        self._pages: dict[tuple[str, int], _Page] = {}
        self._request_lengths: dict[str, int] = {}
        self._clock = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _victim(self) -> tuple[str, int]:
        if self.policy == "lru":
            key = min(self._pages, key=lambda item: (self._pages[item].touched_at, item))
        elif self.policy == "lfu":
            key = min(
                self._pages,
                key=lambda item: (self._pages[item].frequency, self._pages[item].touched_at, item),
            )
        else:
            key = min(
                self._pages,
                key=lambda item: (
                    self._pages[item].frequency / max(self._clock - self._pages[item].touched_at + 1, 1),
                    self._pages[item].touched_at,
                    item,
                ),
            )
        return key

    def put(self, request_id: str, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store ``[heads,tokens,head_dim]`` K/V as independently reusable pages."""

        if not request_id or key.ndim != 3 or key.shape != value.shape:
            raise ValueError("Paged K/V requires a request id and matching [H,T,Dh] tensors.")
        token_count = key.shape[1]
        self._request_lengths[request_id] = token_count
        for page_index, start in enumerate(range(0, token_count, self.page_size)):
            end = min(start + self.page_size, token_count)
            page_key = key[:, start:end, :].contiguous()
            page_value = value[:, start:end, :].contiguous()
            identity = (request_id, page_index)
            if identity not in self._pages and len(self._pages) >= self.capacity_pages:
                del self._pages[self._victim()]
                self.evictions += 1
            self._pages[identity] = _Page(
                request_id,
                page_index,
                page_key,
                page_value,
                end - start,
                self._tick(),
            )

    def gather(self, request_id: str, token_indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather requested tokens in caller order while accounting page reuse."""

        if request_id not in self._request_lengths:
            raise KeyError(request_id)
        length = self._request_lengths[request_id]
        keys, values = [], []
        for token in token_indices:
            if token < 0 or token >= length:
                raise IndexError(token)
            page_index, offset = divmod(int(token), self.page_size)
            identity = (request_id, page_index)
            page = self._pages.get(identity)
            if page is None:
                self.misses += 1
                raise KeyError(f"K/V page was evicted: {identity}")
            self.hits += 1
            page.frequency += 1
            page.touched_at = self._tick()
            keys.append(page.key[:, offset : offset + 1, :])
            values.append(page.value[:, offset : offset + 1, :])
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    @property
    def resident_bytes(self) -> int:
        return sum(page.bytes for page in self._pages.values())

    @property
    def fragmentation_tokens(self) -> int:
        return sum(self.page_size - page.valid_tokens for page in self._pages.values())

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.hits + self.misses, 1)

    def stats(self) -> dict[str, float]:
        return {
            "resident_pages": float(len(self._pages)),
            "resident_bytes": float(self.resident_bytes),
            "fragmentation_tokens": float(self.fragmentation_tokens),
            "hit_rate": self.hit_rate,
            "evictions": float(self.evictions),
        }


@dataclass(frozen=True)
class PackedBatch:
    """Ragged values plus prefix offsets; no padding enters attention memory."""

    values: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor


def pack_ragged(tensors: Sequence[torch.Tensor]) -> PackedBatch:
    """Pack aligned ``[tokens, ...]`` tensors and retain request boundaries."""

    if not tensors or any(tensor.ndim == 0 for tensor in tensors):
        raise ValueError("Ragged packing requires nonempty tensors with a token axis.")
    trailing = tensors[0].shape[1:]
    if any(tensor.shape[1:] != trailing for tensor in tensors):
        raise ValueError("Ragged tensors must share non-token dimensions.")
    lengths = torch.tensor([len(tensor) for tensor in tensors], dtype=torch.long, device=tensors[0].device)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=lengths.device), lengths.cumsum(dim=0)]
    )
    return PackedBatch(torch.cat(list(tensors), dim=0), offsets, lengths)


def unpack_ragged(batch: PackedBatch) -> tuple[torch.Tensor, ...]:
    return tuple(
        batch.values[int(batch.offsets[index]) : int(batch.offsets[index + 1])]
        for index in range(len(batch.lengths))
    )


@dataclass(frozen=True)
class ServingMetrics:
    """Measured request-level runtime metrics with explicit scope metadata."""

    request_id: str
    effort: str
    active_kv_tokens: int
    hbm_bytes: int
    routing_seconds: float
    graph_search_seconds: float
    gather_seconds: float
    h2d_bytes: int
    h2d_seconds: float
    ttft_seconds: float
    tpot_seconds: float
    total_seconds: float
    tokens_per_second: float
    cache_hit_rate: float
    measured: bool = True
    measurement_scope: str = "prototype_component"

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)
