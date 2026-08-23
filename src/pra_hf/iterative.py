"""Bounded associative closure over compact PRA routing gists.

This module deliberately knows nothing about attention or native K/V materialization.
It traverses a layer-local gist index and returns stable chunk identities plus a
versioned retrieval graph.  :mod:`pra_hf.model` maps those identities to native
payloads only after traversal has stopped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from pra_torch.memory import PRACacheEntry, ReferenceChunkMemory, SelectedChunk

from .hybrid_discovery import (
    DiscoveryCandidate,
    HybridDiscoveryPolicy,
    TokenNativeIndex,
)


GRAPH_SCHEMA_VERSION = "2.0"
_PATH_MODES = {"product", "logsum", "last", "min", "mean", "direct"}
_FRONTIER_MODES = {"direct", "residual", "mean", "weighted_mean"}
_FRONTIER_PROJECTIONS = {"memory", "query"}


@dataclass(frozen=True)
class IterativeRoutingConfig:
    """Control bounded gist traversal independently from K/V materialization.

    ``branch_top_k`` limits neighbors proposed by each frontier node,
    ``beam_size`` limits accepted nodes in one expansion, and
    ``max_unique_chunks`` is the hard final memory budget.  ``depth=1`` with
    ``branch_top_k >= max_unique_chunks`` is the matched one-shot Top-B case.
    """

    depth: int = 2
    branch_top_k: int = 2
    beam_size: int = 8
    max_unique_chunks: int = 8
    root_anchor_alpha: float = 0.5
    frontier_mode: str = "direct"
    frontier_projection: str = "memory"
    residual_beta: float = 1.0
    path_score_mode: str = "product"
    min_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth must be non-negative.")
        for name in ("branch_top_k", "beam_size", "max_unique_chunks"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 <= self.root_anchor_alpha <= 1.0:
            raise ValueError("root_anchor_alpha must lie in [0, 1].")
        if self.frontier_mode not in _FRONTIER_MODES:
            raise ValueError(f"Unsupported frontier_mode: {self.frontier_mode}")
        if self.frontier_projection not in _FRONTIER_PROJECTIONS:
            raise ValueError(
                f"Unsupported frontier_projection: {self.frontier_projection}"
            )
        if self.path_score_mode not in _PATH_MODES:
            raise ValueError(f"Unsupported path_score_mode: {self.path_score_mode}")
        if self.residual_beta < 0:
            raise ValueError("residual_beta must be non-negative.")


@dataclass(frozen=True)
class GistIndex:
    """Packed layer-local chunk gists without any native K/V tensor copies.

    ``gists`` has shape ``[chunks,max_gists,routing_width]`` and ``gist_mask``
    marks real rows where chunks carry different numbers of routing gists.
    ``records`` retains stable cache identities; it does not materialize payloads.
    """

    layer_id: int
    records: tuple[tuple[PRACacheEntry, ReferenceChunkMemory], ...]
    gists: torch.Tensor
    gist_mask: torch.Tensor
    query_gists: torch.Tensor | None = None

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[PRACacheEntry],
        layer_id: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GistIndex":
        """Pack only gists belonging to ``layer_id`` in stable identity order."""
        records = []
        for entry in entries:
            memory = entry.layer_memory.get(layer_id)
            if memory is not None:
                records.extend((entry, chunk) for chunk in memory.chunks)
        records.sort(key=lambda pair: (pair[0].uri, pair[1].chunk_id))
        if not records:
            empty = torch.empty((0, 0, 0), device=device, dtype=dtype)
            return cls(layer_id, (), empty, torch.empty((0, 0), device=device, dtype=torch.bool))
        widths = {int(chunk.routing_gist.k.shape[-1]) for _, chunk in records}
        if len(widths) != 1:
            raise ValueError("All layer-local routing gists must have the same width.")
        counts = [int(chunk.routing_gist.k.shape[0]) for _, chunk in records]
        maximum = max(counts)
        width = widths.pop()
        first = records[0][1].routing_gist.k
        target_device = first.device if device is None else torch.device(device)
        packed = torch.zeros((len(records), maximum, width), device=target_device, dtype=dtype)
        mask = torch.zeros((len(records), maximum), device=target_device, dtype=torch.bool)
        for row, ((_, chunk), count) in enumerate(zip(records, counts)):
            packed[row, :count] = chunk.routing_gist.k.to(target_device, dtype)
            mask[row, :count] = True
        query_packed = None
        if all(chunk.routing_gist.query_k is not None for _, chunk in records):
            query_packed = torch.zeros_like(packed)
            for row, ((_, chunk), count) in enumerate(zip(records, counts)):
                query_packed[row, :count] = chunk.routing_gist.query_k.to(
                    target_device, dtype
                )
            query_packed = F.normalize(query_packed, dim=-1, eps=1e-12)
        return cls(
            layer_id,
            tuple(records),
            F.normalize(packed, dim=-1, eps=1e-12),
            mask,
            query_packed,
        )

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for _, chunk in self.records)


@dataclass(frozen=True)
class HierarchicalGistIndex:
    """Contextual parent means plus finer local gists for propagation.

    Parent tensors are ``[P,D_route]``. Local tensors are ``[L,D_route]`` and
    ``local_parent_indices[L]`` maps every local node to its materialization
    parent. Query and memory tensors are aligned projections of the same hidden
    states under the asymmetric routing contract.
    """

    parent_ids: tuple[str, ...]
    parent_spans: tuple[tuple[int, int], ...]
    parent_memory_gists: torch.Tensor
    parent_query_gists: torch.Tensor
    local_spans: tuple[tuple[int, int], ...]
    local_parent_indices: torch.Tensor
    local_memory_gists: torch.Tensor
    local_query_gists: torch.Tensor
    layer_id: int = 0
    records: tuple[tuple[PRACacheEntry, ReferenceChunkMemory], ...] = ()

    def __post_init__(self) -> None:
        parent_count = len(self.parent_ids)
        local_count = len(self.local_spans)
        if len(self.parent_spans) != parent_count:
            raise ValueError("Parent identities and spans must align.")
        if self.parent_memory_gists.ndim != 2:
            raise ValueError("Parent memory gists must have shape [parents,width].")
        if self.parent_query_gists.shape != self.parent_memory_gists.shape:
            raise ValueError("Parent query and memory gists must align.")
        if self.parent_memory_gists.shape[0] != parent_count:
            raise ValueError("Parent tensors must align with parent identities.")
        if self.local_memory_gists.ndim != 2:
            raise ValueError("Local memory gists must have shape [locals,width].")
        if self.local_query_gists.shape != self.local_memory_gists.shape:
            raise ValueError("Local query and memory gists must align.")
        if self.local_memory_gists.shape[0] != local_count:
            raise ValueError("Local tensors must align with local spans.")
        if self.local_memory_gists.shape[-1] != self.parent_memory_gists.shape[-1]:
            raise ValueError("Parent and local routing widths must match.")
        if self.local_parent_indices.shape != (local_count,):
            raise ValueError("local_parent_indices must have shape [locals].")
        if local_count and (
            int(self.local_parent_indices.min()) < 0
            or int(self.local_parent_indices.max()) >= parent_count
        ):
            raise ValueError("Every local node must map to a valid parent.")

    @property
    def device(self) -> torch.device:
        return self.parent_memory_gists.device

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[PRACacheEntry],
        layer_id: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "HierarchicalGistIndex":
        """Build parents and local nodes from one contextual segment-gist pass."""
        records = []
        for entry in entries:
            memory = entry.layer_memory.get(layer_id)
            if memory is not None:
                records.extend((entry, chunk) for chunk in memory.chunks)
        records.sort(key=lambda pair: (pair[0].uri, pair[1].chunk_id))
        if not records:
            raise ValueError("Hierarchical routing requires at least one cached parent.")
        if not all(chunk.routing_gist.query_k is not None for _, chunk in records):
            raise ValueError("Hierarchical routing requires query-projected local gists.")
        target = records[0][1].routing_gist.k.device if device is None else torch.device(device)
        parent_ids, parent_spans, local_spans, local_parents = [], [], [], []
        parent_memory, parent_query, local_memory, local_query = [], [], [], []
        for parent_index, (_, chunk) in enumerate(records):
            memory_gists = chunk.routing_gist.k.to(target, dtype)
            query_gists = chunk.routing_gist.query_k.to(target, dtype)
            spans = chunk.routing_gist.metadata.get("segment_token_spans")
            if not spans or len(spans) != len(memory_gists):
                raise ValueError(
                    "Hierarchical routing requires segment_token_spans for every local gist."
                )
            occupancy = torch.tensor(
                [int(end) - int(start) for start, end in spans],
                device=target,
                dtype=dtype,
            )
            weights = occupancy / occupancy.sum()
            parent_memory.append(
                chunk.routing_gist.parent_k[0].to(target, dtype)
                if chunk.routing_gist.parent_k is not None
                else (memory_gists * weights[:, None]).sum(0)
            )
            parent_query.append(
                chunk.routing_gist.parent_query_k[0].to(target, dtype)
                if chunk.routing_gist.parent_query_k is not None
                else (query_gists * weights[:, None]).sum(0)
            )
            parent_ids.append(chunk.chunk_id)
            parent_spans.append((int(chunk.logical_start), int(chunk.logical_end)))
            for memory_gist, query_gist, (start, end) in zip(
                memory_gists, query_gists, spans
            ):
                local_memory.append(memory_gist)
                local_query.append(query_gist)
                local_parents.append(parent_index)
                local_spans.append(
                    (
                        int(chunk.logical_start) + int(start),
                        int(chunk.logical_start) + int(end),
                    )
                )
        return cls(
            tuple(parent_ids), tuple(parent_spans),
            F.normalize(torch.stack(parent_memory), dim=-1),
            F.normalize(torch.stack(parent_query), dim=-1),
            tuple(local_spans), torch.tensor(local_parents, device=target),
            F.normalize(torch.stack(local_memory), dim=-1),
            F.normalize(torch.stack(local_query), dim=-1), layer_id, tuple(records),
        )


def _entropy(scores: torch.Tensor) -> float:
    """Return entropy of a softmax over finite diagnostic scores."""
    finite = scores[torch.isfinite(scores)]
    if finite.numel() <= 1:
        return 0.0
    probabilities = torch.softmax(finite.float(), dim=0)
    return float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())


@dataclass
class RetrievalNode:
    """One chunk discovered in a query-conditioned latent memory graph."""

    node_id: str
    reference_uri: str
    hop: int
    parent_ids: list[str]
    direct_query_score: float
    edge_score: float
    path_score: float
    winning_gist_index: int
    selection_reason: str = "frontier_topk"
    final_selected: bool = True
    materialized: bool = False
    evidence: bool | None = None
    parent_chunk_id: str | None = None
    local_span: tuple[int, int] | None = None
    resolution_level: str = "chunk"
    representation_type: str = "semantic_gist"
    projection_type: str = "memory"
    discovery_channels: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    confidence_calibrated: bool = False


@dataclass
class RetrievalEdge:
    """Directed discovery relation retained for later graph-guided budgeting."""

    source: str
    target: str
    hop: int
    edge_score: float
    anchored_score: float
    path_score: float
    selected: bool
    edge_type: str = "semantic_similarity"
    representation_type: str = "semantic_gist"
    projection_type: str = "memory_to_memory"
    head_id: int | None = None
    query_head: int | None = None
    kv_head: int | None = None
    score: float | None = None
    threshold: float | None = None
    source_span: tuple[int, int] | None = None
    target_span: tuple[int, int] | None = None
    semantic_candidate_rank: int | None = None
    accepted: bool = True
    source_node: str | None = None
    target_node: str | None = None

    def __post_init__(self) -> None:
        """Expose explicit schema-v2 aliases while retaining v1 field names."""
        self.source_node = self.source if self.source_node is None else self.source_node
        self.target_node = self.target if self.target_node is None else self.target_node


@dataclass
class RetrievalGraph:
    """Versioned interface between iterative routing and later materialization."""

    example_id: str | None
    layer_id: int
    root: dict[str, Any]
    nodes: list[RetrievalNode] = field(default_factory=list)
    edges: list[RetrievalEdge] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = "depth"
    schema_version: str = GRAPH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterativeRoutingResult:
    """Selected index rows, graph trace, and one-shot root scores for one query."""

    selected_indices: tuple[int, ...]
    direct_scores: tuple[float, ...]
    graph: RetrievalGraph


@dataclass(frozen=True)
class _Frontier:
    index: int | None
    node_id: str
    query: torch.Tensor
    path_score: float
    edge_scores: tuple[float, ...]
    token_ids: tuple[int, ...] = ()


def _path_score(mode: str, direct: float, edges: tuple[float, ...]) -> float:
    """Reduce edge affinities in ``[0,1]`` to one transparent path score."""
    if mode == "direct":
        return direct
    if mode == "last":
        return edges[-1]
    if mode == "min":
        return min(edges)
    if mode == "mean":
        return sum(edges) / len(edges)
    if mode == "logsum":
        return sum(math.log(max(value, 1e-12)) for value in edges)
    return math.prod(edges)


class IterativeGistRouter:
    """Perform Level-0/1/2 retrieval using lazy tensorized gist comparisons."""

    def __init__(self, index: GistIndex):
        self.index = index

    def _scores(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return max cosine scores and winning gist rows as ``[frontier,chunks]``."""
        queries = F.normalize(queries.to(self.index.gists), dim=-1, eps=1e-12)
        if queries.ndim != 2 or queries.shape[-1] != self.index.gists.shape[-1]:
            raise ValueError("Queries must have shape [frontier,routing_width].")
        scores = torch.einsum("fd,cgd->fcg", queries, self.index.gists)
        scores = scores.masked_fill(~self.index.gist_mask.unsqueeze(0), float("-inf"))
        return scores.max(dim=-1)

    @staticmethod
    def _topk(values: torch.Tensor, k: int) -> list[int]:
        """Use torch.topk with an index tie-break for CPU/GPU-stable identities."""
        if k <= 0 or values.numel() == 0:
            return []
        finite = torch.isfinite(values)
        k = min(k, int(finite.sum().item()))
        if k == 0:
            return []
        work = values.to(torch.float64)
        tie = torch.arange(work.numel(), device=work.device, dtype=work.dtype)
        work = work - tie * torch.finfo(work.dtype).eps
        return torch.topk(work, k=k, sorted=True).indices.detach().cpu().tolist()

    def route(
        self,
        root_query: torch.Tensor,
        config: IterativeRoutingConfig,
        *,
        example_id: str | None = None,
        evidence_chunk_ids: set[str] | None = None,
        token_index: TokenNativeIndex | None = None,
        root_token_ids: Iterable[int] | None = None,
        tokenizer: Any | None = None,
        discovery_policy: HybridDiscoveryPolicy | None = None,
        explicit_reference_uris: set[str] | None = None,
        token_embedding_weight: torch.Tensor | None = None,
    ) -> IterativeRoutingResult:
        """Traverse compact gists and stop before any native K/V is requested.

        Supplying a token index enables lexical or hybrid scoring inside this
        same bounded loop.  The sidecar must align exactly with the semantic
        index; the default call path remains semantic-only.
        """
        hybrid_enabled = token_index is not None
        if hybrid_enabled:
            if root_token_ids is None or tokenizer is None:
                raise ValueError(
                    "token_index requires root_token_ids and tokenizer."
                )
            token_index.validate_alignment(self.index)
            discovery_policy = discovery_policy or HybridDiscoveryPolicy()
        elif any(
            value is not None
            for value in (
                root_token_ids,
                tokenizer,
                discovery_policy,
                token_embedding_weight,
            )
        ):
            raise ValueError("Token discovery arguments require token_index.")
        if root_query.ndim == 2:
            if root_query.shape[0] != 1:
                raise ValueError("route handles one query row; call route_batch for batches.")
            root_query = root_query[0]
        if self.index.gists.numel() == 0:
            graph = RetrievalGraph(
                example_id, self.index.layer_id, {"node_id": "__root__"},
                budget=asdict(config), stop_reason="empty_index",
            )
            return IterativeRoutingResult((), (), graph)
        root = F.normalize(root_query.to(self.index.gists), dim=-1, eps=1e-12)
        if config.frontier_projection == "query" and self.index.query_gists is None:
            raise ValueError(
                "Query-projected closure requires aligned query_gists in the index."
            )
        root_scores_t, _ = self._scores(root.unsqueeze(0))
        semantic_direct_scores = root_scores_t[0]
        root_candidates: dict[int, DiscoveryCandidate] | None = None
        if hybrid_enabled:
            root_candidates = token_index.score(
                root_token_ids,
                semantic_direct_scores,
                tokenizer,
                discovery_policy,
                hop=1,
                parent_id="__root__",
                explicit_reference_uris=explicit_reference_uris,
                token_embedding_weight=token_embedding_weight,
                sparse=True,
            )
            direct_scores = semantic_direct_scores.new_full(
                semantic_direct_scores.shape, float("-inf")
            )
            for index, candidate in root_candidates.items():
                direct_scores[index] = 2.0 * candidate.selected_score - 1.0
        else:
            direct_scores = semantic_direct_scores
        graph = RetrievalGraph(
            example_id=example_id,
            layer_id=self.index.layer_id,
            root={
                "node_id": "__root__",
                "query_norm": float(root.norm().item()),
                "discovery_mode": (
                    discovery_policy.mode if discovery_policy is not None else "gist_only"
                ),
            },
            budget=asdict(config),
        )
        if config.depth == 0 or config.max_unique_chunks == 0 or config.branch_top_k == 0:
            graph.stop_reason = "zero_limit"
            return IterativeRoutingResult((), tuple(direct_scores.cpu().tolist()), graph)

        visited: set[int] = set()
        frontier = [
            _Frontier(
                None,
                "__root__",
                root,
                1.0,
                (),
                tuple(int(value) for value in (root_token_ids or ())),
            )
        ]
        comparisons = duplicates = proposals = 0
        token_comparisons = 0
        token_index_queries = int(hybrid_enabled)
        indexed_token_comparisons = int(
            token_index.last_search_stats.get("expensive_comparisons", 0)
            if hybrid_enabled
            else 0
        )
        overlap_ratios: list[float] = []
        accepted_per_hop: list[int] = []
        stop_reason = "depth"

        for hop in range(1, config.depth + 1):
            queries = torch.stack([item.query for item in frontier])
            semantic_scores, frontier_winners = self._scores(queries)
            frontier_scores = semantic_scores
            frontier_candidates: list[dict[int, DiscoveryCandidate] | None] = [
                None for _ in frontier
            ]
            if hybrid_enabled:
                hybrid_rows = []
                for parent_row, parent in enumerate(frontier):
                    if hop == 1 and parent.index is None:
                        candidates = root_candidates
                    else:
                        candidates = token_index.score(
                            parent.token_ids,
                            semantic_scores[parent_row],
                            tokenizer,
                            discovery_policy,
                            hop=hop,
                            parent_id=parent.node_id,
                            explicit_reference_uris=explicit_reference_uris,
                            token_embedding_weight=token_embedding_weight,
                            sparse=True,
                        )
                        indexed_token_comparisons += int(
                            token_index.last_search_stats.get("expensive_comparisons", 0)
                        )
                        token_index_queries += 1
                    frontier_candidates[parent_row] = candidates
                    hybrid = semantic_scores.new_full(
                        (len(self.index.records),), float("-inf")
                    )
                    for index, candidate in candidates.items():
                        hybrid[index] = 2.0 * candidate.selected_score - 1.0
                    hybrid_rows.append(hybrid)
                    token_comparisons += len(candidates)
                frontier_scores = torch.stack(hybrid_rows)
            comparisons += int(frontier_scores.numel())
            candidate_parents: dict[
                int, list[tuple[float, int, float, float, int, DiscoveryCandidate | None]]
            ] = {}
            proposed_rows: list[set[int]] = []
            for parent_row, parent in enumerate(frontier):
                scores = frontier_scores[parent_row]
                anchored = (
                    config.root_anchor_alpha * direct_scores
                    + (1.0 - config.root_anchor_alpha) * scores
                )
                masked = anchored.clone()
                if visited:
                    masked[list(visited)] = float("-inf")
                if parent.index is not None:
                    masked[parent.index] = float("-inf")
                rows = set(self._topk(masked, config.branch_top_k))
                proposed_rows.append(rows)
                proposals += len(rows)
                for index in rows:
                    raw_edge = float(scores[index].item())
                    anchored_score = float(anchored[index].item())
                    affinity = max(0.0, min(1.0, (anchored_score + 1.0) / 2.0))
                    edges = (*parent.edge_scores, affinity)
                    path = _path_score(
                        config.path_score_mode, float(direct_scores[index].item()), edges
                    )
                    winner = int(frontier_winners[parent_row, index].item())
                    candidate = (
                        frontier_candidates[parent_row].get(index)
                        if frontier_candidates[parent_row] is not None
                        else None
                    )
                    candidate_parents.setdefault(index, []).append(
                        (path, parent_row, raw_edge, anchored_score, winner, candidate)
                    )
            if len(proposed_rows) > 1:
                total = sum(len(rows) for rows in proposed_rows)
                union = len(set().union(*proposed_rows))
                duplicates += total - union
                overlap_ratios.append((total - union) / max(total, 1))
            if not candidate_parents:
                stop_reason = "no_new_nodes"
                break

            best = {
                index: max(rows, key=lambda row: (row[0], -row[1]))
                for index, rows in candidate_parents.items()
            }
            ranking = direct_scores.new_full((len(self.index.records),), float("-inf"))
            for index, row in best.items():
                ranking[index] = row[0]
            remaining = config.max_unique_chunks - len(visited)
            accepted_indices = self._topk(ranking, min(remaining, config.beam_size))
            if config.min_confidence is not None:
                accepted_indices = [
                    index for index in accepted_indices if best[index][0] >= config.min_confidence
                ]
            if not accepted_indices:
                stop_reason = "confidence_threshold" if config.min_confidence is not None else "no_new_nodes"
                break

            next_frontier: list[_Frontier] = []
            for index in accepted_indices:
                path, parent_row, edge, anchored_score, winner, candidate = best[index]
                parent = frontier[parent_row]
                entry, chunk = self.index.records[index]
                node_id = chunk.chunk_id
                evidence = node_id in evidence_chunk_ids if evidence_chunk_ids is not None else None
                alternatives = sorted(
                    {frontier[row[1]].node_id for row in candidate_parents[index]}
                )
                graph.nodes.append(
                    RetrievalNode(
                        node_id=node_id,
                        reference_uri=entry.uri,
                        hop=hop,
                        parent_ids=alternatives,
                        direct_query_score=float(direct_scores[index].item()),
                        edge_score=edge,
                        path_score=path,
                        winning_gist_index=winner,
                        evidence=evidence,
                        parent_chunk_id=node_id,
                        local_span=(int(chunk.logical_start), int(chunk.logical_end)),
                        projection_type=(
                            "root_query" if hop == 1 else config.frontier_projection
                        ),
                        representation_type=(
                            "token_semantic_hybrid" if candidate is not None else "semantic_gist"
                        ),
                        discovery_channels=(candidate.to_dict() if candidate is not None else {}),
                        confidence=(candidate.confidence if candidate is not None else None),
                        confidence_calibrated=(
                            candidate.confidence_calibrated if candidate is not None else False
                        ),
                    )
                )
                for (
                    alt_path,
                    alt_parent_row,
                    alt_edge,
                    alt_anchored,
                    _,
                    alt_candidate,
                ) in candidate_parents[index]:
                    graph.edges.append(
                        RetrievalEdge(
                            source=frontier[alt_parent_row].node_id,
                            target=node_id,
                            hop=hop,
                            edge_score=alt_edge,
                            anchored_score=alt_anchored,
                            path_score=alt_path,
                            selected=alt_parent_row == parent_row,
                            projection_type=(
                                "query_to_memory"
                                if hop == 1 or config.frontier_projection == "query"
                                else "memory_to_memory"
                            ),
                            score=alt_edge,
                            accepted=alt_parent_row == parent_row,
                            edge_type=(
                                alt_candidate.selected_channel
                                if alt_candidate is not None
                                else "semantic_similarity"
                            ),
                            representation_type=(
                                "token_semantic_hybrid"
                                if alt_candidate is not None
                                else "semantic_gist"
                            ),
                        )
                    )
                gist = (
                    self.index.query_gists[index, winner]
                    if config.frontier_projection == "query"
                    else self.index.gists[index, winner]
                )
                if config.frontier_mode == "residual":
                    query = F.normalize(parent.query + config.residual_beta * gist, dim=-1)
                else:
                    query = gist
                next_frontier.append(
                    _Frontier(
                        index,
                        node_id,
                        query,
                        path,
                        (*parent.edge_scores, max(0.0, min(1.0, (anchored_score + 1.0) / 2.0))),
                        (token_index.records[index].token_ids if hybrid_enabled else ()),
                    )
                )
                visited.add(index)
            accepted_per_hop.append(len(accepted_indices))

            if config.frontier_mode in {"mean", "weighted_mean"}:
                selected_gists = torch.stack([item.query for item in next_frontier])
                if config.frontier_mode == "weighted_mean":
                    weights = torch.softmax(
                        torch.tensor([item.path_score for item in next_frontier], device=root.device), dim=0
                    )
                    aggregate = (selected_gists * weights.unsqueeze(-1)).sum(dim=0)
                else:
                    aggregate = selected_gists.mean(dim=0)
                aggregate = F.normalize(root + config.residual_beta * aggregate, dim=-1)
                best_parent = max(next_frontier, key=lambda item: item.path_score)
                frontier = [
                    _Frontier(
                        best_parent.index,
                        best_parent.node_id,
                        aggregate,
                        best_parent.path_score,
                        best_parent.edge_scores,
                        best_parent.token_ids,
                    )
                ]
            else:
                frontier = next_frontier
            if len(visited) >= config.max_unique_chunks:
                stop_reason = "unique_budget"
                break

        graph.stop_reason = stop_reason
        parent_counts: dict[str, int] = {}
        for edge in graph.edges:
            if edge.selected:
                parent_counts[edge.source] = parent_counts.get(edge.source, 0) + 1
        parent_total = sum(parent_counts.values())
        branch_entropy = -sum(
            (count / parent_total) * math.log(count / parent_total)
            for count in parent_counts.values()
        ) if parent_total else 0.0
        graph.costs = {
            "unique_gist_comparisons": comparisons,
            "candidate_proposals": proposals,
            "duplicate_proposals": duplicates,
            "visited_nodes": len(visited),
            "frontier_sizes": accepted_per_hop,
            "unique_parents": len({edge.source for edge in graph.edges if edge.selected}),
            "branch_entropy": branch_entropy,
            "candidate_overlap_mean": sum(overlap_ratios) / max(len(overlap_ratios), 1),
            "semantic_gist_comparisons": comparisons,
            "token_index_comparisons": token_comparisons,
            "token_index_queries": token_index_queries,
            "token_index_scored_candidates": indexed_token_comparisons,
            "token_index_bytes": token_index.memory_bytes() if hybrid_enabled else 0,
            "native_qk_comparisons": 0,
            "local_nodes_explored": 0,
        }
        selected_indices = tuple(
            self.index.chunk_ids.index(node.node_id) for node in graph.nodes if node.final_selected
        )
        return IterativeRoutingResult(
            selected_indices, tuple(direct_scores.detach().cpu().tolist()), graph
        )

    def route_batch(
        self, root_queries: torch.Tensor, config: IterativeRoutingConfig
    ) -> list[IterativeRoutingResult]:
        """Route independent batch rows without allowing cross-row frontier state."""
        if root_queries.ndim != 2:
            raise ValueError("root_queries must have shape [batch,routing_width].")
        return [self.route(row, config) for row in root_queries]

    def selected_chunks(self, result: IterativeRoutingResult) -> list[SelectedChunk]:
        """Convert final graph identities to payload handles without reading K/V tensors."""
        selected = []
        for rank, index in enumerate(result.selected_indices, start=1):
            entry, chunk = self.index.records[index]
            node = next(node for node in result.graph.nodes if node.node_id == chunk.chunk_id)
            selected.append(
                SelectedChunk(
                    entry=entry,
                    chunk=chunk,
                    reference_score=node.path_score,
                    chunk_score=node.direct_query_score,
                    layer_id=self.index.layer_id,
                    reference_rank=rank,
                    rank_within_reference=rank,
                    winning_gist_index=node.winning_gist_index,
                    winning_gist_score=node.edge_score,
                    gist_count=int(chunk.routing_gist.k.shape[0]),
                    metadata={
                        "selection_policy": node.discovery_channels.get(
                            "selected_channel", "iterative_closure"
                        ),
                        "hop": node.hop,
                        "candidate_confidence": node.confidence,
                        "confidence_calibrated": node.confidence_calibrated,
                    },
                )
            )
        return selected


class HierarchicalLocalGistRouter:
    """Traverse local semantic gists while budgeting unique K/V parents."""

    def __init__(self, index: HierarchicalGistIndex):
        self.index = index

    def _best_local(self, scores: torch.Tensor, parent: int) -> tuple[int, float]:
        rows = torch.nonzero(self.index.local_parent_indices == parent).flatten()
        winner = rows[torch.argmax(scores[rows])]
        return int(winner), float(scores[winner])

    def route(self, root_query, config, *, example_id=None, evidence_parent_ids=None):
        """Route root->parent/local then local->local with parent deduplication."""
        root = F.normalize(root_query.reshape(-1).to(self.index.device).float(), dim=-1)
        pm = F.normalize(self.index.parent_memory_gists.float(), dim=-1)
        pq = F.normalize(self.index.parent_query_gists.float(), dim=-1)
        lm = F.normalize(self.index.local_memory_gists.float(), dim=-1)
        lq = F.normalize(self.index.local_query_gists.float(), dim=-1)
        direct_parent, direct_local = pm @ root, lm @ root
        graph = RetrievalGraph(
            example_id, self.index.layer_id,
            {"node_id": "__root__", "representation_type": "semantic_gist", "projection_type": "query"},
            budget=asdict(config),
        )
        if config.depth == 0 or config.max_unique_chunks == 0:
            graph.stop_reason = "zero_limit"
            return IterativeRoutingResult((), tuple(direct_parent.cpu().tolist()), graph)
        first = IterativeGistRouter._topk(
            direct_parent,
            min(config.branch_top_k, config.beam_size, config.max_unique_chunks),
        )
        visited, frontier = set(first), []
        comparisons, explored = int(direct_parent.numel()), 0
        cross_parent = repeated_parent = 0
        local_edges, parent_edges, local_entropies = [], [], []
        parent_entropies = [_entropy(direct_parent)]

        def add_node(parent, local, hop, source, edge, anchored, path):
            parent_id = self.index.parent_ids[parent]
            node_id = f"{parent_id}#local={local}"
            graph.nodes.append(RetrievalNode(
                node_id, example_id or "memory", hop, [source],
                float(direct_parent[parent]), edge, path, local,
                evidence=(parent_id in evidence_parent_ids if evidence_parent_ids is not None else None),
                parent_chunk_id=parent_id, local_span=self.index.local_spans[local],
                resolution_level="local", projection_type="root_query" if hop == 1 else "query",
            ))
            graph.edges.append(RetrievalEdge(
                source, node_id, hop, edge, anchored, path, True,
                edge_type="root_to_local" if hop == 1 else "local_to_local",
                projection_type="query_to_memory", score=edge,
            ))
            return node_id

        for parent in first:
            local, edge = self._best_local(direct_local, parent)
            explored += int((self.index.local_parent_indices == parent).sum())
            affinity = max(0.0, min(1.0, (float(direct_parent[parent]) + 1.0) / 2.0))
            add_node(parent, local, 1, "__root__", edge, float(direct_parent[parent]), affinity)
            frontier.append((parent, local, affinity, (affinity,)))

        stop = "unique_budget" if len(visited) >= config.max_unique_chunks else "depth"
        for hop in range(2, config.depth + 1):
            if len(visited) >= config.max_unique_chunks or not frontier:
                break
            proposals = {}
            for source_row, (source_parent, source_local, _, source_edges) in enumerate(frontier):
                scores = lm @ lq[source_local]
                comparisons += int(scores.numel())
                explored += int(scores.numel())
                local_entropies.append(_entropy(scores))
                parent_scores = direct_parent.new_full((len(self.index.parent_ids),), float("-inf"))
                winners = {}
                for parent in range(len(self.index.parent_ids)):
                    winners[parent], value = self._best_local(scores, parent)
                    parent_scores[parent] = value
                parent_entropies.append(_entropy(parent_scores))
                anchored = config.root_anchor_alpha * direct_parent + (1 - config.root_anchor_alpha) * parent_scores
                ranked = IterativeGistRouter._topk(anchored, config.branch_top_k + len(visited))
                repeated_parent += sum(parent in visited for parent in ranked[:config.branch_top_k])
                for parent in [p for p in ranked if p not in visited][:config.branch_top_k]:
                    affinity = max(0.0, min(1.0, (float(anchored[parent]) + 1.0) / 2.0))
                    path = _path_score(config.path_score_mode, float(direct_parent[parent]), (*source_edges, affinity))
                    proposals.setdefault(parent, []).append((path, source_row, winners[parent], float(parent_scores[parent]), float(anchored[parent])))
                    cross_parent += int(parent != source_parent)
            if not proposals:
                stop = "no_new_parents"
                break
            best = {parent: max(rows, key=lambda row: (row[0], -row[1])) for parent, rows in proposals.items()}
            ranking = direct_parent.new_full((len(self.index.parent_ids),), float("-inf"))
            for parent, row in best.items(): ranking[parent] = row[0]
            accepted = IterativeGistRouter._topk(ranking, min(config.beam_size, config.max_unique_chunks - len(visited)))
            next_frontier = []
            for parent in accepted:
                path, source_row, local, edge, anchored = best[parent]
                source_parent, source_local, _, source_edges = frontier[source_row]
                source_id = f"{self.index.parent_ids[source_parent]}#local={source_local}"
                add_node(parent, local, hop, source_id, edge, anchored, path)
                local_edges.append(edge)
                parent_edges.append(float(pm[parent] @ pq[source_parent]))
                affinity = max(0.0, min(1.0, (anchored + 1.0) / 2.0))
                next_frontier.append((parent, local, path, (*source_edges, affinity)))
                visited.add(parent)
            frontier = next_frontier
            stop = "unique_budget" if len(visited) >= config.max_unique_chunks else "depth"
        graph.stop_reason = stop
        local_mean = sum(local_edges) / max(len(local_edges), 1)
        parent_mean = sum(parent_edges) / max(len(parent_edges), 1)
        graph.costs = {
            "semantic_gist_comparisons": comparisons, "native_qk_comparisons": 0,
            "local_nodes_explored": explored, "unique_parents_selected": len(visited),
            "local_nodes_activated_per_parent": len(graph.nodes) / max(len(visited), 1),
            "cross_parent_transitions": cross_parent, "repeated_parent_transitions": repeated_parent,
            "local_entropy": sum(local_entropies) / max(len(local_entropies), 1),
            "parent_entropy": sum(parent_entropies) / max(len(parent_entropies), 1),
            "path_depth": max((node.hop for node in graph.nodes), default=0),
            "local_vs_parent_similarity_ratio": (local_mean + 1.0) / max(parent_mean + 1.0, 1e-6),
            "bridge_locality_score": local_mean - parent_mean,
        }
        return IterativeRoutingResult(tuple(sorted(visited)), tuple(direct_parent.cpu().tolist()), graph)

    def selected_chunks(self, result: IterativeRoutingResult) -> list[SelectedChunk]:
        """Map deduplicated parent identities to lazy native-K/V payload handles."""
        if not self.index.records:
            raise ValueError("This hierarchical index has no cache payload records.")
        selected = []
        for rank, parent_index in enumerate(result.selected_indices, start=1):
            entry, chunk = self.index.records[parent_index]
            nodes = [
                node for node in result.graph.nodes
                if node.parent_chunk_id == chunk.chunk_id
            ]
            node = max(nodes, key=lambda value: value.path_score)
            selected.append(SelectedChunk(
                entry=entry, chunk=chunk, reference_score=node.path_score,
                chunk_score=node.direct_query_score, layer_id=self.index.layer_id,
                reference_rank=rank, rank_within_reference=rank,
                winning_gist_index=node.winning_gist_index,
                winning_gist_score=node.edge_score,
                gist_count=int(chunk.routing_gist.k.shape[0]),
                metadata={"selection_policy": "local_iterative_closure", "hop": node.hop},
            ))
        return selected
