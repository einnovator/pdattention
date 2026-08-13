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


GRAPH_SCHEMA_VERSION = "1.0"
_PATH_MODES = {"product", "logsum", "last", "min", "mean", "direct"}
_FRONTIER_MODES = {"direct", "residual", "mean", "weighted_mean"}


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
        return cls(layer_id, tuple(records), F.normalize(packed, dim=-1, eps=1e-12), mask)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for _, chunk in self.records)


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
    ) -> IterativeRoutingResult:
        """Traverse compact gists and stop before any native K/V is requested."""
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
        root_scores_t, _ = self._scores(root.unsqueeze(0))
        direct_scores = root_scores_t[0]
        graph = RetrievalGraph(
            example_id=example_id,
            layer_id=self.index.layer_id,
            root={"node_id": "__root__", "query_norm": float(root.norm().item())},
            budget=asdict(config),
        )
        if config.depth == 0 or config.max_unique_chunks == 0 or config.branch_top_k == 0:
            graph.stop_reason = "zero_limit"
            return IterativeRoutingResult((), tuple(direct_scores.cpu().tolist()), graph)

        visited: set[int] = set()
        frontier = [_Frontier(None, "__root__", root, 1.0, ())]
        comparisons = duplicates = proposals = 0
        overlap_ratios: list[float] = []
        accepted_per_hop: list[int] = []
        stop_reason = "depth"

        for hop in range(1, config.depth + 1):
            queries = torch.stack([item.query for item in frontier])
            frontier_scores, frontier_winners = self._scores(queries)
            comparisons += int(frontier_scores.numel())
            candidate_parents: dict[int, list[tuple[float, int, float, float, int]]] = {}
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
                    candidate_parents.setdefault(index, []).append(
                        (path, parent_row, raw_edge, anchored_score, winner)
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
                path, parent_row, edge, anchored_score, winner = best[index]
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
                    )
                )
                for alt_path, alt_parent_row, alt_edge, alt_anchored, _ in candidate_parents[index]:
                    graph.edges.append(
                        RetrievalEdge(
                            source=frontier[alt_parent_row].node_id,
                            target=node_id,
                            hop=hop,
                            edge_score=alt_edge,
                            anchored_score=alt_anchored,
                            path_score=alt_path,
                            selected=alt_parent_row == parent_row,
                        )
                    )
                gist = self.index.gists[index, winner]
                if config.frontier_mode == "residual":
                    query = F.normalize(parent.query + config.residual_beta * gist, dim=-1)
                else:
                    query = gist
                next_frontier.append(
                    _Frontier(index, node_id, query, path, (*parent.edge_scores, max(0.0, min(1.0, (anchored_score + 1.0) / 2.0))))
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
                frontier = [_Frontier(best_parent.index, best_parent.node_id, aggregate, best_parent.path_score, best_parent.edge_scores)]
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
                    metadata={"selection_policy": "iterative_closure", "hop": node.hop},
                )
            )
        return selected
