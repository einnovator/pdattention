"""Chunk-topology and post-hoc evidence diagnostics for memory-graph studies.

The functions in this module do not participate in routing. They construct
deterministic memory boundaries and evaluate an already-produced discovery
trace against evidence annotations. Keeping these operations separate makes
the oracle-free execution contract explicit and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


Span = tuple[int, int]


@dataclass(frozen=True)
class EvidenceTopology:
    """Evidence groups remapped onto one deterministic parent partition."""

    parent_spans: tuple[Span, ...]
    evidence_spans: tuple[Span, ...]
    evidence_parent_groups: tuple[tuple[int, ...], ...]
    oracle_parent_ids: tuple[int, ...]
    later_oracle_parent_ids: tuple[int, ...]
    root_parent_id: int
    root_oracle_fraction: float
    root_contains_all_evidence: bool
    root_contains_multiple_groups: bool
    root_contains_only_initial_evidence: bool
    evidence_group_collisions: int
    evidence_tokens_per_parent: tuple[int, ...]


@dataclass(frozen=True)
class OracleRecovery:
    """Post-hoc annotation metrics for an oracle-free visited-parent trace."""

    oracle_recall: float
    later_oracle_recall: float
    complete_oracle: bool
    visited_oracle_parents: int
    visited_later_oracle_parents: int


@dataclass(frozen=True)
class FacetParentStatistics:
    """Summary of one query-facet score column for a memory parent."""

    maximum: float
    top2_mean: float
    winning_facet: int
    normalized_entropy: float
    concentration: float


def chunk_spans(token_count: int, chunk_size: int, overlap: int = 0) -> tuple[Span, ...]:
    """Partition ``[0, token_count)`` into stable half-open token spans."""
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")
    stride = chunk_size - overlap
    spans: list[Span] = []
    start = 0
    while start < token_count:
        end = min(start + chunk_size, token_count)
        spans.append((start, end))
        if end == token_count:
            break
        start += stride
    return tuple(spans)


def span_overlap(left: Span, right: Span) -> int:
    """Return the number of token positions shared by two half-open spans."""
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _validate_spans(spans: Sequence[Span], token_count: int, name: str) -> None:
    for start, end in spans:
        if not 0 <= int(start) < int(end) <= token_count:
            raise ValueError(f"Invalid {name} span {(start, end)} for {token_count} tokens.")


def _merged_token_count(spans: Iterable[Span]) -> int:
    ordered = sorted((int(start), int(end)) for start, end in spans)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def evidence_topology(
    token_count: int,
    evidence_spans: Sequence[Span],
    *,
    chunk_size: int,
    overlap: int = 0,
) -> EvidenceTopology:
    """Map ordered evidence groups to parents and choose a deterministic root.

    Each input evidence span is one annotation group. The oracle root is the
    parent with greatest token overlap with the first group; lower parent IDs
    break ties. Distinct groups collide when their mapped parent sets overlap.
    """
    if not evidence_spans:
        raise ValueError("At least one evidence span is required.")
    parents = chunk_spans(token_count, chunk_size, overlap)
    _validate_spans(evidence_spans, token_count, "evidence")
    groups: list[tuple[int, ...]] = []
    for evidence in evidence_spans:
        mapped = tuple(
            parent_id
            for parent_id, parent in enumerate(parents)
            if span_overlap(evidence, parent) > 0
        )
        if not mapped:
            raise ValueError(f"Evidence span {evidence} did not map to a parent.")
        groups.append(mapped)

    first = evidence_spans[0]
    root = max(
        groups[0],
        key=lambda parent_id: (span_overlap(first, parents[parent_id]), -parent_id),
    )
    oracle = tuple(sorted({parent for group in groups for parent in group}))
    later = tuple(sorted({parent for group in groups[1:] for parent in group} - {root}))
    collisions = sum(
        bool(set(groups[left]).intersection(groups[right]))
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    )

    evidence_tokens = tuple(
        _merged_token_count(
            (max(start, parent[0]), min(end, parent[1]))
            for start, end in evidence_spans
            if span_overlap((start, end), parent) > 0
        )
        for parent in parents
    )
    total_evidence = _merged_token_count(evidence_spans)
    root_fraction = evidence_tokens[root] / total_evidence
    root_group_count = sum(root in group for group in groups)
    return EvidenceTopology(
        parent_spans=parents,
        evidence_spans=tuple((int(start), int(end)) for start, end in evidence_spans),
        evidence_parent_groups=tuple(groups),
        oracle_parent_ids=oracle,
        later_oracle_parent_ids=later,
        root_parent_id=root,
        root_oracle_fraction=root_fraction,
        root_contains_all_evidence=math.isclose(root_fraction, 1.0),
        root_contains_multiple_groups=root_group_count > 1,
        root_contains_only_initial_evidence=root_group_count == 1,
        evidence_group_collisions=collisions,
        evidence_tokens_per_parent=evidence_tokens,
    )


def evaluate_oracle_recovery(
    visited_parent_ids: Iterable[int], topology: EvidenceTopology
) -> OracleRecovery:
    """Evaluate discovery after execution without exposing labels to search."""
    visited = {int(parent) for parent in visited_parent_ids}
    oracle = set(topology.oracle_parent_ids)
    later = set(topology.later_oracle_parent_ids)
    found = len(visited.intersection(oracle))
    found_later = len(visited.intersection(later))
    return OracleRecovery(
        oracle_recall=found / len(oracle),
        later_oracle_recall=found_later / len(later) if later else 1.0,
        complete_oracle=oracle.issubset(visited),
        visited_oracle_parents=found,
        visited_later_oracle_parents=found_later,
    )


def minimum_recovery_depth(
    visited_by_depth: Mapping[int, Iterable[int]], topology: EvidenceTopology
) -> int | None:
    """Return the first measured hop limit that contains every oracle parent."""
    for depth in sorted(int(depth) for depth in visited_by_depth):
        if evaluate_oracle_recovery(visited_by_depth[depth], topology).complete_oracle:
            return depth
    return None


def facet_parent_statistics(scores: torch.Tensor) -> tuple[FacetParentStatistics, ...]:
    """Summarize a ``[facets, parents]`` semantic score matrix by parent."""
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("scores must have shape [facets, parents] with non-zero axes.")
    probabilities = torch.softmax(scores.float(), dim=0)
    facet_count = int(scores.shape[0])
    if facet_count == 1:
        entropy = torch.zeros(scores.shape[1], dtype=torch.float32, device=scores.device)
    else:
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=0)
        entropy = entropy / math.log(facet_count)
    top_count = min(2, facet_count)
    top = torch.topk(scores.float(), top_count, dim=0).values
    return tuple(
        FacetParentStatistics(
            maximum=float(scores[:, parent].max()),
            top2_mean=float(top[:, parent].mean()),
            winning_facet=int(torch.argmax(scores[:, parent])),
            normalized_entropy=float(entropy[parent]),
            concentration=float(1.0 - entropy[parent]),
        )
        for parent in range(scores.shape[1])
    )


def normalize_facet_scores(scores: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each facet over parents into an interpretable [0,1] scale."""
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("scores must have shape [facets, parents].")
    values = scores.float()
    lower = values.min(dim=1, keepdim=True).values
    span = values.max(dim=1, keepdim=True).values - lower
    return torch.where(span > 0, (values - lower) / span, torch.zeros_like(values))


def path_facet_coverage(normalized_scores: torch.Tensor, parents: Iterable[int]) -> float:
    """Average each facet's best normalized activation over a selected path."""
    if normalized_scores.ndim != 2 or normalized_scores.shape[0] == 0:
        raise ValueError("normalized_scores must have shape [facets, parents].")
    selected = tuple(dict.fromkeys(int(parent) for parent in parents))
    if not selected:
        return 0.0
    if min(selected) < 0 or max(selected) >= normalized_scores.shape[1]:
        raise ValueError("Path parent is outside normalized_scores.")
    return float(normalized_scores[:, selected].max(dim=1).values.mean())


def incremental_facet_coverage(
    normalized_scores: torch.Tensor,
    path: Iterable[int],
    added_parent: int,
) -> float:
    """Measure complementary facet coverage contributed by one parent."""
    base = tuple(dict.fromkeys(int(parent) for parent in path))
    extended = base + (() if int(added_parent) in base else (int(added_parent),))
    return path_facet_coverage(normalized_scores, extended) - path_facet_coverage(
        normalized_scores, base
    )


def contracted_chain_depth(edge_count: int, nodes_per_chunk: int) -> int:
    """Return observed graph depth after grouping adjacent chain nodes."""
    if edge_count < 0:
        raise ValueError("edge_count must be non-negative.")
    if nodes_per_chunk <= 0:
        raise ValueError("nodes_per_chunk must be positive.")
    node_count = edge_count + 1
    return math.ceil(node_count / nodes_per_chunk) - 1
