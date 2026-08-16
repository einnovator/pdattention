"""Rank-only query grounding for bounded associative proposals.

Association generates a small candidate set in its own score family. Query
facets validate only those identities. The module combines ranks or applies a
query threshold; raw native and semantic scores are never added together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


_MODES = {
    "association",
    "query_rerank",
    "rank_conjunction",
    "threshold_conjunction",
}


@dataclass(frozen=True)
class AssociativeCandidateSet:
    """A bounded A-to-memory proposal list in association-score order."""

    parent_indices: tuple[int, ...]
    association_scores: tuple[float, ...]
    comparisons: int = 0


@dataclass(frozen=True)
class QueryValidation:
    """Facet relevance for an associative candidate set."""

    scores: torch.Tensor
    validating_facets: torch.Tensor
    comparisons: int
    active_facet_count: int


@dataclass(frozen=True)
class GroundedCandidate:
    """Provenance for one candidate after query-facet validation."""

    parent_index: int
    association_rank: int
    association_score: float
    query_rank: int
    query_score: float
    validating_facet: int
    validating_facet_is_root: bool
    admitted: bool
    joint_rank_score: float


@dataclass(frozen=True)
class GroundedRanking:
    """Ordered admitted identities plus every bounded decision record."""

    selected: tuple[int, ...]
    candidates: tuple[GroundedCandidate, ...]
    association_comparisons: int
    validation_comparisons: int


def _ordered(scores: torch.Tensor, candidates: Sequence[int]) -> list[int]:
    finite = [
        int(index)
        for index in candidates
        if 0 <= int(index) < scores.numel()
        and math.isfinite(float(scores[int(index)]))
    ]
    finite.sort(key=lambda index: (-float(scores[index]), index))
    return finite


def generate_associative_candidates(
    association_scores: torch.Tensor,
    *,
    source_parents: set[int],
    candidate_k: int,
    comparisons: int = 0,
) -> AssociativeCandidateSet:
    """Generate bounded proposals without consulting query or oracle labels."""
    if association_scores.ndim != 1:
        raise ValueError("Association scores must have shape [parents].")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")
    pool = [
        index
        for index in range(association_scores.numel())
        if index not in source_parents
    ]
    ordered = _ordered(association_scores, pool)[:candidate_k]
    return AssociativeCandidateSet(
        tuple(ordered),
        tuple(float(association_scores[index]) for index in ordered),
        int(comparisons),
    )


def query_validate_candidates(
    query_facet_parent_scores: torch.Tensor,
    candidates: AssociativeCandidateSet,
    *,
    root_facet: int | None = None,
    residual_only: bool = False,
) -> QueryValidation:
    """Max-reduce a packed ``[facets,candidates]`` validation matrix."""
    if query_facet_parent_scores.ndim != 2:
        raise ValueError("Query validation scores must have shape [facets,parents].")
    facet_count, parent_count = query_facet_parent_scores.shape
    if residual_only:
        if root_facet is None or not 0 <= root_facet < facet_count:
            raise ValueError("Residual grounding requires a valid root facet.")
        active = [index for index in range(facet_count) if index != root_facet]
        # A one-facet query has no unresolved state; retaining the only facet is
        # safer than manufacturing an empty score family.
        if not active:
            active = [root_facet]
    else:
        active = list(range(facet_count))
    candidate_ids = torch.tensor(
        candidates.parent_indices,
        dtype=torch.long,
        device=query_facet_parent_scores.device,
    )
    if candidate_ids.numel() and int(candidate_ids.max()) >= parent_count:
        raise ValueError("Associative candidate is outside query score matrix.")
    values = query_facet_parent_scores[active][:, candidate_ids]
    scores, local_winners = values.max(dim=0)
    active_tensor = torch.tensor(active, dtype=torch.long, device=values.device)
    return QueryValidation(
        scores=scores,
        validating_facets=active_tensor[local_winners],
        comparisons=int(values.numel()),
        active_facet_count=len(active),
    )


def rank_grounded_candidates(
    candidates: AssociativeCandidateSet,
    validation: QueryValidation,
    *,
    mode: str,
    final_k: int = 1,
    rank_lambda: float = 1.0,
    query_threshold: float = float("-inf"),
    root_facet: int | None = None,
) -> GroundedRanking:
    """Apply association, query rerank, rank conjunction, or an AND gate."""
    if mode not in _MODES:
        raise ValueError(f"Unsupported grounding mode: {mode}")
    if final_k <= 0:
        raise ValueError("final_k must be positive.")
    count = len(candidates.parent_indices)
    if validation.scores.shape != (count,) or validation.validating_facets.shape != (
        count,
    ):
        raise ValueError("Query validation must align with associative candidates.")
    query_order = sorted(
        range(count),
        key=lambda index: (
            -float(validation.scores[index]),
            candidates.parent_indices[index],
        ),
    )
    query_ranks = {index: rank + 1 for rank, index in enumerate(query_order)}
    records = []
    for index, parent in enumerate(candidates.parent_indices):
        association_rank = index + 1
        admitted = (
            mode != "threshold_conjunction"
            or float(validation.scores[index]) >= query_threshold
        )
        if mode in {"association", "threshold_conjunction"}:
            joint = float(association_rank)
        elif mode == "query_rerank":
            joint = float(query_ranks[index])
        else:
            joint = float(association_rank + rank_lambda * query_ranks[index])
        validating_facet = int(validation.validating_facets[index])
        records.append(
            GroundedCandidate(
                parent_index=parent,
                association_rank=association_rank,
                association_score=candidates.association_scores[index],
                query_rank=query_ranks[index],
                query_score=float(validation.scores[index]),
                validating_facet=validating_facet,
                validating_facet_is_root=(
                    root_facet is not None and validating_facet == root_facet
                ),
                admitted=admitted,
                joint_rank_score=joint,
            )
        )
    admitted_rows = [row for row in records if row.admitted]
    admitted_rows.sort(
        key=lambda row: (
            row.joint_rank_score,
            -row.query_score if mode == "query_rerank" else row.association_rank,
            row.parent_index,
        )
    )
    return GroundedRanking(
        selected=tuple(row.parent_index for row in admitted_rows[:final_k]),
        candidates=tuple(records),
        association_comparisons=candidates.comparisons,
        validation_comparisons=validation.comparisons,
    )
