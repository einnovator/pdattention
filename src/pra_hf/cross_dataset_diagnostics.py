"""Post-hoc diagnostics for cross-dataset PRA graph experiments.

These helpers consume frozen scores, token spans, and dataset annotations. They
do not participate in native graph search, which keeps oracle identities out of
the runtime routing path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from .chunk_granularity import Span, span_overlap
from .query_facets import QueryFacetSet, build_span_query_facets


@dataclass(frozen=True)
class GroupFacetRank:
    """Best post-hoc rank of an annotated parent group over query facets."""

    rank: int
    facet_index: int
    parent_index: int
    target_score: float
    best_distractor_score: float
    margin: float


def merged_token_count(spans: Iterable[Span]) -> int:
    """Count the union of half-open token spans without double counting."""
    ordered = sorted((int(start), int(end)) for start, end in spans if end > start)
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


def evidence_token_metrics(
    evidence_spans: Sequence[Span],
    parent_spans: Sequence[Span],
    selected_parent_ids: Iterable[int],
    root_parent_ids: Iterable[int] = (),
) -> dict[str, float | int | bool]:
    """Measure annotation-supported density within selected source parents."""
    selected_ids = tuple(dict.fromkeys(int(parent) for parent in selected_parent_ids))
    root_ids = tuple(dict.fromkeys(int(parent) for parent in root_parent_ids))
    evidence = tuple((int(start), int(end)) for start, end in evidence_spans)
    evidence_tokens = merged_token_count(evidence)

    def selected_evidence(ids: Iterable[int]) -> int:
        intersections = []
        for parent in ids:
            parent_span = parent_spans[parent]
            intersections.extend(
                (max(parent_span[0], start), min(parent_span[1], end))
                for start, end in evidence
                if span_overlap(parent_span, (start, end)) > 0
            )
        return merged_token_count(intersections)

    selected_tokens = sum(
        int(parent_spans[parent][1]) - int(parent_spans[parent][0])
        for parent in selected_ids
    )
    selected_evidence_tokens = selected_evidence(selected_ids)
    root_evidence_tokens = selected_evidence(root_ids)
    return {
        "evidence_tokens": evidence_tokens,
        "selected_parent_tokens": selected_tokens,
        "selected_evidence_tokens": selected_evidence_tokens,
        "non_evidence_selected_tokens": selected_tokens - selected_evidence_tokens,
        "evidence_density": (
            selected_evidence_tokens / selected_tokens if selected_tokens else 0.0
        ),
        "root_evidence_fraction": (
            root_evidence_tokens / evidence_tokens if evidence_tokens else 0.0
        ),
        "root_contains_all_evidence": bool(
            evidence_tokens and root_evidence_tokens == evidence_tokens
        ),
    }


def annotated_paths(
    node_ids: Sequence[str], edges: Sequence[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    """Enumerate deterministic root-to-terminal paths in an annotation DAG."""
    incoming: dict[str, list[str]] = {str(node): [] for node in node_ids}
    outgoing: dict[str, list[str]] = {str(node): [] for node in node_ids}
    for source, target in edges:
        incoming[str(target)].append(str(source))
        outgoing[str(source)].append(str(target))
    roots = sorted(node for node in incoming if not incoming[node])
    paths: list[tuple[str, ...]] = []

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in path:
            raise ValueError("Annotated evidence graph contains a cycle.")
        extended = path + (node,)
        successors = sorted(outgoing[node])
        if not successors:
            paths.append(extended)
            return
        for successor in successors:
            visit(successor, extended)

    for root in roots:
        visit(root, ())
    return tuple(paths)


def product_model_path_survival(
    path_steps: Sequence[Sequence[int]], edge_probability_by_step: Mapping[int, float]
) -> float:
    """Average independent-edge survival over paths described by edge steps."""
    if not path_steps:
        return 1.0
    probabilities = []
    for steps in path_steps:
        probability = 1.0
        for step in steps:
            probability *= float(edge_probability_by_step.get(int(step), 0.0))
        probabilities.append(probability)
    return sum(probabilities) / len(probabilities)


def all_offset_multiscale_facets(
    hidden_states: torch.Tensor,
    support_span: Span,
    *,
    scales: Sequence[int] = (1, 2, 4, 8, 16),
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Pool every valid stride-one query span plus the contextual global state."""
    support_start, support_end = map(int, support_span)
    if support_start < 0 or support_end <= support_start or support_end > hidden_states.shape[0]:
        raise ValueError("Support span must fit the contextual query states.")
    spans: list[tuple[int, int, str]] = []
    support_tokens = support_end - support_start
    for scale in scales:
        scale = int(scale)
        if scale <= 0:
            raise ValueError("Facet scales must be positive.")
        if scale > support_tokens:
            continue
        spans.extend(
            (start, start + scale, f"window_{scale}")
            for start in range(support_start, support_end - scale + 1)
        )
    return build_span_query_facets(
        hidden_states,
        spans,
        include_global=True,
        family="all_offset_multiscale",
        native_query=native_query,
    )


def best_group_facet_rank(scores: torch.Tensor, parent_group: Iterable[int]) -> GroupFacetRank:
    """Find a group's best rank over facets without altering the score tensor."""
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("Scores must have shape [facets,parents].")
    group = tuple(sorted(set(int(parent) for parent in parent_group)))
    if not group or min(group) < 0 or max(group) >= scores.shape[1]:
        raise ValueError("Parent group must contain valid parent indices.")
    group_set = set(group)
    best: tuple[int, int, int] | None = None
    best_scores: tuple[float, float] | None = None
    for facet in range(scores.shape[0]):
        values = scores[facet].float()
        order = torch.argsort(values, descending=True, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        parent = min(group, key=lambda item: (int(inverse[item]), item))
        rank = int(inverse[parent]) + 1
        candidate = (rank, facet, parent)
        distractors = [index for index in range(scores.shape[1]) if index not in group_set]
        distractor = (
            float(values[distractors].max()) if distractors else float("-inf")
        )
        target = float(values[parent])
        if best is None or candidate < best:
            best = candidate
            best_scores = (target, distractor)
    assert best is not None and best_scores is not None
    target, distractor = best_scores
    return GroupFacetRank(
        rank=best[0],
        facet_index=best[1],
        parent_index=best[2],
        target_score=target,
        best_distractor_score=distractor,
        margin=target - distractor if math.isfinite(distractor) else math.inf,
    )


def token_jaccard_parent_scores(
    query_token_ids: Iterable[int],
    source_token_ids: Sequence[int],
    parent_spans: Sequence[Span],
) -> torch.Tensor:
    """Score source parents by token-set Jaccard overlap with the question."""
    query = set(int(token) for token in query_token_ids)
    scores = []
    for start, end in parent_spans:
        parent = set(int(token) for token in source_token_ids[int(start) : int(end)])
        union = query | parent
        scores.append(len(query & parent) / len(union) if union else 0.0)
    return torch.tensor(scores, dtype=torch.float32)


def group_rank(scores: torch.Tensor, parent_group: Iterable[int]) -> int:
    """Return the best stable one-based rank of any parent in a group."""
    if scores.ndim != 1:
        raise ValueError("Scores must have shape [parents].")
    group = tuple(sorted(set(int(parent) for parent in parent_group)))
    if not group:
        raise ValueError("Parent group cannot be empty.")
    order = torch.argsort(scores.float(), descending=True, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return min(int(inverse[parent]) + 1 for parent in group)


def bounded_multiscale_candidates(
    scores: torch.Tensor, *, proposal_width: int, global_budget: int
) -> tuple[tuple[int, ...], int]:
    """Union per-facet proposals, then enforce one shared root budget."""
    if scores.ndim != 2 or proposal_width <= 0 or global_budget <= 0:
        raise ValueError("Valid facet scores and positive budgets are required.")
    width = min(int(proposal_width), int(scores.shape[1]))
    nominations = torch.topk(scores.float(), width, dim=1).indices.flatten().tolist()
    candidates = tuple(sorted(set(int(parent) for parent in nominations)))
    values = scores.float()
    lower = values.min(dim=1, keepdim=True).values
    scale = values.max(dim=1, keepdim=True).values - lower
    normalized = torch.where(scale > 0, (values - lower) / scale, torch.zeros_like(values))
    candidate_scores = normalized[:, candidates].max(dim=0).values
    order = torch.argsort(candidate_scores, descending=True, stable=True)
    selected = tuple(candidates[int(index)] for index in order[:global_budget])
    return selected, len(candidates)
