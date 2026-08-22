"""Tensor clustering primitives for sparse query-side graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .query_graph import QueryGraph, threshold_query_graph


@dataclass(frozen=True)
class ClusterResult:
    """One hard partition over ``N`` query units."""

    labels: torch.Tensor
    iterations: int
    converged: bool
    method: str

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or self.labels.dtype != torch.long:
            raise ValueError("Cluster labels must be LongTensor[N].")

    @property
    def cluster_count(self) -> int:
        return int(torch.unique(self.labels).numel())


@dataclass(frozen=True)
class FiltrationLevel:
    """Connected components retained at one threshold of a fixed edge set."""

    threshold: float
    result: ClusterResult
    split_count: int
    persistent_pair_fraction: float


@dataclass(frozen=True)
class FacetRecoveryMetrics:
    """Hard-partition recovery metrics for a controlled labelled query."""

    ari: float
    nmi: float
    pairwise_f1: float
    boundary_f1: float
    cluster_count_error: int


def canonicalize_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map arbitrary labels to contiguous IDs ordered by first occurrence."""

    if labels.ndim != 1:
        raise ValueError("labels must have shape [N].")
    if labels.numel() == 0:
        return labels.to(dtype=torch.long)
    unique = torch.unique(labels)
    first = torch.stack(
        [torch.nonzero(labels == value, as_tuple=False)[0, 0] for value in unique]
    )
    ordered = unique[torch.argsort(first, stable=True)]
    matches = labels.unsqueeze(1) == ordered.unsqueeze(0)
    return matches.to(torch.long).argmax(dim=1)


def connected_components(graph: QueryGraph, *, max_iter: int | None = None) -> ClusterResult:
    """Propagate minimum labels over the graph's explicit edge directions.

    For undirected connected components, callers must use a graph policy that
    emits reciprocal edges. A directed graph intentionally computes directed
    minimum-label reachability and can therefore produce a different result.
    """

    count = graph.node_count
    limit = int(max_iter or max(1, count))
    labels = torch.arange(count, dtype=torch.long, device=graph.node_ids.device)
    converged = False
    iterations = 0
    for iteration in range(limit):
        iterations = iteration + 1
        received = torch.full_like(labels, count)
        if graph.edge_count:
            received.scatter_reduce_(
                0,
                graph.dst,
                labels[graph.src],
                reduce="amin",
                include_self=False,
            )
        updated = torch.minimum(labels, received)
        if torch.equal(updated, labels):
            converged = True
            break
        labels = updated
    return ClusterResult(
        labels=canonicalize_labels(labels),
        iterations=iterations,
        converged=converged,
        method="connected_components",
    )


def weighted_label_propagation(
    graph: QueryGraph,
    *,
    max_iter: int = 64,
    self_retention: float = 0.05,
) -> ClusterResult:
    """Apply deterministic weighted label propagation with tensor reductions.

    Each iteration gathers source labels and accumulates edge weights into a
    dense ``[nodes, active_labels]`` vote table using ``index_put_``. Query
    graphs are short, so this bounded table avoids Python loops over nodes or
    edges while preserving deterministic lowest-label tie breaking.
    """

    if max_iter <= 0 or self_retention < 0:
        raise ValueError("max_iter must be positive and self_retention non-negative.")
    count = graph.node_count
    device = graph.node_ids.device
    labels = torch.arange(count, dtype=torch.long, device=device)
    if graph.edge_count:
        incoming = torch.full((count, count), float("-inf"), device=device)
        incoming[graph.dst, graph.src] = graph.weight
        strongest = incoming.argmax(dim=1)
        has_neighbor = torch.isfinite(incoming.max(dim=1).values)
        labels = torch.where(has_neighbor, torch.minimum(labels, strongest), labels)
        labels = canonicalize_labels(labels)
    converged = False
    iterations = 0
    previous_states: list[torch.Tensor] = []
    for iteration in range(max_iter):
        iterations = iteration + 1
        active = int(labels.max()) + 1
        votes = graph.weight.new_zeros((count, active))
        if graph.edge_count:
            votes.index_put_((graph.dst, labels[graph.src]), graph.weight, accumulate=True)
            incident = graph.weight.new_zeros(count)
            incident.scatter_reduce_(
                0, graph.dst, graph.weight, reduce="amax", include_self=False
            )
            votes[torch.arange(count, device=device), labels] += self_retention * incident
        winners = votes.argmax(dim=1)
        isolated = votes.sum(dim=1) == 0
        winners = torch.where(isolated, labels, winners)
        updated = canonicalize_labels(winners)
        if torch.equal(updated, labels):
            converged = True
            labels = updated
            break
        if any(torch.equal(updated, prior) for prior in previous_states[-2:]):
            # Resolve a synchronous two-cycle reproducibly without claiming
            # fixed-point convergence.
            labels = canonicalize_labels(torch.minimum(labels, updated))
            break
        previous_states.append(labels.clone())
        labels = updated
    return ClusterResult(
        labels=labels,
        iterations=iterations,
        converged=converged,
        method="weighted_label_propagation",
    )


def threshold_filtration(
    graph: QueryGraph,
    thresholds: Sequence[float],
    *,
    max_iter: int | None = None,
) -> tuple[FiltrationLevel, ...]:
    """Track component refinement as a fixed graph's threshold rises."""

    ordered = tuple(float(value) for value in thresholds)
    if not ordered or tuple(sorted(ordered)) != ordered or len(set(ordered)) != len(ordered):
        raise ValueError("thresholds must be non-empty, unique, and increasing.")
    levels: list[FiltrationLevel] = []
    previous: torch.Tensor | None = None
    for threshold in ordered:
        result = connected_components(
            threshold_query_graph(graph, threshold), max_iter=max_iter
        )
        split_count = result.cluster_count
        persistent = 1.0
        if previous is not None:
            for cluster in torch.unique(result.labels):
                parents = torch.unique(previous[result.labels == cluster])
                if parents.numel() != 1:
                    raise AssertionError("Threshold filtration merged prior components.")
            split_count = result.cluster_count - int(torch.unique(previous).numel())
            old_pairs = previous[:, None] == previous[None, :]
            retained_pairs = old_pairs & (result.labels[:, None] == result.labels[None, :])
            persistent = float(retained_pairs.sum() / old_pairs.sum().clamp_min(1))
        levels.append(
            FiltrationLevel(
                threshold=threshold,
                result=result,
                split_count=max(0, split_count),
                persistent_pair_fraction=persistent,
            )
        )
        previous = result.labels
    return tuple(levels)


def deterministic_kmeans(
    values: torch.Tensor,
    cluster_count: int,
    *,
    max_iter: int = 64,
) -> ClusterResult:
    """Embedding-only baseline with deterministic farthest-first seeds."""

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must have shape [N,width].")
    if not 0 < cluster_count <= values.shape[0] or max_iter <= 0:
        raise ValueError("cluster_count and max_iter must fit the input.")
    normalized = F.normalize(values.float(), dim=-1)
    seeds = [0]
    nearest = 1.0 - normalized @ normalized[0]
    for _ in range(1, cluster_count):
        candidate = int(torch.argmax(nearest))
        seeds.append(candidate)
        nearest = torch.minimum(nearest, 1.0 - normalized @ normalized[candidate])
    centroids = normalized[torch.tensor(seeds, device=values.device)]
    labels = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    converged = False
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        updated = torch.argmax(normalized @ centroids.T, dim=1)
        if iteration and torch.equal(updated, labels):
            converged = True
            break
        labels = updated
        sums = normalized.new_zeros((cluster_count, normalized.shape[1]))
        counts = normalized.new_zeros(cluster_count)
        sums.index_add_(0, labels, normalized)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=normalized.dtype))
        empty = counts == 0
        centroids = F.normalize(sums / counts.clamp_min(1).unsqueeze(1), dim=-1)
        if bool(empty.any()):
            centroids[empty] = normalized[torch.tensor(seeds, device=values.device)][empty]
    return ClusterResult(
        labels=canonicalize_labels(labels),
        iterations=iterations,
        converged=converged,
        method="deterministic_kmeans",
    )


def _combination2(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64)
    return values * (values - 1.0) / 2.0


def facet_recovery_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> FacetRecoveryMetrics:
    """Compute ARI, NMI, pairwise F1, boundary F1, and count error."""

    if predicted.shape != target.shape or predicted.ndim != 1 or predicted.numel() == 0:
        raise ValueError("predicted and target must be aligned non-empty vectors.")
    predicted = canonicalize_labels(predicted.cpu())
    target = canonicalize_labels(target.cpu())
    pred_count = int(predicted.max()) + 1
    true_count = int(target.max()) + 1
    contingency = torch.zeros((pred_count, true_count), dtype=torch.float64)
    contingency.index_put_((predicted, target), torch.ones_like(predicted, dtype=torch.float64), accumulate=True)
    rows, columns = contingency.sum(1), contingency.sum(0)
    pair_index = _combination2(contingency).sum()
    row_pairs, column_pairs = _combination2(rows).sum(), _combination2(columns).sum()
    total_pairs = _combination2(torch.tensor(float(predicted.numel())))
    expected = row_pairs * column_pairs / total_pairs.clamp_min(1.0)
    maximum = (row_pairs + column_pairs) / 2.0
    ari = float((pair_index - expected) / (maximum - expected).clamp_min(1e-12))

    probabilities = contingency / contingency.sum()
    row_probability = probabilities.sum(1, keepdim=True)
    column_probability = probabilities.sum(0, keepdim=True)
    valid = probabilities > 0
    mutual_information = (
        probabilities[valid]
        * torch.log(
            probabilities[valid]
            / (row_probability @ column_probability)[valid].clamp_min(1e-12)
        )
    ).sum()
    pred_entropy = -(row_probability[row_probability > 0] * torch.log(row_probability[row_probability > 0])).sum()
    true_entropy = -(column_probability[column_probability > 0] * torch.log(column_probability[column_probability > 0])).sum()
    nmi = float(2.0 * mutual_information / (pred_entropy + true_entropy).clamp_min(1e-12))

    same_pred = predicted[:, None] == predicted[None, :]
    same_true = target[:, None] == target[None, :]
    upper = torch.triu(torch.ones_like(same_pred, dtype=torch.bool), diagonal=1)
    true_positive = int((same_pred & same_true & upper).sum())
    false_positive = int((same_pred & ~same_true & upper).sum())
    false_negative = int((~same_pred & same_true & upper).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    pairwise_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    pred_boundaries = predicted[1:] != predicted[:-1]
    true_boundaries = target[1:] != target[:-1]
    boundary_tp = int((pred_boundaries & true_boundaries).sum())
    boundary_precision = boundary_tp / max(int(pred_boundaries.sum()), 1)
    boundary_recall = boundary_tp / max(int(true_boundaries.sum()), 1)
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-12)
    )
    return FacetRecoveryMetrics(
        ari=ari,
        nmi=nmi,
        pairwise_f1=pairwise_f1,
        boundary_f1=boundary_f1,
        cluster_count_error=abs(pred_count - true_count),
    )
