"""Graph-topology and correlation metrics for frozen layerwise PRA analyses."""

from __future__ import annotations

import math
import statistics

import torch


def topk_neighbors(scores: torch.Tensor, k: int) -> tuple[tuple[int, ...], ...]:
    """Return deterministic directed neighbors after excluding each self-edge."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix")
    if k <= 0:
        raise ValueError("k must be positive")
    rows = []
    for source in range(scores.shape[0]):
        values = scores[source].clone()
        values[source] = float("-inf")
        order = torch.argsort(values, descending=True, stable=True)
        rows.append(
            tuple(
                int(target)
                for target in order[: min(k, max(0, scores.shape[0] - 1))]
                if torch.isfinite(values[target])
            )
        )
    return tuple(rows)


def shortest_distance(
    neighbors: tuple[tuple[int, ...], ...],
    sources,
    targets,
    *,
    max_hops: int = 8,
) -> int | None:
    """Find a bounded directed distance without consulting targets during expansion."""
    frontier = set(map(int, sources))
    targets = set(map(int, targets)) - frontier
    if not targets:
        return None
    visited = set(frontier)
    for depth in range(1, max_hops + 1):
        frontier = {
            target
            for source in frontier
            for target in neighbors[source]
            if target not in visited
        }
        if frontier.intersection(targets):
            return depth
        visited.update(frontier)
        if not frontier:
            break
    return None


def topology_metrics(scores: torch.Tensor, k: int) -> dict[str, float]:
    """Summarize one exact directed Top-K native parent graph."""
    neighbors = topk_neighbors(scores, k)
    nodes = len(neighbors)
    edges = {(source, target) for source, row in enumerate(neighbors) for target in row}
    out_degrees = [len(row) for row in neighbors]
    reciprocal = sum((target, source) in edges for source, target in edges)
    destinations = [target for row in neighbors for target in row]
    duplicate_neighbor_rate = 1.0 - len(set(destinations)) / max(len(destinations), 1)
    undirected = [set() for _ in range(nodes)]
    for source, target in edges:
        undirected[source].add(target)
        undirected[target].add(source)
    components = []
    unseen = set(range(nodes))
    while unseen:
        seed = min(unseen)
        frontier = {seed}
        component = set()
        while frontier:
            component.update(frontier)
            unseen.difference_update(frontier)
            frontier = {
                target
                for source in frontier
                for target in undirected[source]
                if target in unseen
            }
        components.append(component)
    two_hop_growth = []
    for source in range(nodes):
        reached = set(neighbors[source])
        reached.update(
            target for middle in neighbors[source] for target in neighbors[middle]
        )
        reached.discard(source)
        two_hop_growth.append(math.sqrt(len(reached)))
    return {
        "mean_out_degree": statistics.fmean(out_degrees) if out_degrees else 0.0,
        "effective_branching_factor": (
            statistics.fmean(two_hop_growth) if two_hop_growth else 0.0
        ),
        "unique_neighbor_count": float(len(set(destinations))),
        "duplicate_neighbor_rate": duplicate_neighbor_rate,
        "connected_component_count": float(len(components)),
        "giant_component_fraction": max(map(len, components), default=0) / max(nodes, 1),
        "graph_density": len(edges) / max(nodes * (nodes - 1), 1),
        "reciprocal_edge_rate": reciprocal / max(len(edges), 1),
    }


def pearson(left, right) -> float:
    """Return a finite population Pearson correlation or NaN for no variance."""
    x = torch.as_tensor(list(left), dtype=torch.float64)
    y = torch.as_tensor(list(right), dtype=torch.float64)
    if x.numel() != y.numel() or x.numel() < 2:
        raise ValueError("correlation inputs must have equal length >= 2")
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x * y).sum() / denominator) if denominator else float("nan")


def _average_ranks(values) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(left, right) -> float:
    """Return Spearman correlation with deterministic average ranks for ties."""
    left = list(left)
    right = list(right)
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")
    return pearson(_average_ranks(left), _average_ranks(right))


def bootstrap_mean_ci(values, *, replicates: int = 2000, seed: int = 20260814):
    """Bootstrap example-level means; callers must aggregate tokens beforehand."""
    values = [float(value) for value in values if not math.isnan(float(value))]
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(values, dtype=torch.float64)
    draws = torch.randint(len(values), (replicates, len(values)), generator=generator)
    samples = tensor[draws].mean(dim=1).sort().values
    return {
        "mean": statistics.fmean(values),
        "ci_low": float(samples[int(0.025 * replicates)]),
        "ci_high": float(samples[int(0.975 * replicates) - 1]),
        "n": len(values),
    }
