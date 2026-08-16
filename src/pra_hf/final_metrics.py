"""Reusable measurements for the Paper 2.5 experiment-freeze gate.

These helpers analyze frozen scores and execution rows.  They deliberately do
not implement retrieval, graph traversal, or materialization policies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch


def require_disjoint_identifiers(
    fit_identifiers: Iterable[str], heldout_identifiers: Iterable[str]
) -> None:
    """Reject selector fitting when validation and held-out identities overlap."""
    overlap = set(map(str, fit_identifiers)).intersection(map(str, heldout_identifiers))
    if overlap:
        raise ValueError(f"fit/held-out identity overlap: {sorted(overlap)[:3]}")


def selected_facet_group_rank(
    scores: torch.Tensor, facet_index: int, parent_group: Iterable[int]
) -> int:
    """Return a stable one-based target-group rank for one preselected facet.

    ``scores`` has shape ``[facets, parents]``.  The facet must be selected
    without ``parent_group``; the group is consulted only by this evaluator.
    """
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("scores must have shape [facets,parents]")
    facet_index = int(facet_index)
    if not 0 <= facet_index < scores.shape[0]:
        raise ValueError("facet_index is outside the score matrix")
    group = tuple(sorted(set(int(parent) for parent in parent_group)))
    if not group or min(group) < 0 or max(group) >= scores.shape[1]:
        raise ValueError("parent_group must contain valid parent indices")
    order = torch.argsort(scores[facet_index].float(), descending=True, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return min(int(inverse[parent]) + 1 for parent in group)


def facet_confidence_features(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute label-free confidence features for every facet.

    The returned vectors have shape ``[facets]``.  Margin and entropy inspect
    only the facet's score distribution over candidate parents; no target or
    evidence identity is accepted by this API.
    """
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must have shape [facets,parents>=2]")
    values = scores.float()
    top = torch.topk(values, 2, dim=1).values
    probabilities = torch.softmax(values, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(values.shape[1])
    return {
        "top_parent_score": top[:, 0],
        "top_parent_margin": top[:, 0] - top[:, 1],
        "parent_score_entropy": entropy,
        "normalized_parent_score_entropy": normalized_entropy,
    }


def fit_linear_selector(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    ridge: float = 1.0,
) -> dict[str, torch.Tensor | int]:
    """Fit deterministic ridge least-squares classification on validation rows.

    This intentionally tiny linear diagnostic predicts a facet scale, not a
    parent.  Callers own the validation/held-out split and pass no target IDs as
    features.
    """
    x = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or not x.shape[0]:
        raise ValueError("features and labels must align as [N,F] and [N]")
    if class_count <= 1 or int(y.min()) < 0 or int(y.max()) >= class_count:
        raise ValueError("labels must fit class_count")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-8)
    normalized = (x - mean) / scale
    design = torch.cat((normalized, torch.ones((x.shape[0], 1), dtype=x.dtype)), dim=1)
    targets = torch.nn.functional.one_hot(y, class_count).to(x.dtype)
    penalty = torch.eye(design.shape[1], dtype=x.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "class_count": int(class_count),
    }


def predict_linear_selector(
    model: Mapping[str, torch.Tensor | int], features: torch.Tensor
) -> torch.Tensor:
    """Predict scale classes from a model returned by :func:`fit_linear_selector`."""
    x = torch.as_tensor(features, dtype=torch.float64)
    mean = torch.as_tensor(model["mean"], dtype=x.dtype)
    scale = torch.as_tensor(model["scale"], dtype=x.dtype)
    weights = torch.as_tensor(model["weights"], dtype=x.dtype)
    if x.ndim != 2 or x.shape[1] != mean.numel():
        raise ValueError("features do not match the fitted selector")
    design = torch.cat(((x - mean) / scale, torch.ones((x.shape[0], 1))), dim=1)
    return (design @ weights).argmax(dim=1)


def decompose_path_survival(
    edge_recall: float, product_survival: float, observed_survival: float
) -> dict[str, float | str]:
    """Separate local absence, compounding, and search competition losses.

    Observed survival can exceed the independent-edge product because edges are
    correlated or redundant paths help.  That gain is reported separately and
    never mislabeled as negative search loss.
    """
    edge = float(edge_recall)
    product = float(product_survival)
    observed = float(observed_survival)
    if any(not 0.0 <= value <= 1.0 for value in (edge, product, observed)):
        raise ValueError("survival metrics must lie in [0,1]")
    missing = 1.0 - edge
    compounded = max(0.0, edge - product)
    competition = max(0.0, product - observed)
    redundancy = max(0.0, observed - product)
    dominant = max(
        (
            ("missing_local_edge", missing),
            ("compounded_local_errors", compounded),
            ("frontier_search_competition", competition),
            ("correlation_or_redundancy_gain", redundancy),
        ),
        key=lambda item: (item[1], item[0]),
    )[0]
    return {
        "missing_local_edge_loss": missing,
        "compounded_local_error_loss": compounded,
        "additional_frontier_search_loss": competition,
        "correlation_or_redundancy_gain": redundancy,
        "dominant_effect": dominant,
    }


def pareto_flags(
    rows: Sequence[Mapping[str, object]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[bool]:
    """Mark non-dominated rows under explicit quality and cost directions."""
    if not rows or not maximize or not minimize:
        raise ValueError("rows and both metric direction sets are required")
    flags = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            no_worse = all(float(other[key]) >= float(row[key]) for key in maximize)
            no_worse &= all(float(other[key]) <= float(row[key]) for key in minimize)
            strictly_better = any(float(other[key]) > float(row[key]) for key in maximize)
            strictly_better |= any(float(other[key]) < float(row[key]) for key in minimize)
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def exact_join(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    keys: Sequence[str],
) -> list[dict[str, object]]:
    """Perform a one-to-one inner join and reject duplicate or missing keys."""
    if not keys:
        raise ValueError("join keys are required")
    index: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in right:
        key = tuple(row[name] for name in keys)
        if key in index:
            raise ValueError(f"duplicate right join key: {key}")
        index[key] = row
    joined = []
    seen_left: set[tuple[object, ...]] = set()
    for row in left:
        key = tuple(row[name] for name in keys)
        if key in seen_left:
            raise ValueError(f"duplicate left join key: {key}")
        seen_left.add(key)
        if key not in index:
            raise ValueError(f"missing right join key: {key}")
        overlap = set(row).intersection(index[key]) - set(keys)
        if overlap:
            raise ValueError(f"overlapping non-key fields: {sorted(overlap)}")
        joined.append({**row, **{k: v for k, v in index[key].items() if k not in keys}})
    return joined
