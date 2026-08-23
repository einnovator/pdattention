"""Per-facet root and successor method selection for adaptive PRA.

Facet construction and memory retrieval are deliberately separate.  This
module maps an already constructed facet to a retrieval channel; it never
changes the facet embedding or materializes memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .adaptive_facets import AdaptiveFacetTree, HierarchicalFacetNode
from .adaptive_search import (
    AdaptiveSearchAction,
    MATCHED_SUCCESSOR,
    ROOT_METHODS,
    SUCCESSOR_METHODS,
)


PER_FACET_BASELINES = ("fixed", "type_rule", "learned", "oracle")
_METHOD_PAIRS = tuple(
    (root, successor) for root in ROOT_METHODS for successor in SUCCESSOR_METHODS
)


@dataclass(frozen=True)
class FacetRoute:
    """One facet's retrieval decision and auditable decision source."""

    facet_id: str
    facet_type: str
    root_method: str
    successor_method: str
    top_k_roots: int
    top_k_successors: int
    fusion_method: str
    channel_profile: str
    policy: str
    confidence: float

    def __post_init__(self) -> None:
        if self.root_method not in ROOT_METHODS or self.successor_method not in SUCCESSOR_METHODS:
            raise ValueError("A facet route contains an unsupported search method.")
        if self.top_k_roots <= 0 or self.top_k_successors <= 0:
            raise ValueError("Per-facet top-k budgets must be positive.")
        if self.policy not in PER_FACET_BASELINES or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Facet-route policy/confidence is invalid.")


def facet_feature_vector(node: HierarchicalFacetNode) -> torch.Tensor:
    """Encode observable lexical, structural, and graph metadata as ``[14]``."""

    lexical = node.lexical_features
    stats = node.graph_statistics
    types = ("entity", "relational", "mixed", "semantic")
    return torch.tensor(
        [
            float(lexical.get("token_count", 0.0)),
            float(lexical.get("unique_token_count", 0.0)),
            float(lexical.get("rare_token_fraction", 0.0)),
            float(lexical.get("entity_count", 0.0)),
            float(lexical.get("relation_cue_count", 0.0)),
            float(node.confidence),
            float(stats.node_count),
            float(stats.density),
            float(stats.mean_edge_weight),
            float(len(node.token_spans)),
            *(float(node.facet_type == name) for name in types),
        ],
        dtype=torch.float32,
    )


@dataclass(frozen=True)
class LinearPerFacetRouter:
    """Small ridge classifier for a root/successor pair from facet metadata."""

    classes: tuple[tuple[str, str], ...]
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def fit(
        cls,
        nodes: Sequence[HierarchicalFacetNode],
        targets: Sequence[tuple[str, str]],
        *,
        ridge: float = 0.1,
    ) -> "LinearPerFacetRouter":
        if not nodes or len(nodes) != len(targets) or ridge <= 0:
            raise ValueError("Linear facet routing needs aligned data and positive ridge.")
        classes = tuple(pair for pair in _METHOD_PAIRS if pair in set(targets))
        if not classes:
            raise ValueError("No valid per-facet method target was supplied.")
        features = torch.stack([facet_feature_vector(node) for node in nodes]).double()
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False).clamp_min(1e-8)
        design = torch.cat(
            [(features - mean) / scale, torch.ones((len(features), 1), dtype=torch.double)],
            dim=1,
        )
        target = torch.zeros((len(features), len(classes)), dtype=torch.double)
        class_index = {pair: index for index, pair in enumerate(classes)}
        for row, pair in enumerate(targets):
            if pair not in class_index:
                raise ValueError(f"Unsupported per-facet method target: {pair}")
            target[row, class_index[pair]] = 1.0
        penalty = torch.eye(design.shape[1], dtype=torch.double) * ridge
        penalty[-1, -1] = 0.0
        weights = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
        return cls(classes, mean, scale, weights)

    def predict(self, node: HierarchicalFacetNode) -> tuple[str, str, float]:
        values = facet_feature_vector(node).double()
        design = torch.cat(((values - self.mean) / self.scale, torch.ones(1, dtype=torch.double)))
        logits = design @ self.weights
        probabilities = torch.softmax(logits, dim=0)
        selected = int(torch.argmax(probabilities))
        root, successor = self.classes[selected]
        return root, successor, float(probabilities[selected])


def _type_rule(node: HierarchicalFacetNode, action: AdaptiveSearchAction) -> tuple[str, str, float]:
    if node.facet_type == "entity":
        root = "exact" if node.rare_tokens else "bm25"
        return root, MATCHED_SUCCESSOR[root], 0.8
    if node.facet_type == "relational":
        return "semantic", "native_semantic", 0.75
    if node.facet_type == "mixed":
        return "hybrid", "hybrid_state", 0.8
    return action.root_method, action.successor_method, 0.6


def select_per_facet_oracle(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str, float]]:
    """Select evaluator-only best measured methods independently per facet."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["facet_id"]), []).append(row)
    selected: dict[str, tuple[str, str, float]] = {}
    for facet_id, candidates in grouped.items():
        best = min(
            candidates,
            key=lambda row: (
                -float(row.get("recall", 0.0)),
                -float(row.get("precision", 0.0)),
                -float(row.get("mrr", 0.0)),
                float(row.get("comparisons", 0.0)),
                str(row.get("root_method", "")),
                str(row.get("successor_method", "")),
            ),
        )
        selected[facet_id] = (
            str(best["root_method"]),
            str(best["successor_method"]),
            1.0,
        )
    return selected


def route_query_facets(
    tree: AdaptiveFacetTree,
    action: AdaptiveSearchAction,
    *,
    policy: str | None = None,
    learned_router: LinearPerFacetRouter | None = None,
    oracle_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[FacetRoute, ...]:
    """Route every non-root scoring facet under one of four fair baselines."""

    policy = policy or action.per_facet_policy
    if policy not in PER_FACET_BASELINES:
        raise ValueError(f"Unsupported per-facet policy: {policy}")
    candidates = tuple(node for node in tree.nodes if node.facet_id != tree.root_id)
    if not candidates:
        candidates = (next(node for node in tree.nodes if node.facet_id == tree.root_id),)
    oracle = select_per_facet_oracle(oracle_rows or ()) if policy == "oracle" else {}
    if policy == "learned" and learned_router is None:
        raise ValueError("Learned per-facet routing requires a fitted router.")
    if policy == "oracle" and set(oracle) != {node.facet_id for node in candidates}:
        raise ValueError("Oracle rows must cover every routed facet exactly by ID.")

    output = []
    for node in candidates:
        if policy == "fixed":
            root, successor, confidence = action.root_method, action.successor_method, 1.0
        elif policy == "type_rule":
            root, successor, confidence = _type_rule(node, action)
        elif policy == "learned":
            assert learned_router is not None
            root, successor, confidence = learned_router.predict(node)
        else:
            root, successor, confidence = oracle[node.facet_id]
        output.append(
            FacetRoute(
                node.facet_id,
                node.facet_type,
                root,
                successor,
                action.effort.roots,
                action.effort.neighbors,
                action.fusion_method,
                action.channel_profile,
                policy,
                confidence,
            )
        )
    return tuple(output)
