"""Sparse query-side graphs for discovered PRA retrieval facets.

This module operates on units from one already encoded query. It does not
traverse memory, select references, materialize K/V, or run another encoder.
Native attention may be supplied as a directed edge component; any
symmetrization is an explicit graph policy.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


GRAPH_POLICIES = {"directed", "union", "mutual", "reciprocal_average"}
EDGE_COMPONENTS = ("contextual", "lexical", "attention", "position", "residual")


@dataclass(frozen=True)
class QueryUnitProvenance:
    """Audit metadata for one contextual query unit.

    ``unit_id`` is stable within the encoded query. Token bounds use the
    original prompt coordinate system even when only a question slice becomes
    graph nodes. Text is retained for audit and lexical preprocessing only.
    """

    unit_id: int
    token_start: int
    token_end: int
    text: str = ""
    layer: int | None = None
    head: int | None = None
    vector_source: str = "causal_hidden_state"


@dataclass(frozen=True)
class QueryGraph:
    """Tensor-native sparse graph over contextual query units.

    Node tensors have shape ``[N]``. ``src``, ``dst``, ``weight``, and every
    component tensor have shape ``[E]``. An edge ``src[e] -> dst[e]`` means
    that the source unit may contribute its current cluster label to the
    destination during directed propagation.
    """

    node_ids: torch.Tensor
    token_start: torch.Tensor
    token_end: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    components: Mapping[str, torch.Tensor]
    provenance: tuple[QueryUnitProvenance, ...]
    policy: str
    top_k: int
    threshold: float

    def __post_init__(self) -> None:
        if self.node_ids.ndim != 1 or self.node_ids.dtype != torch.long:
            raise ValueError("node_ids must be a LongTensor[N].")
        count = int(self.node_ids.numel())
        if count == 0 or len(self.provenance) != count:
            raise ValueError("A query graph requires aligned non-empty provenance.")
        if self.token_start.shape != (count,) or self.token_end.shape != (count,):
            raise ValueError("Token bounds must have shape [N].")
        if not bool(torch.all(self.token_end > self.token_start)):
            raise ValueError("Every query unit needs a non-empty token span.")
        edge_count = int(self.src.numel())
        if self.src.dtype != torch.long or self.dst.dtype != torch.long:
            raise ValueError("src and dst must be LongTensor[E].")
        if self.dst.shape != (edge_count,) or self.weight.shape != (edge_count,):
            raise ValueError("Sparse edge tensors must align on E.")
        if edge_count and (
            int(self.src.min()) < 0
            or int(self.dst.min()) < 0
            or int(self.src.max()) >= count
            or int(self.dst.max()) >= count
        ):
            raise ValueError("Sparse edge endpoints are outside the node range.")
        if edge_count and bool(torch.any(self.src == self.dst)):
            raise ValueError("Query graphs do not retain self edges.")
        if self.policy not in GRAPH_POLICIES:
            raise ValueError(f"Unsupported graph policy: {self.policy}")
        for name, values in self.components.items():
            if name not in EDGE_COMPONENTS or values.shape != (edge_count,):
                raise ValueError("Every named edge component must have shape [E].")
        devices = {
            tensor.device
            for tensor in (
                self.node_ids,
                self.token_start,
                self.token_end,
                self.src,
                self.dst,
                self.weight,
                *self.components.values(),
            )
        }
        if len(devices) != 1:
            raise ValueError("All query-graph tensors must share one device.")

    @property
    def node_count(self) -> int:
        return int(self.node_ids.numel())

    @property
    def edge_count(self) -> int:
        return int(self.src.numel())


def lexical_feature_matrix(
    token_texts: Sequence[str],
    *,
    buckets: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Hash observable token and character n-grams into ``[N,buckets]``.

    String normalization is preprocessing; graph construction and clustering
    remain tensor-native. Stable BLAKE2 hashes avoid Python hash randomization.
    """

    if buckets <= 0:
        raise ValueError("buckets must be positive.")
    features = torch.zeros((len(token_texts), buckets), dtype=torch.float32)
    for row, text in enumerate(token_texts):
        normalized = re.sub(r"[^a-z0-9]+", "", str(text).casefold())
        if not normalized:
            continue
        grams = {f"token:{normalized}"}
        for width in (2, 3):
            grams.update(
                normalized[index : index + width]
                for index in range(max(0, len(normalized) - width + 1))
            )
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            features[row, int.from_bytes(digest, "little") % buckets] += 1.0
    return F.normalize(features, dim=-1).to(device=device)


def _validate_square(name: str, values: torch.Tensor | None, count: int) -> None:
    if values is not None and values.shape != (count, count):
        raise ValueError(f"{name} must have shape [N,N].")


def _similarity(values: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(values.float(), dim=-1)
    return (normalized @ normalized.T).clamp(min=0.0, max=1.0)


def _symmetrize(
    selected: torch.Tensor,
    values: torch.Tensor,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy == "directed":
        return selected, values
    reciprocal = selected & selected.T
    if policy == "mutual":
        mask = reciprocal
        weights = (values + values.T) / 2.0
    else:
        mask = selected | selected.T
        if policy == "reciprocal_average":
            weights = torch.where(
                reciprocal,
                (values + values.T) / 2.0,
                torch.maximum(values, values.T),
            )
        else:
            weights = torch.maximum(values, values.T)
    return mask, weights


def _component_for_policy(
    component: torch.Tensor,
    selected: torch.Tensor,
    policy: str,
) -> torch.Tensor:
    if policy == "directed":
        return component
    reciprocal = selected & selected.T
    if policy in {"mutual", "reciprocal_average"}:
        return torch.where(
            reciprocal,
            (component + component.T) / 2.0,
            torch.maximum(component, component.T),
        )
    return torch.maximum(component, component.T)


def build_query_graph(
    hidden_states: torch.Tensor,
    *,
    lexical_features: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
    residual_updates: torch.Tensor | None = None,
    provenance: Sequence[QueryUnitProvenance] | None = None,
    contextual_weight: float = 1.0,
    lexical_weight: float = 0.0,
    attention_weight: float = 0.0,
    position_weight: float = 0.0,
    residual_weight: float = 0.0,
    position_scale: float = 4.0,
    top_k: int = 4,
    threshold: float = 0.0,
    policy: str = "union",
) -> QueryGraph:
    """Construct and immediately sparsify a weighted query-unit graph."""

    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("hidden_states must have shape [N,width].")
    count = int(hidden_states.shape[0])
    if top_k <= 0 or position_scale <= 0:
        raise ValueError("top_k and position_scale must be positive.")
    if policy not in GRAPH_POLICIES:
        raise ValueError(f"Unsupported graph policy: {policy}")
    weights = {
        "contextual": float(contextual_weight),
        "lexical": float(lexical_weight),
        "attention": float(attention_weight),
        "position": float(position_weight),
        "residual": float(residual_weight),
    }
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Edge-family weights must be non-negative with positive mass.")
    if lexical_features is not None and lexical_features.shape[0] != count:
        raise ValueError("lexical_features must have shape [N,features].")
    if residual_updates is not None and residual_updates.shape[0] != count:
        raise ValueError("residual_updates must have shape [N,width].")
    _validate_square("attention", attention, count)

    device = hidden_states.device
    positions = torch.arange(count, device=device)
    components = {
        "contextual": _similarity(hidden_states),
        "lexical": (
            _similarity(lexical_features.to(device))
            if lexical_features is not None
            else hidden_states.new_zeros((count, count), dtype=torch.float32)
        ),
        "attention": (
            attention.to(device=device, dtype=torch.float32).clamp(min=0.0)
            if attention is not None
            else hidden_states.new_zeros((count, count), dtype=torch.float32)
        ),
        "position": torch.exp(
            -(positions[:, None] - positions[None, :]).abs().float() / position_scale
        ),
        "residual": (
            _similarity(residual_updates.to(device))
            if residual_updates is not None
            else hidden_states.new_zeros((count, count), dtype=torch.float32)
        ),
    }
    combined = sum(weights[name] * components[name] for name in EDGE_COMPONENTS)
    combined = combined / sum(weights.values())
    combined.fill_diagonal_(float("-inf"))

    if count == 1:
        selected = torch.zeros((1, 1), dtype=torch.bool, device=device)
    else:
        effective_k = min(int(top_k), count - 1)
        # A tiny index preference makes top-k ties reproducible across devices.
        preference = (
            torch.arange(count, device=device, dtype=torch.float64)
            * torch.finfo(torch.float64).eps
            * 8.0
        )
        ranking_values = combined.double() - preference.unsqueeze(0)
        neighbors = torch.topk(ranking_values, k=effective_k, dim=1).indices
        selected = torch.zeros((count, count), dtype=torch.bool, device=device)
        selected.scatter_(1, neighbors, True)
        selected &= combined > float(threshold)

    edge_mask, policy_weights = _symmetrize(selected, combined, policy)
    edge_mask.fill_diagonal_(False)
    src, dst = edge_mask.nonzero(as_tuple=True)
    order = torch.argsort(src * count + dst, stable=True)
    src, dst = src[order], dst[order]
    edge_values = policy_weights[src, dst]
    edge_components = {
        name: _component_for_policy(values, selected, policy)[src, dst]
        for name, values in components.items()
        if weights[name] > 0 or name in {"contextual", "lexical"}
    }

    if provenance is None:
        provenance = tuple(
            QueryUnitProvenance(index, index, index + 1) for index in range(count)
        )
    if len(provenance) != count:
        raise ValueError("provenance must align with hidden_states.")
    node_ids = torch.tensor(
        [int(row.unit_id) for row in provenance], dtype=torch.long, device=device
    )
    if len(set(node_ids.tolist())) != count:
        raise ValueError("Query-unit IDs must be unique.")
    token_start = torch.tensor(
        [int(row.token_start) for row in provenance], dtype=torch.long, device=device
    )
    token_end = torch.tensor(
        [int(row.token_end) for row in provenance], dtype=torch.long, device=device
    )
    return QueryGraph(
        node_ids=node_ids,
        token_start=token_start,
        token_end=token_end,
        src=src,
        dst=dst,
        weight=edge_values,
        components=edge_components,
        provenance=tuple(provenance),
        policy=policy,
        top_k=min(int(top_k), max(0, count - 1)),
        threshold=float(threshold),
    )


def threshold_query_graph(graph: QueryGraph, threshold: float) -> QueryGraph:
    """Filter one fixed weighted edge set without rebuilding its topology."""

    keep = graph.weight > float(threshold)
    return QueryGraph(
        node_ids=graph.node_ids,
        token_start=graph.token_start,
        token_end=graph.token_end,
        src=graph.src[keep],
        dst=graph.dst[keep],
        weight=graph.weight[keep],
        components={name: values[keep] for name, values in graph.components.items()},
        provenance=graph.provenance,
        policy=graph.policy,
        top_k=graph.top_k,
        threshold=float(threshold),
    )


def graph_memory_bytes(graph: QueryGraph) -> int:
    """Return exact bytes occupied by graph tensors, excluding audit strings."""

    tensors = (
        graph.node_ids,
        graph.token_start,
        graph.token_end,
        graph.src,
        graph.dst,
        graph.weight,
        *graph.components.values(),
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
