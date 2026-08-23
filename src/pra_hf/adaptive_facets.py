"""Serving-time query faceting for adaptive PRA request/reply control.

The builders in this module consume states from one completed query forward
pass.  They do not inspect memory labels or evidence annotations.  This keeps
facet construction available to a one-shot controller without leaking oracle
information and gives callback policies the same interpretation surface later.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .query_facets import (
    QueryFacetProvenance,
    QueryFacetSet,
    build_multiscale_query_facets,
    build_span_query_facets,
    deterministic_phrase_spans,
    global_query_facet,
)
from .query_graph import (
    QueryGraph,
    QueryUnitProvenance,
    build_query_graph,
    graph_memory_bytes,
    lexical_feature_matrix,
)
from .query_graph_cluster import (
    ClusterResult,
    canonicalize_labels,
    connected_components,
    weighted_label_propagation,
)


FACET_MODES = ("global", "syntactic", "multiscale", "graph", "syntactic_graph")
FACET_MODE_ALIASES = {
    "last_span": "global",
    "multi_span": "syntactic",
    "multi_scale": "multiscale",
    "syntactic->graph": "syntactic_graph",
    "syntactic_to_graph": "syntactic_graph",
}
COARSE_PARTITION_MODES = ("clause", "sentence", "delimiter", "fixed")
GRAPH_SIMILARITY_MODES = ("contextual", "lexical", "hybrid", "contextual_position")
GRAPH_CLUSTER_METHODS = ("connected_components", "label_propagation")

_RELATION_CUES = {
    "after", "before", "between", "both", "during", "from", "how", "than",
    "what", "when", "where", "which", "who", "why", "with",
}
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def normalize_facet_mode(mode: str) -> str:
    """Map legacy profile names and arrow spellings to one stable mode name."""

    normalized = FACET_MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())
    if normalized not in FACET_MODES:
        raise ValueError(f"Unsupported facet_mode={mode!r}.")
    return normalized


@dataclass(frozen=True)
class GraphFacetConfig:
    """Observable controls for one query-side graph construction.

    ``top_k`` and ``threshold`` bound graph edges. Component constraints are
    applied after clustering and before pooling. They are query-only controls,
    not memory retrieval budgets.
    """

    similarity_mode: str = "contextual"
    threshold: float = 0.45
    top_k: int = 2
    min_component_size: int = 1
    max_component_size: int | None = None
    graph_policy: str = "union"
    cluster_method: str = "connected_components"

    def __post_init__(self) -> None:
        if self.similarity_mode not in GRAPH_SIMILARITY_MODES:
            raise ValueError(f"Unsupported graph similarity mode: {self.similarity_mode}")
        if self.top_k <= 0 or self.min_component_size <= 0:
            raise ValueError("Graph top_k and minimum component size must be positive.")
        if self.max_component_size is not None and self.max_component_size < self.min_component_size:
            raise ValueError("Maximum component size cannot be smaller than the minimum.")
        if self.cluster_method not in GRAPH_CLUSTER_METHODS:
            raise ValueError(f"Unsupported graph cluster method: {self.cluster_method}")


@dataclass(frozen=True)
class FacetGraphStatistics:
    """Graph measurements attached to one facet node."""

    node_count: int = 0
    internal_edges: int = 0
    density: float = 0.0
    mean_edge_weight: float = 0.0
    component_confidence: float = 1.0


@dataclass(frozen=True)
class HierarchicalFacetNode:
    """One auditable node in the global/coarse/semantic query-facet tree.

    Embeddings have shape ``[hidden_width]``. ``token_indices`` can be
    non-contiguous for graph communities; ``token_spans`` preserves that fact
    rather than pretending the community is one contiguous phrase.
    """

    facet_id: str
    parent_id: str | None
    children: tuple[str, ...]
    kind: str
    facet_type: str
    token_indices: tuple[int, ...]
    token_spans: tuple[tuple[int, int], ...]
    embedding: torch.Tensor
    semantic_centroid: torch.Tensor
    entities: tuple[str, ...]
    rare_tokens: tuple[str, ...]
    lexical_features: Mapping[str, float]
    graph_statistics: FacetGraphStatistics
    confidence: float

    def __post_init__(self) -> None:
        if not self.facet_id or self.embedding.ndim != 1:
            raise ValueError("A facet node requires an ID and a rank-one embedding.")
        if self.semantic_centroid.shape != self.embedding.shape:
            raise ValueError("Facet embedding and semantic centroid must align.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Facet confidence must lie in [0, 1].")

    def audit_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance while omitting the full embedding values."""

        return {
            "facet_id": self.facet_id,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "kind": self.kind,
            "facet_type": self.facet_type,
            "token_indices": list(self.token_indices),
            "token_spans": [list(span) for span in self.token_spans],
            "embedding_shape": list(self.embedding.shape),
            "embedding_norm": float(torch.linalg.vector_norm(self.embedding.float())),
            "entities": list(self.entities),
            "rare_tokens": list(self.rare_tokens),
            "lexical_features": dict(self.lexical_features),
            "graph_statistics": asdict(self.graph_statistics),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class FacetConstructionMetrics:
    """Measured query-facet work, independent of memory search and K/V admission."""

    facet_mode: str
    construction_latency_ms: float
    graph_construction_ms: float
    graph_clustering_ms: float
    graph_calls: int
    graph_nodes: int
    graph_edges: int
    graph_density: float
    graph_memory_bytes: int
    pairwise_similarity_evaluations: int
    facet_count: int
    tree_node_count: int
    mean_facet_overlap: float


@dataclass(frozen=True)
class AdaptiveFacetTree:
    """Rooted facet hierarchy plus the flat scoring view consumed by PRA."""

    mode: str
    root_id: str
    nodes: tuple[HierarchicalFacetNode, ...]
    scoring_facets: QueryFacetSet
    metrics: FacetConstructionMetrics

    def __post_init__(self) -> None:
        by_id = {node.facet_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes) or self.root_id not in by_id:
            raise ValueError("Facet-tree IDs must be unique and include the root.")
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id not in by_id:
                raise ValueError(f"Unknown parent {node.parent_id!r} for {node.facet_id!r}.")
            if any(child not in by_id for child in node.children):
                raise ValueError(f"Facet {node.facet_id!r} has an unknown child.")

    @property
    def leaves(self) -> tuple[HierarchicalFacetNode, ...]:
        return tuple(node for node in self.nodes if not node.children)


def _spans_from_indices(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return ()
    spans: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            spans.append((start, previous + 1))
            start = index
        previous = index
    spans.append((start, previous + 1))
    return tuple(spans)


def _normalized_tokens(token_texts: Sequence[str], indices: Sequence[int]) -> list[str]:
    values = []
    for index in indices:
        match = _TOKEN_PATTERN.search(str(token_texts[index]))
        if match:
            values.append(match.group(0))
    return values


def _node_metadata(
    token_texts: Sequence[str], indices: Sequence[int], query_frequencies: Mapping[str, int]
) -> tuple[str, tuple[str, ...], tuple[str, ...], dict[str, float]]:
    values = _normalized_tokens(token_texts, indices)
    lowered = [value.casefold() for value in values]
    entities = tuple(dict.fromkeys(value for value in values if value[:1].isupper() or value.isdigit()))
    rare = tuple(dict.fromkeys(value for value in values if query_frequencies.get(value.casefold(), 0) == 1))
    relation_count = sum(value in _RELATION_CUES for value in lowered)
    if entities and relation_count:
        facet_type = "mixed"
    elif entities or (rare and not relation_count):
        facet_type = "entity"
    elif relation_count:
        facet_type = "relational"
    else:
        facet_type = "semantic"
    count = max(len(values), 1)
    lexical = {
        "token_count": float(len(values)),
        "unique_token_count": float(len(set(lowered))),
        "rare_token_fraction": len(rare) / count,
        "entity_count": float(len(entities)),
        "relation_cue_count": float(relation_count),
    }
    return facet_type, entities, rare, lexical


def _component_statistics(graph: QueryGraph, members: Sequence[int]) -> FacetGraphStatistics:
    member_set = torch.zeros(graph.node_count, dtype=torch.bool, device=graph.node_ids.device)
    member_set[torch.tensor(tuple(members), dtype=torch.long, device=graph.node_ids.device)] = True
    keep = member_set[graph.src] & member_set[graph.dst]
    edge_count = int(keep.sum())
    node_count = len(members)
    possible = node_count * max(node_count - 1, 0)
    mean_weight = float(graph.weight[keep].mean()) if edge_count else 0.0
    density = edge_count / possible if possible else 0.0
    return FacetGraphStatistics(
        node_count=node_count,
        internal_edges=edge_count,
        density=density,
        mean_edge_weight=mean_weight,
        component_confidence=max(0.0, min(1.0, mean_weight)),
    )


def _graph_weights(mode: str) -> dict[str, float]:
    if mode == "contextual":
        return {"contextual_weight": 1.0}
    if mode == "lexical":
        return {"contextual_weight": 0.0, "lexical_weight": 1.0}
    if mode == "hybrid":
        return {"contextual_weight": 0.75, "lexical_weight": 0.25}
    return {"contextual_weight": 0.85, "position_weight": 0.15}


def _constrain_components(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    minimum: int,
    maximum: int | None,
) -> torch.Tensor:
    """Deterministically split oversized and merge undersized communities."""

    groups: list[list[int]] = []
    for label in torch.unique(labels, sorted=True).tolist():
        members = torch.nonzero(labels == label, as_tuple=False).flatten().tolist()
        width = int(maximum or len(members))
        groups.extend([members[offset : offset + width] for offset in range(0, len(members), width)])
    while len(groups) > 1:
        small_index = next((index for index, group in enumerate(groups) if len(group) < minimum), None)
        if small_index is None:
            break
        source = groups[small_index]
        source_center = F.normalize(hidden[source].float().mean(dim=0), dim=0)
        candidates = []
        for index, target in enumerate(groups):
            if index == small_index or (maximum is not None and len(source) + len(target) > maximum):
                continue
            center = F.normalize(hidden[target].float().mean(dim=0), dim=0)
            candidates.append((float(torch.dot(source_center, center)), -index, index))
        if not candidates:
            break
        target_index = max(candidates)[2]
        groups[target_index] = sorted(groups[target_index] + source)
        groups.pop(small_index)
    constrained = torch.empty_like(labels)
    for label, members in enumerate(groups):
        constrained[torch.tensor(members, device=labels.device)] = label
    return canonicalize_labels(constrained)


def _build_graph_region(
    hidden_states: torch.Tensor,
    token_texts: Sequence[str],
    span: tuple[int, int],
    config: GraphFacetConfig,
) -> tuple[QueryGraph, ClusterResult, torch.Tensor, float, float]:
    start, end = span
    support = hidden_states[start:end]
    texts = [str(token_texts[index]) for index in range(start, end)]
    provenance = tuple(
        QueryUnitProvenance(index, index, index + 1, texts[index - start])
        for index in range(start, end)
    )
    lexical = lexical_feature_matrix(texts, device=hidden_states.device)
    graph_started = time.perf_counter()
    graph = build_query_graph(
        support,
        lexical_features=lexical,
        provenance=provenance,
        top_k=config.top_k,
        threshold=config.threshold,
        policy=config.graph_policy,
        **_graph_weights(config.similarity_mode),
    )
    graph_ms = (time.perf_counter() - graph_started) * 1000.0
    cluster_started = time.perf_counter()
    clusters = (
        connected_components(graph)
        if config.cluster_method == "connected_components"
        else weighted_label_propagation(graph)
    )
    labels = _constrain_components(
        support,
        clusters.labels,
        minimum=config.min_component_size,
        maximum=config.max_component_size,
    )
    clusters = ClusterResult(labels, clusters.iterations, clusters.converged, clusters.method)
    cluster_ms = (time.perf_counter() - cluster_started) * 1000.0
    return graph, clusters, support, graph_ms, cluster_ms


def _coarse_spans(
    token_texts: Sequence[str], support_span: tuple[int, int], mode: str
) -> tuple[tuple[int, int, str], ...]:
    if mode not in COARSE_PARTITION_MODES:
        raise ValueError(f"Unsupported coarse_partition_mode={mode!r}.")
    start, end = support_span
    if mode == "fixed":
        width = 8
        return tuple((left, min(left + width, end), "fixed") for left in range(start, end, width))
    pattern = r"[?!.\n]" if mode == "sentence" else r"[|\n]" if mode == "delimiter" else r"[?!.;:\n]"
    spans: list[tuple[int, int, str]] = []
    left = start
    for index in range(start, end):
        if re.search(pattern, str(token_texts[index])):
            if left < index + 1:
                spans.append((left, index + 1, mode))
            left = index + 1
    if left < end:
        spans.append((left, end, mode))
    return tuple(spans or ((start, end, mode),))


def _mean_overlap(nodes: Sequence[HierarchicalFacetNode]) -> float:
    # Include coarse parents as well as leaves: parent/child overlap is a real
    # redundancy cost of hierarchical faceting and should not be hidden by a
    # leaf-only statistic. Hard graph communities alone remain disjoint.
    leaves = [set(node.token_indices) for node in nodes if node.kind != "global"]
    values = []
    for left in range(len(leaves)):
        for right in range(left + 1, len(leaves)):
            union = leaves[left] | leaves[right]
            values.append(len(leaves[left] & leaves[right]) / len(union) if union else 0.0)
    return sum(values) / len(values) if values else 0.0


def build_adaptive_query_facets(
    hidden_states: torch.Tensor,
    token_texts: Sequence[str],
    *,
    mode: str,
    support_span: tuple[int, int] | None = None,
    native_query: torch.Tensor | None = None,
    coarse_partition_mode: str = "clause",
    graph_config: GraphFacetConfig | None = None,
    multiscale_windows: Sequence[int] = (2, 4, 8, 16),
) -> AdaptiveFacetTree:
    """Build one of the five request/reply facet modes and its audit tree."""

    started = time.perf_counter()
    mode = normalize_facet_mode(mode)
    if hidden_states.ndim != 2 or hidden_states.shape[0] != len(token_texts):
        raise ValueError("Hidden states and token_texts must align as [tokens,width].")
    if native_query is not None and (native_query.ndim != 3 or native_query.shape[0] != len(token_texts)):
        raise ValueError("Native query states must align as [tokens,heads,head_dim].")
    support_span = support_span or (0, len(token_texts))
    start, end = support_span
    if start < 0 or end <= start or end > len(token_texts):
        raise ValueError("support_span must fit the non-empty query sequence.")
    graph_config = graph_config or GraphFacetConfig()
    normalized = [value.casefold() for value in _normalized_tokens(token_texts, range(start, end))]
    frequencies = {value: normalized.count(value) for value in set(normalized)}

    root_indices = tuple(range(start, end))
    root_embedding = hidden_states[-1]
    root_type, entities, rare, lexical = _node_metadata(token_texts, root_indices, frequencies)
    root = HierarchicalFacetNode(
        "query", None, (), "global", root_type, root_indices, ((start, end),),
        root_embedding, root_embedding, entities, rare, lexical, FacetGraphStatistics(), 1.0,
    )
    nodes: list[HierarchicalFacetNode] = [root]
    flat_hidden = [root_embedding]
    flat_native = [native_query[-1]] if native_query is not None else []
    flat_provenance = [QueryFacetProvenance("global", end - 1, end, "global", 1)]
    graph_calls = graph_nodes = graph_edges = graph_bytes = pairwise = 0
    graph_ms = cluster_ms = 0.0

    def add_node(
        facet_id: str,
        parent_id: str,
        kind: str,
        indices: Sequence[int],
        *,
        statistics: FacetGraphStatistics | None = None,
        confidence: float = 1.0,
        add_to_scoring: bool = True,
    ) -> None:
        selected = tuple(sorted(set(int(index) for index in indices)))
        embedding = hidden_states[list(selected)].mean(dim=0)
        facet_type, local_entities, local_rare, local_lexical = _node_metadata(
            token_texts, selected, frequencies
        )
        nodes.append(
            HierarchicalFacetNode(
                facet_id, parent_id, (), kind, facet_type, selected,
                _spans_from_indices(selected), embedding, embedding,
                local_entities, local_rare, local_lexical,
                statistics or FacetGraphStatistics(), confidence,
            )
        )
        if add_to_scoring:
            flat_hidden.append(embedding)
            if native_query is not None:
                flat_native.append(native_query[list(selected)].mean(dim=0))
            span_start, span_end = min(selected), max(selected) + 1
            flat_provenance.append(
                QueryFacetProvenance("local", span_start, span_end, kind, len(selected))
            )

    if mode == "syntactic":
        spans = deterministic_phrase_spans(token_texts, support_span)
        for index, (left, right, label) in enumerate(spans or ((start, end, "clause"),)):
            add_node(f"syntax.{index}", "query", label, range(left, right))
    elif mode == "multiscale":
        facets = build_multiscale_query_facets(
            hidden_states, support_span, windows=multiscale_windows,
            include_global=False, native_query=native_query,
        )
        for index, provenance in enumerate(facets.provenance):
            add_node(
                f"scale.{index}", "query", provenance.family,
                range(provenance.token_start, provenance.token_end),
            )
    elif mode in {"graph", "syntactic_graph"}:
        regions = (
            ((start, end, "query"),)
            if mode == "graph"
            else _coarse_spans(token_texts, support_span, coarse_partition_mode)
        )
        for region_index, (left, right, label) in enumerate(regions):
            parent_id = "query"
            if mode == "syntactic_graph":
                parent_id = f"coarse.{region_index}"
                add_node(parent_id, "query", label, range(left, right), add_to_scoring=True)
            graph, clusters, _, local_graph_ms, local_cluster_ms = _build_graph_region(
                hidden_states, token_texts, (left, right), graph_config
            )
            graph_calls += 1
            graph_nodes += graph.node_count
            graph_edges += graph.edge_count
            graph_bytes += graph_memory_bytes(graph)
            pairwise += graph.node_count * max(graph.node_count - 1, 0)
            graph_ms += local_graph_ms
            cluster_ms += local_cluster_ms
            for component_index, label_id in enumerate(torch.unique(clusters.labels, sorted=True).tolist()):
                local_members = torch.nonzero(
                    clusters.labels == label_id, as_tuple=False
                ).flatten().tolist()
                global_members = [left + index for index in local_members]
                statistics = _component_statistics(graph, local_members)
                add_node(
                    f"graph.{region_index}.{component_index}", parent_id, "graph_community",
                    global_members, statistics=statistics,
                    confidence=statistics.component_confidence,
                )

    child_map: dict[str, list[str]] = {node.facet_id: [] for node in nodes}
    for node in nodes:
        if node.parent_id is not None:
            child_map[node.parent_id].append(node.facet_id)
    nodes = [
        HierarchicalFacetNode(
            node.facet_id, node.parent_id, tuple(child_map[node.facet_id]), node.kind,
            node.facet_type, node.token_indices, node.token_spans, node.embedding,
            node.semantic_centroid, node.entities, node.rare_tokens,
            node.lexical_features, node.graph_statistics, node.confidence,
        )
        for node in nodes
    ]
    scoring = QueryFacetSet(
        hidden=torch.stack(flat_hidden),
        native_query=torch.stack(flat_native) if native_query is not None else None,
        provenance=tuple(flat_provenance),
    )
    possible_edges = graph_nodes * max(graph_nodes - 1, 0)
    metrics = FacetConstructionMetrics(
        facet_mode=mode,
        construction_latency_ms=(time.perf_counter() - started) * 1000.0,
        graph_construction_ms=graph_ms,
        graph_clustering_ms=cluster_ms,
        graph_calls=graph_calls,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        graph_density=graph_edges / possible_edges if possible_edges else 0.0,
        graph_memory_bytes=graph_bytes,
        pairwise_similarity_evaluations=pairwise,
        facet_count=int(scoring.hidden.shape[0]),
        tree_node_count=len(nodes),
        mean_facet_overlap=_mean_overlap(nodes),
    )
    return AdaptiveFacetTree(mode, "query", tuple(nodes), scoring, metrics)
