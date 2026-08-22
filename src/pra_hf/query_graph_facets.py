"""Pool sparse query-graph communities into existing PRA query facets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .query_facets import QueryFacetProvenance, QueryFacetSet
from .query_graph import QueryGraph
from .query_graph_cluster import ClusterResult, canonicalize_labels


@dataclass(frozen=True)
class GraphFacetProvenance:
    """Exact graph membership retained behind one pooled facet."""

    facet_id: int
    kind: str
    member_unit_ids: tuple[int, ...]
    token_spans: tuple[tuple[int, int], ...]
    member_texts: tuple[str, ...]
    graph_method: str


@dataclass(frozen=True)
class GraphQueryFacetSet:
    """Graph-pooled states and memberships.

    ``hidden`` is ``[facets,width]`` and ``membership`` is ``[nodes,facets]``.
    A requested global row is column zero; discovered communities follow it.
    Native query heads, when supplied, remain ``[facets,heads,head_dim]``.
    """

    hidden: torch.Tensor
    native_query: torch.Tensor | None
    membership: torch.Tensor
    provenance: tuple[GraphFacetProvenance, ...]

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2 or self.hidden.shape[0] == 0:
            raise ValueError("Graph facets must have shape [facets,width].")
        if self.membership.ndim != 2 or self.membership.shape[1] != self.hidden.shape[0]:
            raise ValueError("membership must have shape [nodes,facets].")
        if len(self.provenance) != self.hidden.shape[0]:
            raise ValueError("Facet provenance must align with pooled states.")
        if self.native_query is not None and (
            self.native_query.ndim != 3
            or self.native_query.shape[0] != self.hidden.shape[0]
        ):
            raise ValueError("Native graph facets must have shape [facets,heads,dim].")

    def as_query_facet_set(self) -> QueryFacetSet:
        """Adapt graph facets to the frozen Paper 2.5 scoring interface."""

        rows = []
        for row in self.provenance:
            start = min(span[0] for span in row.token_spans)
            end = max(span[1] for span in row.token_spans)
            rows.append(
                QueryFacetProvenance(
                    kind="global" if row.kind == "global" else "local",
                    token_start=start,
                    token_end=end,
                    family=f"query_graph_{row.graph_method}",
                    scale=len(row.member_unit_ids),
                )
            )
        return QueryFacetSet(
            hidden=self.hidden,
            native_query=self.native_query,
            provenance=tuple(rows),
        )


def _pool(values: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    mass = membership.sum(dim=0).clamp_min(1e-12)
    if values.ndim == 2:
        return membership.T @ values / mass.unsqueeze(1)
    if values.ndim == 3:
        return torch.einsum("nf,nhd->fhd", membership, values) / mass[:, None, None]
    raise ValueError("Poolable query states must be rank two or three.")


def _provenance(
    graph: QueryGraph,
    membership: torch.Tensor,
    *,
    method: str,
    include_global: bool,
) -> tuple[GraphFacetProvenance, ...]:
    rows = []
    for facet in range(membership.shape[1]):
        members = torch.nonzero(membership[:, facet] > 0, as_tuple=False).flatten().tolist()
        if not members:
            raise ValueError("Every graph facet must contain at least one query unit.")
        source = [graph.provenance[index] for index in members]
        rows.append(
            GraphFacetProvenance(
                facet_id=facet,
                kind="global" if include_global and facet == 0 else "community",
                member_unit_ids=tuple(int(row.unit_id) for row in source),
                token_spans=tuple((int(row.token_start), int(row.token_end)) for row in source),
                member_texts=tuple(str(row.text) for row in source),
                graph_method=method,
            )
        )
    return tuple(rows)


def pool_hard_graph_facets(
    graph: QueryGraph,
    hidden_states: torch.Tensor,
    clusters: ClusterResult | torch.Tensor,
    *,
    native_query: torch.Tensor | None = None,
    include_global: bool = True,
) -> GraphQueryFacetSet:
    """Mean-pool one facet per hard community without losing membership."""

    if hidden_states.ndim != 2 or hidden_states.shape[0] != graph.node_count:
        raise ValueError("hidden_states must align with graph nodes as [N,width].")
    if native_query is not None and (
        native_query.ndim != 3 or native_query.shape[0] != graph.node_count
    ):
        raise ValueError("native_query must align with graph nodes.")
    labels = clusters.labels if isinstance(clusters, ClusterResult) else clusters
    labels = canonicalize_labels(labels.to(device=hidden_states.device, dtype=torch.long))
    if labels.shape != (graph.node_count,):
        raise ValueError("Cluster labels must align with graph nodes.")
    count = int(labels.max()) + 1
    membership = torch.nn.functional.one_hot(labels, num_classes=count).to(hidden_states.dtype)
    method = clusters.method if isinstance(clusters, ClusterResult) else "hard_membership"
    if include_global:
        membership = torch.cat(
            [torch.ones((graph.node_count, 1), device=hidden_states.device, dtype=hidden_states.dtype), membership],
            dim=1,
        )
    return GraphQueryFacetSet(
        hidden=_pool(hidden_states, membership),
        native_query=_pool(native_query, membership) if native_query is not None else None,
        membership=membership,
        provenance=_provenance(
            graph, membership, method=method, include_global=include_global
        ),
    )


def pool_soft_graph_facets(
    graph: QueryGraph,
    hidden_states: torch.Tensor,
    membership: torch.Tensor,
    *,
    native_query: torch.Tensor | None = None,
    include_global: bool = True,
    method: str = "soft_propagation",
) -> GraphQueryFacetSet:
    """Pool bounded soft memberships ``Q[N,K]`` without selecting oracle K."""

    if membership.ndim != 2 or membership.shape[0] != graph.node_count:
        raise ValueError("membership must have shape [N,K].")
    if membership.shape[1] == 0 or bool((membership < 0).any()):
        raise ValueError("Soft memberships must be non-negative with K > 0.")
    membership = membership.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if bool((membership.sum(dim=1) <= 0).any()):
        raise ValueError("Every query unit must have positive membership mass.")
    membership = membership / membership.sum(dim=1, keepdim=True)
    if include_global:
        membership = torch.cat(
            [torch.ones((graph.node_count, 1), device=hidden_states.device, dtype=hidden_states.dtype), membership],
            dim=1,
        )
    return GraphQueryFacetSet(
        hidden=_pool(hidden_states, membership),
        native_query=_pool(native_query, membership) if native_query is not None else None,
        membership=membership,
        provenance=_provenance(
            graph, membership, method=method, include_global=include_global
        ),
    )


def suppress_graph_facet(facets: GraphQueryFacetSet, facet_index: int) -> GraphQueryFacetSet:
    """Remove one discovered facet for causal retrieval ablations."""

    if not 0 <= facet_index < facets.hidden.shape[0]:
        raise ValueError("facet_index is outside the facet set.")
    if facets.provenance[facet_index].kind == "global":
        raise ValueError("The global control is not a discovered facet ablation.")
    keep = [index for index in range(facets.hidden.shape[0]) if index != facet_index]
    return GraphQueryFacetSet(
        hidden=facets.hidden[keep],
        native_query=facets.native_query[keep] if facets.native_query is not None else None,
        membership=facets.membership[:, keep],
        provenance=tuple(facets.provenance[index] for index in keep),
    )
