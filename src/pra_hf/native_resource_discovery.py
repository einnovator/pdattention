"""Deployment-aware model-native resource discovery for Paper 6.5 M6.

Generic agent integrations should use external discovery. Native-Q/K search is
an optional server-side mode: raw vectors remain inside the model runtime and
only stable resource identities, scores, and provenance cross the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

import torch


class NativeDiscoveryDeployment(str, Enum):
    """Architectural boundary for access to model-native query state."""

    COLOCATED = "co_located"
    SHARED_MEMORY = "shared_memory"
    MODEL_SERVER = "model_server"
    REPLICATED_QUERY = "replicated_query"


@dataclass(frozen=True)
class NativeResourceSearchRequest:
    """Boundary-safe request accepted by a model-server native resolver."""

    collection: str
    query_context: str
    top_k: int = 4
    routing_mode: str = "native_mean_k"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")


@dataclass(frozen=True)
class NativeResourceHit:
    """Stable identity returned without raw query/key vectors."""

    uri: str
    score: float
    rank: int
    provenance: str


@dataclass(frozen=True)
class NativeResourceSearchResult:
    """Server-safe reply plus declared deployment and index identity."""

    hits: tuple[NativeResourceHit, ...]
    deployment: NativeDiscoveryDeployment
    index_fingerprint: str
    raw_state_exported: bool = False


@dataclass(frozen=True)
class ProjectedQueryExport:
    """Minimal low-rank query payload allowed in shared-memory mode."""

    values: torch.Tensor
    model_revision: str
    routing_layer: int
    projection_fingerprint: str

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("A projected query export must have shape [rank].")


def repeat_kv_heads(keys: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Expand ``[...,kv_heads,head_dim]`` keys to the query-head count."""

    if keys.ndim < 2 or query_heads <= 0:
        raise ValueError("Keys need head dimensions and query_heads must be positive.")
    kv_heads = keys.shape[-2]
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by K/V heads.")
    return keys.repeat_interleave(query_heads // kv_heads, dim=-2)


def native_mean_k_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Score one ``[H,D]`` query against one mean K gist per resource.

    ``keys`` has shape ``[resources,tokens,kv_heads,head_dim]`` and
    ``token_mask`` has shape ``[resources,tokens]``.
    """

    if query.ndim != 2 or keys.ndim != 4 or token_mask.shape != keys.shape[:2]:
        raise ValueError("Expected query [H,D], keys [R,T,KVH,D], mask [R,T].")
    expanded = repeat_kv_heads(keys, query.shape[0])
    weights = token_mask.to(expanded.dtype)[..., None, None]
    means = (expanded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return torch.einsum("hd,rhd->r", query, means) / math.sqrt(query.shape[-1])


def native_token_qk_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    top_r: int = 4,
) -> torch.Tensor:
    """Score resources from their strongest physical native-K token responses."""

    if top_r <= 0:
        raise ValueError("top_r must be positive.")
    expanded = repeat_kv_heads(keys, query.shape[0])
    dots = torch.einsum("hd,rthd->rth", query, expanded) / math.sqrt(query.shape[-1])
    dots = dots.mean(dim=-1).masked_fill(~token_mask, float("-inf"))
    count = min(top_r, dots.shape[1])
    values = dots.topk(count, dim=1).values
    finite = torch.isfinite(values)
    return values.masked_fill(~finite, 0.0).sum(dim=1) / finite.sum(dim=1).clamp_min(1)


class NativeResolverEndpoint:
    """Conceptual model-server endpoint that never returns native tensors."""

    def __init__(
        self,
        resources: Sequence[str],
        scorer: Callable[[str, str], Sequence[float]],
        *,
        index_fingerprint: str,
        deployment: NativeDiscoveryDeployment | str = NativeDiscoveryDeployment.MODEL_SERVER,
    ) -> None:
        self.resources = tuple(resources)
        self.scorer = scorer
        self.index_fingerprint = index_fingerprint
        self.deployment = NativeDiscoveryDeployment(deployment)

    def search(self, request: NativeResourceSearchRequest) -> NativeResourceSearchResult:
        """Compute internally and return only ranked typed resource identities."""

        scores = tuple(float(value) for value in self.scorer(request.query_context, request.routing_mode))
        if len(scores) != len(self.resources):
            raise ValueError("Native scorer output does not match the resource collection.")
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], self.resources[index]))
        hits = tuple(
            NativeResourceHit(
                self.resources[index], scores[index], rank, f"native:{request.routing_mode}"
            )
            for rank, index in enumerate(order[: request.top_k], start=1)
        )
        return NativeResourceSearchResult(
            hits,
            self.deployment,
            self.index_fingerprint,
            raw_state_exported=False,
        )
