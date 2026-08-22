"""Public PRA-HF API for bounded sparse native-K/V memory."""

from .config import PRAConfig
from .evaluation import evaluate_router_features
from .memory_adapter import PRAMemoryAdapter
from .model import GenerationResult, PRAForCausalLM, ReferenceHandle
from .router import PRARouter
from .hybrid_discovery import (
    DiscoveryCandidate,
    HybridDiscoveryPolicy,
    TokenChunkRecord,
    TokenNativeIndex,
)
from .iterative import (
    GistIndex,
    HierarchicalGistIndex,
    HierarchicalLocalGistRouter,
    IterativeGistRouter,
    IterativeRoutingConfig,
    IterativeRoutingResult,
    RetrievalGraph,
)
from .query_graph import QueryGraph, QueryUnitProvenance, build_query_graph
from .query_graph_cluster import connected_components, weighted_label_propagation
from .query_graph_facets import GraphQueryFacetSet, pool_hard_graph_facets

__version__ = "0.2.0rc1"

__all__ = [
    "GenerationResult",
    "DiscoveryCandidate",
    "GistIndex",
    "HierarchicalGistIndex",
    "HierarchicalLocalGistRouter",
    "IterativeGistRouter",
    "IterativeRoutingConfig",
    "IterativeRoutingResult",
    "HybridDiscoveryPolicy",
    "PRAConfig",
    "PRAForCausalLM",
    "PRAMemoryAdapter",
    "PRARouter",
    "ReferenceHandle",
    "RetrievalGraph",
    "GraphQueryFacetSet",
    "QueryGraph",
    "QueryUnitProvenance",
    "TokenChunkRecord",
    "TokenNativeIndex",
    "build_query_graph",
    "connected_components",
    "evaluate_router_features",
    "pool_hard_graph_facets",
    "weighted_label_propagation",
]
