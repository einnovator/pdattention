"""Public PRA-HF API for bounded sparse native-K/V memory."""

from .config import PRAConfig
from .evaluation import evaluate_router_features
from .memory_adapter import PRAMemoryAdapter
from .model import GenerationResult, PRAForCausalLM, ReferenceHandle
from .router import PRARouter
from .adaptive_runtime import (
    AdaptiveRetryAgent,
    ControllerFeatures,
    EffortProfile,
    HandRuleController,
    LinearEffortController,
    StopPolicy,
    default_effort_profiles,
)
from .serving_runtime import NativeQKIndex, PagedKVCache, fused_gather_kv, pack_ragged
from .effort_router import (
    ActionField,
    AutoregressiveEffortRouter,
    EffortDecision,
    HashingQueryEncoder,
    MultiHeadEffortRouter,
    RouterActionSpace,
    profile_actions,
)
from .factorized_control import (
    BUDGET_LEVELS,
    FACET_LEVELS,
    HOP_LEVELS,
    NEIGHBOR_LEVELS,
    ROOT_LEVELS,
    FactorizedEffortAction,
    allocation_outcome,
    changed_control,
    cheapest_sufficient,
    dominates,
    evidence_kv_metrics,
    factorized_action_space,
    factorized_cost,
    pareto_frontier,
)
from .adaptive_search import (
    AdaptiveSearchAction,
    ROOT_METHODS,
    SUCCESSOR_METHODS,
    SearchMethodActionSpec,
    SearchTransition,
    choose_successor_cascade,
    load_search_method_action_spec,
    method_retry_action,
)
from .adaptive_facets import (
    AdaptiveFacetTree,
    FacetConstructionMetrics,
    GraphFacetConfig,
    HierarchicalFacetNode,
    build_adaptive_query_facets,
)
from .facet_routing import (
    FacetRoute,
    LinearPerFacetRouter,
    route_query_facets,
    select_per_facet_oracle,
)
from .self_router import (
    QueryPrefillAccounting,
    QwenPrefixState,
    ValidationProjector,
    decode_grouped_action,
    native_qk_representation,
    pool_query_tokens,
    query_span_mask,
    qwen_prefill_continue,
    qwen_prefill_prefix,
    reuse_is_semantically_valid,
)
from .query_regions import (
    PromptSegment,
    QueryRegion,
    QueryRegionSelection,
    QueryRegionSelector,
    render_segments,
    token_offsets,
)
from .query_graph import QueryGraph, QueryUnitProvenance, build_query_graph
from .query_graph_cluster import connected_components, weighted_label_propagation
from .query_graph_facets import GraphQueryFacetSet, pool_hard_graph_facets
from .iterative import (
    GistIndex,
    HierarchicalGistIndex,
    HierarchicalLocalGistRouter,
    IterativeGistRouter,
    IterativeRoutingConfig,
    IterativeRoutingResult,
    RetrievalGraph,
)

__version__ = "0.2.0rc1"

__all__ = [
    "GenerationResult",
    "ActionField",
    "AdaptiveRetryAgent",
    "AdaptiveSearchAction",
    "AdaptiveFacetTree",
    "AutoregressiveEffortRouter",
    "ControllerFeatures",
    "EffortProfile",
    "EffortDecision",
    "FacetConstructionMetrics",
    "FacetRoute",
    "GistIndex",
    "HierarchicalGistIndex",
    "HierarchicalLocalGistRouter",
    "HandRuleController",
    "HashingQueryEncoder",
    "GraphFacetConfig",
    "GraphQueryFacetSet",
    "HierarchicalFacetNode",
    "IterativeGistRouter",
    "IterativeRoutingConfig",
    "IterativeRoutingResult",
    "LinearEffortController",
    "LinearPerFacetRouter",
    "MultiHeadEffortRouter",
    "NativeQKIndex",
    "PRAConfig",
    "PRAForCausalLM",
    "PRAMemoryAdapter",
    "PRARouter",
    "PromptSegment",
    "PagedKVCache",
    "ReferenceHandle",
    "ROOT_METHODS",
    "QueryRegion",
    "QueryRegionSelection",
    "QueryRegionSelector",
    "QueryGraph",
    "QueryUnitProvenance",
    "RouterActionSpace",
    "SUCCESSOR_METHODS",
    "SearchMethodActionSpec",
    "SearchTransition",
    "RetrievalGraph",
    "StopPolicy",
    "default_effort_profiles",
    "choose_successor_cascade",
    "build_adaptive_query_facets",
    "build_query_graph",
    "connected_components",
    "evaluate_router_features",
    "fused_gather_kv",
    "pack_ragged",
    "pool_hard_graph_facets",
    "load_search_method_action_spec",
    "method_retry_action",
    "route_query_facets",
    "select_per_facet_oracle",
    "profile_actions",
    "render_segments",
    "token_offsets",
    "weighted_label_propagation",
]
