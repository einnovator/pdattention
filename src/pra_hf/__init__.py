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
from .query_regions import (
    PromptSegment,
    QueryRegion,
    QueryRegionSelection,
    QueryRegionSelector,
    render_segments,
    token_offsets,
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

__version__ = "0.2.0rc1"

__all__ = [
    "GenerationResult",
    "ActionField",
    "AdaptiveRetryAgent",
    "AutoregressiveEffortRouter",
    "ControllerFeatures",
    "EffortProfile",
    "EffortDecision",
    "GistIndex",
    "HierarchicalGistIndex",
    "HierarchicalLocalGistRouter",
    "HandRuleController",
    "HashingQueryEncoder",
    "IterativeGistRouter",
    "IterativeRoutingConfig",
    "IterativeRoutingResult",
    "LinearEffortController",
    "MultiHeadEffortRouter",
    "NativeQKIndex",
    "PRAConfig",
    "PRAForCausalLM",
    "PRAMemoryAdapter",
    "PRARouter",
    "PromptSegment",
    "PagedKVCache",
    "ReferenceHandle",
    "QueryRegion",
    "QueryRegionSelection",
    "QueryRegionSelector",
    "RouterActionSpace",
    "RetrievalGraph",
    "StopPolicy",
    "default_effort_profiles",
    "evaluate_router_features",
    "fused_gather_kv",
    "pack_ragged",
    "profile_actions",
    "render_segments",
    "token_offsets",
]
