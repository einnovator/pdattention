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
    "AdaptiveRetryAgent",
    "ControllerFeatures",
    "EffortProfile",
    "GistIndex",
    "HierarchicalGistIndex",
    "HierarchicalLocalGistRouter",
    "HandRuleController",
    "IterativeGistRouter",
    "IterativeRoutingConfig",
    "IterativeRoutingResult",
    "LinearEffortController",
    "NativeQKIndex",
    "PRAConfig",
    "PRAForCausalLM",
    "PRAMemoryAdapter",
    "PRARouter",
    "PagedKVCache",
    "ReferenceHandle",
    "RetrievalGraph",
    "StopPolicy",
    "default_effort_profiles",
    "evaluate_router_features",
    "fused_gather_kv",
    "pack_ragged",
]
