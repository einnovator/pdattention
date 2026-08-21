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
    "TokenChunkRecord",
    "TokenNativeIndex",
    "evaluate_router_features",
]
