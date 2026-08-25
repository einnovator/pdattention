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
from .qk_compression import LOW_RANK_INDEX_DTYPES, LowRankRoutingIndex
from .external_memory import (
    AuthContext,
    EncodingContext,
    ExternalMemoryManager,
    FileResourceResolver,
    HotMemoryHandle,
    NativeEncoding,
    PRASession,
    ResolverRegistry,
    ResourceRecord,
    ResourceResolver,
    ResourceStat,
)
from .pra_aware_training import (
    GemmaLayerTopology,
    gemma_layer_topology,
    hf_parameter_summary,
    install_hf_adaptation_regime,
)
from .agent_execution import (
    ExecutionAuthorization,
    SafeToolExecutor,
    ToolCall,
    ToolExecutionResult,
    parse_tool_call,
)
from .agent_resources import (
    AgentResource,
    DiscoveryDecision,
    DiscoveryMode,
    DiscoveryRequest,
    DiscoveryTrace,
    PersistentResourceIndex,
    ResourceDiscoveryEngine,
    SideEffectClass,
)
from .runtime import (
    CompilationMode,
    HuggingFaceBackend,
    KVInterval,
    KVLayout,
    KVMaterializer,
    MaterializationPlan,
    MaterializedKV,
    NativeKV,
    PackedNativeKVStore,
    PRARuntime,
    PRARuntimeConfig,
    RuntimeBackend,
    RuntimeKVCache,
    RuntimeProfiler,
    SelectedKVGather,
    VLLMThinBackend,
    VLLMThinRequest,
    runtime_capabilities,
)

__version__ = "0.2.0rc1"

__all__ = [
    "GenerationResult",
    "AuthContext",
    "DiscoveryCandidate",
    "GistIndex",
    "HierarchicalGistIndex",
    "HierarchicalLocalGistRouter",
    "IterativeGistRouter",
    "IterativeRoutingConfig",
    "IterativeRoutingResult",
    "LOW_RANK_INDEX_DTYPES",
    "LowRankRoutingIndex",
    "EncodingContext",
    "ExternalMemoryManager",
    "FileResourceResolver",
    "HotMemoryHandle",
    "HybridDiscoveryPolicy",
    "PRAConfig",
    "PRAForCausalLM",
    "PRASession",
    "PRAMemoryAdapter",
    "PRARouter",
    "ReferenceHandle",
    "ResolverRegistry",
    "ResourceRecord",
    "ResourceResolver",
    "ResourceStat",
    "NativeEncoding",
    "RetrievalGraph",
    "GraphQueryFacetSet",
    "GemmaLayerTopology",
    "QueryGraph",
    "QueryUnitProvenance",
    "TokenChunkRecord",
    "TokenNativeIndex",
    "build_query_graph",
    "connected_components",
    "evaluate_router_features",
    "gemma_layer_topology",
    "hf_parameter_summary",
    "install_hf_adaptation_regime",
    "pool_hard_graph_facets",
    "weighted_label_propagation",
]

__all__ += [
    "AgentResource",
    "CompilationMode",
    "DiscoveryDecision",
    "DiscoveryMode",
    "DiscoveryRequest",
    "DiscoveryTrace",
    "ExecutionAuthorization",
    "HuggingFaceBackend",
    "KVInterval",
    "KVLayout",
    "KVMaterializer",
    "MaterializationPlan",
    "MaterializedKV",
    "NativeKV",
    "PackedNativeKVStore",
    "PRARuntime",
    "PRARuntimeConfig",
    "PersistentResourceIndex",
    "ResourceDiscoveryEngine",
    "RuntimeBackend",
    "RuntimeKVCache",
    "RuntimeProfiler",
    "SelectedKVGather",
    "SafeToolExecutor",
    "SideEffectClass",
    "ToolCall",
    "ToolExecutionResult",
    "VLLMThinBackend",
    "VLLMThinRequest",
    "parse_tool_call",
    "runtime_capabilities",
]
