"""vLLM transport and logical-memory bridges for PRA."""

from .adapter import VLLMEngineAdapter, VLLMNativeExecutor
from .cuda_scheduler_alias import (
    SchedulerAliasTelemetry,
    SchedulerPageSelection,
    VLLMCudaSchedulerPageRegistry,
    install_vllm_scheduler_page_alias_hooks,
)
from .metal_native import VLLMMetalBlockHandle, VLLMMetalPRAStore
from .metal_live_kv import (
    VLLMMetalLiveKVRuntime,
    VLLMMetalLiveRequest,
    VLLMMetalLiveSelection,
    VLLMMetalLiveSource,
    VLLMMetalOffloadedSource,
)
from .v1_metadata import (
    VLLMNativeBlockSet,
    VLLMNativeStep,
    VLLMNativeStepRegistry,
)
from .v1_native import (
    VLLMMetalV1NativeBridge,
    VLLMPageHotBridge,
    augment_paged_context,
    capture_paged_memory,
    native_request_cache_salt,
)

__all__ = [
    "SchedulerAliasTelemetry",
    "SchedulerPageSelection",
    "VLLMEngineAdapter",
    "VLLMCudaSchedulerPageRegistry",
    "VLLMMetalBlockHandle",
    "VLLMMetalLiveKVRuntime",
    "VLLMMetalLiveRequest",
    "VLLMMetalLiveSelection",
    "VLLMMetalLiveSource",
    "VLLMMetalOffloadedSource",
    "VLLMMetalPRAStore",
    "VLLMMetalV1NativeBridge",
    "VLLMPageHotBridge",
    "VLLMNativeBlockSet",
    "VLLMNativeExecutor",
    "VLLMNativeStep",
    "VLLMNativeStepRegistry",
    "augment_paged_context",
    "capture_paged_memory",
    "install_vllm_scheduler_page_alias_hooks",
    "native_request_cache_salt",
]
