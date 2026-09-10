"""PRA integration contracts for llama.cpp and llama-server."""

from .adapter import (
    LlamaCppEngineAdapter,
    LlamaCppLivePrefixPlan,
    LlamaCppLivePrefixRange,
    LlamaCppNativeExecutor,
    LlamaCppNativeServerExecutor,
    LlamaCppSlotClient,
    LlamaCppSlotState,
)
from .runtime_provider import LlamaCppRuntimeProvider
from .resource_slots import (
    LlamaCppResourceIdentity,
    LlamaCppResourceSlotAllocator,
    LlamaCppSlotCapacityError,
    LlamaCppSlotIsolationError,
    LlamaCppSlotLease,
)

__all__ = [
    "LlamaCppEngineAdapter",
    "LlamaCppLivePrefixPlan",
    "LlamaCppLivePrefixRange",
    "LlamaCppNativeExecutor",
    "LlamaCppNativeServerExecutor",
    "LlamaCppRuntimeProvider",
    "LlamaCppResourceIdentity",
    "LlamaCppResourceSlotAllocator",
    "LlamaCppSlotCapacityError",
    "LlamaCppSlotIsolationError",
    "LlamaCppSlotLease",
    "LlamaCppSlotClient",
    "LlamaCppSlotState",
]
