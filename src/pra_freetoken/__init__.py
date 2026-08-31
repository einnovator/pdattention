"""PRA integration contracts for bandwidth-adaptive FreeToken serving.

The scheduler model is dependency-light. Runtime adapter imports stay lazy so
FreeToken hosts can run coordination diagnostics before installing the full SDK.
"""

from typing import TYPE_CHECKING

from .coordinator import (
    PrefetchDemand,
    PrefetchOutcome,
    PrefetchStrategy,
    coordinate_prefetch,
)

if TYPE_CHECKING:
    from .adapter import FreeTokenEngineAdapter, FreeTokenNativeExecutor
    from .runtime_provider import FreeTokenRuntimeProvider

__all__ = [
    "FreeTokenEngineAdapter",
    "FreeTokenNativeExecutor",
    "FreeTokenRuntimeProvider",
    "PrefetchDemand",
    "PrefetchOutcome",
    "PrefetchStrategy",
    "coordinate_prefetch",
]


def __getattr__(name: str):
    if name in {"FreeTokenEngineAdapter", "FreeTokenNativeExecutor"}:
        from .adapter import FreeTokenEngineAdapter, FreeTokenNativeExecutor

        return {
            "FreeTokenEngineAdapter": FreeTokenEngineAdapter,
            "FreeTokenNativeExecutor": FreeTokenNativeExecutor,
        }[name]
    if name == "FreeTokenRuntimeProvider":
        from .runtime_provider import FreeTokenRuntimeProvider

        return FreeTokenRuntimeProvider
    raise AttributeError(name)
