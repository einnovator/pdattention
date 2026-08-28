"""MLX transport and unified-memory logical bridges for PRA."""

from .adapter import MLXEngineAdapter, MLXNativeExecutor
from .native import (
    MLXInProcessNativeExecutor,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXSelectedKVCache,
    combine_native_memories,
    encode_native_memory,
    make_native_prompt_cache,
)

__all__ = [
    "MLXEngineAdapter",
    "MLXInProcessNativeExecutor",
    "MLXNativeExecutor",
    "MLXNativeLayerKV",
    "MLXNativeMemory",
    "MLXSelectedKVCache",
    "combine_native_memories",
    "encode_native_memory",
    "make_native_prompt_cache",
]
