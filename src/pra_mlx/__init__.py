"""MLX transport and unified-memory logical bridges for PRA."""

from .adapter import MLXEngineAdapter, MLXNativeExecutor
from .native import (
    MLXInProcessNativeExecutor,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXNativeFingerprint,
    MLXPositionedKVCache,
    MLXSelectedKVCache,
    combine_native_memories,
    encode_native_memory,
    make_native_prompt_cache,
    load_native_memory,
    save_native_memory,
)

__all__ = [
    "MLXEngineAdapter",
    "MLXInProcessNativeExecutor",
    "MLXNativeExecutor",
    "MLXNativeLayerKV",
    "MLXNativeMemory",
    "MLXNativeFingerprint",
    "MLXPositionedKVCache",
    "MLXSelectedKVCache",
    "combine_native_memories",
    "encode_native_memory",
    "make_native_prompt_cache",
    "load_native_memory",
    "save_native_memory",
]
