"""MLX transport and unified-memory logical bridges for PRA."""

from .adapter import MLXEngineAdapter, MLXNativeExecutor
from .native import (
    MLXInProcessNativeExecutor,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXNativeColdCodec,
    MLXNativeFingerprint,
    MLXPositionedKVCache,
    MLXQuantizedLayerKV,
    MLXQuantizedMemory,
    MLXSelectedKVCache,
    combine_native_memories,
    dequantize_native_memory,
    deserialize_native_memory,
    encode_native_memory,
    make_native_prompt_cache,
    load_native_memory,
    quantize_native_memory,
    save_native_memory,
    serialize_native_memory,
)
from .native_storage import MLXNativeSegmentStore

__all__ = [
    "MLXEngineAdapter",
    "MLXInProcessNativeExecutor",
    "MLXNativeExecutor",
    "MLXNativeLayerKV",
    "MLXNativeMemory",
    "MLXNativeSegmentStore",
    "MLXNativeColdCodec",
    "MLXNativeFingerprint",
    "MLXPositionedKVCache",
    "MLXQuantizedLayerKV",
    "MLXQuantizedMemory",
    "MLXSelectedKVCache",
    "combine_native_memories",
    "dequantize_native_memory",
    "deserialize_native_memory",
    "encode_native_memory",
    "make_native_prompt_cache",
    "load_native_memory",
    "quantize_native_memory",
    "save_native_memory",
    "serialize_native_memory",
]
