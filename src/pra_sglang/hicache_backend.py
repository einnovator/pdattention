"""Adapters from PRA logical memory to SGLang HiCache storage backends."""

from __future__ import annotations

import hashlib
import io
import json
from typing import Protocol

from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory


class PRAHiCacheL3Backend(Protocol):
    """Immutable object interface required by :class:`SGLangPRAHiCache`."""

    def put(self, logical_key: str, memory: MLXNativeMemory) -> int: ...

    def get(self, logical_key: str) -> MLXNativeMemory: ...

    def exists(self, logical_key: str) -> bool: ...

    def size(self, logical_key: str) -> int: ...

    def remove(self, logical_key: str) -> None: ...


def _serialize_memory(memory: MLXNativeMemory, *, compress: bool = False) -> bytes:
    """Serialize K/V shapes and logical dtypes into one backend-neutral blob."""

    import numpy as np
    from pra_sglang.hicache import _HostArray, _default_to_host

    host = _default_to_host(memory)
    arrays: dict[str, object] = {}
    logical_dtypes: dict[str, str] = {}
    for index, layer in enumerate(host.layers):
        for suffix, array in (("k", layer.keys), ("v", layer.values)):
            name = f"layer_{index:04d}_{suffix}"
            arrays[name] = array.data if isinstance(array, _HostArray) else array
            logical_dtypes[name] = (
                array.logical_dtype if isinstance(array, _HostArray) else str(array.dtype)
            )
    metadata = json.dumps(
        {
            "schema": "pra-sglang-hicache-blob-v1",
            "source_tokens": memory.source_tokens,
            "layer_count": len(memory.layers),
            "logical_dtypes": logical_dtypes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    arrays["__pra_metadata__"] = np.frombuffer(metadata, dtype=np.uint8)
    stream = io.BytesIO()
    writer = np.savez_compressed if compress else np.savez
    writer(stream, **arrays)
    return stream.getvalue()


def _deserialize_memory(payload: bytes) -> MLXNativeMemory:
    """Restore a host-owned memory; L3-to-L1 promotion moves it to MLX."""

    import numpy as np
    from pra_sglang.hicache import _HostArray

    with np.load(io.BytesIO(payload)) as arrays:
        metadata = json.loads(bytes(arrays["__pra_metadata__"]).decode("utf-8"))
        if metadata.get("schema") != "pra-sglang-hicache-blob-v1":
            raise RuntimeError("Unsupported PRA HiCache blob schema.")
        layers = tuple(
            MLXNativeLayerKV(
                _HostArray(
                    arrays[f"layer_{index:04d}_k"].copy(),
                    metadata["logical_dtypes"][f"layer_{index:04d}_k"],
                ),
                _HostArray(
                    arrays[f"layer_{index:04d}_v"].copy(),
                    metadata["logical_dtypes"][f"layer_{index:04d}_v"],
                ),
            )
            for index in range(int(metadata["layer_count"]))
        )
    return MLXNativeMemory(layers, int(metadata["source_tokens"]))


class SGLangHiCacheStorageBackend:
    """Store PRA blobs through SGLang's actual ``HiCacheStorage`` API.

    The supplied storage object may be the built-in file backend or a
    distributed backend created by ``StorageBackendFactory``. A fixed-size
    header records the payload length because ``HiCacheStorage.get`` requires a
    preallocated destination tensor. Logical removal revokes the key from this
    adapter; portable backend APIs intentionally leave physical reclamation to
    their own eviction or administrative-clear policy.
    """

    header_bytes = 512

    def __init__(
        self,
        storage: object,
        *,
        namespace: str = "pra",
        compress: bool = False,
    ) -> None:
        self.storage = storage
        self.namespace = str(namespace)
        self.compress = bool(compress)
        self._sizes: dict[str, int] = {}
        self._revoked: set[str] = set()

    def _stem(self, logical_key: str) -> str:
        digest = hashlib.sha256(str(logical_key).encode("utf-8")).hexdigest()
        return f"{self.namespace}-{digest}"

    def _keys(self, logical_key: str) -> tuple[str, str]:
        stem = self._stem(logical_key)
        return f"{stem}-header", f"{stem}-payload"

    @staticmethod
    def _tensor_from_bytes(payload: bytes):
        import numpy as np
        import torch

        return torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())

    def put(self, logical_key: str, memory: MLXNativeMemory) -> int:
        payload = _serialize_memory(memory, compress=self.compress)
        header = json.dumps(
            {"schema": "pra-hicache-header-v1", "payload_bytes": len(payload)},
            separators=(",", ":"),
        ).encode("utf-8")
        if len(header) >= self.header_bytes:
            raise RuntimeError("PRA HiCache header exceeds its fixed allocation.")
        header += b"\0" * (self.header_bytes - len(header))
        header_key, payload_key = self._keys(logical_key)
        if not self.storage.set(header_key, self._tensor_from_bytes(header)):
            raise IOError("SGLang HiCache rejected the PRA metadata header.")
        if not self.storage.set(payload_key, self._tensor_from_bytes(payload)):
            raise IOError("SGLang HiCache rejected the PRA memory payload.")
        self._sizes[str(logical_key)] = len(header) + len(payload)
        self._revoked.discard(str(logical_key))
        return self._sizes[str(logical_key)]

    def _read_tensor(self, key: str, size: int) -> bytes:
        import torch

        target = torch.empty(size, dtype=torch.uint8)
        result = self.storage.get(key, target)
        if result is None:
            raise KeyError(key)
        return bytes(result.numpy())

    def get(self, logical_key: str) -> MLXNativeMemory:
        if str(logical_key) in self._revoked:
            raise KeyError(logical_key)
        header_key, payload_key = self._keys(logical_key)
        raw_header = self._read_tensor(header_key, self.header_bytes)
        header = json.loads(raw_header.split(b"\0", 1)[0].decode("utf-8"))
        if header.get("schema") != "pra-hicache-header-v1":
            raise RuntimeError("Unsupported PRA HiCache storage header.")
        payload_bytes = int(header["payload_bytes"])
        payload = self._read_tensor(payload_key, payload_bytes)
        self._sizes[str(logical_key)] = self.header_bytes + payload_bytes
        return _deserialize_memory(payload)

    def exists(self, logical_key: str) -> bool:
        if str(logical_key) in self._revoked:
            return False
        header_key, payload_key = self._keys(logical_key)
        return bool(self.storage.exists(header_key) and self.storage.exists(payload_key))

    def size(self, logical_key: str) -> int:
        size = self._sizes.get(str(logical_key))
        if size is not None:
            return size
        if not self.exists(logical_key):
            return 0
        header_key, _ = self._keys(logical_key)
        raw_header = self._read_tensor(header_key, self.header_bytes)
        header = json.loads(raw_header.split(b"\0", 1)[0].decode("utf-8"))
        size = self.header_bytes + int(header["payload_bytes"])
        self._sizes[str(logical_key)] = size
        return size

    def remove(self, logical_key: str) -> None:
        key = str(logical_key)
        self._sizes.pop(key, None)
        self._revoked.add(key)
