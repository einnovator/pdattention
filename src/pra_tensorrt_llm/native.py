"""Engine-facing logical block ownership for TensorRT-LLM PRA memory.

This module deliberately contains no fake attention implementation.  It owns
resource identity, topology checks, connector residency, request pinning, and
exactly-once attachment.  A version-specific engine patch consumes the returned
block handles when constructing paged-context attention metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Protocol, Sequence

from pra_hf.engine_invariants import EnginePRAIsolationGuard


@dataclass(frozen=True)
class TensorRTLLMTopology:
    """Execution geometry that must match when native K/V is reused."""

    model: str
    revision: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    kv_dtype: str = "float16"
    tokens_per_block: int = 32

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0 or self.pipeline_parallel_size <= 0:
            raise ValueError("Tensor and pipeline parallel sizes must be positive.")
        if self.tokens_per_block <= 0:
            raise ValueError("TensorRT-LLM tokens_per_block must be positive.")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TensorRTLLMBlockHandle:
    """Logical identity plus connector-managed physical paged-block handles."""

    logical_key: str
    block_ids: tuple[int, ...]
    token_count: int
    byte_count: int
    topology_fingerprint: str

    def __post_init__(self) -> None:
        if not self.logical_key or not self.topology_fingerprint:
            raise ValueError("TensorRT-LLM block handles require stable identity.")
        if not self.block_ids or len(self.block_ids) != len(set(self.block_ids)):
            raise ValueError("TensorRT-LLM block IDs must be non-empty and unique.")
        if self.token_count <= 0 or self.byte_count <= 0:
            raise ValueError("TensorRT-LLM block size accounting must be positive.")


class TensorRTConnector(Protocol):
    """Minimal PRA view of an official TensorRT-LLM KV connector."""

    def put(self, handle: TensorRTLLMBlockHandle) -> None: ...

    def get(self, logical_key: str) -> TensorRTLLMBlockHandle: ...

    def contains(self, logical_key: str) -> bool: ...

    def pin(self, logical_key: str, request_id: str) -> None: ...

    def unpin(self, logical_key: str, request_id: str) -> None: ...

    def remove(self, logical_key: str) -> None: ...


class InMemoryTensorRTConnector:
    """Deterministic connector double used for lifecycle and isolation tests."""

    def __init__(self) -> None:
        self._handles: dict[str, TensorRTLLMBlockHandle] = {}
        self._pins: dict[str, set[str]] = {}

    def put(self, handle: TensorRTLLMBlockHandle) -> None:
        existing = self._handles.get(handle.logical_key)
        if existing is not None and existing != handle:
            raise RuntimeError("A logical PRA key cannot change physical identity in place.")
        self._handles[handle.logical_key] = handle

    def get(self, logical_key: str) -> TensorRTLLMBlockHandle:
        return self._handles[logical_key]

    def contains(self, logical_key: str) -> bool:
        return logical_key in self._handles

    def pin(self, logical_key: str, request_id: str) -> None:
        if logical_key not in self._handles:
            raise KeyError(logical_key)
        self._pins.setdefault(logical_key, set()).add(request_id)

    def unpin(self, logical_key: str, request_id: str) -> None:
        pins = self._pins.get(logical_key)
        if pins is None:
            return
        pins.discard(request_id)
        if not pins:
            self._pins.pop(logical_key, None)

    def remove(self, logical_key: str) -> None:
        if self._pins.get(logical_key):
            raise RuntimeError("Pinned TensorRT-LLM PRA blocks cannot be removed.")
        self._handles.pop(logical_key, None)

    def pin_count(self, logical_key: str) -> int:
        return len(self._pins.get(logical_key, ()))


@dataclass(frozen=True)
class _RegisteredResource:
    handle: TensorRTLLMBlockHandle
    tenant_id: str
    authorization_scope: str | None


class TensorRTLLMNativeAttachmentManager:
    """Authorize, pin, attach once, and clean selected TRT-LLM block sets."""

    def __init__(
        self,
        connector: TensorRTConnector,
        topology: TensorRTLLMTopology,
    ) -> None:
        self.connector = connector
        self.topology = topology
        self.guard = EnginePRAIsolationGuard()
        self._resources: dict[str, _RegisteredResource] = {}
        self._request_tenants: dict[str, str] = {}
        self._lock = RLock()

    def register(
        self,
        handle: TensorRTLLMBlockHandle,
        *,
        tenant_id: str,
        authorization_scope: str | None = None,
    ) -> None:
        """Register connector residency only for matching model topology."""

        if handle.topology_fingerprint != self.topology.fingerprint:
            raise ValueError("PRA K/V topology does not match the TensorRT-LLM runtime.")
        with self._lock:
            current = self._resources.get(handle.logical_key)
            resource = _RegisteredResource(handle, tenant_id, authorization_scope)
            if current is not None and current != resource:
                raise RuntimeError("A logical PRA key cannot be rebound across tenants.")
            self.connector.put(handle)
            self._resources[handle.logical_key] = resource

    def open_request(
        self,
        request_id: str,
        logical_keys: Sequence[str],
        *,
        tenant_id: str,
        authorization_scopes: Sequence[str] = (),
    ) -> None:
        """Authorize and pin the complete routed set before scheduler admission."""

        allowed = set(authorization_scopes)
        keys = tuple(logical_keys)
        with self._lock:
            for key in keys:
                resource = self._resources.get(key)
                if resource is None or not self.connector.contains(key):
                    raise KeyError(f"Unknown TensorRT-LLM PRA resource: {key}")
                if resource.tenant_id != tenant_id:
                    raise PermissionError("PRA resource belongs to another tenant.")
                if resource.authorization_scope and resource.authorization_scope not in allowed:
                    raise PermissionError("PRA resource authorization scope was not granted.")
            self.guard.open_request(request_id, keys)
            try:
                for key in keys:
                    self.connector.pin(key, request_id)
            except Exception:
                for key in keys:
                    self.connector.unpin(key, request_id)
                self.guard.close_request(request_id, require_attached=False)
                raise
            self._request_tenants[request_id] = tenant_id

    def attach_once(self, request_id: str) -> tuple[TensorRTLLMBlockHandle, ...]:
        """Return the exact selected handles once for attention metadata assembly."""

        with self._lock:
            keys = self.guard.visible_keys(request_id)
            self.guard.attach_once(request_id, keys)
            return tuple(self.connector.get(key) for key in keys)

    def close_request(self, request_id: str, *, require_attached: bool = True) -> None:
        """Unpin all blocks and erase request visibility, including on failure."""

        with self._lock:
            keys = self.guard.visible_keys(request_id)
            for key in keys:
                self.connector.unpin(key, request_id)
            self.guard.close_request(request_id, require_attached=require_attached)
            self._request_tenants.pop(request_id, None)

    def assert_prefix_pool_safe(self, logical_keys: Sequence[str]) -> None:
        """Prevent selected PRA detail from entering ordinary prefix reuse."""

        self.guard.assert_ordinary_pool_safe(logical_keys)

    def metrics(self) -> Mapping[str, int]:
        return {
            "registered_resources": len(self._resources),
            "active_requests": sum(
                self.guard.view(request_id) is not None
                for request_id in self._request_tenants
            ),
            "resident_bytes": sum(
                resource.handle.byte_count for resource in self._resources.values()
            ),
        }
