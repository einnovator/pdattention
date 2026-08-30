"""Request-safe contracts for future OpenVINO native non-prefix K/V."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Protocol, Sequence

from pra_hf.engine_invariants import EnginePRAIsolationGuard


@dataclass(frozen=True)
class OpenVINOTopology:
    """Model/device geometry that makes compiled K/V representations valid."""

    model: str
    revision: str | None = None
    device: str = "CPU"
    precision: str = "f16"
    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpenVINOKVHandle:
    """Logical PRA identity plus engine-owned per-layer tensor identities."""

    logical_key: str
    tensor_ids: tuple[str, ...]
    token_count: int
    byte_count: int
    topology_fingerprint: str

    def __post_init__(self) -> None:
        if not self.logical_key or not self.topology_fingerprint:
            raise ValueError("OpenVINO K/V handles require stable identity.")
        if not self.tensor_ids or len(self.tensor_ids) != len(set(self.tensor_ids)):
            raise ValueError("OpenVINO tensor identities must be non-empty and unique.")
        if self.token_count <= 0 or self.byte_count <= 0:
            raise ValueError("OpenVINO K/V accounting must be positive.")


class OpenVINOStore(Protocol):
    """Physical-store operations needed by PRA lifecycle management."""

    def put(self, handle: OpenVINOKVHandle) -> None: ...
    def get(self, logical_key: str) -> OpenVINOKVHandle: ...
    def contains(self, logical_key: str) -> bool: ...
    def pin(self, logical_key: str, request_id: str) -> None: ...
    def unpin(self, logical_key: str, request_id: str) -> None: ...
    def remove(self, logical_key: str) -> None: ...


class InMemoryOpenVINOStore:
    """Deterministic store double for topology, lifecycle, and isolation tests."""

    def __init__(self) -> None:
        self._handles: dict[str, OpenVINOKVHandle] = {}
        self._pins: dict[str, set[str]] = {}

    def put(self, handle: OpenVINOKVHandle) -> None:
        current = self._handles.get(handle.logical_key)
        if current is not None and current != handle:
            raise RuntimeError("A logical PRA key cannot change physical identity in place.")
        self._handles[handle.logical_key] = handle

    def get(self, logical_key: str) -> OpenVINOKVHandle:
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
            raise RuntimeError("Pinned OpenVINO PRA tensors cannot be removed.")
        self._handles.pop(logical_key, None)

    def pin_count(self, logical_key: str) -> int:
        return len(self._pins.get(logical_key, ()))


@dataclass(frozen=True)
class _RegisteredResource:
    handle: OpenVINOKVHandle
    tenant_id: str
    authorization_scope: str | None


class OpenVINONativeAttachmentManager:
    """Authorize, pin, expose once, and clean request-selected K/V handles."""

    def __init__(self, store: OpenVINOStore, topology: OpenVINOTopology) -> None:
        self.store = store
        self.topology = topology
        self.guard = EnginePRAIsolationGuard()
        self._resources: dict[str, _RegisteredResource] = {}
        self._active_requests: set[str] = set()
        self._lock = RLock()

    def register(
        self,
        handle: OpenVINOKVHandle,
        *,
        tenant_id: str,
        authorization_scope: str | None = None,
    ) -> None:
        """Publish a physical handle only when compiled topology matches."""

        if handle.topology_fingerprint != self.topology.fingerprint:
            raise ValueError("PRA K/V topology does not match the OpenVINO pipeline.")
        resource = _RegisteredResource(handle, tenant_id, authorization_scope)
        with self._lock:
            current = self._resources.get(handle.logical_key)
            if current is not None and current != resource:
                raise RuntimeError("A logical PRA key cannot be rebound across tenants.")
            self.store.put(handle)
            self._resources[handle.logical_key] = resource

    def open_request(
        self,
        request_id: str,
        logical_keys: Sequence[str],
        *,
        tenant_id: str,
        authorization_scopes: Sequence[str] = (),
    ) -> None:
        """Authorize and pin the entire routed set before scheduler admission."""

        keys = tuple(logical_keys)
        allowed = set(authorization_scopes)
        with self._lock:
            for key in keys:
                resource = self._resources.get(key)
                if resource is None or not self.store.contains(key):
                    raise KeyError(f"Unknown OpenVINO PRA resource: {key}")
                if resource.tenant_id != tenant_id:
                    raise PermissionError("PRA resource belongs to another tenant.")
                if resource.authorization_scope and resource.authorization_scope not in allowed:
                    raise PermissionError("PRA resource authorization scope was not granted.")
            self.guard.open_request(request_id, keys)
            try:
                for key in keys:
                    self.store.pin(key, request_id)
            except Exception:
                for key in keys:
                    self.store.unpin(key, request_id)
                self.guard.close_request(request_id, require_attached=False)
                raise
            self._active_requests.add(request_id)

    def attach_once(self, request_id: str) -> tuple[OpenVINOKVHandle, ...]:
        """Return exactly the authorized handles for attention metadata assembly."""

        with self._lock:
            keys = self.guard.visible_keys(request_id)
            self.guard.attach_once(request_id, keys)
            return tuple(self.store.get(key) for key in keys)

    def close_request(self, request_id: str, *, require_attached: bool = True) -> None:
        """Unpin selected tensors and erase visibility even after failed decode."""

        with self._lock:
            keys = self.guard.visible_keys(request_id)
            for key in keys:
                self.store.unpin(key, request_id)
            self.guard.close_request(request_id, require_attached=require_attached)
            self._active_requests.discard(request_id)

    def assert_prefix_pool_safe(self, logical_keys: Sequence[str]) -> None:
        """Prevent PRA detail from entering ordinary OpenVINO prefix reuse."""

        self.guard.assert_ordinary_pool_safe(logical_keys)

    def metrics(self) -> Mapping[str, int]:
        return {
            "registered_resources": len(self._resources),
            "active_requests": len(self._active_requests),
            "resident_bytes": sum(
                resource.handle.byte_count for resource in self._resources.values()
            ),
        }
