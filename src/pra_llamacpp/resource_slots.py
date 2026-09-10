"""Tenant-scoped control-plane allocation for patched llama-server slots."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable


class LlamaCppSlotCapacityError(RuntimeError):
    """Raised when every request or evictable resource slot is occupied."""


class LlamaCppSlotIsolationError(RuntimeError):
    """Raised when a request attempts to cross an identity boundary."""


@dataclass(frozen=True)
class LlamaCppResourceIdentity:
    """Authority and compatibility boundary for one encoded resource sequence."""

    tenant_id: str
    session_id: str
    model_fingerprint: str
    resource_digest: str


@dataclass(frozen=True)
class LlamaCppSlotLease:
    """Resource/request slot pair held for one in-flight request."""

    request_id: str
    identity: LlamaCppResourceIdentity
    resource_slot: int
    request_slot: int


@dataclass
class _ResourceEntry:
    identity: LlamaCppResourceIdentity
    slot: int
    active_requests: int
    last_used: int


class LlamaCppResourceSlotAllocator:
    """Allocate disjoint server slots without trusting client-provided slot IDs.

    This allocator owns the legacy detached-resource slot. Stateful-agent PRA
    no longer funnels records through it: the live-prefix protocol attaches an
    arbitrary validated list of record ranges from one session source slot.
    The allocator still scopes detached reuse by tenant/session/model/content
    and only evicts entries that have no in-flight request lease.
    """

    max_detached_resources_per_request = 1
    max_live_prefix_records_per_request = None

    def __init__(
        self,
        *,
        resource_slots: Iterable[int],
        request_slots: Iterable[int],
    ) -> None:
        resources = tuple(dict.fromkeys(int(slot) for slot in resource_slots))
        requests = tuple(dict.fromkeys(int(slot) for slot in request_slots))
        if not resources or not requests:
            raise ValueError("At least one resource slot and request slot are required.")
        if set(resources).intersection(requests):
            raise ValueError("Resource and request slot pools must be disjoint.")
        self.resource_slots = resources
        self.request_slots = requests
        self._resources: dict[LlamaCppResourceIdentity, _ResourceEntry] = {}
        self._requests: dict[str, LlamaCppSlotLease] = {}
        self._clock = 0
        self._lock = threading.RLock()

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _resource_entry(self, identity: LlamaCppResourceIdentity) -> _ResourceEntry:
        current = self._resources.get(identity)
        if current is not None:
            current.last_used = self._tick()
            return current

        occupied = {entry.slot for entry in self._resources.values()}
        free = next((slot for slot in self.resource_slots if slot not in occupied), None)
        if free is None:
            candidates = [
                entry for entry in self._resources.values() if entry.active_requests == 0
            ]
            if not candidates:
                raise LlamaCppSlotCapacityError("All resource slots have active leases.")
            victim = min(candidates, key=lambda entry: entry.last_used)
            free = victim.slot
            del self._resources[victim.identity]

        current = _ResourceEntry(identity, free, 0, self._tick())
        self._resources[identity] = current
        return current

    def acquire(
        self,
        *,
        request_id: str,
        tenant_id: str,
        identity: LlamaCppResourceIdentity,
    ) -> LlamaCppSlotLease:
        """Acquire one resource/request pair and enforce tenant ownership."""

        with self._lock:
            if tenant_id != identity.tenant_id:
                raise LlamaCppSlotIsolationError(
                    "Request tenant does not own the selected resource identity."
                )
            if request_id in self._requests:
                raise LlamaCppSlotIsolationError(
                    f"Request {request_id!r} already owns a slot lease."
                )
            occupied = {lease.request_slot for lease in self._requests.values()}
            request_slot = next(
                (slot for slot in self.request_slots if slot not in occupied), None
            )
            if request_slot is None:
                raise LlamaCppSlotCapacityError("All request slots have active leases.")
            resource = self._resource_entry(identity)
            resource.active_requests += 1
            lease = LlamaCppSlotLease(
                request_id=str(request_id),
                identity=identity,
                resource_slot=resource.slot,
                request_slot=request_slot,
            )
            self._requests[request_id] = lease
            return lease

    def release(self, request_id: str) -> None:
        """Release an in-flight request lease; repeated release is harmless."""

        with self._lock:
            lease = self._requests.pop(request_id, None)
            if lease is None:
                return
            resource = self._resources.get(lease.identity)
            if resource is not None:
                resource.active_requests -= 1
                resource.last_used = self._tick()

    def invalidate_session(self, session_id: str) -> tuple[int, ...]:
        """Forget inactive resources for a closed session and return their slots."""

        with self._lock:
            matches = [
                entry
                for entry in self._resources.values()
                if entry.identity.session_id == session_id
            ]
            if any(entry.active_requests for entry in matches):
                raise LlamaCppSlotIsolationError(
                    "Cannot invalidate a session with active request leases."
                )
            for entry in matches:
                del self._resources[entry.identity]
            return tuple(sorted(entry.slot for entry in matches))

    def snapshot(self) -> dict[str, object]:
        """Return non-secret allocator state for traces and lifecycle audits."""

        with self._lock:
            return {
                "resource_capacity": len(self.resource_slots),
                "request_capacity": len(self.request_slots),
                "resident_resources": len(self._resources),
                "active_requests": len(self._requests),
                "max_detached_resources_per_request": (
                    self.max_detached_resources_per_request
                ),
                "max_live_prefix_records_per_request": (
                    self.max_live_prefix_records_per_request
                ),
                "tenant_scoped": True,
                "session_scoped": True,
                "model_scoped": True,
            }
