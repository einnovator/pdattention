"""Engine-neutral identity and residency control for PRA detail blocks.

The objects in this module describe logical PRA memory.  They deliberately do
not import vLLM, SGLang, MLX, or tensor libraries: an engine bridge maps the
stable identity to its own physical cache handles and performs actual K/V
movement.  Keeping that boundary explicit prevents a metadata-only adapter
from being reported as native PRA execution.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping


class PRAResidencyState(str, Enum):
    """Physical availability of one logical detail block."""

    INDEXED_ONLY = "INDEXED_ONLY"
    OFF_DEVICE = "OFF_DEVICE"
    PREFETCHING = "PREFETCHING"
    RESIDENT = "RESIDENT"
    PINNED = "PINNED"
    EVICTABLE = "EVICTABLE"
    INVALID = "INVALID"


_TRANSITIONS = {
    PRAResidencyState.INDEXED_ONLY: {
        PRAResidencyState.OFF_DEVICE,
        PRAResidencyState.PREFETCHING,
        PRAResidencyState.RESIDENT,
        PRAResidencyState.INVALID,
    },
    PRAResidencyState.OFF_DEVICE: {
        PRAResidencyState.PREFETCHING,
        PRAResidencyState.RESIDENT,
        PRAResidencyState.INVALID,
    },
    PRAResidencyState.PREFETCHING: {
        PRAResidencyState.OFF_DEVICE,
        PRAResidencyState.RESIDENT,
        PRAResidencyState.INVALID,
    },
    PRAResidencyState.RESIDENT: {
        PRAResidencyState.PINNED,
        PRAResidencyState.EVICTABLE,
        PRAResidencyState.OFF_DEVICE,
        PRAResidencyState.INVALID,
    },
    PRAResidencyState.PINNED: {
        PRAResidencyState.RESIDENT,
        PRAResidencyState.INVALID,
    },
    PRAResidencyState.EVICTABLE: {
        PRAResidencyState.RESIDENT,
        PRAResidencyState.OFF_DEVICE,
        PRAResidencyState.INVALID,
    },
    PRAResidencyState.INVALID: set(),
}


@dataclass(frozen=True)
class LogicalPRABlockId:
    """Stable semantic identity, independent of an engine's block numbers.

    ``token_start`` and ``token_end`` identify the half-open source interval.
    ``layer`` identifies the consumer/detail layer whose K/V layout is named by
    ``dtype`` and ``layout``.  The source positional frame is represented by
    ``position_policy`` and must never be inferred from a physical cache slot.
    """

    tenant_id: str
    session_id: str | None
    resource_id: str
    resource_version: str
    record_type: str
    token_start: int
    token_end: int
    layer: int
    model_revision: str
    dtype: str
    layout: str
    materialization_profile: str
    position_policy: str
    security_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.resource_id or not self.resource_version:
            raise ValueError("Logical PRA identity requires tenant, resource, and version.")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("Logical PRA token intervals must be non-empty and half-open.")
        if self.layer < 0 or not self.model_revision:
            raise ValueError("Logical PRA identity requires a layer and model revision.")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def digest(self) -> str:
        """Return a deterministic engine-safe identity without exposing content."""

        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LogicalPRABlock:
    """Current control-plane record for one logical native-detail block."""

    identity: LogicalPRABlockId
    address_bytes: int
    detail_bytes: int
    state: PRAResidencyState = PRAResidencyState.INDEXED_ONLY
    physical_handles: tuple[str, ...] = ()
    storage_tier: str | None = None
    selections: int = 0
    reuses: int = 0
    last_access_ns: int = field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        if self.address_bytes < 0 or self.detail_bytes < 0:
            raise ValueError("PRA block byte counts cannot be negative.")
        if self.state in {PRAResidencyState.RESIDENT, PRAResidencyState.PINNED}:
            if not self.physical_handles:
                raise ValueError("Resident PRA blocks require physical engine handles.")


@dataclass(frozen=True)
class PRAResidencySnapshot:
    """Disjoint logical, resident, and active-memory counters."""

    logical_blocks: int
    valid_blocks: int
    address_bytes: int
    detail_bytes: int
    resident_detail_bytes: int
    indexed_only_detail_bytes: int
    selections: int
    reuses: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class LogicalPRABlockStore:
    """Thread-safe logical namespace used by engine-specific physical bridges."""

    def __init__(self) -> None:
        self._blocks: dict[str, LogicalPRABlock] = {}
        self._lock = threading.RLock()

    def register(self, block: LogicalPRABlock) -> str:
        """Register an immutable identity or return its existing digest."""

        key = block.identity.digest()
        with self._lock:
            previous = self._blocks.get(key)
            if previous is not None and previous.identity != block.identity:
                raise ValueError("Logical PRA digest collision.")
            if previous is None:
                self._blocks[key] = block
        return key

    def get(self, key: str) -> LogicalPRABlock:
        with self._lock:
            return self._blocks[key]

    def record_detail_bytes(self, key: str, detail_bytes: int) -> LogicalPRABlock:
        """Record measured physical detail size without changing logical identity."""

        if detail_bytes < 0:
            raise ValueError("PRA block byte counts cannot be negative.")
        with self._lock:
            current = self._blocks[key]
            if current.state in {PRAResidencyState.PINNED, PRAResidencyState.INVALID}:
                raise RuntimeError("Cannot resize pinned or invalid PRA detail.")
            if current.detail_bytes not in {0, detail_bytes}:
                raise ValueError("Immutable PRA detail changed physical size.")
            updated = replace(current, detail_bytes=detail_bytes)
            self._blocks[key] = updated
            return updated

    def transition(
        self,
        key: str,
        state: PRAResidencyState | str,
        *,
        physical_handles: Iterable[str] | None = None,
        storage_tier: str | None = None,
    ) -> LogicalPRABlock:
        """Apply a validated lifecycle transition after physical work succeeds."""

        target = PRAResidencyState(state)
        with self._lock:
            current = self._blocks[key]
            if target == current.state:
                return current
            if target not in _TRANSITIONS[current.state]:
                raise ValueError(f"Invalid PRA residency transition {current.state} -> {target}.")
            handles = (
                current.physical_handles
                if physical_handles is None
                else tuple(str(value) for value in physical_handles)
            )
            if target in {PRAResidencyState.RESIDENT, PRAResidencyState.PINNED} and not handles:
                raise ValueError("A resident PRA block requires physical engine handles.")
            if target in {
                PRAResidencyState.INDEXED_ONLY,
                PRAResidencyState.OFF_DEVICE,
                PRAResidencyState.INVALID,
            }:
                handles = ()
            updated = replace(
                current,
                state=target,
                physical_handles=handles,
                storage_tier=storage_tier if storage_tier is not None else current.storage_tier,
                last_access_ns=time.monotonic_ns(),
            )
            self._blocks[key] = updated
            return updated

    def select(
        self,
        keys: Iterable[str],
        *,
        tenant_id: str,
        authorization_scopes: Iterable[str] = (),
    ) -> tuple[LogicalPRABlock, ...]:
        """Authorize and account for a request's selected logical block set."""

        scopes = set(map(str, authorization_scopes))
        selected: list[LogicalPRABlock] = []
        with self._lock:
            for key in keys:
                current = self._blocks[key]
                identity = current.identity
                if identity.tenant_id != tenant_id:
                    raise PermissionError("Cross-tenant PRA block selection is forbidden.")
                if identity.security_scope and identity.security_scope not in scopes:
                    raise PermissionError("The request is not authorized for this PRA block.")
                if current.state == PRAResidencyState.INVALID:
                    raise RuntimeError("Invalidated PRA blocks cannot be selected.")
                reused = current.selections > 0
                updated = replace(
                    current,
                    selections=current.selections + 1,
                    reuses=current.reuses + int(reused),
                    last_access_ns=time.monotonic_ns(),
                )
                self._blocks[key] = updated
                selected.append(updated)
        return tuple(selected)

    def invalidate_resource(
        self, tenant_id: str, resource_id: str, *, resource_version: str | None = None
    ) -> int:
        """Invalidate all matching physical realizations after update or removal."""

        count = 0
        with self._lock:
            for key, block in tuple(self._blocks.items()):
                identity = block.identity
                if identity.tenant_id != tenant_id or identity.resource_id != resource_id:
                    continue
                if resource_version is not None and identity.resource_version != resource_version:
                    continue
                if block.state != PRAResidencyState.INVALID:
                    self._blocks[key] = replace(
                        block,
                        state=PRAResidencyState.INVALID,
                        physical_handles=(),
                        last_access_ns=time.monotonic_ns(),
                    )
                    count += 1
        return count

    def snapshot(self) -> PRAResidencySnapshot:
        with self._lock:
            values = tuple(self._blocks.values())
        valid = tuple(row for row in values if row.state != PRAResidencyState.INVALID)
        resident_states = {
            PRAResidencyState.RESIDENT,
            PRAResidencyState.PINNED,
            PRAResidencyState.EVICTABLE,
        }
        return PRAResidencySnapshot(
            logical_blocks=len(values),
            valid_blocks=len(valid),
            address_bytes=sum(row.address_bytes for row in valid),
            detail_bytes=sum(row.detail_bytes for row in valid),
            resident_detail_bytes=sum(
                row.detail_bytes for row in valid if row.state in resident_states
            ),
            indexed_only_detail_bytes=sum(
                row.detail_bytes
                for row in valid
                if row.state == PRAResidencyState.INDEXED_ONLY
            ),
            selections=sum(row.selections for row in valid),
            reuses=sum(row.reuses for row in valid),
        )
