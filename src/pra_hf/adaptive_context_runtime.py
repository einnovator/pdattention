"""Runtime policy, materialization events, and stateful cursors for Paper 7."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .context_records import RecordType, RecordViewName
from .context_store import LocalBackingStore, RecordAccessDenied, RecordScope
from .typed_context import (
    AdaptiveContextRecord,
    CompressorRegistry,
    create_adaptive_record,
)


class StoragePolicy(str, Enum):
    """Where full backing bytes reside relative to the model runtime."""

    UPFRONT = "upfront"
    ON_DEMAND = "on_demand"
    ADAPTIVE = "adaptive"


class RetrievalMode(str, Enum):
    """How omitted state becomes model-visible."""

    NATIVE_EVENT = "native_event"
    TOOL = "tool"
    MIXED = "mixed"
    PROACTIVE = "proactive"


class CursorOperation(str, Enum):
    """Model-selectable operations over one authorized persistent cursor."""

    NEXT = "next"
    PREVIOUS = "previous"
    RANGE = "range"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    SEARCH = "search"
    SAMPLE = "sample"
    MATERIALIZE_FIELDS = "materialize_fields"
    CLOSE = "close"


class DeploymentTopology(str, Enum):
    """Location relationship between agent, model, and backing state."""

    SAME_PROCESS = "same_process"
    LOCAL_PROCESS = "local_process"
    REMOTE_MODEL = "remote_model"
    DISTRIBUTED_STORE = "distributed_store"


@dataclass(frozen=True)
class CursorPolicy:
    """Bounded cursor lifetime and page-size defaults."""

    page_size: int = 20
    ttl_seconds: float = 900.0
    max_page_size: int = 200

    def __post_init__(self) -> None:
        if self.page_size <= 0 or self.max_page_size <= 0:
            raise ValueError("Cursor page sizes must be positive.")
        if self.page_size > self.max_page_size:
            raise ValueError("page_size cannot exceed max_page_size.")
        if self.ttl_seconds <= 0:
            raise ValueError("Cursor TTL must be positive.")


@dataclass(frozen=True)
class TypeContextPolicy:
    """Per-result overrides for compaction, transport, and native indexing.

    Native-index limits inherit from :class:`ContextPolicy` unless either
    threshold is set. Set ``override_native_index_limits`` to disable or replace
    both inherited bounds explicitly, including with two ``None`` values.
    """

    unit_limit: int = 8
    compact_target_tokens: int | None = None
    compact_max_tokens: int | None = None
    compact_ratio_target: float | None = None
    storage: StoragePolicy | str | None = None
    max_native_index_tokens: int | None = None
    max_native_index_bytes: int | None = None
    override_native_index_limits: bool = False
    defer_native_index: bool | None = None

    def __post_init__(self) -> None:
        if self.unit_limit <= 0:
            raise ValueError("unit_limit must be positive.")
        if self.storage is not None:
            object.__setattr__(self, "storage", StoragePolicy(self.storage))
        for name in ("max_native_index_tokens", "max_native_index_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None.")
        for name in ("compact_target_tokens", "compact_max_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None.")
        if self.compact_ratio_target is not None and not 0 < self.compact_ratio_target <= 1:
            raise ValueError("compact_ratio_target must be in (0, 1].")


@dataclass(frozen=True)
class ContextPolicy:
    """Public policy joining storage, retrieval, compaction, and cursors.

    The finite native-index defaults are a conservative reference profile, not
    universal optima. Deployments should tune them from measured ingestion,
    resident-state, and recovery costs. Setting either limit to ``None``
    disables that dimension of the gate.
    """

    storage: StoragePolicy | str = StoragePolicy.ADAPTIVE
    local_store: str | Path | None = None
    retrieval_mode: RetrievalMode | str = RetrievalMode.MIXED
    topology: DeploymentTopology | str = DeploymentTopology.SAME_PROCESS
    record_policies: Mapping[RecordType | str, TypeContextPolicy] = field(default_factory=dict)
    cursor_policy: CursorPolicy = field(default_factory=CursorPolicy)
    upfront_max_bytes: int = 100_000
    adaptive_reuse_max_bytes: int = 1_000_000
    native_kv_bytes_per_token: int = 0
    max_native_index_tokens: int | None = 4_096
    max_native_index_bytes: int | None = 65_536
    defer_native_index: bool = False
    store_max_bytes: int | None = None
    persistent_store: bool = False
    allow_proactive_expansion: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage", StoragePolicy(self.storage))
        object.__setattr__(self, "retrieval_mode", RetrievalMode(self.retrieval_mode))
        object.__setattr__(self, "topology", DeploymentTopology(self.topology))
        policies = {
            RecordType(key): value if isinstance(value, TypeContextPolicy) else TypeContextPolicy(**value)
            for key, value in self.record_policies.items()
        }
        object.__setattr__(self, "record_policies", policies)
        if self.upfront_max_bytes <= 0 or self.adaptive_reuse_max_bytes <= 0:
            raise ValueError("Adaptive transport thresholds must be positive.")
        if self.upfront_max_bytes > self.adaptive_reuse_max_bytes:
            raise ValueError("upfront_max_bytes cannot exceed adaptive_reuse_max_bytes.")
        if self.native_kv_bytes_per_token < 0:
            raise ValueError("native_kv_bytes_per_token must be non-negative.")
        for name in ("max_native_index_tokens", "max_native_index_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None.")

    def native_index_policy(
        self, record_type: RecordType | str
    ) -> tuple[int | None, int | None, bool]:
        """Resolve global and type-specific native-index ingestion policy."""

        type_policy = self.record_policies.get(RecordType(record_type))
        if type_policy is None:
            return (
                self.max_native_index_tokens,
                self.max_native_index_bytes,
                self.defer_native_index,
            )
        overrides_limits = type_policy.override_native_index_limits or any(
            value is not None
            for value in (
                type_policy.max_native_index_tokens,
                type_policy.max_native_index_bytes,
            )
        )
        token_limit = (
            type_policy.max_native_index_tokens
            if overrides_limits
            else self.max_native_index_tokens
        )
        byte_limit = (
            type_policy.max_native_index_bytes
            if overrides_limits
            else self.max_native_index_bytes
        )
        deferred = (
            type_policy.defer_native_index
            if type_policy.defer_native_index is not None
            else self.defer_native_index
        )
        return token_limit, byte_limit, deferred

    def to_dict(self) -> dict[str, object]:
        """Serialize policy values without leaking enum or ``Path`` objects."""

        return {
            "storage": self.storage.value,
            "local_store": str(self.local_store) if self.local_store is not None else None,
            "retrieval_mode": self.retrieval_mode.value,
            "topology": self.topology.value,
            "record_policies": {
                record_type.value: _serialized_type_policy(policy)
                for record_type, policy in self.record_policies.items()
            },
            "cursor_policy": {
                "page_size": self.cursor_policy.page_size,
                "ttl_seconds": self.cursor_policy.ttl_seconds,
                "max_page_size": self.cursor_policy.max_page_size,
            },
            "upfront_max_bytes": self.upfront_max_bytes,
            "adaptive_reuse_max_bytes": self.adaptive_reuse_max_bytes,
            "native_kv_bytes_per_token": self.native_kv_bytes_per_token,
            "max_native_index_tokens": self.max_native_index_tokens,
            "max_native_index_bytes": self.max_native_index_bytes,
            "defer_native_index": self.defer_native_index,
            "store_max_bytes": self.store_max_bytes,
            "persistent_store": self.persistent_store,
            "allow_proactive_expansion": self.allow_proactive_expansion,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "ContextPolicy":
        """Restore a policy emitted by :meth:`to_dict`."""

        data = dict(values)
        data["cursor_policy"] = CursorPolicy(**dict(data.get("cursor_policy", {})))
        data["record_policies"] = {
            RecordType(record_type): TypeContextPolicy(**dict(policy))
            for record_type, policy in dict(data.get("record_policies", {})).items()
        }
        return cls(**data)


def _serialized_type_policy(policy: TypeContextPolicy) -> dict[str, object]:
    """Serialize optional compact budgets only when explicitly configured."""

    values: dict[str, object] = {
                    "unit_limit": policy.unit_limit,
                    "storage": policy.storage.value if policy.storage is not None else None,
                    "max_native_index_tokens": policy.max_native_index_tokens,
                    "max_native_index_bytes": policy.max_native_index_bytes,
                    "override_native_index_limits": policy.override_native_index_limits,
                    "defer_native_index": policy.defer_native_index,
    }
    for name in ("compact_target_tokens", "compact_max_tokens", "compact_ratio_target"):
        value = getattr(policy, name)
        if value is not None:
            values[name] = value
    return values


@dataclass(frozen=True)
class TransportDecision:
    """Auditable placement decision for one newly ingested result."""

    record_id: str
    storage: StoragePolicy
    topology: DeploymentTopology
    bytes_transferred: int
    expected_round_trips: int
    reason: str


@dataclass(frozen=True)
class MaterializationEvent:
    """Typed exact-identity request emitted by a model or local runtime."""

    record_id: str
    level: RecordViewName | str = RecordViewName.FULL
    selector: Mapping[str, object] | None = None
    cursor_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", RecordViewName(self.level))
        if not self.record_id:
            raise ValueError("MATERIALIZE requires record_id.")


@dataclass(frozen=True)
class MaterializationResult:
    """Payload and measured cost of one authorized expansion."""

    record_id: str
    level: RecordViewName
    payload: object
    payload_bytes: int
    network_bytes: int
    round_trips: int
    active_kv_bytes: int
    latency_seconds: float
    cache_hit: bool


@dataclass(frozen=True)
class RecordSearchResult:
    """Authorized bounded search result over one exact backing identity."""

    record_id: str
    query: str
    payload: object
    payload_bytes: int
    latency_seconds: float


@dataclass(frozen=True)
class CursorRecord:
    """Scoped continuation state for bounded access to one backing collection."""

    cursor_id: str
    record_id: str
    scope: RecordScope
    source: str
    query: str | None
    schema: tuple[str, ...]
    collection: str
    position: int
    page_size: int
    filters: Mapping[str, object]
    order: str | None
    total_estimate: int
    created_at: float
    expires_at: float
    provenance: Mapping[str, object]
    authorization: str
    continuation_handle: str
    closed: bool = False


@dataclass(frozen=True)
class CursorPage:
    """One bounded page plus updated cursor metadata."""

    cursor_id: str
    items: tuple[object, ...]
    start: int
    stop: int
    total_estimate: int
    has_more: bool


@dataclass(frozen=True)
class CursorAction:
    """One structured cursor decision emitted by a model or host policy."""

    cursor_id: str
    operation: CursorOperation | str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cursor_id:
            raise ValueError("cursor_id is required.")
        object.__setattr__(self, "operation", CursorOperation(self.operation))
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class CursorActionResult:
    """Result and accounting for one attempted model-selected cursor action."""

    action: CursorAction
    success: bool
    payload: object | None
    payload_bytes: int
    latency_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class RuntimeAccounting:
    """Cumulative transport and materialization counters."""

    records: int
    expansions: int
    cursor_fetches: int
    network_bytes: int
    materialized_bytes: int
    active_kv_bytes: int
    round_trips: int
    cache_hits: int


class CursorManager:
    """In-process cursor control plane over exact authorized backing records."""

    def __init__(self, store: LocalBackingStore, policy: CursorPolicy) -> None:
        self.store = store
        self.policy = policy
        self._cursors: dict[str, CursorRecord] = {}

    @staticmethod
    def _collection(payload: object, requested: str | None) -> tuple[str, list[object]]:
        if requested is not None:
            if not isinstance(payload, Mapping) or requested not in payload:
                raise ValueError(f"Cursor collection {requested!r} is not present.")
            values = payload[requested]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError("Cursor collection must be a sequence.")
            return requested, list(values)
        if isinstance(payload, Mapping):
            for name in ("rows", "nodes", "edges", "chunks", "results", "items", "events"):
                values = payload.get(name)
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    return name, list(values)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            return "items", list(payload)
        if isinstance(payload, str):
            return "lines", payload.splitlines()
        raise TypeError("Record has no cursor-compatible collection.")

    def open(
        self,
        record: AdaptiveContextRecord,
        *,
        scope: RecordScope,
        collection: str | None = None,
        page_size: int | None = None,
        source: str = "typed_record",
        query: str | None = None,
        filters: Mapping[str, object] | None = None,
        order: str | None = None,
    ) -> CursorRecord:
        payload = self.store.get(record.record_id, scope=scope)
        collection_name, values = self._collection(payload, collection)
        resolved_page_size = page_size or self.policy.page_size
        if resolved_page_size <= 0 or resolved_page_size > self.policy.max_page_size:
            raise ValueError("Cursor page_size is outside the configured bound.")
        now = time.time()
        identity = hashlib.sha256(
            f"{record.record_id}\0{scope.fingerprint}\0{now}\0{collection_name}".encode("utf-8")
        ).hexdigest()
        cursor = CursorRecord(
            cursor_id=f"pra-cursor://{scope.fingerprint}/{identity[:24]}",
            record_id=record.record_id,
            scope=scope,
            source=source,
            query=query,
            schema=self._schema(values),
            collection=collection_name,
            position=0,
            page_size=resolved_page_size,
            filters=dict(filters or {}),
            order=order,
            total_estimate=len(values),
            created_at=now,
            expires_at=now + self.policy.ttl_seconds,
            provenance=dict(record.backing.provenance),
            authorization=scope.fingerprint,
            continuation_handle=identity,
        )
        self._cursors[cursor.cursor_id] = cursor
        return cursor

    @staticmethod
    def _schema(values: Sequence[object]) -> tuple[str, ...]:
        fields = {
            str(key)
            for item in values[:32]
            if isinstance(item, Mapping)
            for key in item
        }
        return tuple(sorted(fields))

    def _resolve(self, cursor_id: str, scope: RecordScope) -> CursorRecord:
        cursor = self._cursors.get(cursor_id)
        if cursor is None or cursor.closed or time.time() >= cursor.expires_at:
            self._cursors.pop(cursor_id, None)
            raise KeyError(f"Cursor is missing, closed, or expired: {cursor_id}")
        if cursor.scope != scope or cursor.authorization != scope.fingerprint:
            raise RecordAccessDenied(cursor_id)
        return cursor

    def describe(self, cursor_id: str, *, scope: RecordScope) -> CursorRecord:
        """Return authorized cursor metadata without reading another page."""

        return self._resolve(cursor_id, scope)

    @property
    def cursor_ids(self) -> tuple[str, ...]:
        """Return live cursor identities for structured-output validation."""

        now = time.time()
        return tuple(sorted(
            cursor_id
            for cursor_id, cursor in self._cursors.items()
            if not cursor.closed and cursor.expires_at > now
        ))

    def _values(self, cursor: CursorRecord) -> list[object]:
        payload = self.store.get(cursor.record_id, scope=cursor.scope)
        _, values = self._collection(payload, cursor.collection)
        if cursor.filters:
            values = [item for item in values if _matches(item, cursor.filters)]
        if cursor.order:
            reverse = cursor.order.startswith("-")
            field_name = cursor.order.lstrip("+-")
            values.sort(key=lambda item: _field(item, field_name), reverse=reverse)
        return values

    def page(
        self,
        cursor_id: str,
        *,
        scope: RecordScope,
        direction: str = "next",
    ) -> CursorPage:
        cursor = self._resolve(cursor_id, scope)
        values = self._values(cursor)
        if direction == "next":
            start = cursor.position
        elif direction == "previous":
            start = max(0, cursor.position - 2 * cursor.page_size)
        else:
            raise ValueError("direction must be next or previous.")
        stop = min(start + cursor.page_size, len(values))
        self._cursors[cursor_id] = _replace_cursor(cursor, position=stop, total_estimate=len(values))
        return CursorPage(cursor_id, tuple(values[start:stop]), start, stop, len(values), stop < len(values))

    def range(
        self, cursor_id: str, start: int, stop: int, *, scope: RecordScope
    ) -> CursorPage:
        cursor = self._resolve(cursor_id, scope)
        values = self._values(cursor)
        if start < 0 or stop < start or stop > len(values):
            raise ValueError("Cursor range is outside the result collection.")
        if stop - start > self.policy.max_page_size:
            raise ValueError("Cursor range exceeds max_page_size.")
        self._cursors[cursor_id] = _replace_cursor(cursor, position=stop, total_estimate=len(values))
        return CursorPage(cursor_id, tuple(values[start:stop]), start, stop, len(values), stop < len(values))

    def search(
        self, cursor_id: str, query: str, *, scope: RecordScope, limit: int | None = None
    ) -> tuple[object, ...]:
        cursor = self._resolve(cursor_id, scope)
        terms = tuple(token.casefold() for token in query.split() if token)
        matches = [
            item for item in self._values(cursor)
            if all(term in json.dumps(item, default=str).casefold() for term in terms)
        ]
        return tuple(matches[: min(limit or cursor.page_size, self.policy.max_page_size)])

    def aggregate(
        self, cursor_id: str, field_name: str, *, scope: RecordScope
    ) -> Mapping[str, float | int]:
        cursor = self._resolve(cursor_id, scope)
        values = [
            float(value)
            for item in self._values(cursor)
            for value in [_field(item, field_name)]
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
        }

    def sample(
        self, cursor_id: str, count: int, *, scope: RecordScope
    ) -> tuple[object, ...]:
        cursor = self._resolve(cursor_id, scope)
        values = self._values(cursor)
        count = min(count, self.policy.max_page_size)
        if count <= 0:
            raise ValueError("sample count must be positive.")
        if len(values) <= count:
            return tuple(values)
        indices = sorted({round(index * (len(values) - 1) / (count - 1)) for index in range(count)}) if count > 1 else [0]
        return tuple(values[index] for index in indices)

    def materialize_fields(
        self, cursor_id: str, fields: Sequence[str], *, scope: RecordScope
    ) -> tuple[Mapping[str, object], ...]:
        cursor = self._resolve(cursor_id, scope)
        return tuple(
            {field_name: item[field_name] for field_name in fields if field_name in item}
            for item in self._values(cursor)[: self.policy.max_page_size]
            if isinstance(item, Mapping)
        )

    def refine(
        self,
        cursor_id: str,
        *,
        scope: RecordScope,
        filters: Mapping[str, object] | None = None,
        order: str | None = None,
    ) -> CursorRecord:
        """Replace bounded filter/order state without changing cursor identity."""

        cursor = self._resolve(cursor_id, scope)
        updated = _replace_cursor(
            cursor,
            position=0,
            filters=dict(filters if filters is not None else cursor.filters),
            order=order if order is not None else cursor.order,
        )
        self._cursors[cursor_id] = updated
        return updated

    def close(self, cursor_id: str, *, scope: RecordScope) -> None:
        cursor = self._resolve(cursor_id, scope)
        self._cursors[cursor_id] = _replace_cursor(cursor, closed=True)


def _replace_cursor(cursor: CursorRecord, **changes: object) -> CursorRecord:
    values = {name: getattr(cursor, name) for name in cursor.__dataclass_fields__}
    values.update(changes)
    return CursorRecord(**values)


def _field(item: object, field_name: str) -> object:
    return item.get(field_name) if isinstance(item, Mapping) else None


def _matches(item: object, filters: Mapping[str, object]) -> bool:
    return isinstance(item, Mapping) and all(item.get(name) == value for name, value in filters.items())


class AdaptiveContextRuntime:
    """User-facing typed-result ingestion, retrieval, and accounting facade."""

    def __init__(
        self,
        scope: RecordScope,
        policy: ContextPolicy | None = None,
        *,
        store: LocalBackingStore | None = None,
        registry: CompressorRegistry | None = None,
    ) -> None:
        self.scope = scope
        self.policy = policy or ContextPolicy()
        self.store = store or LocalBackingStore(
            self.policy.local_store,
            max_bytes=self.policy.store_max_bytes,
            persistent=self.policy.persistent_store,
        )
        self.registry = registry or CompressorRegistry()
        self.cursors = CursorManager(self.store, self.policy.cursor_policy)
        self.records: dict[str, AdaptiveContextRecord] = {}
        self.decisions: dict[str, TransportDecision] = {}
        self.audit_events: list[dict[str, object]] = []
        self._expansions = 0
        self._cursor_fetches = 0
        self._network_bytes = 0
        self._materialized_bytes = 0
        self._active_kv_bytes = 0
        self._round_trips = 0
        self._cache_hits = 0
        self._materialized_levels: set[tuple[str, RecordViewName, str]] = set()

    def ingest(
        self,
        payload: object,
        *,
        record_type: RecordType | str,
        provenance: Mapping[str, object] | None = None,
        ttl_seconds: float | None = None,
        expected_reuse: float = 0.0,
    ) -> AdaptiveContextRecord:
        """Persist a typed result and expose only its bounded compact descriptor."""

        record_type = RecordType(record_type)
        type_policy = self.policy.record_policies.get(record_type, TypeContextPolicy())
        record = create_adaptive_record(
            payload,
            record_type=record_type,
            store=self.store,
            scope=self.scope,
            registry=self.registry,
            unit_limit=type_policy.unit_limit,
            compact_target_tokens=type_policy.compact_target_tokens,
            compact_max_tokens=type_policy.compact_max_tokens,
            compact_ratio_target=type_policy.compact_ratio_target,
            provenance=provenance,
            ttl_seconds=ttl_seconds,
        )
        self.records[record.record_id] = record
        decision = self._transport_decision(record, type_policy.storage, expected_reuse)
        self.decisions[record.record_id] = decision
        self._network_bytes += decision.bytes_transferred
        self._round_trips += decision.expected_round_trips
        self._audit("ingest", record.record_id, storage=decision.storage.value)
        return record

    def _transport_decision(
        self,
        record: AdaptiveContextRecord,
        override: StoragePolicy | None,
        expected_reuse: float,
    ) -> TransportDecision:
        requested = override or self.policy.storage
        storage = requested
        reason = "explicit policy"
        if requested == StoragePolicy.ADAPTIVE:
            if self.policy.topology == DeploymentTopology.SAME_PROCESS:
                storage = StoragePolicy.UPFRONT
                reason = "co-located backing state needs no network transfer"
            elif record.backing.size_bytes <= self.policy.upfront_max_bytes:
                storage = StoragePolicy.UPFRONT
                reason = "payload is below the configured upfront threshold"
            elif expected_reuse >= 0.5 and record.backing.size_bytes <= self.policy.adaptive_reuse_max_bytes:
                storage = StoragePolicy.UPFRONT
                reason = "bounded payload has high expected reuse"
            else:
                storage = StoragePolicy.ON_DEMAND
                reason = "large or low-reuse payload remains agent-side"
        remote = self.policy.topology != DeploymentTopology.SAME_PROCESS
        transferred = record.backing.size_bytes if storage == StoragePolicy.UPFRONT and remote else 0
        return TransportDecision(
            record.record_id,
            storage,
            self.policy.topology,
            transferred,
            1 if transferred else 0,
            reason,
        )

    def materialize(
        self,
        event: MaterializationEvent,
        *,
        scope: RecordScope | None = None,
    ) -> MaterializationResult:
        """Execute a native typed event against exact known record identity."""

        caller_scope = scope or self.scope
        record = self.records.get(event.record_id)
        if record is None:
            raise KeyError(f"Unknown adaptive-context record: {event.record_id}")
        started = time.perf_counter()
        try:
            payload = record.materialize(
                self.store,
                scope=caller_scope,
                level=event.level,
                selector=event.selector,
            )
        except RecordAccessDenied:
            self._audit("materialize_denied", event.record_id, caller=caller_scope.fingerprint)
            raise
        payload_bytes = _payload_bytes(payload)
        selector_key = json.dumps(event.selector, sort_keys=True, default=str)
        cache_key = (event.record_id, event.level, selector_key)
        cache_hit = cache_key in self._materialized_levels
        self._materialized_levels.add(cache_key)
        decision = self.decisions[event.record_id]
        remote = self.policy.topology != DeploymentTopology.SAME_PROCESS
        fetch_needed = decision.storage == StoragePolicy.ON_DEMAND and remote and not cache_hit
        network_bytes = payload_bytes if fetch_needed else 0
        round_trips = 1 if fetch_needed else 0
        token_estimate = math.ceil(payload_bytes / 4)
        active_kv_bytes = token_estimate * self.policy.native_kv_bytes_per_token
        self._expansions += 1
        self._network_bytes += network_bytes
        self._materialized_bytes += payload_bytes
        self._active_kv_bytes += active_kv_bytes
        self._round_trips += round_trips
        self._cache_hits += int(cache_hit)
        self._audit("materialize", event.record_id, level=event.level.value, cache_hit=cache_hit)
        return MaterializationResult(
            event.record_id,
            event.level,
            payload,
            payload_bytes,
            network_bytes,
            round_trips,
            active_kv_bytes,
            time.perf_counter() - started,
            cache_hit,
        )

    def retrieve_record(
        self,
        record_id: str,
        *,
        level: RecordViewName | str = RecordViewName.FULL,
        selector: Mapping[str, object] | None = None,
        scope: RecordScope | None = None,
    ) -> MaterializationResult:
        """Tool-compatible wrapper over the same authorized event path."""

        return self.materialize(
            MaterializationEvent(record_id, level=level, selector=selector), scope=scope
        )

    def proactive_materialize(
        self,
        event: MaterializationEvent,
        *,
        reason: str,
        scope: RecordScope | None = None,
    ) -> MaterializationResult:
        """Expand hidden state only when host policy authorizes proactive reads."""

        if not self.policy.allow_proactive_expansion:
            self._audit("proactive_materialize_denied", event.record_id, reason=reason)
            raise RecordAccessDenied(
                "Proactive materialization is disabled by ContextPolicy."
            )
        result = self.materialize(event, scope=scope)
        self._audit("proactive_materialize", event.record_id, reason=reason)
        return result

    def search_records(
        self,
        query: str,
        *,
        top_k: int = 5,
        address_kinds: Sequence[str] | None = None,
    ) -> tuple[AdaptiveContextRecord, ...]:
        """Search selected retrieval-only views without exposing originals."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        supported = {"lexical", "entity", "rare_term", "schema", "summary", "dense"}
        kinds = tuple(address_kinds or ("lexical", "entity", "rare_term", "schema", "summary"))
        unknown = set(kinds) - supported
        if unknown:
            raise ValueError(f"Unsupported address views: {sorted(unknown)}")
        terms = _query_terms(query)
        scored = []
        for record in self.records.values():
            addresses = record.address_views()
            indexed = _address_terms(addresses, kinds)
            overlap = terms & indexed
            phrase_bonus = int(bool(query.strip()) and query.casefold() in " ".join(sorted(indexed)))
            score = len(overlap) + phrase_bonus
            if score:
                scored.append((score, record.record_id, record))
        scored.sort(key=lambda row: (-row[0], row[1]))
        self._audit(
            "address_search",
            "",
            query=query,
            address_kinds=list(kinds),
            matches=len(scored),
        )
        return tuple(row[2] for row in scored[:top_k])

    def search_record(
        self,
        record_id: str,
        query: str,
        *,
        limit: int = 4,
        scope: RecordScope | None = None,
    ) -> RecordSearchResult:
        """Search one known record after enforcing its tenant/session scope."""

        if record_id not in self.records:
            raise KeyError(record_id)
        if limit <= 0:
            raise ValueError("limit must be positive.")
        started = time.perf_counter()
        caller_scope = scope or self.scope
        try:
            payload = self.store.get(record_id, scope=caller_scope)
        except RecordAccessDenied:
            self._audit("record_search_denied", record_id, caller=caller_scope.fingerprint)
            raise
        selected = _search_payload(payload, query, limit=limit)
        payload_bytes = _payload_bytes(selected)
        self._materialized_bytes += payload_bytes
        remote = self.policy.topology != DeploymentTopology.SAME_PROCESS
        if remote:
            self._network_bytes += payload_bytes
            self._round_trips += 1
        self._audit(
            "record_search",
            record_id,
            query=query,
            limit=limit,
            payload_bytes=payload_bytes,
        )
        return RecordSearchResult(
            record_id,
            query,
            selected,
            payload_bytes,
            time.perf_counter() - started,
        )

    def open_cursor(self, record_id: str, **kwargs: object) -> CursorRecord:
        """Open a scoped stateful cursor over an exact record collection."""

        if record_id not in self.records:
            raise KeyError(record_id)
        cursor = self.cursors.open(self.records[record_id], scope=self.scope, **kwargs)
        self._audit("cursor_open", record_id, cursor_id=cursor.cursor_id)
        return cursor

    def fetch_cursor(
        self,
        cursor_id: str,
        *,
        direction: str = "next",
        scope: RecordScope | None = None,
    ) -> CursorPage:
        """Fetch one bounded cursor page and account for the local operation."""

        caller_scope = scope or self.scope
        page = self.cursors.page(cursor_id, scope=caller_scope, direction=direction)
        self._cursor_fetches += 1
        self._account_cursor_payload(page)
        self._audit("cursor_fetch", "", cursor_id=cursor_id, start=page.start, stop=page.stop)
        return page

    def execute_cursor_action(
        self,
        action: CursorAction,
        *,
        scope: RecordScope | None = None,
    ) -> CursorActionResult:
        """Validate, execute, and account one structured model cursor decision."""

        caller_scope = scope or self.scope
        started = time.perf_counter()
        args = action.arguments
        try:
            if action.operation in {CursorOperation.NEXT, CursorOperation.PREVIOUS}:
                payload = self.cursors.page(
                    action.cursor_id,
                    scope=caller_scope,
                    direction=action.operation.value,
                )
            elif action.operation == CursorOperation.RANGE:
                payload = self.cursors.range(
                    action.cursor_id,
                    int(args["start"]),
                    int(args["stop"]),
                    scope=caller_scope,
                )
            elif action.operation == CursorOperation.FILTER:
                filters = args.get("filters")
                if not isinstance(filters, Mapping):
                    raise ValueError("filter requires a filters mapping.")
                payload = self.cursors.refine(
                    action.cursor_id,
                    scope=caller_scope,
                    filters=filters,
                    order=str(args["order"]) if args.get("order") is not None else None,
                )
            elif action.operation == CursorOperation.AGGREGATE:
                payload = self.cursors.aggregate(
                    action.cursor_id, str(args["field"]), scope=caller_scope
                )
            elif action.operation == CursorOperation.SEARCH:
                payload = self.cursors.search(
                    action.cursor_id,
                    str(args["query"]),
                    scope=caller_scope,
                    limit=int(args["limit"]) if args.get("limit") is not None else None,
                )
            elif action.operation == CursorOperation.SAMPLE:
                payload = self.cursors.sample(
                    action.cursor_id, int(args["count"]), scope=caller_scope
                )
            elif action.operation == CursorOperation.MATERIALIZE_FIELDS:
                fields = args.get("fields")
                if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
                    raise ValueError("materialize_fields requires a fields sequence.")
                payload = self.cursors.materialize_fields(
                    action.cursor_id,
                    tuple(str(value) for value in fields),
                    scope=caller_scope,
                )
            elif action.operation == CursorOperation.CLOSE:
                self.cursors.close(action.cursor_id, scope=caller_scope)
                payload = {"closed": True}
            else:  # pragma: no cover - Enum construction prevents this branch.
                raise ValueError(f"Unsupported cursor operation: {action.operation}")
        except (KeyError, TypeError, ValueError, RecordAccessDenied) as exc:
            self._audit(
                "cursor_action_error",
                "",
                cursor_id=action.cursor_id,
                operation=action.operation.value,
                error=str(exc),
            )
            return CursorActionResult(
                action,
                False,
                None,
                0,
                time.perf_counter() - started,
                f"{type(exc).__name__}: {exc}",
            )
        self._cursor_fetches += int(action.operation != CursorOperation.CLOSE)
        payload_bytes = self._account_cursor_payload(payload)
        self._audit(
            "cursor_action",
            "",
            cursor_id=action.cursor_id,
            operation=action.operation.value,
            payload_bytes=payload_bytes,
        )
        return CursorActionResult(
            action,
            True,
            payload,
            payload_bytes,
            time.perf_counter() - started,
        )

    def _account_cursor_payload(self, payload: object) -> int:
        payload_bytes = _payload_bytes(payload)
        self._materialized_bytes += payload_bytes
        if self.policy.topology != DeploymentTopology.SAME_PROCESS:
            self._network_bytes += payload_bytes
            self._round_trips += 1
        return payload_bytes

    def account_selected_payload(self, payload: object, *, action: str) -> tuple[int, int, int]:
        """Account a bounded agent-side search result returned to the model."""

        payload_bytes = _payload_bytes(payload)
        remote = self.policy.topology != DeploymentTopology.SAME_PROCESS
        network_bytes = payload_bytes if remote else 0
        round_trips = int(remote)
        self._expansions += 1
        self._materialized_bytes += payload_bytes
        self._network_bytes += network_bytes
        self._round_trips += round_trips
        self._audit(action, "", payload_bytes=payload_bytes)
        return payload_bytes, network_bytes, round_trips

    def accounting(self) -> RuntimeAccounting:
        return RuntimeAccounting(
            records=len(self.records),
            expansions=self._expansions,
            cursor_fetches=self._cursor_fetches,
            network_bytes=self._network_bytes,
            materialized_bytes=self._materialized_bytes,
            active_kv_bytes=self._active_kv_bytes,
            round_trips=self._round_trips,
            cache_hits=self._cache_hits,
        )

    def _audit(self, action: str, record_id: str, **details: object) -> None:
        self.audit_events.append({
            "timestamp": time.time(),
            "action": action,
            "record_id": record_id,
            "tenant_id": self.scope.tenant_id,
            "session_id": self.scope.session_id,
            **details,
        })


def _payload_bytes(payload: object) -> int:
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


_SEARCH_TOKEN = re.compile(r"[A-Za-z0-9_./:@+-]+")


def _query_terms(value: object) -> set[str]:
    return {token.casefold() for token in _SEARCH_TOKEN.findall(str(value))}


def _address_terms(addresses: Mapping[str, object], kinds: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for kind in kinds:
        value = addresses.get(kind, ())
        values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
        for item in values:
            terms.update(_query_terms(item))
    return terms


def _search_payload(payload: object, query: str, *, limit: int) -> object:
    """Return bounded matching units and stable source positions."""

    collection = "items"
    values: list[object]
    if isinstance(payload, Mapping):
        for name in ("rows", "nodes", "edges", "events", "results", "chunks", "items"):
            candidate = payload.get(name)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                collection, values = name, list(candidate)
                break
        else:
            collection, values = "fields", [
                {"field": key, "value": value} for key, value in payload.items()
            ]
    elif isinstance(payload, str):
        collection, values = "lines", payload.splitlines()
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
    else:
        values = [payload]
    terms = _query_terms(query)
    ranked = []
    for index, value in enumerate(values):
        overlap = len(terms & _query_terms(value))
        if overlap:
            ranked.append((overlap, index, value))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    chosen = ranked[:limit]
    return {
        "collection": collection,
        "query": query,
        "matches": [row[2] for row in chosen],
        "match_indices": [row[1] for row in chosen],
    }
