"""Session-aware realization shared by gateways and embedded runtimes.

The ledger records what the model can still see.  It deliberately does not
claim that an engine retained prefix K/V or that PRA native memory is resident;
those are separate observations consumed by policy and diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_count(text: str | None) -> int:
    return len((text or "").split())


class VisibilityState(str, Enum):
    """Whether one materialized occurrence remains in active model context."""

    VISIBLE = "VISIBLE"
    DROPPED_FROM_ACTIVE_CONTEXT = "DROPPED_FROM_ACTIVE_CONTEXT"
    SUPERSEDED = "SUPERSEDED"
    REMOVED = "REMOVED"
    UNKNOWN = "UNKNOWN"


class RealizationDecision(str, Enum):
    """Action required for one selected resource on the current turn."""

    ALREADY_VISIBLE = "ALREADY_VISIBLE"
    NATIVE_AVAILABLE = "NATIVE_AVAILABLE"
    MUST_MATERIALIZE = "MUST_MATERIALIZE"
    OPTIONAL_REFRESH = "OPTIONAL_REFRESH"
    INVALID = "INVALID"


class PrefixReuseStatus(str, Enum):
    """Confidence in an engine's physical sequential-prefix reuse."""

    CONFIRMED = "CONFIRMED"
    LIKELY = "LIKELY"
    UNKNOWN = "UNKNOWN"
    ABSENT = "ABSENT"


@dataclass(frozen=True)
class MaterializationIdentity:
    """Compatibility key for one rendered selected interval."""

    resource_id: str
    resource_version: str
    selected_interval: tuple[int, int]
    rendering_profile: str
    rendering_digest: str

    @classmethod
    def from_resource(cls, resource: Any, rendering_profile: str) -> "MaterializationIdentity":
        metadata = dict(getattr(resource, "metadata", {}) or {})
        raw_interval = metadata.get("selected_interval", metadata.get("selected_span"))
        if raw_interval is None:
            interval = (0, _token_count(getattr(resource, "text", None)))
        else:
            interval = tuple(int(value) for value in raw_interval)
            if len(interval) != 2 or interval[0] < 0 or interval[1] < interval[0]:
                raise ValueError("Selected materialization intervals must be [start, end].")
        version = str(
            metadata.get("version")
            or getattr(resource, "version", None)
            or metadata.get("content_digest")
            or _digest(getattr(resource, "text", None))
        )
        rendered = {
            "uri": getattr(resource, "uri", ""),
            "text": getattr(resource, "text", None),
            "interval": interval,
            "profile": rendering_profile,
        }
        return cls(
            resource_id=str(resource.resource_id),
            resource_version=version,
            selected_interval=(interval[0], interval[1]),
            rendering_profile=rendering_profile,
            rendering_digest=_digest(rendered),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaterializationIdentity":
        """Restore an identity from a durable gateway-session snapshot."""

        interval = tuple(int(item) for item in value["selected_interval"])
        if len(interval) != 2:
            raise ValueError("Materialization snapshot intervals must contain two values.")
        return cls(
            resource_id=str(value["resource_id"]),
            resource_version=str(value["resource_version"]),
            selected_interval=(interval[0], interval[1]),
            rendering_profile=str(value["rendering_profile"]),
            rendering_digest=str(value["rendering_digest"]),
        )


@dataclass(frozen=True)
class VisibleMaterialization:
    """One provenance-bearing occurrence in the active serialized context."""

    identity: MaterializationIdentity
    tenant_id: str
    session_id: str
    record_type: str
    introduced_turn: int
    last_visible_turn: int
    serialized_message_index: int
    serialized_message_digest: str
    token_count: int
    task_id: str | None = None
    authorization_scope: str | None = None
    state: VisibilityState | str = VisibilityState.VISIBLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", VisibilityState(self.state))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisibleMaterialization":
        """Restore provenance without assuming that engine cache state survived."""

        return cls(
            identity=MaterializationIdentity.from_dict(value["identity"]),
            tenant_id=str(value["tenant_id"]),
            session_id=str(value["session_id"]),
            record_type=str(value["record_type"]),
            introduced_turn=int(value["introduced_turn"]),
            last_visible_turn=int(value["last_visible_turn"]),
            serialized_message_index=int(value["serialized_message_index"]),
            serialized_message_digest=str(value["serialized_message_digest"]),
            token_count=int(value["token_count"]),
            task_id=value.get("task_id"),
            authorization_scope=value.get("authorization_scope"),
            state=value.get("state", VisibilityState.UNKNOWN.value),
        )


@dataclass(frozen=True)
class PrefixReuseObservation:
    """Evidence reported about physical prefix-cache reuse for one request."""

    status: PrefixReuseStatus | str = PrefixReuseStatus.UNKNOWN
    engine_cache_handle: str | None = None
    prefix_token_count: int | None = None
    cached_token_count: int | None = None
    prefix_digest: str | None = None
    worker_identity: str | None = None
    model_fingerprint: str | None = None
    continuity_reason: str | None = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PrefixReuseStatus(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status.value}

    @classmethod
    def from_result(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        prefix_cache_mode: str,
        prior_engine_session: bool,
        prefix_digest: str | None,
        prefix_token_count: int | None,
        model_fingerprint: str | None,
        prior_worker_identity: str | None = None,
        prior_model_fingerprint: str | None = None,
        engine_restarted: bool = False,
    ) -> "PrefixReuseObservation":
        raw = dict(raw or {})
        cached = raw.get("prefix_cached_tokens", raw.get("cached_tokens"))
        hit = raw.get("prefix_cache_hit")
        worker_identity = raw.get("worker_identity", raw.get("runner_identity"))
        observed_model = raw.get("model_fingerprint", model_fingerprint)
        continuity_reason = None
        if hit is True or (cached is not None and int(cached) > 0):
            status = PrefixReuseStatus.CONFIRMED
        elif hit is False or prefix_cache_mode == "stateless":
            status = PrefixReuseStatus.ABSENT
        elif engine_restarted:
            status = PrefixReuseStatus.UNKNOWN
            continuity_reason = "engine_restart"
        elif (
            prior_worker_identity is not None
            and worker_identity is not None
            and prior_worker_identity != worker_identity
        ):
            status = PrefixReuseStatus.UNKNOWN
            continuity_reason = "worker_changed"
        elif (
            prior_model_fingerprint is not None
            and observed_model is not None
            and prior_model_fingerprint != observed_model
        ):
            status = PrefixReuseStatus.UNKNOWN
            continuity_reason = "model_fingerprint_changed"
        elif prior_engine_session and prefix_cache_mode in {
            "automatic_prefix_cache",
            "explicit_prefix_handle",
            "session_state",
        }:
            status = PrefixReuseStatus.LIKELY
        else:
            status = PrefixReuseStatus.UNKNOWN
        return cls(
            status=status,
            engine_cache_handle=raw.get("prefix_cache_handle"),
            prefix_token_count=(
                int(raw["prefix_token_count"])
                if raw.get("prefix_token_count") is not None
                else prefix_token_count
            ),
            cached_token_count=int(cached) if cached is not None else None,
            prefix_digest=prefix_digest,
            worker_identity=worker_identity,
            model_fingerprint=observed_model,
            continuity_reason=continuity_reason,
        )


@dataclass(frozen=True)
class RealizationItem:
    """Planner decision and accounting for one selected resource."""

    identity: MaterializationIdentity
    decision: RealizationDecision | str
    token_count: int
    native_bytes: int = 0
    reason: str = ""
    resource: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", RealizationDecision(self.decision))


@dataclass(frozen=True)
class RealizationPlan:
    """Disjoint per-turn decisions with non-overlapping reuse accounting."""

    requested_mode: str
    resolved_mode: str
    items: tuple[RealizationItem, ...]
    resolution_reason: str = "Execution mode was explicitly resolved by gateway capabilities."
    fallback_used: bool = False
    fallback_reason: str | None = None

    def resources_for(self, *decisions: RealizationDecision) -> tuple[Any, ...]:
        allowed = set(decisions)
        return tuple(item.resource for item in self.items if item.decision in allowed)

    @property
    def diagnostics(self) -> dict[str, Any]:
        def ids(decision: RealizationDecision) -> list[str]:
            return [
                item.identity.resource_id
                for item in self.items
                if item.decision == decision
            ]

        visible = [item for item in self.items if item.decision == RealizationDecision.ALREADY_VISIBLE]
        materialized = [item for item in self.items if item.decision in {
            RealizationDecision.MUST_MATERIALIZE, RealizationDecision.OPTIONAL_REFRESH
        }]
        native = [item for item in self.items if item.decision == RealizationDecision.NATIVE_AVAILABLE]
        return {
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "resolution_reason": self.resolution_reason,
            "selected_resources": [item.identity.resource_id for item in self.items],
            "already_visible_resources": ids(RealizationDecision.ALREADY_VISIBLE),
            "native_available_resources": ids(RealizationDecision.NATIVE_AVAILABLE),
            "newly_materialized_resources": [item.identity.resource_id for item in materialized],
            "invalid_resources": ids(RealizationDecision.INVALID),
            "visible_reuse_tokens": sum(item.token_count for item in visible),
            "new_materialized_tokens": sum(item.token_count for item in materialized),
            "native_reuse_tokens": sum(item.token_count for item in native),
            "native_attach_bytes": sum(item.native_bytes for item in native),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


class VisibleMaterializationLedger:
    """Pure helpers for reconciling and updating immutable ledger entries."""

    @staticmethod
    def reconcile(
        entries: Sequence[VisibleMaterialization],
        serialized_messages: Sequence[Mapping[str, Any]],
        *,
        turn: int,
    ) -> tuple[VisibleMaterialization, ...]:
        message_digests = {_digest(dict(message)) for message in serialized_messages}
        values = []
        for entry in entries:
            if entry.state != VisibilityState.VISIBLE:
                values.append(entry)
            elif entry.serialized_message_digest in message_digests:
                values.append(replace(entry, last_visible_turn=turn))
            else:
                values.append(replace(entry, state=VisibilityState.DROPPED_FROM_ACTIVE_CONTEXT))
        return tuple(values)

    @staticmethod
    def mark_resource_state(
        entries: Sequence[VisibleMaterialization],
        resource_id: str,
        state: VisibilityState,
    ) -> tuple[VisibleMaterialization, ...]:
        return tuple(
            replace(entry, state=state)
            if entry.identity.resource_id == resource_id and entry.state == VisibilityState.VISIBLE
            else entry
            for entry in entries
        )

    @staticmethod
    def add_materializations(
        entries: Sequence[VisibleMaterialization],
        resources: Sequence[Any],
        serialized_messages: Sequence[Mapping[str, Any]],
        *,
        tenant_id: str,
        session_id: str,
        rendering_profile: str,
        turn: int,
    ) -> tuple[VisibleMaterialization, ...]:
        values = list(entries)
        for resource in resources:
            identity = MaterializationIdentity.from_resource(resource, rendering_profile)
            values = [
                replace(entry, state=VisibilityState.SUPERSEDED)
                if entry.state == VisibilityState.VISIBLE
                and entry.identity.resource_id == identity.resource_id
                and entry.identity != identity
                else entry
                for entry in values
            ]
            location = VisibleMaterializationLedger._find_location(resource, serialized_messages)
            if location is None:
                continue
            index, message_digest = location
            values.append(
                VisibleMaterialization(
                    identity=identity,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    record_type=str(getattr(resource, "record_type", "document")),
                    introduced_turn=turn,
                    last_visible_turn=turn,
                    serialized_message_index=index,
                    serialized_message_digest=message_digest,
                    token_count=_token_count(getattr(resource, "text", None)),
                    task_id=getattr(resource, "task_id", None),
                    authorization_scope=getattr(resource, "authorization_scope", None),
                )
            )
        return tuple(values)

    @staticmethod
    def _find_location(
        resource: Any, serialized_messages: Sequence[Mapping[str, Any]]
    ) -> tuple[int, str] | None:
        marker = f"[PRA resource {resource.uri}]\n{resource.text}"
        for index in range(len(serialized_messages) - 1, -1, -1):
            message = serialized_messages[index]
            if marker in str(message.get("content", "")):
                return index, _digest(dict(message))
        return None


class RealizationPlanner:
    """Classify selected resources without consulting guessed prefix-cache hits."""

    def plan(
        self,
        resources: Sequence[Any],
        ledger: Sequence[VisibleMaterialization],
        *,
        requested_mode: str,
        resolved_mode: str,
        rendering_profile: str,
        tenant_id: str,
        native_capable: bool = False,
        fallback_reason: str | None = None,
    ) -> RealizationPlan:
        visible = {
            entry.identity: entry
            for entry in ledger
            if entry.state == VisibilityState.VISIBLE and entry.tenant_id == tenant_id
        }
        items = []
        for resource in resources:
            identity = MaterializationIdentity.from_resource(resource, rendering_profile)
            metadata = dict(getattr(resource, "metadata", {}) or {})
            token_count = _token_count(getattr(resource, "text", None))
            terminal_task = str(
                getattr(resource, "task_status", "") or metadata.get("task_status", "")
            ).lower() in {
                "closed", "complete", "completed", "cancelled", "canceled"
            }
            if metadata.get("authorized") is False:
                decision = RealizationDecision.INVALID
                reason = "authorization_rejected"
            elif identity in visible and not terminal_task:
                decision = RealizationDecision.ALREADY_VISIBLE
                reason = "compatible_occurrence_is_active"
            elif resolved_mode in {"native-memory", "native-serving"} and native_capable:
                residency = str(metadata.get("native_residency", "SOURCE")).upper()
                if residency in {"HOT", "WARM"}:
                    decision = RealizationDecision.NATIVE_AVAILABLE
                    reason = f"native_{residency.lower()}"
                else:
                    decision = RealizationDecision.MUST_MATERIALIZE
                    reason = "native_source_requires_encoding"
            else:
                decision = RealizationDecision.MUST_MATERIALIZE
                reason = "not_visible"
            items.append(
                RealizationItem(
                    identity=identity,
                    decision=decision,
                    token_count=token_count,
                    native_bytes=int(metadata.get("native_bytes", 0) or 0),
                    reason=reason,
                    resource=resource,
                )
            )
        return RealizationPlan(
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            items=tuple(items),
            resolution_reason=(
                fallback_reason
                or f"{resolved_mode} is the gateway capability-resolved execution mode."
            ),
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
        )
