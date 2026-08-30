"""Shared logical-turn and ephemeral engine-session state for PRA runtimes.

Durable agent state remains owned by :mod:`pra_hf.session_service`. This module
tracks reconstructible engine handles, canonical message history, resource
versions, and prefix fingerprints without storing resource bodies in traces.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .session_service import SessionConflict, SessionService


class HistoryMode(str, Enum):
    """Meaning of messages supplied to or emitted by one session turn."""

    FULL = "FULL"
    DELTA = "DELTA"
    AUTO = "AUTO"


class ResourceOperation(str, Enum):
    """Change to one stable PRA resource identity."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    REMOVE = "REMOVE"
    UNCHANGED = "UNCHANGED"


def canonical_digest(value: object) -> str:
    """Hash a canonical JSON representation for trace-safe identity."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resource_version(resource: Any) -> str:
    """Return an explicit version or a digest of logical resource content."""

    metadata = dict(getattr(resource, "metadata", {}) or {})
    explicit = metadata.get("version") or metadata.get("content_digest")
    if explicit is not None:
        return str(explicit)
    return canonical_digest(
        {
            "resource_id": resource.resource_id,
            "uri": resource.uri,
            "record_type": resource.record_type,
            "text": resource.text,
            "authorization_scope": resource.authorization_scope,
        }
    )


def cache_affinity_key(tenant_id: str, session_id: str, model: str) -> str:
    """Derive a tenant/model/session-scoped scheduler hint."""

    digest = canonical_digest((tenant_id, session_id, model))[:24]
    return f"pra-affinity:{digest}"


@dataclass(frozen=True)
class ResourceDelta:
    """Content-free operation metadata plus an optional transport resource."""

    operation: ResourceOperation | str
    resource_id: str
    uri: str
    version: str
    record_type: str = "document"
    authorization_scope: str | None = None
    resource: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", ResourceOperation(self.operation))

    def to_dict(self, *, include_resource: bool = True) -> dict[str, Any]:
        values = {
            "operation": self.operation.value,
            "resource_id": self.resource_id,
            "uri": self.uri,
            "version": self.version,
            "record_type": self.record_type,
            "authorization_scope": self.authorization_scope,
        }
        if include_resource and self.resource is not None:
            values["resource"] = self.resource.to_dict() if hasattr(self.resource, "to_dict") else self.resource
        return values


@dataclass(frozen=True)
class GatewaySessionState:
    """Ephemeral engine-side state keyed by tenant, logical session, and model."""

    tenant_id: str
    session_id: str
    model: str
    canonical_messages: tuple[Mapping[str, Any], ...] = ()
    serialized_messages: tuple[Mapping[str, Any], ...] = ()
    prefix_digest: str = ""
    last_serialized_prefix_digest: str = ""
    prefix_message_count: int = 0
    prefix_token_count: int | None = None
    engine_session_id: str | None = None
    prefix_cache_handle: str | None = None
    known_resources: Mapping[str, str] = field(default_factory=dict)
    known_resource_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    known_resource_bodies: Mapping[str, Any] = field(default_factory=dict, repr=False)
    last_profile: str | None = None
    last_policy_digest: str | None = None
    model_revision: str | None = None
    chat_template_digest: str | None = None
    visible_prefix_profile: str | None = None
    cache_affinity_key: str = ""
    last_invalidation_reason: str | None = None
    turns: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_messages", tuple(dict(row) for row in self.canonical_messages))
        object.__setattr__(self, "serialized_messages", tuple(dict(row) for row in self.serialized_messages))
        object.__setattr__(self, "known_resources", dict(self.known_resources))
        object.__setattr__(
            self,
            "known_resource_metadata",
            {key: dict(value) for key, value in self.known_resource_metadata.items()},
        )
        object.__setattr__(self, "known_resource_bodies", dict(self.known_resource_bodies))
        if not self.cache_affinity_key:
            object.__setattr__(
                self,
                "cache_affinity_key",
                cache_affinity_key(self.tenant_id, self.session_id, self.model),
            )

    @property
    def key(self) -> tuple[str, str, str]:
        return self.tenant_id, self.session_id, self.model

    def inspect(self) -> dict[str, Any]:
        """Return non-sensitive session metadata for debugging."""

        return {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "model": self.model,
            "engine_session_present": self.engine_session_id is not None,
            "prefix_cache_handle_present": self.prefix_cache_handle is not None,
            "prefix_digest": self.prefix_digest,
            "last_serialized_prefix_digest": self.last_serialized_prefix_digest,
            "prefix_message_count": self.prefix_message_count,
            "prefix_token_count": self.prefix_token_count,
            "known_resources": dict(self.known_resources),
            "cache_affinity_key": self.cache_affinity_key,
            "last_profile": self.last_profile,
            "model_revision": self.model_revision,
            "chat_template_digest": self.chat_template_digest,
            "visible_prefix_profile": self.visible_prefix_profile,
            "last_invalidation_reason": self.last_invalidation_reason,
            "turns": self.turns,
        }


@dataclass(frozen=True)
class ResolvedSessionTurn:
    """Canonical and outbound state resolved before adapter execution."""

    state: GatewaySessionState
    canonical_messages: tuple[Mapping[str, Any], ...]
    outbound_messages: tuple[Mapping[str, Any], ...]
    outbound_history_mode: HistoryMode
    resource_deltas: tuple[ResourceDelta, ...]
    active_resources: tuple[Any, ...]
    gateway_prefix_stable: bool
    prefix_changed_reason: str | None
    message_bytes_sent: int
    resource_bytes_sent: int
    session_delta_bytes: int


class GatewaySessionRegistry:
    """Thread-safe session sidecar shared by gateway and local runtime paths."""

    def __init__(self, session_service: SessionService | None = None) -> None:
        self.session_service = session_service
        self._states: dict[tuple[str, str, str], GatewaySessionState] = {}
        self._lock = threading.RLock()

    def get(self, tenant_id: str, session_id: str, model: str) -> GatewaySessionState | None:
        with self._lock:
            return self._states.get((tenant_id, session_id, model))

    def resolve_turn(
        self,
        request: Any,
        *,
        incremental_messages: bool,
        resource_delta: bool,
    ) -> ResolvedSessionTurn:
        """Resolve explicit input history and capability-driven outbound deltas."""

        if request.session_id is None:
            messages = tuple(dict(row) for row in request.messages)
            return ResolvedSessionTurn(
                GatewaySessionState(request.tenant_id, "", request.model),
                messages,
                messages,
                HistoryMode.FULL,
                (),
                tuple(request.resources),
                False,
                "no_session_id",
                _bytes(messages),
                _bytes([resource.to_dict() for resource in request.resources]),
                0,
            )
        key = (request.tenant_id, request.session_id, request.model)
        with self._lock:
            previous = self._states.get(key)
            if previous is None:
                previous = GatewaySessionState(*key)
                self._states[key] = previous
                self._ensure_durable(request)

            input_mode = HistoryMode(request.history_mode)
            if input_mode == HistoryMode.DELTA:
                canonical = (*previous.canonical_messages, *request.messages)
                new_messages = tuple(request.messages)
                stable = bool(previous.canonical_messages)
                reason = None if stable else "first_turn"
            else:
                canonical = tuple(request.messages)
                prefix = previous.canonical_messages
                stable = bool(prefix) and tuple(canonical[: len(prefix)]) == prefix
                reason = None if stable else (
                    "first_turn" if not prefix else "history_rewrite"
                )
                new_messages = tuple(canonical[len(prefix) :]) if stable else canonical

            requested_outbound = input_mode
            if input_mode == HistoryMode.AUTO:
                requested_outbound = (
                    HistoryMode.DELTA if incremental_messages and previous.turns > 0
                    else HistoryMode.FULL
                )
            outbound = new_messages if requested_outbound == HistoryMode.DELTA else canonical
            if request.resource_ops:
                active_resources, deltas = self._apply_resource_ops(
                    previous, request.resource_ops, request.resources
                )
            else:
                active_resources = tuple(request.resources)
                deltas = self._resource_deltas(previous, active_resources)
            outbound_deltas = deltas if resource_delta else ()
            resource_payload = [delta.to_dict() for delta in outbound_deltas]
            if not resource_delta:
                resource_payload = [resource.to_dict() for resource in active_resources]
            message_bytes = _bytes(outbound)
            resource_bytes = _bytes(resource_payload)
            delta_bytes = message_bytes + resource_bytes if requested_outbound == HistoryMode.DELTA else 0
            return ResolvedSessionTurn(
                previous,
                tuple(canonical),
                tuple(outbound),
                requested_outbound,
                tuple(outbound_deltas),
                tuple(active_resources),
                stable,
                reason,
                message_bytes,
                resource_bytes,
                delta_bytes,
            )

    def commit(
        self,
        turn: ResolvedSessionTurn,
        request: Any,
        *,
        engine_session_id: str | None,
        prefix_cache_handle: str | None = None,
        prefix_token_count: int | None = None,
        serialized_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> GatewaySessionState:
        """Commit one successful turn without persisting engine cache contents."""

        if request.session_id is None:
            return turn.state
        resources = dict(turn.state.known_resources)
        metadata = dict(turn.state.known_resource_metadata)
        bodies = dict(turn.state.known_resource_bodies)
        active_by_id = {
            resource.resource_id: resource for resource in turn.active_resources
        }
        committed_deltas = (
            list(request.resource_ops)
            if request.resource_ops
            else self._resource_deltas(turn.state, turn.active_resources)
        )
        for delta in committed_deltas:
            if delta.operation == ResourceOperation.REMOVE:
                resources.pop(delta.resource_id, None)
                metadata.pop(delta.resource_id, None)
                bodies.pop(delta.resource_id, None)
            else:
                resources[delta.resource_id] = delta.version
                metadata[delta.resource_id] = {
                    "uri": delta.uri,
                    "record_type": delta.record_type,
                    "authorization_scope": delta.authorization_scope,
                }
                body = delta.resource or active_by_id.get(delta.resource_id)
                if body is not None:
                    bodies[delta.resource_id] = body
        policy_digest = canonical_digest(dict(request.pra_policy))
        profile = request.pra_policy.get("profile") if request.pra_policy else None
        updated = replace(
            turn.state,
            canonical_messages=turn.canonical_messages,
            serialized_messages=(
                turn.canonical_messages
                if serialized_messages is None
                else tuple(serialized_messages)
            ),
            prefix_digest=canonical_digest(turn.canonical_messages),
            last_serialized_prefix_digest=canonical_digest(
                turn.canonical_messages if serialized_messages is None else serialized_messages
            ),
            prefix_message_count=len(
                turn.canonical_messages if serialized_messages is None else serialized_messages
            ),
            prefix_token_count=prefix_token_count,
            engine_session_id=engine_session_id or turn.state.engine_session_id,
            prefix_cache_handle=prefix_cache_handle or turn.state.prefix_cache_handle,
            known_resources=resources,
            known_resource_metadata=metadata,
            known_resource_bodies=bodies,
            last_profile=(
                turn.state.last_profile if profile is None else str(profile)
            ),
            last_policy_digest=(
                turn.state.last_policy_digest
                if not request.pra_policy
                else policy_digest
            ),
            model_revision=(
                _optional(request.metadata.get("model_revision"))
                or turn.state.model_revision
            ),
            chat_template_digest=(
                _optional(request.metadata.get("chat_template_digest"))
                or turn.state.chat_template_digest
            ),
            visible_prefix_profile=(
                _optional(request.metadata.get("visible_prefix_profile"))
                or turn.state.visible_prefix_profile
            ),
            last_invalidation_reason=(
                turn.prefix_changed_reason or turn.state.last_invalidation_reason
            ),
            turns=turn.state.turns + 1,
        )
        with self._lock:
            self._states[updated.key] = updated
        return updated

    def invalidate(self, tenant_id: str, session_id: str, model: str, reason: str) -> None:
        with self._lock:
            key = (tenant_id, session_id, model)
            state = self._states.get(key)
            if state is not None:
                self._states[key] = replace(
                    state,
                    engine_session_id=None,
                    prefix_cache_handle=None,
                    last_invalidation_reason=reason,
                )

    def close(self, tenant_id: str, session_id: str, model: str) -> GatewaySessionState | None:
        with self._lock:
            return self._states.pop((tenant_id, session_id, model), None)

    def inspect(self, tenant_id: str, session_id: str, model: str) -> dict[str, Any] | None:
        state = self.get(tenant_id, session_id, model)
        return None if state is None else state.inspect()

    def inspect_all(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(state.inspect() for _, state in sorted(self._states.items()))

    @staticmethod
    def _resource_deltas(previous: GatewaySessionState, resources: Sequence[Any]) -> list[ResourceDelta]:
        current = {resource.resource_id: resource for resource in resources}
        deltas: list[ResourceDelta] = []
        for resource_id, resource in current.items():
            version = resource_version(resource)
            old = previous.known_resources.get(resource_id)
            operation = (
                ResourceOperation.ADD if old is None
                else ResourceOperation.UNCHANGED if old == version
                else ResourceOperation.UPDATE
            )
            deltas.append(ResourceDelta(
                operation,
                resource_id,
                resource.uri,
                version,
                resource.record_type,
                resource.authorization_scope,
                resource,
            ))
        for resource_id, version in previous.known_resources.items():
            if resource_id in current:
                continue
            info = previous.known_resource_metadata.get(resource_id, {})
            deltas.append(ResourceDelta(
                ResourceOperation.REMOVE,
                resource_id,
                str(info.get("uri", "")),
                version,
                str(info.get("record_type", "document")),
                info.get("authorization_scope"),
            ))
        return deltas

    @staticmethod
    def _apply_resource_ops(
        previous: GatewaySessionState,
        operations: Sequence[ResourceDelta],
        supplied_resources: Sequence[Any] = (),
    ) -> tuple[tuple[Any, ...], list[ResourceDelta]]:
        """Apply client deltas to the gateway inventory without leaking bodies."""

        active = dict(previous.known_resource_bodies)
        supplied = {resource.resource_id: resource for resource in supplied_resources}
        normalized: list[ResourceDelta] = []
        for delta in operations:
            if delta.operation in {ResourceOperation.ADD, ResourceOperation.UPDATE}:
                body = delta.resource or supplied.get(delta.resource_id)
                if body is None:
                    raise ValueError(
                        f"{delta.operation.value} requires a resource body for {delta.resource_id}."
                    )
                if body.resource_id != delta.resource_id:
                    raise ValueError("Resource operation identity does not match its body.")
                active[delta.resource_id] = body
            elif delta.operation == ResourceOperation.REMOVE:
                active.pop(delta.resource_id, None)
            elif delta.operation == ResourceOperation.UNCHANGED:
                if delta.resource_id not in active:
                    raise ValueError(
                        f"Unknown unchanged resource {delta.resource_id}; full resync required."
                    )
            normalized.append(delta)
        return tuple(active[key] for key in sorted(active)), normalized

    def _ensure_durable(self, request: Any) -> None:
        if self.session_service is None:
            return
        user_id = str(request.metadata.get("user_id", "gateway"))
        try:
            existing = self.session_service.get_session(user_id, request.session_id)
            if existing.tenant_id != request.tenant_id:
                raise PermissionError("Durable session belongs to another tenant.")
        except KeyError:
            try:
                self.session_service.create_session(
                    user_id,
                    request.session_id,
                    tenant_id=request.tenant_id,
                    metadata={"gateway_model": request.model},
                )
            except SessionConflict:
                pass


def _bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _optional(value: object) -> str | None:
    return None if value is None else str(value)
