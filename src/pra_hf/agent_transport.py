"""Negotiated agent context transport over the OpenAI-compatible PRA wire.

The agent builds semantic messages and detached records. This module alone
decides whether those records remain logical wire resources or are rendered as
ordinary text for an endpoint that does not implement the PRA extension.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .context_records import ContextRecord, RecordViewName, serialize_record
from .deployment import PRAWireRequest, PRAWireResource
from .gateway_session import HistoryMode, ResourceDelta, ResourceOperation


class ContextTransportMode(str, Enum):
    """User policy for preserving or rendering detached context records."""

    AUTO = "auto"
    PRA = "pra"
    TEXT = "text"


class AgentWireMode(str, Enum):
    """Resolved transport used for one endpoint/session."""

    TEXT = "TEXT"
    PRA_FULL = "PRA_FULL"
    PRA_DELTA = "PRA_DELTA"


class AgentTransportError(RuntimeError):
    """Base error for negotiation and logical record transport."""


class EndpointUnavailableError(AgentTransportError):
    """The endpoint could not be reached; this is not a capability downgrade."""


class PRAProtocolRequiredError(AgentTransportError):
    """The selected policy requires unsupported PRA wire capabilities."""


@dataclass(frozen=True)
class AgentTurnContext:
    """Provider-neutral conversational spine plus selected detached records."""

    messages: tuple[Mapping[str, Any], ...]
    records: tuple[ContextRecord, ...] = ()
    tool_records: tuple[ContextRecord, ...] = ()
    skill_records: tuple[ContextRecord, ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    task_id: str | None = None
    task_metadata: Mapping[str, Any] = field(default_factory=dict)
    selected_record_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("An agent turn requires at least one conversational message.")
        object.__setattr__(self, "messages", tuple(dict(row) for row in self.messages))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "tool_records", tuple(self.tool_records))
        object.__setattr__(self, "skill_records", tuple(self.skill_records))
        object.__setattr__(self, "tools", tuple(dict(row) for row in self.tools))
        object.__setattr__(self, "task_metadata", dict(self.task_metadata))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self, "selected_record_ids", tuple(dict.fromkeys(self.selected_record_ids))
        )

    @property
    def detached_records(self) -> tuple[ContextRecord, ...]:
        """Return unique context, tool, and skill records in stable order."""

        values: dict[str, ContextRecord] = {}
        for record in (*self.records, *self.tool_records, *self.skill_records):
            values.setdefault(record.record_id, record)
        return tuple(values.values())


@dataclass(frozen=True)
class AgentTransportCapabilities:
    """Normalized immediate-endpoint features used by an agent transport."""

    endpoint_type: str = "openai"
    protocol_version: str | None = None
    gateway_mode: str | None = None
    engine_type: str | None = None
    integration_level: str | None = None
    logical_refs: bool = False
    typed_records: bool = False
    task_metadata: bool = False
    resource_delta: bool = False
    session_state: bool = False
    incremental_messages: bool = False
    cache_affinity: bool = False
    streaming: bool = False
    native_kv: bool = False
    capability_source: str = "openai_404"

    @property
    def pra_supported(self) -> bool:
        return bool(self.logical_refs and self.typed_records)

    def supports(self, name: str) -> bool:
        if not hasattr(self, name) or name in {
            "endpoint_type", "protocol_version", "gateway_mode", "engine_type",
            "integration_level", "capability_source",
        }:
            raise ValueError(f"Unknown agent transport capability: {name}")
        return bool(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_type": self.endpoint_type,
            "protocol_version": self.protocol_version,
            "gateway_mode": self.gateway_mode,
            "engine_type": self.engine_type,
            "integration_level": self.integration_level,
            "logical_refs": self.logical_refs,
            "typed_records": self.typed_records,
            "task_metadata": self.task_metadata,
            "resource_delta": self.resource_delta,
            "session_state": self.session_state,
            "incremental_messages": self.incremental_messages,
            "cache_affinity": self.cache_affinity,
            "streaming": self.streaming,
            "native_kv": self.native_kv,
            "capability_source": self.capability_source,
        }

    @classmethod
    def ordinary_openai(cls, *, source: str = "openai_404") -> "AgentTransportCapabilities":
        return cls(capability_source=source)

    @classmethod
    def from_response(cls, value: Mapping[str, Any]) -> "AgentTransportCapabilities":
        if value.get("protocol") != "pra":
            raise AgentTransportError("Capability response is not a PRA protocol response.")
        version = str(value.get("protocol_version", ""))
        if not version or version.split(".", 1)[0] != "1":
            raise AgentTransportError(f"Unsupported PRA protocol version: {version or 'missing'}")
        effective = dict(
            value.get("effective_capabilities")
            or value.get("capabilities")
            or {}
        )
        engine = dict(value.get("engine") or {})
        gateway = dict(value.get("gateway") or {})
        return cls(
            endpoint_type=str(value.get("endpoint_type", "engine")),
            protocol_version=version,
            gateway_mode=(gateway.get("mode") or value.get("gateway_mode")),
            engine_type=engine.get("type") or engine.get("engine_type"),
            integration_level=engine.get("integration_level"),
            logical_refs=bool(effective.get("logical_refs")),
            typed_records=bool(effective.get("typed_records")),
            task_metadata=bool(effective.get("task_metadata")),
            resource_delta=bool(effective.get("resource_delta")),
            session_state=bool(effective.get("session_state")),
            incremental_messages=bool(effective.get("incremental_messages")),
            cache_affinity=bool(effective.get("cache_affinity")),
            streaming=bool(effective.get("streaming")),
            native_kv=bool(effective.get("native_kv")),
            capability_source="pra_handshake",
        )


def context_record_to_wire_resource(
    record: ContextRecord,
    *,
    include_body: bool = True,
    selected_view: RecordViewName | str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> PRAWireResource:
    """Project portable record semantics without exposing Python policy or K/V."""

    provenance = dict(record.selection_provenance)
    task = provenance.get("task") if isinstance(provenance.get("task"), Mapping) else {}
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    view = RecordViewName(selected_view or record.policy.selected_view)
    uri = str(payload.get("uri") or provenance.get("uri") or f"pra://record/{record.record_id}")
    authorization_scope = provenance.get("authorization_scope")
    shareable = bool(provenance.get("shareable", False))
    text = serialize_record(record, view=view) if include_body else None
    return PRAWireResource(
        resource_id=record.record_id,
        uri=uri,
        record_type=record.record_type.value,
        text=text,
        version=record.version,
        source_fingerprint=record.source_fingerprint,
        provenance=provenance,
        authorization_scope=(
            None if authorization_scope is None else str(authorization_scope)
        ),
        task_id=(
            None
            if not task or task.get("task_id") is None
            else str(task["task_id"])
        ),
        task_status=(
            None if not task or task.get("task_status") is None
            else str(task.get("task_status"))
        ),
        available_views=tuple(view_name.value for view_name in record.views),
        initial_view=record.policy.initial_view.value,
        selected_view=view.value,
        shareable=shareable,
        session_bound=not shareable,
        metadata={
            "tenant_id": tenant_id,
            "session_id": None if shareable else session_id,
            "parent_id": record.parent_id,
            "child_ids": list(record.child_ids),
        },
    )


def wire_resource_identity(resource: PRAWireResource) -> str:
    """Hash every semantic field that requires an ADD or UPDATE body."""

    identity = {
        "resource_id": resource.resource_id,
        "uri": resource.uri,
        "record_type": resource.record_type,
        "version": resource.version,
        "source_fingerprint": resource.source_fingerprint,
        "authorization_scope": resource.authorization_scope,
        "task_id": resource.task_id,
        "task_status": resource.task_status,
        "available_views": resource.available_views,
        "initial_view": resource.initial_view,
        "selected_view": resource.selected_view,
        "shareable": resource.shareable,
        "session_bound": resource.session_bound,
        "metadata": resource.metadata,
        "text": resource.text,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_wire_resources_as_text(
    messages: Sequence[Mapping[str, Any]],
    resources: Sequence[PRAWireResource],
) -> tuple[Mapping[str, Any], ...]:
    """Apply the canonical G10-compatible text projection to chat messages."""

    # Resource order is not semantic. Canonicalize it so full and delta
    # reconstruction produce byte-identical model input.
    blocks = [
        f"[PRA resource {resource.uri}]\n{resource.text}"
        for resource in sorted(resources, key=lambda item: item.resource_id)
        if resource.text
    ]
    if not blocks:
        return tuple(dict(message) for message in messages)
    context = "PRA text fallback context (not native K/V):\n\n" + "\n\n".join(blocks)
    values = [dict(message) for message in messages]
    user_index = next(
        (
            index
            for index in range(len(values) - 1, -1, -1)
            if values[index].get("role") == "user"
        ),
        len(values),
    )
    if user_index < len(values):
        original = str(values[user_index].get("content", ""))
        values[user_index]["content"] = f"{context}\n\n{original}"
    else:
        values.append({"role": "user", "content": context})
    return tuple(values)


def render_text_messages(turn: AgentTurnContext) -> tuple[Mapping[str, Any], ...]:
    """Render detached records once while preserving the chat message spine."""

    if not turn.detached_records:
        return turn.messages
    resources = tuple(
        context_record_to_wire_resource(record)
        for record in turn.detached_records
    )
    return render_wire_resources_as_text(turn.messages, resources)


def render_text_prompt(turn: AgentTurnContext) -> str:
    """Flatten canonical text messages only at an embedded text-model boundary."""

    return "\n".join(
        f"{str(message.get('role', 'user')).capitalize()}: {message.get('content', '')}"
        for message in render_text_messages(turn)
    ) + "\nAssistant:"


def resolve_wire_mode(
    requested: ContextTransportMode | str,
    capabilities: AgentTransportCapabilities,
    *,
    required: Sequence[str] = (),
    allow_text_fallback: bool = True,
) -> AgentWireMode:
    """Resolve transport from features, never from an endpoint product name."""

    requested = ContextTransportMode(requested)
    if requested == ContextTransportMode.TEXT:
        return AgentWireMode.TEXT
    missing = tuple(name for name in required if not capabilities.supports(name))
    pra_ready = capabilities.pra_supported and not missing
    if pra_ready:
        if capabilities.resource_delta and capabilities.incremental_messages:
            return AgentWireMode.PRA_DELTA
        return AgentWireMode.PRA_FULL
    if requested == ContextTransportMode.PRA and not allow_text_fallback:
        detail = ", ".join(missing or ("logical_refs", "typed_records"))
        raise PRAProtocolRequiredError(f"Endpoint lacks required PRA capabilities: {detail}")
    return AgentWireMode.TEXT


class CapabilityNegotiator:
    """Cache the immediate endpoint handshake until explicit invalidation."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 10.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._cached: AgentTransportCapabilities | None = None
        self._lock = threading.RLock()

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None

    def negotiate(self, *, refresh: bool = False) -> AgentTransportCapabilities:
        with self._lock:
            if self._cached is not None and not refresh:
                return self._cached
            request = urllib.request.Request(
                self.endpoint + "/v1/pra/capabilities", method="GET"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    value = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code in {404, 405, 501}:
                    self._cached = AgentTransportCapabilities.ordinary_openai()
                    return self._cached
                raise EndpointUnavailableError(
                    f"Capability handshake failed with HTTP {error.code}."
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise EndpointUnavailableError(
                    f"Endpoint is unreachable: {self.endpoint}"
                ) from error
            self._cached = AgentTransportCapabilities.from_response(value)
            return self._cached


class NegotiatedRemoteBackend:
    """Remote model backend preserving typed records when the endpoint can."""

    name = "negotiated-remote"

    def __init__(
        self,
        endpoint: str,
        model: str | None,
        *,
        transport: ContextTransportMode | str = ContextTransportMode.AUTO,
        allow_text_fallback: bool = True,
        required_capabilities: Sequence[str] = (),
        credentials_file: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.transport = ContextTransportMode(transport)
        self.allow_text_fallback = bool(allow_text_fallback)
        self.required_capabilities = tuple(required_capabilities)
        self.credentials_file = credentials_file
        self.timeout_seconds = float(timeout_seconds)
        self.negotiator = CapabilityNegotiator(self.endpoint)
        self._resource_versions: dict[str, dict[str, str]] = {}
        self._message_history: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self._last_trace: dict[str, Any] = {}

    def add_reference(self, reference: str, *, text: str | None = None, uri: str | None = None):
        raise RuntimeError("Remote references are transported as logical PRA resources.")

    def refresh_capabilities(self) -> AgentTransportCapabilities:
        """Re-handshake and force full message/resource inventory resynchronization."""

        self.negotiator.invalidate()
        capabilities = self.negotiator.negotiate(refresh=True)
        self._resource_versions.clear()
        self._message_history.clear()
        return capabilities

    def capabilities(self) -> AgentTransportCapabilities:
        return self.negotiator.negotiate()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("PRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _resource_identity(resource: PRAWireResource) -> str:
        return wire_resource_identity(resource)

    def _delta_resources(
        self, session_key: str, resources: Sequence[PRAWireResource]
    ) -> tuple[tuple[PRAWireResource, ...], tuple[ResourceDelta, ...]]:
        previous = self._resource_versions.get(session_key, {})
        current = {resource.resource_id: resource for resource in resources}
        changed: list[PRAWireResource] = []
        operations: list[ResourceDelta] = []
        for resource in current.values():
            identity = self._resource_identity(resource)
            old = previous.get(resource.resource_id)
            if old == identity:
                operations.append(ResourceDelta(
                    ResourceOperation.UNCHANGED,
                    resource.resource_id,
                    resource.uri,
                    identity,
                    resource.record_type,
                    resource.authorization_scope,
                ))
                continue
            operation = ResourceOperation.ADD if old is None else ResourceOperation.UPDATE
            changed.append(resource)
            operations.append(ResourceDelta(
                operation, resource.resource_id, resource.uri, identity,
                resource.record_type, resource.authorization_scope, resource,
            ))
        for resource_id, identity in previous.items():
            if resource_id not in current:
                operations.append(ResourceDelta(
                    ResourceOperation.REMOVE, resource_id, "", identity
                ))
        return tuple(changed), tuple(operations)

    def generate_turn(
        self,
        turn: AgentTurnContext,
        *,
        tenant_id: str = "default",
        session_id: str | None = None,
        max_new_tokens: int = 1024,
        **_: Any,
    ) -> str:
        capabilities = self.capabilities()
        wire_mode = resolve_wire_mode(
            self.transport,
            capabilities,
            required=self.required_capabilities,
            allow_text_fallback=self.allow_text_fallback,
        )
        session_key = session_id or "__stateless__"
        all_resources = tuple(
            context_record_to_wire_resource(
                record,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            for record in turn.detached_records
        )
        messages = turn.messages
        resources: tuple[PRAWireResource, ...] = ()
        resource_ops: tuple[ResourceDelta, ...] = ()
        history_mode = HistoryMode.FULL
        if wire_mode == AgentWireMode.TEXT:
            messages = render_text_messages(turn)
        elif wire_mode == AgentWireMode.PRA_FULL:
            resources = all_resources
        else:
            previous_messages = self._message_history.get(session_key, ())
            if previous_messages and tuple(messages[: len(previous_messages)]) == previous_messages:
                messages = tuple(messages[len(previous_messages) :])
                history_mode = HistoryMode.DELTA
            resources, resource_ops = self._delta_resources(session_key, all_resources)
        request = PRAWireRequest(
            model=self.model or "provider-default",
            messages=tuple(messages),
            tools=turn.tools,
            tenant_id=tenant_id,
            session_id=session_id,
            task_id=turn.task_id,
            resources=resources,
            resource_ops=resource_ops,
            required_capabilities=(
                self.required_capabilities if wire_mode != AgentWireMode.TEXT else ()
            ),
            allow_text_fallback=self.allow_text_fallback,
            history_mode=history_mode,
            max_new_tokens=max_new_tokens,
            metadata={
                **turn.metadata,
                "task_metadata": dict(turn.task_metadata),
                "agent_transport": wire_mode.value,
            },
        )
        payload = request.to_openai(stream=False)
        if wire_mode == AgentWireMode.TEXT:
            payload.pop("pra", None)
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=encoded,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        output_text = str(value["choices"][0]["message"]["content"])
        if wire_mode == AgentWireMode.PRA_DELTA:
            self._resource_versions[session_key] = {
                resource.resource_id: self._resource_identity(resource)
                for resource in all_resources
            }
            self._message_history[session_key] = (
                *turn.messages,
                {"role": "assistant", "content": output_text},
            )
        self._last_trace = {
            "negotiated_transport": wire_mode.value,
            "capability_source": capabilities.capability_source,
            "fallback": wire_mode == AgentWireMode.TEXT and self.transport != ContextTransportMode.TEXT,
            "history_mode": history_mode.value,
            "resource_delta_count": len(resource_ops),
            "selected_resource_ids": list(turn.selected_record_ids),
            "native_kv": capabilities.native_kv,
            "integration_level": capabilities.integration_level,
            "full_text_bytes": len(json.dumps(render_text_messages(turn)).encode("utf-8")),
            "wire_bytes": len(encoded),
            "message_bytes": len(
                json.dumps(payload.get("messages", ()), default=str).encode("utf-8")
            ),
            "resource_body_bytes": sum(
                len((resource.text or "").encode("utf-8")) for resource in resources
            ),
            "resource_delta_bytes": len(
                json.dumps(
                    [operation.to_dict(include_resource=False) for operation in resource_ops],
                    default=str,
                ).encode("utf-8")
            ),
            "resource_bodies_sent": sum(resource.text is not None for resource in resources),
            "resource_resynchronization": (
                wire_mode == AgentWireMode.PRA_DELTA
                and history_mode == HistoryMode.FULL
                and bool(resource_ops)
            ),
        }
        return output_text

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Compatibility path for callers that have already chosen plain text."""

        return self.generate_turn(
            AgentTurnContext(messages=({"role": "user", "content": prompt},)),
            **kwargs,
        )

    def inspect(self) -> Mapping[str, Any]:
        capabilities = self.negotiator._cached
        return {
            "backend": self.name,
            "endpoint": self.endpoint,
            "model": self.model,
            "transport_requested": self.transport.value,
            "transport": dict(self._last_trace),
            "capabilities": None if capabilities is None else capabilities.to_dict(),
        }
