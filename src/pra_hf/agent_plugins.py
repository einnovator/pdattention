"""Thin agent-family bridges for the logical PRA gateway protocol.

The bridges consume public harness events, retain stable typed resource
identities, and emit :class:`PRAWireRequest` objects. They never move native
K/V through the agent process. An ordinary engine remains usable through the
gateway's explicit G10 text fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .deployment import PRAEngineResult, PRAWireBudget, PRAWireRequest, PRAWireResource
from .gateway import PRAGateway


class AgentFamily(str, Enum):
    """Agent event vocabulary understood by a public bridge."""

    DEEPSEEK_HARNESS = "deepseek_harness"
    PI_CODING_AGENT = "pi_coding_agent"


@dataclass(frozen=True)
class PRAAgentPluginConfig:
    """Logical request defaults shared by in-process and RPC integrations."""

    model: str
    tenant_id: str = "default"
    max_resources: int = 8
    max_selected_tokens: int = 2048
    allow_text_fallback: bool = True
    require_native_pra: bool = False

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Agent plugin configuration requires a model.")
        if self.max_resources <= 0 or self.max_selected_tokens <= 0:
            raise ValueError("Agent plugin budgets must be positive.")


def _text_content(value: Any) -> str:
    """Flatten common agent text-content envelopes without serializing blobs."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if value.get("type") == "text" and "text" in value:
            return str(value["text"])
        if "content" in value:
            return _text_content(value["content"])
        if "text" in value:
            return str(value["text"])
        return json.dumps(dict(value), sort_keys=True, default=str)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(filter(None, (_text_content(row) for row in value)))
    return str(value)


@dataclass
class PRAAgentPluginAdapter:
    """Stateful adapter from one agent event stream to logical PRA requests."""

    family: AgentFamily | str
    config: PRAAgentPluginConfig
    session_id: str | None = None
    task_id: str | None = None
    _resources: dict[str, PRAWireResource] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.family = AgentFamily(self.family)

    @property
    def resources(self) -> tuple[PRAWireResource, ...]:
        """Return resources in deterministic ingestion order."""

        return tuple(self._resources.values())

    def _resource(
        self,
        *,
        event_id: str,
        event_type: str,
        record_type: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> PRAWireResource | None:
        text = text.strip()
        if not text:
            return None
        identity = event_id or hashlib.sha256(
            f"{self.family.value}:{event_type}:{text}".encode("utf-8")
        ).hexdigest()[:20]
        resource_id = f"{self.family.value}:{identity}"
        resource = PRAWireResource(
            resource_id=resource_id,
            uri=f"pra://agent/{self.family.value}/{identity}",
            record_type=record_type,
            text=text,
            provenance={
                "agent_family": self.family.value,
                "event_type": event_type,
                "event_id": identity,
            },
            metadata={
                "tenant_id": self.config.tenant_id,
                **({"task_id": self.task_id} if self.task_id else {}),
                **dict(metadata or {}),
            },
        )
        self._resources.setdefault(resource_id, resource)
        return self._resources[resource_id]

    def ingest_event(self, event: Mapping[str, Any]) -> PRAWireResource | None:
        """Capture one durable tool/result event from the configured family."""

        event_type = str(event.get("type", ""))
        if self.family == AgentFamily.DEEPSEEK_HARNESS:
            return self._ingest_deepseek(event_type, event)
        return self._ingest_pi(event_type, event)

    def _ingest_deepseek(
        self, event_type: str, event: Mapping[str, Any]
    ) -> PRAWireResource | None:
        if event_type not in {"tool/result", "session/reference", "attachment"}:
            return None
        payload = event.get("result", event.get("content", event.get("payload")))
        return self._resource(
            event_id=str(event.get("id", event.get("eventId", ""))),
            event_type=event_type,
            record_type="tool_result" if event_type == "tool/result" else "document",
            text=_text_content(payload),
            metadata={
                "tool_name": event.get("toolName", event.get("name")),
                "session_sequence": event.get("sequence"),
            },
        )

    def _ingest_pi(
        self, event_type: str, event: Mapping[str, Any]
    ) -> PRAWireResource | None:
        if event_type == "tool_execution_end":
            return self._resource(
                event_id=str(event.get("toolCallId", "")),
                event_type=event_type,
                record_type="tool_result",
                text=_text_content(event.get("result")),
                metadata={
                    "tool_name": event.get("toolName"),
                    "is_error": bool(event.get("isError", False)),
                },
            )
        if event_type == "message_end":
            message = event.get("message", {})
            if not isinstance(message, Mapping) or message.get("role") not in {
                "toolResult",
                "bashExecution",
            }:
                return None
            return self._resource(
                event_id=str(message.get("toolCallId", event.get("id", ""))),
                event_type=event_type,
                record_type="tool_result",
                text=_text_content(message.get("content", message.get("output"))),
                metadata={"tool_name": message.get("toolName")},
            )
        return None

    def ingest_events(
        self, events: Iterable[Mapping[str, Any]]
    ) -> tuple[PRAWireResource, ...]:
        """Ingest an event batch and return only newly recognized resources."""

        recognized = []
        before = set(self._resources)
        for event in events:
            resource = self.ingest_event(event)
            if resource is not None and resource.resource_id not in before:
                recognized.append(resource)
                before.add(resource.resource_id)
        return tuple(recognized)

    def request(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        query_facets: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> PRAWireRequest:
        """Build a tensor-free logical request for a local or remote gateway."""

        required = (
            ("logical_refs", "typed_records", "native_kv")
            if self.config.require_native_pra
            else ("logical_refs", "typed_records")
        )
        return PRAWireRequest(
            model=self.config.model,
            messages=tuple(dict(row) for row in messages),
            tenant_id=self.config.tenant_id,
            session_id=self.session_id,
            task_id=self.task_id,
            resources=self.resources,
            query_facets=tuple(dict(row) for row in query_facets),
            budget=PRAWireBudget(
                self.config.max_resources, self.config.max_selected_tokens
            ),
            required_capabilities=required,
            allow_text_fallback=self.config.allow_text_fallback,
            metadata={"agent_family": self.family.value, **dict(metadata or {})},
        )

    def generate(
        self,
        gateway: PRAGateway,
        messages: Sequence[Mapping[str, Any]],
        **request_kwargs: Any,
    ) -> PRAEngineResult:
        """Submit through the same gateway path used by ordinary HTTP agents."""

        return gateway.generate(self.request(messages, **request_kwargs))


class DeepSeekHarnessPRAAdapter(PRAAgentPluginAdapter):
    """Bridge DeepSeek Harness durable events to the PRA logical protocol."""

    def __init__(
        self,
        config: PRAAgentPluginConfig,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        super().__init__(AgentFamily.DEEPSEEK_HARNESS, config, session_id, task_id)


class PiCodingAgentPRAAdapter(PRAAgentPluginAdapter):
    """Bridge Pi extension/RPC events to the PRA logical protocol."""

    def __init__(
        self,
        config: PRAAgentPluginConfig,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        super().__init__(AgentFamily.PI_CODING_AGENT, config, session_id, task_id)
