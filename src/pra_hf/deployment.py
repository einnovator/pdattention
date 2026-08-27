"""Logical PRA wire contracts and inference-engine adapter boundaries."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Protocol, Sequence


class PRAEngineIntegrationLevel(str, Enum):
    """Depth at which an inference engine implements PRA."""

    E0_FACADE = "E0"
    E1_NATIVE_EXECUTION = "E1"
    E2_MEMORY_RUNTIME = "E2"
    E3_SCHEDULER = "E3"


class PRAGatewayMode(str, Enum):
    """PRA awareness on the request and engine sides, respectively."""

    G00_PASS_THROUGH = "G00"
    G10_TEXT_FALLBACK = "G10"
    G01_UPGRADE = "G01"
    G11_MEDIATION = "G11"


@dataclass(frozen=True)
class PRAEngineCapabilities:
    """Static, inspectable features implemented by one transport adapter."""

    adapter: str
    integration_level: PRAEngineIntegrationLevel | str = PRAEngineIntegrationLevel.E0_FACADE
    logical_refs: bool = False
    typed_records: bool = False
    task_metadata: bool = False
    text_fallback: bool = True
    native_kv: bool = False
    external_kv_residency: bool = False
    cpu_kv: bool = False
    pinned_kv: bool = False
    gpu_kv: bool = False
    semantic_cache: bool = False
    streaming: bool = False
    callback_routing: bool = False
    pra_scheduler: bool = False
    tool_resources: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "integration_level", PRAEngineIntegrationLevel(self.integration_level)
        )
        if self.native_kv and self.integration_level == PRAEngineIntegrationLevel.E0_FACADE:
            raise ValueError("Native K/V requires at least E1 engine integration.")

    def supports(self, capability: str) -> bool:
        if not hasattr(self, capability) or capability in {"adapter", "integration_level"}:
            raise ValueError(f"Unknown PRA engine capability: {capability}")
        return bool(getattr(self, capability))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "integration_level": self.integration_level.value}


@dataclass(frozen=True)
class PRAWireResource:
    """Stable logical record descriptor; never contains model-native tensors."""

    resource_id: str
    uri: str
    record_type: str = "document"
    text: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    authorization_scope: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PRAWireResource":
        return cls(**dict(value))


@dataclass(frozen=True)
class PRAWireBudget:
    """Logical selection limits independent of physical engine allocation."""

    max_resources: int = 8
    max_selected_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.max_resources <= 0 or self.max_selected_tokens <= 0:
            raise ValueError("PRA wire budgets must be positive.")


@dataclass(frozen=True)
class PRAWireRequest:
    """Serializable request shared by harness, gateway, and engine adapters."""

    model: str
    messages: tuple[Mapping[str, Any], ...]
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = "default"
    session_id: str | None = None
    task_id: str | None = None
    resources: tuple[PRAWireResource, ...] = ()
    query_facets: tuple[Mapping[str, Any], ...] = ()
    budget: PRAWireBudget = field(default_factory=PRAWireBudget)
    pra_policy: Mapping[str, Any] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    allow_text_fallback: bool = False
    engine_hints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not self.messages:
            raise ValueError("A PRA request requires a model and at least one message.")
        if any(key.lower() in {"api_key", "password", "token", "secret"} for key in self.metadata):
            raise ValueError("Credentials must not be stored in PRA request metadata.")
        for resource in self.resources:
            owner = resource.metadata.get("tenant_id")
            if owner is not None and str(owner) != self.tenant_id:
                raise PermissionError(
                    f"Resource {resource.resource_id!r} belongs to another tenant."
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PRAWireRequest":
        data = dict(value)
        data["messages"] = tuple(dict(row) for row in data.get("messages", ()))
        data["resources"] = tuple(
            item if isinstance(item, PRAWireResource) else PRAWireResource.from_dict(item)
            for item in data.get("resources", ())
        )
        budget = data.get("budget")
        if budget is not None and not isinstance(budget, PRAWireBudget):
            data["budget"] = PRAWireBudget(**dict(budget))
        for key in ("query_facets", "required_capabilities"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def from_openai(cls, value: Mapping[str, Any]) -> "PRAWireRequest":
        """Read the stable subset plus an optional ``pra`` extension envelope."""

        envelope = dict(value.get("pra", {}))
        envelope.update(
            {"model": value.get("model"), "messages": value.get("messages", ())}
        )
        if "request_id" not in envelope and value.get("id"):
            envelope["request_id"] = str(value["id"])
        return cls.from_dict(envelope)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PRAEngineResult:
    """Normalized non-streaming engine response and component-local trace."""

    text: str
    raw: Mapping[str, Any] = field(default_factory=dict)
    trace: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PRAEngineAdapter(Protocol):
    """Transport contract implemented by local and remote inference engines."""

    def capabilities(self) -> PRAEngineCapabilities: ...
    def prepare_session(self, request: PRAWireRequest) -> str | None: ...
    def generate(self, request: PRAWireRequest) -> PRAEngineResult: ...
    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, Any]]: ...
    def close_session(self, session_id: str) -> None: ...


class OpenAICompatibleEngineAdapter:
    """E0 HTTP adapter for ordinary OpenAI-compatible inference servers."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 120.0, name: str = "openai"):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.name = name

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(adapter=self.name, text_fallback=True)

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        payload = json.dumps(
            {"model": request.model, "messages": list(request.messages), "stream": False}
        ).encode("utf-8")
        started = time.perf_counter()
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        text = str(raw["choices"][0]["message"]["content"])
        return PRAEngineResult(
            text,
            raw,
            ({"stage": "engine_request", "seconds": time.perf_counter() - started},),
        )

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, Any]]:
        raise NotImplementedError("This E0 adapter exposes non-streaming generation only.")

    def close_session(self, session_id: str) -> None:
        return None


class HuggingFaceEngineAdapter:
    """E1 in-process adapter using the HF semantic-reference implementation."""

    def __init__(self, model) -> None:
        self.model = model

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter="huggingface_eager",
            integration_level="E1",
            logical_refs=True,
            typed_records=True,
            task_metadata=True,
            text_fallback=True,
            native_kv=True,
            external_kv_residency=True,
            cpu_kv=True,
            gpu_kv=True,
            semantic_cache=True,
            tool_resources=True,
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        handles = []
        try:
            for resource in request.resources:
                if resource.text:
                    handles.append(
                        self.model.add_reference(resource.uri, text=resource.text)
                    )
            text = self.model.chat(
                request.messages,
                return_details=True,
                pra_policy=request.pra_policy or None,
            )
            return PRAEngineResult(
                text.text,
                {"stats": text.stats},
                ({"stage": "engine_native_materialization", "native_kv": True},),
            )
        finally:
            for handle in handles:
                self.model.remove_reference(handle)

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, Any]]:
        raise NotImplementedError("HF gateway streaming is a planned extension.")

    def close_session(self, session_id: str) -> None:
        return None


def inferred_resource(message: Mapping[str, Any], index: int) -> PRAWireResource | None:
    """Derive one stable logical resource from structured ordinary-agent traffic."""

    role = str(message.get("role", ""))
    if role not in {"system", "tool"} and not message.get("attachment"):
        return None
    content = str(message.get("content", ""))
    if not content:
        return None
    digest = hashlib.sha256(f"{index}:{role}:{content}".encode("utf-8")).hexdigest()[:16]
    return PRAWireResource(
        resource_id=f"inferred-{digest}",
        uri=f"pra://inferred/{digest}",
        record_type="tool_result" if role == "tool" else "system_record",
        text=content,
        provenance={"message_index": index, "role": role, "inferred": True},
    )
