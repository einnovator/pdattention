"""Logical PRA wire contracts and inference-engine adapter boundaries."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Protocol, Sequence

from .engine_profiles import EngineProfileRegistry, EngineType, PrefixCacheMode
from .gateway_session import HistoryMode, ResourceDelta
from .observability import (
    DISABLED_OBSERVABILITY,
    Observability,
    engine_observability_capabilities,
)


class PRAEngineIntegrationLevel(str, Enum):
    """Depth at which an inference engine implements PRA."""

    E0_SELECTED_TEXT = "E0"
    E1_LOGICAL_PRA = "E1"
    E2_NATIVE_ATTENTION = "E2"
    E3_SCHEDULER = "E3"

    # Source-compatible aliases retained for callers written against the first
    # naming pass. Their canonical meanings are the definitions above.
    E0_FACADE = "E0"
    E1_NATIVE_EXECUTION = "E1"
    E2_MEMORY_RUNTIME = "E2"


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
    engine_type: EngineType | str = EngineType.CUSTOM
    integration_level: PRAEngineIntegrationLevel | str = PRAEngineIntegrationLevel.E0_FACADE
    prefix_cache_mode: PrefixCacheMode | str = PrefixCacheMode.UNKNOWN
    automatic_prefix_cache: bool = False
    explicit_prefix_cache: bool = False
    session_state: bool = False
    incremental_messages: bool = False
    resource_delta: bool = False
    cache_affinity: bool = False
    prefix_cache_handle: bool = False
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
    selected_interval_materialization: bool = False
    request_lifetime: bool = False
    phase_selection: bool = False
    host_device_residency: bool = False
    scheduler_hints: bool = False
    tenant_isolation: bool = False
    callback_routing: bool = False
    pra_scheduler: bool = False
    tool_resources: bool = False
    observability: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "integration_level", PRAEngineIntegrationLevel(self.integration_level)
        )
        object.__setattr__(self, "engine_type", EngineType(self.engine_type))
        object.__setattr__(self, "prefix_cache_mode", PrefixCacheMode(self.prefix_cache_mode))
        object.__setattr__(self, "observability", dict(self.observability))
        if self.native_kv and self.integration_level not in {
            PRAEngineIntegrationLevel.E2_NATIVE_ATTENTION,
            PRAEngineIntegrationLevel.E3_SCHEDULER,
        }:
            raise ValueError("Native K/V requires E2 or E3 engine integration.")

    def supports(self, capability: str) -> bool:
        if not hasattr(self, capability) or capability in {"adapter", "integration_level"}:
            raise ValueError(f"Unknown PRA engine capability: {capability}")
        return bool(getattr(self, capability))

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "engine_type": self.engine_type.value,
            "integration_level": self.integration_level.value,
            "prefix_cache_mode": self.prefix_cache_mode.value,
        }


@dataclass(frozen=True)
class PRAWireResource:
    """Stable logical record descriptor; never contains model-native tensors."""

    resource_id: str
    uri: str
    record_type: str = "document"
    text: str | None = None
    version: str = "v1"
    source_fingerprint: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    authorization_scope: str | None = None
    task_id: str | None = None
    task_status: str | None = None
    available_views: tuple[str, ...] = ()
    initial_view: str | None = None
    selected_view: str | None = None
    shareable: bool = False
    session_bound: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "available_views", tuple(dict.fromkeys(self.available_views)))
        if not self.resource_id or not self.uri:
            raise ValueError("PRA wire resources require resource_id and uri.")
        forbidden = {"api_key", "password", "token", "secret", "credential"}
        if any(str(key).lower() in forbidden for key in (*self.metadata, *self.provenance)):
            raise ValueError("Credentials must not be stored in PRA resource metadata.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PRAWireResource":
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    tools: tuple[Mapping[str, Any], ...] = ()
    protocol_version: str = "1"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = "default"
    session_id: str | None = None
    task_id: str | None = None
    resources: tuple[PRAWireResource, ...] = ()
    resource_ops: tuple[ResourceDelta, ...] = ()
    query_facets: tuple[Mapping[str, Any], ...] = ()
    budget: PRAWireBudget = field(default_factory=PRAWireBudget)
    pra_policy: Mapping[str, Any] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    allow_text_fallback: bool = False
    history_mode: HistoryMode | str = HistoryMode.AUTO
    engine_session_id: str | None = None
    prefix_cache_handle: str | None = None
    cache_affinity_key: str | None = None
    max_new_tokens: int | None = None
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
        object.__setattr__(self, "history_mode", HistoryMode(self.history_mode))
        object.__setattr__(self, "tools", tuple(dict(tool) for tool in self.tools))
        if self.protocol_version.split(".", 1)[0] != "1":
            raise ValueError(
                f"Unsupported PRA protocol major version: {self.protocol_version}"
            )
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

    @property
    def resolved_max_new_tokens(self) -> int:
        """Prefer the typed field while retaining the pre-contract hint."""

        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return int(self.engine_hints.get("max_new_tokens", 64))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PRAWireRequest":
        data = dict(value)
        data["messages"] = tuple(dict(row) for row in data.get("messages", ()))
        data["tools"] = tuple(dict(row) for row in data.get("tools", ()))
        data["resources"] = tuple(
            item if isinstance(item, PRAWireResource) else PRAWireResource.from_dict(item)
            for item in data.get("resources", ())
        )
        operations = []
        for item in data.get("resource_ops", ()):
            if isinstance(item, ResourceDelta):
                operations.append(item)
                continue
            operation = dict(item)
            resource = operation.get("resource")
            if resource is not None and not isinstance(resource, PRAWireResource):
                operation["resource"] = PRAWireResource.from_dict(resource)
            operations.append(ResourceDelta(**operation))
        data["resource_ops"] = tuple(operations)
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
            {
                "model": value.get("model"),
                "messages": value.get("messages", ()),
                "tools": value.get("tools", ()),
            }
        )
        if "max_new_tokens" not in envelope and value.get("max_tokens") is not None:
            envelope["max_new_tokens"] = int(value["max_tokens"])
        if "request_id" not in envelope and value.get("id"):
            envelope["request_id"] = str(value["id"])
        return cls.from_dict(envelope)

    def to_openai(self, *, stream: bool = False) -> dict[str, Any]:
        """Return an OpenAI-compatible request with one versioned PRA extension."""

        envelope = self.to_dict()
        envelope.pop("model", None)
        envelope.pop("messages", None)
        tools = envelope.pop("tools", None)
        payload = {
            "model": self.model,
            "messages": list(self.messages),
            "stream": bool(stream),
            "max_tokens": self.resolved_max_new_tokens,
            "pra": envelope,
        }
        if tools:
            payload["tools"] = list(tools)
        return payload

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["history_mode"] = self.history_mode.value
        values["resource_ops"] = [
            row.to_dict(include_resource=False) for row in self.resource_ops
        ]
        return values


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

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        name: str | None = None,
        engine_type: EngineType | str = EngineType.OPENAI_GENERIC,
        pra_level: str = "auto",
        prefix_cache_mode: PrefixCacheMode | str = "auto",
        session_state: bool | None = None,
        incremental_messages: bool | None = None,
        resource_delta: bool | None = None,
        cache_affinity: bool | None = None,
        model_override: str | None = None,
        observability: Observability | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.engine_type = EngineType(engine_type)
        self.name = name or self.engine_type.value
        profile = EngineProfileRegistry.default().resolve(self.engine_type)
        self.pra_level = profile.default_pra_level if pra_level == "auto" else pra_level
        self.prefix_cache_mode = (
            profile.default_prefix_cache_mode
            if prefix_cache_mode == "auto"
            else PrefixCacheMode(prefix_cache_mode)
        )
        self.session_state = profile.explicit_session if session_state is None else bool(session_state)
        self.incremental_messages = profile.incremental_messages if incremental_messages is None else bool(incremental_messages)
        self.resource_delta = profile.resource_delta if resource_delta is None else bool(resource_delta)
        self.cache_affinity = profile.cache_affinity if cache_affinity is None else bool(cache_affinity)
        self.model_override = model_override.strip() if model_override else None
        self.observability = observability or DISABLED_OBSERVABILITY

    def capabilities(self) -> PRAEngineCapabilities:
        level = PRAEngineIntegrationLevel(self.pra_level)
        logical = level != PRAEngineIntegrationLevel.E0_SELECTED_TEXT
        native = level in {
            PRAEngineIntegrationLevel.E2_NATIVE_ATTENTION,
            PRAEngineIntegrationLevel.E3_SCHEDULER,
        }
        return PRAEngineCapabilities(
            adapter=self.name,
            engine_type=self.engine_type,
            integration_level=level,
            prefix_cache_mode=self.prefix_cache_mode,
            automatic_prefix_cache=self.prefix_cache_mode == PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            explicit_prefix_cache=self.prefix_cache_mode == PrefixCacheMode.EXPLICIT_PREFIX_HANDLE,
            session_state=self.session_state,
            incremental_messages=self.incremental_messages,
            resource_delta=self.resource_delta,
            cache_affinity=self.cache_affinity,
            prefix_cache_handle=self.prefix_cache_mode == PrefixCacheMode.EXPLICIT_PREFIX_HANDLE,
            logical_refs=logical,
            typed_records=logical,
            text_fallback=True,
            native_kv=native,
            streaming=True,
            observability=engine_observability_capabilities(self.engine_type.value),
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id if self.session_state else None

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        payload = json.dumps(self._payload(request)).encode("utf-8")
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        self.observability.inject(headers)
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        text = str(raw["choices"][0]["message"].get("content") or "")
        return PRAEngineResult(
            text,
            raw,
            ({"stage": "engine_request", "seconds": time.perf_counter() - started},),
        )

    def _payload(
        self, request: PRAWireRequest, *, stream: bool = False
    ) -> dict[str, Any]:
        """Build an ordinary OpenAI request plus a typed PRA envelope at E1+."""

        payload: dict[str, Any] = {
            "model": self.model_override or request.model,
            "messages": list(request.messages),
            "stream": stream,
        }
        if request.max_new_tokens is not None or "max_new_tokens" in request.engine_hints:
            payload["max_tokens"] = request.resolved_max_new_tokens
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.tools:
            payload["tools"] = list(request.tools)
        if self.capabilities().logical_refs:
            envelope = request.to_dict()
            envelope.pop("model", None)
            envelope.pop("messages", None)
            envelope.pop("tools", None)
            payload["pra"] = envelope
        return payload

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, Any]]:
        """Translate an OpenAI-compatible SSE stream into portable PRA rows."""

        payload = json.dumps(self._payload(request, stream=True)).encode("utf-8")
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        self.observability.inject(headers)
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            last_event: Mapping[str, Any] | None = None
            finish_reason: str | None = None
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    yield {
                        "type": "done",
                        "request_id": request.request_id,
                        "raw": last_event,
                        "finish_reason": finish_reason,
                    }
                    return
                event = json.loads(data)
                last_event = event
                choice = next(iter(event.get("choices", ())), {})
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    yield {
                        "type": "delta",
                        "request_id": request.request_id,
                        "text": str(content),
                        "raw": event,
                    }
                reasoning = delta.get("reasoning_content", delta.get("reasoning"))
                if reasoning:
                    yield {
                        "type": "reasoning_delta",
                        "request_id": request.request_id,
                        "text": str(reasoning),
                        "raw": event,
                    }
                for tool_call in delta.get("tool_calls", ()) or ():
                    function = tool_call.get("function", {})
                    yield {
                        "type": "tool_call_delta",
                        "request_id": request.request_id,
                        "index": int(tool_call.get("index", 0)),
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": str(function.get("arguments", "")),
                        "raw": event,
                    }
        yield {
            "type": "done",
            "request_id": request.request_id,
            "raw": last_event,
            "finish_reason": finish_reason,
        }

    def close_session(self, session_id: str) -> None:
        return None


class HuggingFaceEngineAdapter:
    """E1 in-process adapter using the HF semantic-reference implementation."""

    def __init__(self, model, *, storage_manager=None, observability: Observability | None = None) -> None:
        self.model = model
        self.storage = storage_manager
        self.observability = observability or DISABLED_OBSERVABILITY
        self._reference_lock = threading.RLock()

    def _logical_key(self, request: PRAWireRequest, resource: PRAWireResource) -> str:
        shareable = bool(resource.metadata.get("shareable", False))
        identity = {
            "tenant": request.tenant_id,
            "session": None if shareable else request.session_id,
            "uri": resource.uri,
            "version": resource.metadata.get("version"),
            "text_sha256": hashlib.sha256((resource.text or "").encode()).hexdigest(),
            "model": getattr(self.model.model.config, "_name_or_path", "hf-model"),
            "pra": self.model.config.to_dict(),
        }
        return "hf:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _add_stored_reference(
        self, request: PRAWireRequest, resource: PRAWireResource, key: str
    ):
        from .hf_storage import serialize_reference
        from .storage_lifecycle import PRARetentionClass, PRAStorageEntry

        handle = self.model.add_reference(resource.uri, text=resource.text)
        payload = serialize_reference(self.model, handle.uri)
        entry = PRAStorageEntry(
            logical_key=key,
            record_type=resource.record_type,
            retention_class=PRARetentionClass.RECONSTRUCTABLE,
            tenant_id=request.tenant_id,
            session_id=(
                None if resource.metadata.get("shareable", False) else request.session_id
            ),
            task_id=(
                None
                if resource.metadata.get("task_id") is None
                else str(resource.metadata["task_id"])
            ),
            task_status=(
                None
                if resource.metadata.get("task_status") is None
                else str(resource.metadata["task_status"])
            ),
            resource_version=str(resource.metadata.get("version", "source-hash")),
            detail_bytes=len(payload),
            security_scope=resource.authorization_scope,
            source_reconstructable=resource.text is not None,
            reconstruction_cost_ms=float(
                resource.metadata.get("reconstruction_cost_ms", 0.0)
            ),
            shared_reference_count=int(bool(resource.metadata.get("shareable", False))),
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "model": getattr(
                        self.model.model.config, "_name_or_path", "hf-model"
                    ),
                    "pra": self.model.config.to_dict(),
                    "resource_version": entry.resource_version,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        self.storage.register(
            entry,
            payload,
            hot_value=handle,
            source_loader=lambda payload=payload: payload,
            fingerprint=fingerprint,
        )
        return handle

    @contextmanager
    def _request_references(self, request: PRAWireRequest):
        """Expose only this request's authorized native references to routing."""

        with self._reference_lock:
            handles = []
            keys = []
            try:
                for resource in request.resources:
                    if not resource.text:
                        continue
                    if self.storage is None:
                        handles.append(
                            self.model.add_reference(resource.uri, text=resource.text)
                        )
                        continue
                    key = self._logical_key(request, resource)
                    keys.append(key)
                    if (
                        key not in self.storage.entries
                        or self.storage.entries[key].current_tier.value == "source"
                    ):
                        handle = self._add_stored_reference(request, resource, key)
                    else:
                        handle = self.storage.promote(
                            key,
                            tenant_id=request.tenant_id,
                            authorization_scopes=request.metadata.get(
                                "authorization_scopes", ()
                            ),
                        )
                    handles.append(handle)
                    self.storage.record_access(key, selected=True)
                if self.storage is None or not keys:
                    yield handles
                else:
                    with self.storage.pin_request(
                        request.request_id,
                        keys,
                        tenant_id=request.tenant_id,
                        authorization_scopes=request.metadata.get(
                            "authorization_scopes", ()
                        ),
                    ):
                        yield handles
            finally:
                if self.storage is None:
                    for handle in handles:
                        self.model.remove_reference(handle)
                else:
                    from .hf_storage import serialize_reference
                    from .storage_lifecycle import PRAStorageTier

                    for key in keys:
                        entry = self.storage.entries.get(key)
                        if entry is None or entry.current_tier != PRAStorageTier.HOT:
                            continue
                        handle = self.storage.hot.get_hot(key)
                        payload = serialize_reference(self.model, handle.uri)
                        self.storage.demote_hot(key, payload=payload)

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter="huggingface_eager",
            engine_type=EngineType.HUGGINGFACE,
            integration_level="E2",
            prefix_cache_mode=PrefixCacheMode.STATELESS,
            logical_refs=True,
            typed_records=True,
            task_metadata=True,
            text_fallback=True,
            native_kv=True,
            external_kv_residency=True,
            cpu_kv=True,
            gpu_kv=True,
            semantic_cache=True,
            streaming=True,
            selected_interval_materialization=True,
            request_lifetime=True,
            phase_selection=True,
            host_device_residency=True,
            scheduler_hints=False,
            tenant_isolation=True,
            tool_resources=True,
            observability=engine_observability_capabilities("hf"),
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        with self._request_references(request):
            text = self.model.chat(
                request.messages,
                return_details=True,
                pra_policy=request.pra_policy or None,
                max_new_tokens=request.resolved_max_new_tokens,
            )
            return PRAEngineResult(
                text.text,
                {"stats": text.stats},
                ({"stage": "engine_native_materialization", "native_kv": True},),
            )

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, Any]]:
        """Stream HF text deltas while retaining native references for the request.

        Cancellation is cooperative at the token boundary. The generator's
        ``finally`` block waits for the worker before releasing request-owned
        reference handles, so native K/V cannot disappear during decode.
        """

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class _Cancelled(StoppingCriteria):
            def __init__(self, event: threading.Event) -> None:
                self.event = event

            def __call__(self, input_ids, scores, **kwargs):
                del scores, kwargs
                return torch.full(
                    (input_ids.shape[0],), self.event.is_set(), device=input_ids.device
                )

        cancel = threading.Event()
        errors: list[BaseException] = []
        streamer = TextIteratorStreamer(
            self.model.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
        )
        with self._request_references(request):
            def run() -> None:
                try:
                    self.model.chat(
                        request.messages,
                        pra_policy=request.pra_policy or None,
                        max_new_tokens=request.resolved_max_new_tokens,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_Cancelled(cancel)]),
                    )
                except BaseException as error:  # propagated on the consumer thread
                    errors.append(error)
                    streamer.on_finalized_text("", stream_end=True)

            worker = threading.Thread(target=run, name=f"pra-hf-{request.request_id}", daemon=True)
            worker.start()
            index = 0
            try:
                for delta in streamer:
                    if delta:
                        yield {
                            "type": "delta",
                            "index": index,
                            "text": delta,
                            "request_id": request.request_id,
                            "session_id": request.session_id,
                        }
                        index += 1
                worker.join()
                if errors:
                    raise errors[0]
                yield {
                    "type": "done",
                    "request_id": request.request_id,
                    "session_id": request.session_id,
                    "trace": {"stage": "engine_native_stream", "native_kv": True},
                }
            finally:
                cancel.set()
                worker.join()

    def close_session(self, session_id: str) -> None:
        if self.storage is not None:
            self.storage.close_session(session_id)


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
