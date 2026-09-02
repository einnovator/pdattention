"""Open management API for one PRA gateway and its upstream transport state.

The gateway contract intentionally differs from the engine-management API: it
owns upstream selection, capability handshakes, session realization, transport
fallbacks, and gateway policy, but never engine scheduler internals.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import socket
import ssl
import threading
import time
import urllib.request
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from fastapi import Depends, FastAPI, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAGatewayMode,
    PRAWireRequest,
)
from .engine_profiles import EngineType
from .gateway import FallbackInjectionPolicy, PRAGateway
from .management import (
    Actor,
    AuthMode,
    ManagementAPIError,
    ManagementAuthConfig,
    _Authenticator,
    _is_loopback,
    _redact,
)


GATEWAY_MANAGEMENT_PROTOCOL = "pra-gateway-management/1"
GATEWAY_API_PREFIX = "/v1/pra/gateway"
GATEWAY_SCOPES = (
    "pra-gateway:read",
    "pra-gateway:configure",
    "pra-gateway:sessions",
    "pra-gateway:upstreams",
    "pra-gateway:admin",
)


class GatewayManagementAuthConfig(ManagementAuthConfig):
    """Gateway-scoped credentials; values are never serialized by the API."""

    scopes: tuple[str, ...] = GATEWAY_SCOPES


class GatewayRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    url: str | None = None
    token_env: str | None = None
    deployment_id: str | None = None
    model_id: str | None = None
    environment: str = "development"
    cluster: str = "gateway"
    heartbeat_seconds: int = Field(default=30, ge=5, le=3600)

    def token(self) -> str | None:
        return os.environ.get(self.token_env or "") or None

    def validate_registration(self) -> None:
        if self.enabled and not all((self.url, self.deployment_id, self.model_id)):
            raise ValueError("gateway Registry registration requires url, deployment_id, and model_id")


class GatewayManagementAPIConfig(BaseModel):
    """Separate listener configuration; disabled is the default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=9150, ge=1, le=65535)
    environment: str = "development"
    auth: GatewayManagementAuthConfig = Field(default_factory=GatewayManagementAuthConfig)
    registry: GatewayRegistryConfig = Field(default_factory=GatewayRegistryConfig)
    metrics_url: str | None = None
    trace_backend_url: str | None = None
    grafana_url: str | None = None
    tls_certfile: str | None = None
    tls_keyfile: str | None = None
    tls_ca_certs: str | None = None

    @field_validator("host")
    @classmethod
    def nonempty_host(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Gateway management host cannot be empty")
        return value

    def validate_binding(self) -> None:
        if self.enabled and self.auth.mode == AuthMode.NONE and not _is_loopback(self.host):
            raise ValueError("unauthenticated gateway management may bind only to loopback")
        if self.enabled and self.auth.mode == AuthMode.STATIC_BEARER and not self.auth.resolved_token():
            raise ValueError("static bearer authentication requires a token or token_env")
        if self.enabled and self.auth.mode == AuthMode.MTLS and not all((self.tls_certfile, self.tls_keyfile, self.tls_ca_certs)):
            raise ValueError("mTLS requires certificate, key, and CA files")
        self.registry.validate_registration()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> "GatewayManagementAPIConfig":
        data = dict(value or {})
        auth = dict(data.get("auth") or {})
        registry = dict(data.get("registry") or {})
        scalar = {
            "PRA_GATEWAY_MANAGEMENT_ENABLED": ("enabled", _bool),
            "PRA_GATEWAY_MANAGEMENT_HOST": ("host", str),
            "PRA_GATEWAY_MANAGEMENT_PORT": ("port", int),
        }
        for name, (field, cast) in scalar.items():
            if os.environ.get(name) is not None:
                data[field] = cast(os.environ[name])
        if os.environ.get("PRA_GATEWAY_MANAGEMENT_AUTH_MODE"):
            auth["mode"] = os.environ["PRA_GATEWAY_MANAGEMENT_AUTH_MODE"]
        if os.environ.get("PRA_GATEWAY_MANAGEMENT_TOKEN"):
            auth["token"] = os.environ["PRA_GATEWAY_MANAGEMENT_TOKEN"]
        if os.environ.get("PRA_GATEWAY_REGISTRY_URL"):
            registry.update({"enabled": True, "url": os.environ["PRA_GATEWAY_REGISTRY_URL"]})
        if os.environ.get("PRA_GATEWAY_REGISTRY_TOKEN_ENV"):
            registry["token_env"] = os.environ["PRA_GATEWAY_REGISTRY_TOKEN_ENV"]
        if os.environ.get("PRA_GATEWAY_REGISTRY_DEPLOYMENT"):
            registry["deployment_id"] = os.environ["PRA_GATEWAY_REGISTRY_DEPLOYMENT"]
        if os.environ.get("PRA_GATEWAY_REGISTRY_MODEL"):
            registry["model_id"] = os.environ["PRA_GATEWAY_REGISTRY_MODEL"]
        if auth:
            data["auth"] = auth
        if registry:
            data["registry"] = registry
        return cls.model_validate(data)


class NegotiatedCapability(BaseModel):
    upstream_id: str
    protocol: str | None = None
    protocol_version: str | None = None
    selected_context: str = "unknown"
    typed_transport: str = "unknown"
    native_memory: str = "unknown"
    native_serving: str = "unknown"
    storage_lifecycle: str = "unknown"
    model_fingerprint: str | None = None
    bundle_profile_compatibility: str = "unknown"
    last_negotiated_at: float | None = None
    expires_at: float | None = None
    fallback_state: str | None = None
    rejection_reason: str | None = None


class UpstreamCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upstream_id: str = Field(min_length=1, max_length=255)
    name: str
    base_url: str
    provider: str = "openai"
    inference_api_type: str = "openai-compatible"
    management_url: str | None = None
    models: tuple[str, ...] = ()
    auth_reference: str | None = None
    priority: int = 100
    weight: float = Field(default=1.0, gt=0)
    enabled: bool = True
    labels: Mapping[str, str] = Field(default_factory=dict)


class UpstreamPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    base_url: str | None = None
    management_url: str | None = None
    models: tuple[str, ...] | None = None
    auth_reference: str | None = None
    priority: int | None = None
    weight: float | None = Field(default=None, gt=0)
    enabled: bool | None = None
    labels: Mapping[str, str] | None = None


class UpstreamEndpoint(BaseModel):
    upstream_id: str
    name: str
    base_url: str
    provider: str
    inference_api_type: str
    management_url: str | None
    models: tuple[str, ...]
    execution_modes: tuple[str, ...]
    auth_reference: str | None
    health: str
    last_health_check: float | None
    negotiated: NegotiatedCapability
    latency_ms: float | None
    error_count: int
    priority: int
    weight: float
    enabled: bool
    labels: Mapping[str, str]


class GatewayPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upstream_selection: str = Field(default="static", pattern="^(static|model|capability|tenant|weighted|failover)$")
    default_upstream_id: str | None = None
    fallback_policy: str = "selected-context"
    routing_policy: Mapping[str, Any] = Field(default_factory=dict)
    session_affinity: bool = True
    negotiate_on_start: bool = True
    capability_ttl_seconds: int = Field(default=300, ge=5, le=86_400)
    timeout_seconds: float = Field(default=120.0, gt=0)
    retry_count: int = Field(default=1, ge=0, le=10)
    authorization_policy: Mapping[str, Any] = Field(default_factory=dict)
    observability_policy: Mapping[str, Any] = Field(default_factory=dict)


class GatewayConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_profile: str | None = None
    mode: str | None = None
    fallback_injection: FallbackInjectionPolicy | None = None
    policy: GatewayPolicy | None = None

    @field_validator("mode")
    @classmethod
    def valid_gateway_mode(cls, value: str | None) -> str | None:
        if value is not None:
            _gateway_mode_code(value)
        return value


class GatewayActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="management action", min_length=1)
    idempotency_key: str | None = None


class GatewayAuditEvent(BaseModel):
    timestamp: float
    event: str
    actor: str
    request_id: str
    trace_id: str | None
    result: str
    before: Any = None
    after: Any = None
    reason: str


@dataclass
class _ManagedUpstream:
    config: UpstreamCreate
    adapter: PRAEngineAdapter
    health: str = "unknown"
    last_health_check: float | None = None
    negotiated: NegotiatedCapability | None = None
    latency_ms: float | None = None
    error_count: int = 0


class GatewayUpstreamRouter:
    """Route requests across live upstream adapters using bounded policies."""

    def __init__(self, initial: UpstreamCreate, adapter: PRAEngineAdapter, policy: GatewayPolicy | None = None) -> None:
        self.policy = policy or GatewayPolicy(default_upstream_id=initial.upstream_id)
        self._rows = {initial.upstream_id: _ManagedUpstream(initial, adapter)}
        self._sessions: dict[str, str] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> PRAEngineCapabilities:
        return self._default().adapter.capabilities()

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        row = self._select(request)
        value = row.adapter.prepare_session(request)
        if request.session_id:
            self._sessions[request.session_id] = row.config.upstream_id
        return value

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        selected = self._select(request)
        rows = self._candidates(request)
        if selected in rows:
            rows.remove(selected)
        rows.insert(0, selected)
        last_error: Exception | None = None
        for row in rows:
            started = time.perf_counter()
            try:
                result = row.adapter.generate(request)
                row.health = "healthy"
                row.latency_ms = (time.perf_counter() - started) * 1000
                return result
            except Exception as error:
                row.error_count += 1
                row.health = "degraded"
                last_error = error
                if self.policy.upstream_selection != "failover":
                    raise
        assert last_error is not None
        raise last_error

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, Any]]:
        row = self._select(request)
        if request.session_id:
            self._sessions[request.session_id] = row.config.upstream_id
        return row.adapter.stream(request)

    def close_session(self, session_id: str) -> None:
        upstream_id = self._sessions.pop(session_id, None)
        row = self._rows.get(upstream_id) if upstream_id else None
        (row or self._default()).adapter.close_session(session_id)

    def session_upstream(self, session_id: str) -> str | None:
        """Return the affinity target without exposing tenant or message data."""

        return self._sessions.get(session_id)

    def rows(self) -> list[UpstreamEndpoint]:
        with self._lock:
            return [self._public(row) for _, row in sorted(self._rows.items())]

    def row(self, upstream_id: str) -> UpstreamEndpoint:
        return self._public(self._get(upstream_id))

    def add(self, value: UpstreamCreate, adapter: PRAEngineAdapter | None = None) -> UpstreamEndpoint:
        with self._lock:
            if value.upstream_id in self._rows:
                raise ManagementAPIError(409, "UPSTREAM_EXISTS", "Upstream identity already exists")
            self._rows[value.upstream_id] = _ManagedUpstream(value, adapter or _remote_adapter(value))
            return self._public(self._rows[value.upstream_id])

    def patch(self, upstream_id: str, patch: UpstreamPatch) -> tuple[UpstreamEndpoint, UpstreamEndpoint]:
        with self._lock:
            row = self._get(upstream_id)
            before = self._public(row)
            values = {**row.config.model_dump(), **patch.model_dump(exclude_none=True)}
            updated = UpstreamCreate.model_validate(values)
            rebuild = updated.base_url != row.config.base_url or updated.provider != row.config.provider
            row.config = updated
            if rebuild:
                row.adapter = _remote_adapter(updated)
                row.negotiated = None
                row.health = "unknown"
            return before, self._public(row)

    def remove(self, upstream_id: str) -> UpstreamEndpoint:
        with self._lock:
            if len(self._rows) == 1:
                raise ManagementAPIError(409, "LAST_UPSTREAM", "The active gateway requires at least one upstream")
            row = self._get(upstream_id)
            del self._rows[upstream_id]
            if self.policy.default_upstream_id == upstream_id:
                self.policy = self.policy.model_copy(update={"default_upstream_id": next(iter(self._rows))})
            return self._public(row)

    def health_check(self, upstream_id: str) -> UpstreamEndpoint:
        row = self._get(upstream_id)
        started = time.perf_counter()
        request = urllib.request.Request(f"{row.config.base_url.rstrip('/')}/health", headers=_upstream_headers(row.config))
        try:
            with urllib.request.urlopen(request, timeout=min(self.policy.timeout_seconds, 10)) as response:
                row.health = "healthy" if response.status < 400 else "degraded"
        except Exception:
            row.health = "offline"
            row.error_count += 1
        row.last_health_check = time.time()
        row.latency_ms = (time.perf_counter() - started) * 1000
        return self._public(row)

    def negotiate(self, upstream_id: str) -> NegotiatedCapability:
        row = self._get(upstream_id)
        now = time.time()
        capabilities: Mapping[str, Any]
        protocol = None
        rejection = None
        for path in ("/v1/pra/gateway/capabilities", "/v1/pra/capabilities"):
            try:
                request = urllib.request.Request(f"{row.config.management_url or row.config.base_url}{path}", headers=_upstream_headers(row.config))
                with urllib.request.urlopen(request, timeout=min(self.policy.timeout_seconds, 10)) as response:
                    capabilities = json.loads(response.read().decode("utf-8"))
                protocol = str(capabilities.get("protocol") or capabilities.get("management_api_version") or "pra")
                break
            except Exception as error:
                rejection = str(error)
        else:
            capabilities = row.adapter.capabilities().to_dict()
            protocol = "adapter-local"
        effective = capabilities.get("effective_capabilities", capabilities)
        integration = str(effective.get("integration_level", capabilities.get("integration_level", "")))
        row.negotiated = NegotiatedCapability(
            upstream_id=upstream_id, protocol=protocol, protocol_version="1",
            selected_context=_status(effective.get("text_fallback", True)),
            typed_transport=_status(effective.get("typed_records")),
            native_memory=_status(effective.get("native_kv", effective.get("native_memory"))),
            native_serving=_status(effective.get("native_serving") or integration in {"E2", "E3"}),
            storage_lifecycle=_status(effective.get("storage_lifecycle")),
            model_fingerprint=effective.get("model_fingerprint"),
            bundle_profile_compatibility=str(effective.get("compatibility", "unknown")),
            last_negotiated_at=now, expires_at=now + self.policy.capability_ttl_seconds,
            fallback_state="selected-context" if not effective.get("native_kv") else None,
            rejection_reason=rejection,
        )
        return row.negotiated

    def clear_capabilities(self) -> int:
        count = 0
        for row in self._rows.values():
            count += int(row.negotiated is not None)
            row.negotiated = None
        return count

    def _default(self) -> _ManagedUpstream:
        value = self._rows.get(self.policy.default_upstream_id or "")
        return value or sorted(self._rows.values(), key=lambda row: (row.config.priority, row.config.upstream_id))[0]

    def _get(self, upstream_id: str) -> _ManagedUpstream:
        try:
            return self._rows[upstream_id]
        except KeyError as error:
            raise ManagementAPIError(404, "UPSTREAM_NOT_FOUND", "Gateway upstream was not found") from error

    def _select(self, request: PRAWireRequest) -> _ManagedUpstream:
        if self.policy.session_affinity and request.session_id and request.session_id in self._sessions:
            row = self._rows.get(self._sessions[request.session_id])
            if row and row.config.enabled:
                return row
        return self._candidates(request)[0]

    def _candidates(self, request: PRAWireRequest) -> list[_ManagedUpstream]:
        rows = [row for row in self._rows.values() if row.config.enabled]
        if not rows:
            raise ManagementAPIError(503, "NO_UPSTREAM", "No enabled upstream is available")
        mode = self.policy.upstream_selection
        if mode == "model":
            matched = [row for row in rows if not row.config.models or request.model in row.config.models]
            rows = matched or rows
        elif mode == "capability":
            matched = [row for row in rows if all(row.adapter.capabilities().supports(name) for name in request.required_capabilities)]
            rows = matched or rows
        elif mode == "tenant":
            matched = [row for row in rows if row.config.labels.get("tenant") in {None, request.tenant_id}]
            rows = matched or rows
        elif mode == "weighted":
            seed = int(hashlib.sha256(request.request_id.encode()).hexdigest()[:12], 16)
            expanded = [row for row in rows for _ in range(max(1, int(row.config.weight * 10)))]
            rows = [expanded[seed % len(expanded)]]
        rows.sort(key=lambda row: (row.config.priority, row.config.upstream_id))
        default = self._default()
        if mode == "static" and default.config.enabled:
            return [default]
        if mode == "failover" and default in rows:
            rows.remove(default)
            rows.insert(0, default)
        return rows

    @staticmethod
    def _public(row: _ManagedUpstream) -> UpstreamEndpoint:
        negotiated = row.negotiated or NegotiatedCapability(upstream_id=row.config.upstream_id)
        capabilities = row.adapter.capabilities()
        execution_modes = ["passthrough"]
        if capabilities.text_fallback:
            execution_modes.append("selected-context")
        if capabilities.logical_refs or capabilities.typed_records:
            execution_modes.extend(("upgrade", "typed-transport"))
        return UpstreamEndpoint(
            **row.config.model_dump(), health=row.health, last_health_check=row.last_health_check,
            negotiated=negotiated, latency_ms=row.latency_ms, error_count=row.error_count,
            execution_modes=tuple(execution_modes),
        )


class GatewayMetricRecorder:
    """Observe gateway metrics while forwarding them to the configured exporter."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def increment(self, name: str, value: float = 1, **kwargs: Any) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + float(value)
            aliases = {
                "pra_context_visible_reuse_tokens_total": "pra_gateway_visible_reuse_tokens_total",
                "pra_context_new_materialized_tokens_total": "pra_gateway_new_materialized_tokens_total",
            }
            alias = aliases.get(name)
            if alias:
                self.counters[alias] = self.counters.get(alias, 0) + float(value)
        self.delegate.increment(name, value, **kwargs)
        if alias:
            self.delegate.increment(alias, value)

    def set_gauge(self, name: str, value: float, **kwargs: Any) -> None:
        with self._lock:
            self.gauges[name] = float(value)
        self.delegate.set_gauge(name, value, **kwargs)

    def observe(self, name: str, value: float, **kwargs: Any) -> None:
        with self._lock:
            row = self.histograms.setdefault(name, {"count": 0, "sum": 0, "last": 0})
            row["count"] += 1
            row["sum"] += float(value)
            row["last"] = float(value)
        self.delegate.observe(name, value, **kwargs)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"counters": dict(self.counters), "gauges": dict(self.gauges), "histograms": {key: dict(value) for key, value in self.histograms.items()}}


class GatewayManagementProvider:
    """Project live gateway state into privacy-safe API resources and actions."""

    def __init__(
        self, gateway: PRAGateway, upstreams: GatewayUpstreamRouter,
        settings: GatewayManagementAPIConfig, metrics: GatewayMetricRecorder,
        *, gateway_id: str | None = None,
        policy_loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.gateway = gateway
        self.upstreams = upstreams
        self.settings = settings
        self.metrics = metrics
        self.gateway_id = gateway_id or uuid.uuid4().hex
        self.started_at = time.time()
        self.policy = upstreams.policy
        self.audit: deque[GatewayAuditEvent] = deque(maxlen=1000)
        self.registry_status: dict[str, Any] = {"enabled": settings.registry.enabled, "status": "not_started"}
        self._idempotency: dict[tuple[str, str], Any] = {}
        self.policy_loader = policy_loader

    def info(self) -> dict[str, Any]:
        return {
            "protocol": GATEWAY_MANAGEMENT_PROTOCOL,
            "gateway_id": self.gateway_id, "pra_version": _version("pra-hf"),
            "gateway_version": _version("pra-hf"), "host": socket.gethostname(),
            "process_id": os.getpid(), "started_at": self.started_at, "health": "healthy",
            "environment": self.settings.environment, "registry_registration": self.registry_status,
            "observability": self.observability(),
        }

    def state(self) -> dict[str, Any]:
        return {
            "protocol": GATEWAY_MANAGEMENT_PROTOCOL, "gateway": self.info(),
            "upstream_count": len(self.upstreams.rows()), "session_count": len(self.sessions()),
            "resource_count": len(self.resources()), "transport": self.transport(),
            "policy": self.policy.model_dump(mode="json"),
        }

    def config(self) -> dict[str, Any]:
        return _redact({
            "default_profile": self.gateway.default_profile,
            "mode": _gateway_mode_name(self.gateway.mode.value),
            "fallback_injection": self.gateway.fallback_injection.value,
            "policy": self.policy.model_dump(mode="json"),
            "management": self.settings.model_dump(exclude={"auth", "registry"}, mode="json"),
            "auth": self.settings.auth.public_dict(),
            "registry": {**self.settings.registry.model_dump(exclude={"token_env"}), "token_configured": bool(self.settings.registry.token())},
        })

    def patch_config(
        self, patch: GatewayConfigPatch, actor: Actor, request_id: str,
        reason: str, trace_id: str | None = None,
    ) -> dict[str, Any]:
        before = self.config()
        if patch.default_profile is not None:
            self.gateway.default_profile = patch.default_profile
        if patch.mode is not None:
            self.gateway.mode = PRAGatewayMode(_gateway_mode_code(patch.mode))
        if patch.fallback_injection is not None:
            self.gateway.fallback_injection = patch.fallback_injection
        if patch.policy is not None:
            self.policy = patch.policy
            self.upstreams.policy = patch.policy
        after = self.config()
        event = "FALLBACK_POLICY_CHANGED" if patch.fallback_injection is not None else "POLICY_CHANGED"
        self.record(event, actor, request_id, "success", before, after, reason, trace_id)
        return after

    def sessions(self) -> list[dict[str, Any]]:
        return [self._session(row) for row in self.gateway.sessions.inspect_all()]

    def session(self, safe_id: str) -> dict[str, Any]:
        raw = self._raw_session(safe_id)
        return self._session(raw)

    def resources(self) -> list[dict[str, Any]]:
        values: dict[tuple[str, str], dict[str, Any]] = {}
        for session in self.gateway.sessions.inspect_all():
            for resource_id, version in dict(session.get("known_resources", {})).items():
                metadata = dict(session.get("known_resource_metadata", {}).get(resource_id, {}))
                key = (resource_id, str(version))
                values[key] = {
                    "resource_id": _safe("resource", resource_id), "version": str(version),
                    "body_known": bool(metadata.get("body_known")),
                    "body_free_delta_eligible": bool(metadata.get("body_known")),
                    "record_type": metadata.get("record_type", "document"),
                    "size_bytes": metadata.get("size_bytes"), "token_count": metadata.get("token_count"),
                    "last_transmitted": metadata.get("last_transmitted"),
                    "last_acknowledged": metadata.get("last_acknowledged"),
                    "authorization": "restricted" if metadata.get("authorization_scope") else "default",
                }
        return list(values.values())

    def transport(self) -> dict[str, Any]:
        snapshot = self.metrics.snapshot()
        counters = snapshot["counters"]
        return {
            "requested_mode": _gateway_mode_name(self.gateway.mode.value),
            "resolved_mode": _gateway_mode_name(self.gateway.mode.value),
            "internal_transport": _transport_name(self.gateway.mode.value),
            "wire_bytes": counters.get("pra_gateway_transport_bytes_total", 0),
            "message_bytes": counters.get("pra_gateway_message_bytes_total", 0),
            "resource_bytes": counters.get("pra_gateway_resource_bytes_total", 0),
            "delta_bytes": counters.get("pra_gateway_delta_bytes_total", 0),
            "resync_count": counters.get("pra_gateway_resyncs_total", 0),
            "fallback_count": counters.get("pra_gateway_fallbacks_total", 0),
            "reconnect_count": counters.get("pra_gateway_reconnects_total", 0),
            "visible_reuse_tokens": counters.get("pra_gateway_visible_reuse_tokens_total", 0),
            "new_materialized_tokens": counters.get("pra_gateway_new_materialized_tokens_total", 0),
            "metrics": snapshot,
        }

    def observability(self) -> dict[str, Any]:
        return {
            "otel_enabled": bool(getattr(self.metrics, "tracing_enabled", False)),
            "prometheus_enabled": bool(getattr(self.metrics, "metrics_enabled", False)),
            "metrics_url": self.settings.metrics_url, "grafana_url": self.settings.grafana_url,
            "tempo_url": self.settings.trace_backend_url,
            "sampling": getattr(getattr(self.metrics.delegate, "config", None), "sampling", None),
        }

    def action(
        self, name: str, target: str | None, body: GatewayActionRequest,
        actor: Actor, request_id: str, trace_id: str | None = None,
    ) -> dict[str, Any]:
        cache_key = (actor.identity, body.idempotency_key or "")
        if body.idempotency_key and cache_key in self._idempotency:
            return {**self._idempotency[cache_key], "idempotent_replay": True}
        before: Any = None
        if name == "renegotiate":
            before = self.upstreams.row(target or "").negotiated.model_dump(mode="json")
            after = self.upstreams.negotiate(target or "").model_dump(mode="json")
            event = "CAPABILITY_RENEGOTIATED"
            self.metrics.increment("pra_gateway_capability_negotiations_total")
        elif name == "health-check":
            before = self.upstreams.row(target or "").model_dump(mode="json")
            after = self.upstreams.health_check(target or "").model_dump(mode="json")
            event = "UPSTREAM_HEALTH_CHECKED"
        elif name in {"resync-session", "drop-session"}:
            raw = self._raw_session(target or "")
            before = self._session(raw)
            if name == "resync-session":
                self.gateway.sessions.invalidate(raw["tenant_id"], raw["session_id"], raw["model"], "management_resync")
                self.metrics.increment("pra_gateway_resyncs_total")
                after = self.session(target or "")
                event = "SESSION_RESYNCED"
            else:
                self.gateway.close_session(raw["tenant_id"], raw["session_id"], raw["model"])
                after = {"dropped": True}
                event = "SESSION_DROPPED"
        elif name == "clear-capability-cache":
            after = {"cleared": self.upstreams.clear_capabilities()}
            event = "CAPABILITY_CACHE_CLEARED"
        elif name == "reload-policy":
            reloaded = self.policy_loader is not None
            if self.policy_loader is not None:
                values = dict(self.policy_loader())
                if values:
                    self.policy = GatewayPolicy.model_validate(values)
                    self.upstreams.policy = self.policy
            after = {"policy": self.policy.model_dump(mode="json"), "reloaded": reloaded}
            event = "POLICY_CHANGED"
        else:
            raise ManagementAPIError(404, "ACTION_NOT_FOUND", "Gateway action is not supported")
        result = {"action": name, "status": "success", "target": target, "detail": after, "idempotent_replay": False}
        if body.idempotency_key:
            self._idempotency[cache_key] = result
        self.record(event, actor, request_id, "success", before, after, body.reason, trace_id)
        return result

    def record(self, event: str, actor: Actor, request_id: str, result: str, before: Any, after: Any, reason: str, trace_id: str | None = None) -> None:
        self.audit.append(GatewayAuditEvent(
            timestamp=time.time(), event=event, actor=actor.identity, request_id=request_id,
            trace_id=trace_id, result=result, before=_redact(before), after=_redact(after), reason=reason,
        ))

    def _raw_session(self, safe_id: str) -> dict[str, Any]:
        for row in self.gateway.sessions.inspect_all():
            if _safe("session", f"{row.get('tenant_id')}:{row.get('session_id')}:{row.get('model')}") == safe_id:
                return row
        raise ManagementAPIError(404, "SESSION_NOT_FOUND", "Gateway session was not found")

    def _session(self, row: Mapping[str, Any]) -> dict[str, Any]:
        materializations = list(row.get("visible_materializations", ()))
        selected = sum(int(item.get("token_count", 0)) for item in materializations)
        return {
            "session_id": _safe("session", f"{row.get('tenant_id')}:{row.get('session_id')}:{row.get('model')}"),
            "tenant_id": _safe("tenant", str(row.get("tenant_id"))),
            "upstream_id": self.upstreams.session_upstream(str(row.get("session_id"))),
            "model": row.get("model"),
            "model_fingerprint": row.get("engine_model_fingerprint"), "active_task_count": 0,
            "canonical_message_count": int(row.get("canonical_message_count", row.get("prefix_message_count", 0))),
            "serialized_message_count": int(row.get("serialized_message_count", row.get("prefix_message_count", 0))),
            "prefix_digest": row.get("prefix_digest"), "prefix_message_count": int(row.get("prefix_message_count", 0)),
            "prefix_token_count": row.get("prefix_token_count"),
            "visible_materialization": {"count": len(materializations), "selected_tokens": selected},
            "known_resource_count": len(row.get("known_resources", {})),
            "transport_mode": row.get("outbound_history_mode", "session-aware"),
            "last_activity": row.get("updated_at"), "connection_status": "active",
            "visible_reuse_tokens": selected if int(row.get("turns", 0)) > 1 else 0,
            "new_materialized_tokens": selected, "prefix_digest_stable": row.get("last_invalidation_reason") is None,
        }


class GatewayRegistryReporter:
    """Register and heartbeat gateway state through the open Registry API."""

    def __init__(self, provider: GatewayManagementProvider) -> None:
        self.provider = provider
        self.config = provider.settings.registry
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.enabled:
            return
        self.thread = threading.Thread(target=self._run, name="pra-gateway-registry", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.config.enabled:
            return
        self.stop_event.set()
        self._publish("offline")
        if self.thread:
            self.thread.join(timeout=3)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._publish("healthy")
            self.stop_event.wait(self.config.heartbeat_seconds)

    def _publish(self, health: str) -> None:
        selector = {
            "kind": "gateway", "name": self.provider.gateway_id,
            "management_url": (
                f"http://{self.provider.settings.host}:{self.provider.settings.port}"
                if self.provider.settings.enabled else None
            ),
            "health": health, "protocol": GATEWAY_MANAGEMENT_PROTOCOL,
            "upstreams": [{"id": row.upstream_id, "models": list(row.models), "health": row.health} for row in self.provider.upstreams.rows()],
            "capabilities": [row.negotiated.model_dump(mode="json") for row in self.provider.upstreams.rows()],
            "last_heartbeat": time.time(),
        }
        patch = {"engine_instance_selector": selector, "desired_model_id": self.config.model_id}
        path = f"/v1/deployments/{self.config.deployment_id}"
        try:
            self._request("PATCH", path, patch)
        except Exception:
            create = {
                "id": self.config.deployment_id, "owner": "pra-gateway", "environment": self.config.environment,
                "cluster": self.config.cluster, "engine_instance_selector": selector,
                "desired_model_id": self.config.model_id, "desired_mode": "selected-context",
            }
            try:
                self._request("POST", "/v1/deployments", create)
            except Exception as error:
                self.provider.registry_status = {"enabled": True, "status": "error", "detail": str(error), "last_attempt": time.time()}
                return
        self.provider.registry_status = {"enabled": True, "status": health, "deployment_id": self.config.deployment_id, "last_heartbeat": time.time()}

    def _request(self, method: str, path: str, body: Mapping[str, Any]) -> Any:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.token():
            headers["Authorization"] = f"Bearer {self.config.token()}"
        request = urllib.request.Request(
            f"{self.config.url.rstrip('/')}{path}", data=json.dumps(body).encode(), headers=headers, method=method,
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode())


def create_gateway_management_app(provider: GatewayManagementProvider, settings: GatewayManagementAPIConfig) -> FastAPI:
    """Build the enabled gateway API without starting its separate listener."""
    settings.validate_binding()
    reporter = GatewayRegistryReporter(provider)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        reporter.start()
        try:
            yield
        finally:
            reporter.stop()

    app = FastAPI(
        title="PRA Gateway Management API",
        summary="Open local management API for gateway upstream, transport, and session state.",
        description="This open API manages one PRA gateway. It is not the Enterprise Control Plane.",
        version="1.0.0", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    authenticator = _Authenticator(settings.auth)
    bearer = HTTPBearer(auto_error=False)

    @app.exception_handler(ManagementAPIError)
    async def api_error(_: Request, error: ManagementAPIError):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "detail": error.detail, **error.extra}},
        )

    def require(*scopes: str):
        async def dependency(request: Request, _: HTTPAuthorizationCredentials | None = Security(bearer)) -> Actor:
            actor = authenticator.authorize(request, ())
            if "pra-gateway:admin" not in actor.scopes and not set(scopes).issubset(actor.scopes):
                raise ManagementAPIError(403, "INSUFFICIENT_SCOPE", "Required gateway-management scope is missing")
            return actor
        return dependency

    def request_id(request: Request) -> str:
        return request.headers.get("x-request-id") or uuid.uuid4().hex

    def reason(request: Request, fallback: str) -> str:
        return request.headers.get("x-pra-reason") or fallback

    def trace_id(request: Request) -> str | None:
        return request.headers.get("x-trace-id") or request.headers.get("traceparent")

    def run_action(
        name: str, target: str | None, body: GatewayActionRequest,
        request: Request, actor: Actor,
    ) -> dict[str, Any]:
        events = {
            "renegotiate": "CAPABILITY_RENEGOTIATED",
            "health-check": "UPSTREAM_HEALTH_CHECKED",
            "resync-session": "SESSION_RESYNCED",
            "drop-session": "SESSION_DROPPED",
            "clear-capability-cache": "CAPABILITY_CACHE_CLEARED",
            "reload-policy": "POLICY_CHANGED",
        }
        try:
            return provider.action(
                name, target, body, actor, request_id(request), trace_id(request)
            )
        except Exception as error:
            provider.record(
                events[name], actor, request_id(request), "failure", None,
                {"error": type(error).__name__}, body.reason, trace_id(request),
            )
            raise

    @app.get(f"{GATEWAY_API_PREFIX}/health", tags=["gateway"])
    def health(_: Actor = Depends(require("pra-gateway:read"))):
        return {"status": "healthy", "protocol": GATEWAY_MANAGEMENT_PROTOCOL, "gateway_id": provider.gateway_id}

    @app.get(f"{GATEWAY_API_PREFIX}/info", tags=["gateway"])
    def info(_: Actor = Depends(require("pra-gateway:read"))): return provider.info()

    @app.get(f"{GATEWAY_API_PREFIX}/capabilities", tags=["gateway"])
    def capabilities(_: Actor = Depends(require("pra-gateway:read"))):
        value = provider.gateway.capabilities()
        value["gateway_mode"] = _gateway_mode_name(provider.gateway.mode.value)
        value["gateway"] = {
            **value.get("gateway", {}), "mode": _gateway_mode_name(provider.gateway.mode.value)
        }
        return {**value, "protocol": GATEWAY_MANAGEMENT_PROTOCOL}

    @app.get(f"{GATEWAY_API_PREFIX}/config", tags=["configuration"])
    def config(_: Actor = Depends(require("pra-gateway:read"))): return provider.config()

    @app.patch(f"{GATEWAY_API_PREFIX}/config", tags=["configuration"])
    def patch_config(body: GatewayConfigPatch, request: Request, actor: Actor = Depends(require("pra-gateway:configure"))):
        return provider.patch_config(
            body, actor, request_id(request), reason(request, "gateway configuration update"),
            trace_id(request),
        )

    @app.get(f"{GATEWAY_API_PREFIX}/state", tags=["gateway"])
    def state(_: Actor = Depends(require("pra-gateway:read"))): return provider.state()

    @app.get(f"{GATEWAY_API_PREFIX}/upstreams", tags=["upstreams"])
    def upstreams(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), _: Actor = Depends(require("pra-gateway:read"))): return _page(provider.upstreams.rows(), offset, limit)

    @app.get(f"{GATEWAY_API_PREFIX}/upstreams/{{upstream_id}}", tags=["upstreams"])
    def upstream(upstream_id: str, _: Actor = Depends(require("pra-gateway:read"))): return provider.upstreams.row(upstream_id)

    @app.post(f"{GATEWAY_API_PREFIX}/upstreams", status_code=201, tags=["upstreams"])
    def add_upstream(body: UpstreamCreate, request: Request, actor: Actor = Depends(require("pra-gateway:upstreams"))):
        after = provider.upstreams.add(body)
        provider.record("UPSTREAM_ADDED", actor, request_id(request), "success", None, after.model_dump(mode="json"), reason(request, "upstream added"), trace_id(request))
        return after

    @app.patch(f"{GATEWAY_API_PREFIX}/upstreams/{{upstream_id}}", tags=["upstreams"])
    def patch_upstream(upstream_id: str, body: UpstreamPatch, request: Request, actor: Actor = Depends(require("pra-gateway:upstreams"))):
        before, after = provider.upstreams.patch(upstream_id, body)
        provider.record("UPSTREAM_CHANGED", actor, request_id(request), "success", before.model_dump(mode="json"), after.model_dump(mode="json"), reason(request, "upstream changed"), trace_id(request))
        return after

    @app.delete(f"{GATEWAY_API_PREFIX}/upstreams/{{upstream_id}}", tags=["upstreams"])
    def delete_upstream(upstream_id: str, request: Request, actor: Actor = Depends(require("pra-gateway:upstreams"))):
        before = provider.upstreams.remove(upstream_id)
        provider.record("UPSTREAM_REMOVED", actor, request_id(request), "success", before.model_dump(mode="json"), None, reason(request, "upstream removed"), trace_id(request))
        return {"removed": True, "upstream_id": upstream_id}

    @app.get(f"{GATEWAY_API_PREFIX}/sessions", tags=["sessions"])
    def sessions(model: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), _: Actor = Depends(require("pra-gateway:sessions"))):
        rows = provider.sessions()
        if model: rows = [row for row in rows if row["model"] == model]
        return _page(rows, offset, limit)

    @app.get(f"{GATEWAY_API_PREFIX}/sessions/{{session_id}}", tags=["sessions"])
    def session(session_id: str, _: Actor = Depends(require("pra-gateway:sessions"))): return provider.session(session_id)

    @app.get(f"{GATEWAY_API_PREFIX}/resources", tags=["resources"])
    def resources(record_type: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), _: Actor = Depends(require("pra-gateway:read"))):
        rows = provider.resources()
        if record_type: rows = [row for row in rows if row["record_type"] == record_type]
        return _page(rows, offset, limit)

    @app.get(f"{GATEWAY_API_PREFIX}/transport", tags=["transport"])
    def transport(_: Actor = Depends(require("pra-gateway:read"))): return provider.transport()

    @app.get(f"{GATEWAY_API_PREFIX}/observability", tags=["observability"])
    def observability(_: Actor = Depends(require("pra-gateway:read"))): return provider.observability()

    @app.get(f"{GATEWAY_API_PREFIX}/audit", tags=["audit"])
    def audit(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), _: Actor = Depends(require("pra-gateway:admin"))): return _page(list(reversed(provider.audit)), offset, limit)

    @app.post(f"{GATEWAY_API_PREFIX}/actions/renegotiate/{{upstream_id}}", tags=["actions"])
    def renegotiate(upstream_id: str, body: GatewayActionRequest, request: Request, actor: Actor = Depends(require("pra-gateway:upstreams"))):
        return run_action("renegotiate", upstream_id, body, request, actor)

    @app.post(f"{GATEWAY_API_PREFIX}/actions/health-check/{{upstream_id}}", tags=["actions"])
    def health_check(upstream_id: str, body: GatewayActionRequest, request: Request, actor: Actor = Depends(require("pra-gateway:upstreams"))):
        return run_action("health-check", upstream_id, body, request, actor)

    @app.post(f"{GATEWAY_API_PREFIX}/actions/resync-session/{{session_id}}", tags=["actions"])
    def resync_session(session_id: str, body: GatewayActionRequest, request: Request, actor: Actor = Depends(require("pra-gateway:sessions"))):
        return run_action("resync-session", session_id, body, request, actor)

    @app.post(f"{GATEWAY_API_PREFIX}/actions/drop-session/{{session_id}}", tags=["actions"])
    def drop_session(session_id: str, body: GatewayActionRequest, request: Request, actor: Actor = Depends(require("pra-gateway:sessions"))):
        return run_action("drop-session", session_id, body, request, actor)

    @app.post(f"{GATEWAY_API_PREFIX}/actions/clear-capability-cache", tags=["actions"])
    def clear_capability_cache(body: GatewayActionRequest, request: Request, actor: Actor = Depends(require("pra-gateway:admin"))):
        return run_action("clear-capability-cache", None, body, request, actor)

    @app.post(f"{GATEWAY_API_PREFIX}/actions/reload-policy", tags=["actions"])
    def reload_policy(body: GatewayActionRequest, request: Request, actor: Actor = Depends(require("pra-gateway:configure"))):
        return run_action("reload-policy", None, body, request, actor)
    return app


def start_gateway_management_api(provider: GatewayManagementProvider, settings: GatewayManagementAPIConfig) -> Any | None:
    """Start a background listener only when explicitly enabled."""
    if not settings.enabled:
        if settings.registry.enabled:
            reporter = GatewayRegistryReporter(provider)
            reporter.start()
            return reporter
        return None
    import uvicorn
    settings.validate_binding()
    server = uvicorn.Server(uvicorn.Config(
        create_gateway_management_app(provider, settings), host=settings.host, port=settings.port,
        log_level="warning", ssl_certfile=settings.tls_certfile, ssl_keyfile=settings.tls_keyfile,
        ssl_ca_certs=settings.tls_ca_certs,
        ssl_cert_reqs=ssl.CERT_REQUIRED if settings.auth.mode == AuthMode.MTLS else ssl.CERT_NONE,
    ))
    thread = threading.Thread(target=server.run, name="pra-gateway-management", daemon=True)
    thread.start()
    server.pra_thread = thread
    return server


def stop_gateway_management_api(handle: Any | None, timeout: float = 5) -> None:
    if handle is None: return
    if isinstance(handle, GatewayRegistryReporter):
        handle.stop(); return
    handle.should_exit = True
    thread = getattr(handle, "pra_thread", None)
    if thread: thread.join(timeout=timeout)


def _remote_adapter(value: UpstreamCreate) -> PRAEngineAdapter:
    try: engine = EngineType(value.provider)
    except ValueError: engine = EngineType.OPENAI_GENERIC
    return OpenAICompatibleEngineAdapter(value.base_url, name=value.name, engine_type=engine)


def _upstream_headers(value: UpstreamCreate) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "pra-gateway-management/1"}
    token = os.environ.get(value.auth_reference or "")
    if token: headers["Authorization"] = f"Bearer {token}"
    return headers


def _page(values: Sequence[Any], offset: int, limit: int) -> dict[str, Any]:
    end = min(len(values), offset + limit)
    return {"items": values[offset:end], "total": len(values), "offset": offset, "limit": limit, "next_offset": end if end < len(values) else None}


def _safe(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:24]


def _status(value: Any) -> str:
    return "validated" if bool(value) else "not_supported"


def _transport_name(mode: str) -> str:
    return {"G00": "TEXT", "G10": "TEXT", "G01": "PRA-FULL", "G11": "PRA-DELTA"}.get(mode, "UNKNOWN")


def _gateway_mode_name(mode: str) -> str:
    return {
        "G00": "passthrough", "G10": "selected-context",
        "G01": "upgrade", "G11": "typed-transport",
    }.get(mode.upper(), mode)


def _gateway_mode_code(mode: str) -> str:
    aliases = {
        "passthrough": "G00", "selected-context": "G10",
        "upgrade": "G01", "typed-transport": "G11",
        "g00": "G00", "g10": "G10", "g01": "G01", "g11": "G11",
    }
    try:
        return aliases[mode.lower()]
    except KeyError as error:
        raise ValueError(
            "mode must be passthrough, selected-context, upgrade, or typed-transport"
        ) from error


def _version(name: str) -> str:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return "source"


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
