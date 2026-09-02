"""Versioned local management API shared by PRA engine integrations.

The module intentionally separates the HTTP contract from engine internals.
``ManagementProvider`` projects live runtime objects into privacy-safe models;
engine adapters may also supply bounded action handlers for mechanisms they
actually support.  Importing this module never starts a listener.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import ipaddress
import json
import os
import platform
import socket
import ssl
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from fastapi import Depends, FastAPI, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


MANAGEMENT_PROTOCOL = "pra-management/1"
API_PREFIX = "/v1/pra"


class CapabilityStatus(str, Enum):
    """Qualification vocabulary shared across the PRA paper series."""

    AVAILABLE = "AVAILABLE"
    VALIDATED = "VALIDATED"
    CANDIDATE = "CANDIDATE"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    BLOCKED = "BLOCKED"
    NOT_MEASURED = "NOT_MEASURED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AuthMode(str, Enum):
    NONE = "none"
    STATIC_BEARER = "static_bearer"
    JWT_OIDC = "jwt_oidc"
    MTLS = "mtls"


class ManagementAuthConfig(BaseModel):
    """Authentication settings; secrets are excluded from API responses."""

    model_config = ConfigDict(extra="forbid")

    mode: AuthMode = AuthMode.NONE
    token: SecretStr | None = None
    token_env: str | None = None
    scopes: tuple[str, ...] = (
        "pra:read",
        "pra:configure",
        "pra:storage",
        "pra:sessions",
        "pra:models",
        "pra:admin",
    )
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    mtls_subjects: tuple[str, ...] = ()

    def resolved_token(self) -> str | None:
        if self.token is not None:
            return self.token.get_secret_value()
        if self.token_env:
            return os.environ.get(self.token_env)
        return None

    def public_dict(self) -> dict[str, Any]:
        value = self.model_dump(exclude={"token"})
        value["token_configured"] = self.resolved_token() is not None
        return value


class ManagementAPIConfig(BaseModel):
    """Local listener configuration. Disabled is the package default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=9101, ge=1, le=65535)
    auth: ManagementAuthConfig = Field(default_factory=ManagementAuthConfig)
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
            raise ValueError("Management API host cannot be empty.")
        return value

    def validate_binding(self) -> None:
        if self.auth.mode == AuthMode.NONE and not _is_loopback(self.host):
            raise ValueError(
                "Unauthenticated management API may bind only to a loopback address."
            )
        if self.auth.mode == AuthMode.STATIC_BEARER and not self.auth.resolved_token():
            raise ValueError("Static bearer authentication requires a token or token_env.")
        if self.auth.mode == AuthMode.MTLS and not all(
            (self.tls_certfile, self.tls_keyfile, self.tls_ca_certs)
        ):
            raise ValueError("mTLS requires tls_certfile, tls_keyfile, and tls_ca_certs.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> "ManagementAPIConfig":
        """Load a mapping and apply explicit PRA_MANAGEMENT_* environment overrides."""

        data = dict(value or {})
        auth = dict(data.get("auth") or {})
        environment = {
            "enabled": os.environ.get("PRA_MANAGEMENT_ENABLED"),
            "host": os.environ.get("PRA_MANAGEMENT_HOST"),
            "port": os.environ.get("PRA_MANAGEMENT_PORT"),
        }
        if environment["enabled"] is not None:
            data["enabled"] = environment["enabled"].strip().lower() in {
                "1", "true", "yes", "on"
            }
        if environment["host"]:
            data["host"] = environment["host"]
        if environment["port"]:
            data["port"] = int(environment["port"])
        if os.environ.get("PRA_MANAGEMENT_AUTH_MODE"):
            auth["mode"] = os.environ["PRA_MANAGEMENT_AUTH_MODE"]
        if os.environ.get("PRA_MANAGEMENT_TOKEN"):
            auth["token"] = os.environ["PRA_MANAGEMENT_TOKEN"]
        if auth:
            data["auth"] = auth
        return cls.model_validate(data)


class CapabilityDetail(BaseModel):
    status: CapabilityStatus
    mechanism: str | None = None
    quality: CapabilityStatus = CapabilityStatus.NOT_MEASURED
    economics: CapabilityStatus = CapabilityStatus.NOT_MEASURED
    recommendation: str | None = None


class EngineCapabilities(BaseModel):
    selected_context: CapabilityDetail
    typed_transport: CapabilityDetail
    native_memory: CapabilityDetail
    native_serving: CapabilityDetail
    prefix_cache: CapabilityDetail
    session_aware_realization: CapabilityDetail
    storage_tiers: CapabilityDetail
    observability: CapabilityDetail
    management_api_version: str = MANAGEMENT_PROTOCOL


class EngineInstance(BaseModel):
    instance_id: str
    engine: str
    engine_version: str | None
    pra_version: str
    host: str
    process_id: int | None
    started_at: float
    health: str
    accelerator: Mapping[str, Any]
    models: tuple[str, ...]
    worker_topology: Mapping[str, Any]


class LoadedModel(BaseModel):
    model_id: str
    revision: str | None = None
    model_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None
    quantization: str | None = None
    device_placement: str | None = None
    pra_bundle_id: str | None = None
    pra_bundle_revision: str | None = None
    profile: str | None = None
    execution_mode: str | None = None
    loaded_at: float | None = None
    runtime_state: str = "observed"


class PRAProfileSummary(BaseModel):
    name: str
    version: str = "1"
    source: str = "runtime"
    effective_policy: Mapping[str, Any] = Field(default_factory=dict)
    qualification_status: CapabilityStatus = CapabilityStatus.NOT_MEASURED
    immutable: bool = False
    managed: bool = False


class PRAResourceSummary(BaseModel):
    resource_id: str
    resource_type: str
    version: str
    size_bytes: int
    token_count: int | None = None
    storage_tier: str
    native_resident: bool
    pin_count: int
    last_access: float | None
    task_scoped: bool
    session_scoped: bool
    authorization_scope: str
    checksum: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    created_at: float | None = None
    last_activity: float | None = None
    active_task_count: int = 0
    visible_context: Mapping[str, Any] = Field(default_factory=dict)
    selected_token_total: int = 0
    logical_reuse_total: int = 0
    native_reuse_total: int = 0
    engine_cache: Mapping[str, Any] = Field(default_factory=dict)
    status: str = "active"


class StorageState(BaseModel):
    tiers: Mapping[str, Mapping[str, int]]
    quotas: Mapping[str, Any]
    evictions: int = 0
    reloads: int = 0
    promotions: int = 0
    reconstructions: int = 0
    retention_policy: Mapping[str, Any]
    maintenance_status: str


class ObservabilityState(BaseModel):
    otel_enabled: bool = False
    prometheus_enabled: bool = False
    metrics_url: str | None = None
    trace_backend_url: str | None = None
    grafana_url: str | None = None
    sampling: str | None = None
    engine_native: Mapping[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    items: list[Any]
    total: int
    offset: int
    limit: int
    next_offset: int | None = None


class AuditEvent(BaseModel):
    timestamp: float
    event: str
    actor: str
    request_id: str
    trace_id: str | None = None
    result: str
    changes: Mapping[str, Any] = Field(default_factory=dict)


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    selection_budget: Mapping[str, Any] | None = None
    storage_quota: Mapping[str, Any] | None = None
    retention_policy: Mapping[str, Any] | None = None
    observability: Mapping[str, Any] | None = None
    prefetch_policy: Mapping[str, Any] | None = None
    engine: str | None = None
    model: str | None = None
    device: str | None = None
    topology: Mapping[str, Any] | None = None
    host: str | None = None
    port: int | None = None


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_id: str | None = None
    profile: str | None = None
    bundle: str | None = None
    tenant_id: str | None = None
    idempotency_key: str | None = None
    parameters: Mapping[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    action: str
    status: str
    resource_id: str | None = None
    detail: Mapping[str, Any] = Field(default_factory=dict)
    idempotent_replay: bool = False


class ActionHandler(Protocol):
    def __call__(self, request: ActionRequest) -> Mapping[str, Any] | None: ...


class ManagementAPIError(RuntimeError):
    def __init__(self, status_code: int, code: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.extra = extra


@dataclass(frozen=True)
class Actor:
    identity: str
    scopes: frozenset[str]


class ManagementProvider:
    """Project one local engine into the shared privacy-safe API contract."""

    _restart_fields = frozenset({"engine", "model", "device", "topology", "host", "port"})

    def __init__(
        self,
        *,
        engine: str,
        engine_version: str | None = None,
        capabilities: Mapping[str, Any] | None = None,
        models: Sequence[LoadedModel | Mapping[str, Any]] = (),
        profiles: Sequence[PRAProfileSummary | Mapping[str, Any]] = (),
        effective_config: Mapping[str, Any] | None = None,
        storage_manager: Any | None = None,
        session_source: Any | None = None,
        observability: Mapping[str, Any] | None = None,
        health_probe: Callable[[], str] | None = None,
        config_patch_handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
        action_handlers: Mapping[str, ActionHandler] | None = None,
        instance_id: str | None = None,
    ) -> None:
        self.engine = str(engine)
        self.engine_version = engine_version
        self.raw_capabilities = dict(capabilities or {})
        self.models = [
            row if isinstance(row, LoadedModel) else LoadedModel.model_validate(row)
            for row in models
        ]
        self.profiles = [
            row if isinstance(row, PRAProfileSummary) else PRAProfileSummary.model_validate(row)
            for row in profiles
        ]
        self.effective_config = dict(effective_config or {})
        self.storage_manager = storage_manager
        self.session_source = session_source
        self.observability_config = dict(observability or {})
        self.health_probe = health_probe
        self.config_patch_handler = config_patch_handler
        self.action_handlers = dict(action_handlers or {})
        self.instance_id = instance_id or uuid.uuid4().hex
        self.started_at = time.time()
        self.observed_revision = 1
        self.desired_revision: int | None = None
        self.drift_fields: list[str] = []
        self.audit: deque[AuditEvent] = deque(maxlen=500)
        self._idempotency: dict[tuple[str, str, str], tuple[str, ActionResult]] = {}
        self._lock = threading.RLock()

    def instance(self) -> EngineInstance:
        model_names = tuple(row.model_id for row in self.models)
        accelerator = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        return EngineInstance(
            instance_id=self.instance_id,
            engine=self.engine,
            engine_version=self.engine_version,
            pra_version=_package_version("pra-hf"),
            host=socket.gethostname(),
            process_id=os.getpid(),
            started_at=self.started_at,
            health=self.health_probe() if self.health_probe else "healthy",
            accelerator=accelerator,
            models=model_names,
            worker_topology={"workers": 1, "identity": self.instance_id},
        )

    @staticmethod
    def _detail(enabled: Any, mechanism: str) -> CapabilityDetail:
        status = CapabilityStatus.AVAILABLE if bool(enabled) else CapabilityStatus.NOT_APPLICABLE
        return CapabilityDetail(status=status, mechanism=mechanism)

    def capabilities(self) -> EngineCapabilities:
        value = self.raw_capabilities
        native = value.get("native_kv", value.get("native_memory", False))
        integration = str(value.get("integration_level", "E0"))
        return EngineCapabilities(
            selected_context=self._detail(
                value.get("text_fallback", True) or value.get("logical_refs"),
                "selected text or logical resource selection",
            ),
            typed_transport=self._detail(value.get("typed_records"), "typed PRA records"),
            native_memory=self._detail(native, "detached native K/V"),
            native_serving=self._detail(
                native and integration in {"E2", "E3"}, "native attention consumption"
            ),
            prefix_cache=self._detail(
                value.get("automatic_prefix_cache")
                or value.get("explicit_prefix_cache")
                or value.get("session_state"),
                str(value.get("prefix_cache_mode", "unknown")),
            ),
            session_aware_realization=self._detail(
                value.get("session_state"), "session-aware request realization"
            ),
            storage_tiers=self._detail(self.storage_manager is not None, "HOT/WARM/COLD/SOURCE"),
            observability=self._detail(bool(self.observability_config), "OpenTelemetry and Prometheus"),
        )

    def config_state(self) -> dict[str, Any]:
        return {
            "effective": _redact(self.effective_config),
            "desired_revision": self.desired_revision,
            "observed_revision": self.observed_revision,
            "in_sync": not self.drift_fields,
            "drift_fields": list(self.drift_fields),
        }

    def state(self) -> dict[str, Any]:
        return {
            "protocol": MANAGEMENT_PROTOCOL,
            "instance": self.instance().model_dump(mode="json"),
            "capabilities": self.capabilities().model_dump(mode="json"),
            "config": self.config_state(),
            "model_count": len(self.models),
            "profile_count": len(self.profiles),
            "resource_count": len(self._resource_rows()),
            "session_count": len(self._session_rows()),
        }

    def model(self, model_id: str) -> LoadedModel:
        for row in self.models:
            if row.model_id == model_id:
                return row
        raise ManagementAPIError(404, "MODEL_NOT_FOUND", "Loaded model was not found.")

    def profile(self, name: str) -> PRAProfileSummary:
        for row in self.profiles:
            if row.name == name:
                return row
        raise ManagementAPIError(404, "PROFILE_NOT_FOUND", "PRA profile was not found.")

    @staticmethod
    def _safe_id(kind: str, value: str) -> str:
        return hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()[:24]

    def _resource_rows(self) -> list[PRAResourceSummary]:
        if self.storage_manager is None:
            return []
        rows: list[PRAResourceSummary] = []
        for key, entry in sorted(self.storage_manager.entries.items()):
            safe_id = self._safe_id("resource", str(key))
            tier = getattr(entry.current_tier, "value", str(entry.current_tier))
            rows.append(PRAResourceSummary(
                resource_id=safe_id,
                resource_type=str(entry.record_type),
                version=str(entry.resource_version),
                size_bytes=int(entry.detail_bytes),
                storage_tier=tier,
                native_resident=tier != "source",
                pin_count=int(entry.request_pin_count),
                last_access=float(entry.last_access_ns) / 1e9 if entry.last_access_ns else None,
                task_scoped=entry.task_id is not None,
                session_scoped=entry.session_id is not None,
                authorization_scope="restricted" if entry.security_scope else "default",
                checksum=entry.source_sha256,
            ))
        return rows

    def resource(self, resource_id: str) -> PRAResourceSummary:
        for row in self._resource_rows():
            if row.resource_id == resource_id:
                return row
        raise ManagementAPIError(404, "RESOURCE_NOT_FOUND", "PRA resource was not found.")

    def _resource_key(self, resource_id: str) -> str:
        if self.storage_manager is not None:
            for key in self.storage_manager.entries:
                if self._safe_id("resource", str(key)) == resource_id:
                    return str(key)
        raise ManagementAPIError(404, "RESOURCE_NOT_FOUND", "PRA resource was not found.")

    def _session_rows(self) -> list[SessionSummary]:
        if self.session_source is None:
            return []
        values = self.session_source.inspect_all()
        rows: list[SessionSummary] = []
        for value in values:
            raw_id = str(value.get("session_id", "unknown"))
            materializations = value.get("visible_materializations", ())
            rows.append(SessionSummary(
                session_id=self._safe_id("session", raw_id),
                created_at=value.get("created_at"),
                last_activity=value.get("updated_at"),
                active_task_count=int(value.get("active_task_count", 0)),
                visible_context={
                    "resource_count": len(value.get("known_resources", {})),
                    "materialization_count": len(materializations),
                    "message_count": int(value.get("prefix_message_count", 0)),
                },
                selected_token_total=sum(int(row.get("token_count", 0)) for row in materializations),
                logical_reuse_total=int(value.get("turns", 0)),
                native_reuse_total=int(value.get("native_reuse_total", 0)),
                engine_cache={
                    "session_present": bool(value.get("engine_session_present")),
                    "prefix_handle_present": bool(value.get("prefix_cache_handle_present")),
                    "worker_present": value.get("engine_worker_identity") is not None,
                },
                status=str(value.get("status", "active")),
            ))
        return rows

    def session(self, session_id: str) -> SessionSummary:
        for row in self._session_rows():
            if row.session_id == session_id:
                return row
        raise ManagementAPIError(404, "SESSION_NOT_FOUND", "PRA session was not found.")

    def storage(self) -> StorageState:
        if self.storage_manager is None:
            return StorageState(
                tiers={name: {"bytes": 0, "resources": 0} for name in ("hot", "warm", "cold", "source")},
                quotas={}, retention_policy={}, maintenance_status="not_configured",
            )
        inspected = self.storage_manager.inspect()
        usage = inspected["usage"]
        objects = inspected["objects"]
        metrics = inspected["metrics"]
        policy = inspected["policy"]
        return StorageState(
            tiers={
                "hot": {"bytes": usage["hot_bytes"], "resources": objects["hot"]},
                "warm": {"bytes": usage["warm_bytes"], "resources": objects["warm"]},
                "cold": {"bytes": usage["cold_bytes"], "resources": objects["cold"]},
                "source": {"bytes": 0, "resources": objects["source"]},
            },
            quotas={
                name: details.get("max_bytes")
                for name, details in policy.items()
                if name in {"hot", "warm", "cold"} and isinstance(details, Mapping)
            },
            evictions=int(metrics.get("evictions", 0)),
            reloads=int(metrics.get("reloads", 0)),
            promotions=int(metrics.get("promotions", 0)),
            reconstructions=int(metrics.get("hits", {}).get("source", 0)),
            retention_policy=_redact(policy),
            maintenance_status=(
                "running"
                if getattr(self.storage_manager, "_maintenance_thread", None) is not None
                else "manual"
            ),
        )

    def observability(self, settings: ManagementAPIConfig) -> ObservabilityState:
        config = self.observability_config
        return ObservabilityState(
            otel_enabled=bool(config.get("otel", {}).get("enabled", config.get("otel_enabled", False))),
            prometheus_enabled=bool(config.get("prometheus", {}).get("enabled", config.get("prometheus_enabled", False))),
            metrics_url=settings.metrics_url or config.get("metrics_url"),
            trace_backend_url=settings.trace_backend_url or config.get("trace_backend_url"),
            grafana_url=settings.grafana_url or config.get("grafana_url"),
            sampling=config.get("sampling"),
            engine_native=dict(config.get("engine_native", {})),
        )

    def patch_config(self, patch: ConfigPatch, actor: Actor, request_id: str) -> dict[str, Any]:
        values = patch.model_dump(exclude_none=True)
        restart = sorted(self._restart_fields.intersection(values))
        if restart:
            raise ManagementAPIError(
                409,
                "RESTART_REQUIRED",
                "Requested fields cannot be changed safely while the engine is running.",
                restart_fields=restart,
            )
        if self.config_patch_handler is None:
            raise ManagementAPIError(
                501,
                "CONFIG_PATCH_NOT_SUPPORTED",
                f"The local {self.engine} integration does not support live configuration changes.",
            )
        applied = dict(self.config_patch_handler(values) or values)
        with self._lock:
            old = {key: self.effective_config.get(key) for key in applied}
            self.effective_config.update(applied)
            self.observed_revision += 1
            self.desired_revision = self.observed_revision
            self.drift_fields = []
        self.record_audit("CONFIG_CHANGED", actor, request_id, "success", {"old": old, "new": applied})
        return self.config_state()

    def action(self, name: str, request: ActionRequest, actor: Actor, request_id: str) -> ActionResult:
        idempotency_key = request.idempotency_key
        cache_key = (actor.identity, name, idempotency_key or "")
        request_digest = hashlib.sha256(
            json.dumps(
                request.model_dump(exclude={"idempotency_key"}),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if idempotency_key and cache_key in self._idempotency:
            prior_digest, prior_result = self._idempotency[cache_key]
            if not hmac.compare_digest(request_digest, prior_digest):
                raise ManagementAPIError(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "The idempotency key was already used with a different action payload.",
                )
            return prior_result.model_copy(update={"idempotent_replay": True})

        if name in {"prefetch", "promote", "demote"} and self.storage_manager is not None:
            if not request.resource_id:
                raise ManagementAPIError(422, "RESOURCE_REQUIRED", "resource_id is required.")
            key = self._resource_key(request.resource_id)
            if name in {"prefetch", "promote"}:
                self.storage_manager.promote(
                    key,
                    tenant_id=request.tenant_id,
                    authorization_scopes=actor.scopes,
                )
            else:
                self.storage_manager.demote_hot(key)
            detail: Mapping[str, Any] = {"storage_tier": self.resource(request.resource_id).storage_tier}
        elif name == "maintenance" and self.storage_manager is not None:
            self.storage_manager.run_maintenance()
            detail = {"storage": self.storage().model_dump(mode="json")}
        elif name in self.action_handlers:
            detail = dict(self.action_handlers[name](request) or {})
        else:
            raise ManagementAPIError(
                501,
                "ACTION_NOT_SUPPORTED",
                f"The local {self.engine} integration does not support action '{name}'.",
            )
        result = ActionResult(
            action=name, status="success", resource_id=request.resource_id, detail=detail
        )
        if idempotency_key:
            self._idempotency[cache_key] = (request_digest, result)
        event = {
            "prefetch": "RESOURCE_PREFETCHED",
            "promote": "RESOURCE_PROMOTED",
            "demote": "RESOURCE_DEMOTED",
            "evict": "RESOURCE_EVICTED",
            "reload-profile": "PROFILE_CHANGED",
            "reload-bundle": "BUNDLE_CHANGED",
            "maintenance": "MAINTENANCE_TRIGGERED",
        }[name]
        self.record_audit(event, actor, request_id, "success", {"resource_id": request.resource_id})
        return result

    def record_audit(
        self,
        event: str,
        actor: Actor,
        request_id: str,
        result: str,
        changes: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit.append(AuditEvent(
            timestamp=time.time(), event=event, actor=actor.identity,
            request_id=request_id, result=result, changes=_redact(dict(changes or {})),
        ))


def create_management_app(provider: ManagementProvider, settings: ManagementAPIConfig):
    """Create the enabled FastAPI application without starting a listener."""

    settings.validate_binding()
    app = FastAPI(
        title="PRA Engine Management API",
        summary="Open local management and observed-state API for one PRA engine.",
        description=(
            "The open-source PRA management surface controls one local engine. "
            "It is not an enterprise fleet control plane."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    authenticator = _Authenticator(settings.auth)
    bearer_scheme = HTTPBearer(auto_error=False)

    @app.exception_handler(ManagementAPIError)
    async def management_error(_request: Request, error: ManagementAPIError):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "detail": error.detail, **error.extra}},
        )

    def require(*scopes: str):
        async def dependency(
            request: Request,
            _credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
        ) -> Actor:
            return authenticator.authorize(request, scopes)
        return dependency

    def request_id(request: Request) -> str:
        return request.headers.get("x-request-id") or uuid.uuid4().hex

    @app.get(f"{API_PREFIX}/health", tags=["engine"])
    def health(actor: Actor = Depends(require("pra:read"))):
        instance = provider.instance()
        return {
            "status": instance.health,
            "protocol": MANAGEMENT_PROTOCOL,
            "instance_id": instance.instance_id,
        }

    @app.get(f"{API_PREFIX}/info", response_model=EngineInstance, tags=["engine"])
    def info(actor: Actor = Depends(require("pra:read"))):
        return provider.instance()

    @app.get(f"{API_PREFIX}/capabilities", response_model=EngineCapabilities, tags=["engine"])
    def capabilities(actor: Actor = Depends(require("pra:read"))):
        return provider.capabilities()

    @app.get(f"{API_PREFIX}/config", tags=["configuration"])
    def config(actor: Actor = Depends(require("pra:read"))):
        return provider.config_state()

    @app.patch(f"{API_PREFIX}/config", tags=["configuration"])
    def patch_config(
        patch: ConfigPatch,
        request: Request,
        actor: Actor = Depends(require("pra:configure")),
    ):
        return provider.patch_config(patch, actor, request_id(request))

    @app.get(f"{API_PREFIX}/state", tags=["engine"])
    def state(actor: Actor = Depends(require("pra:read"))):
        return provider.state()

    @app.get(f"{API_PREFIX}/models", response_model=Page, tags=["models"])
    def models(
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:models")),
    ):
        return _page(provider.models, offset, limit)

    @app.get(f"{API_PREFIX}/models/{{model_id:path}}", response_model=LoadedModel, tags=["models"])
    def model(model_id: str, actor: Actor = Depends(require("pra:models"))):
        return provider.model(model_id)

    @app.get(f"{API_PREFIX}/profiles", response_model=Page, tags=["profiles"])
    def profiles(
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:read")),
    ):
        return _page(provider.profiles, offset, limit)

    @app.get(f"{API_PREFIX}/profiles/{{profile}}", response_model=PRAProfileSummary, tags=["profiles"])
    def profile(profile: str, actor: Actor = Depends(require("pra:read"))):
        return provider.profile(profile)

    @app.get(f"{API_PREFIX}/resources", response_model=Page, tags=["resources"])
    def resources(
        resource_type: str | None = None,
        storage_tier: str | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:read")),
    ):
        rows = provider._resource_rows()
        if resource_type:
            rows = [row for row in rows if row.resource_type == resource_type]
        if storage_tier:
            rows = [row for row in rows if row.storage_tier == storage_tier]
        return _page(rows, offset, limit)

    @app.get(f"{API_PREFIX}/resources/{{resource_id}}", response_model=PRAResourceSummary, tags=["resources"])
    def resource(resource_id: str, actor: Actor = Depends(require("pra:read"))):
        return provider.resource(resource_id)

    @app.get(f"{API_PREFIX}/sessions", response_model=Page, tags=["sessions"])
    def sessions(
        status: str | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:sessions")),
    ):
        rows = provider._session_rows()
        if status:
            rows = [row for row in rows if row.status == status]
        return _page(rows, offset, limit)

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}", response_model=SessionSummary, tags=["sessions"])
    def session(session_id: str, actor: Actor = Depends(require("pra:sessions"))):
        return provider.session(session_id)

    @app.get(f"{API_PREFIX}/storage", response_model=StorageState, tags=["storage"])
    def storage(actor: Actor = Depends(require("pra:storage"))):
        return provider.storage()

    @app.get(f"{API_PREFIX}/observability", response_model=ObservabilityState, tags=["observability"])
    def observability(actor: Actor = Depends(require("pra:read"))):
        return provider.observability(settings)

    @app.get(f"{API_PREFIX}/metrics-link", tags=["observability"])
    def metrics_link(actor: Actor = Depends(require("pra:read"))):
        return {"url": provider.observability(settings).metrics_url}

    @app.get(f"{API_PREFIX}/trace-link", tags=["observability"])
    def trace_link(actor: Actor = Depends(require("pra:read"))):
        state = provider.observability(settings)
        return {"url": state.trace_backend_url, "grafana_url": state.grafana_url}

    @app.get(f"{API_PREFIX}/audit", response_model=Page, tags=["audit"])
    def audit(
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:admin")),
    ):
        return _page(list(reversed(provider.audit)), offset, limit)

    action_scopes = {
        "prefetch": "pra:storage", "evict": "pra:storage",
        "promote": "pra:storage", "demote": "pra:storage",
        "reload-profile": "pra:configure", "reload-bundle": "pra:models",
        "maintenance": "pra:storage",
    }
    def register_action(action_name: str, scope: str) -> None:
        def action_endpoint(
            body: ActionRequest,
            request: Request,
            actor: Actor = Depends(require(scope)),
        ):
            return provider.action(action_name, body, actor, request_id(request))
        app.post(
            f"{API_PREFIX}/actions/{action_name}",
            response_model=ActionResult,
            tags=["actions"],
            name=f"action_{action_name.replace('-', '_')}",
        )(action_endpoint)

    for action_name, scope in action_scopes.items():
        register_action(action_name, scope)

    return app


def serve_management_api(
    provider: ManagementProvider,
    settings: ManagementAPIConfig,
    *,
    log_level: str = "info",
) -> None:
    """Serve one explicitly enabled management API on its separate port."""

    if not settings.enabled:
        return
    import uvicorn

    settings.validate_binding()
    uvicorn.run(
        create_management_app(provider, settings),
        host=settings.host,
        port=settings.port,
        log_level=log_level,
        ssl_certfile=settings.tls_certfile,
        ssl_keyfile=settings.tls_keyfile,
        ssl_ca_certs=settings.tls_ca_certs,
        ssl_cert_reqs=ssl.CERT_REQUIRED if settings.auth.mode == AuthMode.MTLS else ssl.CERT_NONE,
    )


def start_management_api(
    provider: ManagementProvider,
    settings: ManagementAPIConfig,
) -> Any | None:
    """Start an owned background Uvicorn server, or do nothing when disabled."""

    if not settings.enabled:
        return None
    import uvicorn

    settings.validate_binding()
    server = uvicorn.Server(uvicorn.Config(
        create_management_app(provider, settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",
        ssl_certfile=settings.tls_certfile,
        ssl_keyfile=settings.tls_keyfile,
        ssl_ca_certs=settings.tls_ca_certs,
        ssl_cert_reqs=ssl.CERT_REQUIRED if settings.auth.mode == AuthMode.MTLS else ssl.CERT_NONE,
    ))
    thread = threading.Thread(target=server.run, name="pra-management", daemon=True)
    thread.start()
    server.pra_thread = thread
    return server


def stop_management_api(server: Any | None, timeout: float = 5.0) -> None:
    if server is None:
        return
    server.should_exit = True
    thread = getattr(server, "pra_thread", None)
    if thread is not None:
        thread.join(timeout=timeout)


class _Authenticator:
    """Scope-aware local auth with optional JWT/OIDC and mTLS adapters."""

    def __init__(self, config: ManagementAuthConfig) -> None:
        self.config = config

    def authorize(self, request: Any, required: Iterable[str]) -> Actor:
        required_set = set(required)
        if self.config.mode == AuthMode.NONE:
            return Actor("local-dev", frozenset(self.config.scopes))
        if self.config.mode == AuthMode.STATIC_BEARER:
            actor = self._static_bearer(request)
        elif self.config.mode == AuthMode.JWT_OIDC:
            actor = self._jwt(request)
        else:
            actor = self._mtls(request)
        scopes = set(actor.scopes)
        if "pra:admin" not in scopes and not required_set.issubset(scopes):
            raise ManagementAPIError(403, "INSUFFICIENT_SCOPE", "Required management scope is missing.")
        return actor

    @staticmethod
    def _bearer(request: Any) -> str:
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not value:
            raise ManagementAPIError(401, "AUTH_REQUIRED", "Bearer authentication is required.")
        return value

    def _static_bearer(self, request: Any) -> Actor:
        supplied = self._bearer(request)
        expected = self.config.resolved_token() or ""
        if not hmac.compare_digest(supplied, expected):
            raise ManagementAPIError(401, "INVALID_TOKEN", "Bearer token is invalid.")
        return Actor("static-token", frozenset(self.config.scopes))

    def _jwt(self, request: Any) -> Actor:
        try:
            import jwt
        except ImportError as error:
            raise ManagementAPIError(503, "JWT_BACKEND_UNAVAILABLE", "Install pra-hf[management-auth].") from error
        token = self._bearer(request)
        if not self.config.oidc_jwks_url:
            raise ManagementAPIError(503, "OIDC_NOT_CONFIGURED", "oidc_jwks_url is required.")
        key = jwt.PyJWKClient(self.config.oidc_jwks_url).get_signing_key_from_jwt(token).key
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256", "ES256"],
                audience=self.config.oidc_audience,
                issuer=self.config.oidc_issuer,
            )
        except jwt.PyJWTError as error:
            raise ManagementAPIError(401, "INVALID_TOKEN", "OIDC token validation failed.") from error
        raw_scopes = claims.get("scope", claims.get("scp", ()))
        scopes = raw_scopes.split() if isinstance(raw_scopes, str) else list(raw_scopes)
        return Actor(str(claims.get("sub", "oidc-client")), frozenset(map(str, scopes)))

    def _mtls(self, request: Any) -> Actor:
        ssl_object = request.scope.get("ssl_object")
        certificate = ssl_object.getpeercert() if ssl_object is not None else None
        if not certificate and request.scope.get("scheme") != "https":
            raise ManagementAPIError(401, "CLIENT_CERT_REQUIRED", "A verified client certificate is required.")
        if not certificate and self.config.mtls_subjects:
            raise ManagementAPIError(503, "CLIENT_SUBJECT_UNAVAILABLE", "The ASGI server did not expose the verified certificate subject.")
        if not certificate:
            return Actor("mtls-client", frozenset(self.config.scopes))
        subject = ",".join("=".join(item) for group in certificate.get("subject", ()) for item in group)
        if self.config.mtls_subjects and subject not in self.config.mtls_subjects:
            raise ManagementAPIError(403, "CLIENT_CERT_FORBIDDEN", "Client certificate subject is not allowed.")
        return Actor(subject or "mtls-client", frozenset(self.config.scopes))


def _page(values: Sequence[Any], offset: int, limit: int) -> Page:
    total = len(values)
    end = min(total, offset + limit)
    return Page(
        items=list(values[offset:end]), total=total, offset=offset, limit=limit,
        next_offset=end if end < total else None,
    )


def _redact(value: Any) -> Any:
    sensitive = {"token", "secret", "password", "credential", "api_key", "authorization"}
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in sensitive else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source"


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
