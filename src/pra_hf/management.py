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
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from fastapi import Depends, FastAPI, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .registry_registration import RegistryRegistrationClient, RuntimeRegistryConfig


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
    registry: RuntimeRegistryConfig = Field(default_factory=RuntimeRegistryConfig)
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
        registry = dict(data.get("registry") or {})
        if os.environ.get("PRA_REGISTRY_URL"):
            registry.update({"enabled": True, "url": os.environ["PRA_REGISTRY_URL"]})
        registry_token_env = os.environ.get("PRA_REGISTRY_TOKEN_ENV")
        if registry_token_env or os.environ.get("PRA_REGISTRY_TOKEN"):
            registry["auth"] = {
                **dict(registry.get("auth") or {}),
                "type": "bearer", "token_env": registry_token_env or "PRA_REGISTRY_TOKEN",
            }
        if registry:
            data["registry"] = registry
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
    multi_model: bool = False
    dynamic_model_load: bool = False
    dynamic_model_unload: bool = False
    dynamic_model_switch: bool = False
    max_loaded_models: int | None = Field(default=1, ge=1)
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
    runtime_model_id: str = "default"
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
    runtime_model_id: str = "default"
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
    runtime_model_id: str = "default"
    model_fingerprint: str | None = None
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
    models: Mapping[str, Any] = Field(default_factory=dict)


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
    runtime_model_id: str | None = None
    model_id: str | None = None
    revision: str | None = None
    profile: str | None = None
    bundle: str | None = None
    execution_mode: str | None = None
    force: bool = False
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


@dataclass
class ModelRuntimeState:
    """Model-local state owned by one engine management container."""

    model: LoadedModel
    storage_manager: Any | None = None
    session_source: Any | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    effective_config: Mapping[str, Any] = field(default_factory=dict)


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
        model_runtimes: Mapping[str, ModelRuntimeState] | None = None,
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
        rows = [
            row if isinstance(row, LoadedModel) else LoadedModel.model_validate(row)
            for row in models
        ]
        if model_runtimes is not None and (
            rows or storage_manager is not None or session_source is not None
        ):
            raise ValueError(
                "model_runtimes cannot be combined with models, storage_manager, or session_source."
            )
        self.model_runtimes: dict[str, ModelRuntimeState] = {}
        if model_runtimes is not None:
            for runtime_id, state in model_runtimes.items():
                if not isinstance(state, ModelRuntimeState):
                    raise TypeError("model_runtimes values must be ModelRuntimeState instances.")
                normalized = state.model.model_copy(update={"runtime_model_id": str(runtime_id)})
                self._add_runtime_state(ModelRuntimeState(
                    model=normalized,
                    storage_manager=state.storage_manager,
                    session_source=state.session_source,
                    capabilities=dict(state.capabilities),
                    effective_config=dict(state.effective_config),
                ))
        else:
            multiple = len(rows) > 1
            for row in rows:
                runtime_id = row.runtime_model_id
                if multiple and "runtime_model_id" not in row.model_fields_set:
                    runtime_id = row.model_id
                normalized = row.model_copy(update={"runtime_model_id": runtime_id})
                self._add_runtime_state(ModelRuntimeState(
                    model=normalized,
                    storage_manager=storage_manager,
                    session_source=session_source,
                    effective_config=dict(effective_config or {}),
                ))
        self._sync_models()
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
        self.model_drift: dict[str, dict[str, Any]] = {}
        self.audit: deque[AuditEvent] = deque(maxlen=500)
        self._idempotency: dict[tuple[str, str, str], tuple[str, ActionResult]] = {}
        self._lock = threading.RLock()
        self.registry_reporter: RegistryRegistrationClient | None = None

    def _add_runtime_state(self, state: ModelRuntimeState) -> None:
        runtime_id = state.model.runtime_model_id
        if not runtime_id:
            raise ValueError("runtime_model_id cannot be empty.")
        if runtime_id in self.model_runtimes:
            raise ValueError(f"Duplicate runtime_model_id: {runtime_id}")
        self.model_runtimes[runtime_id] = state

    def _sync_models(self) -> None:
        self.models = [state.model for state in self.model_runtimes.values()]

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
            storage_tiers=self._detail(
                any(state.storage_manager is not None for state in self.model_runtimes.values())
                or self.storage_manager is not None,
                "HOT/WARM/COLD/SOURCE",
            ),
            observability=self._detail(bool(self.observability_config), "OpenTelemetry and Prometheus"),
            multi_model=bool(value.get("multi_model", len(self.models) > 1)),
            dynamic_model_load=bool(value.get("dynamic_model_load", False)),
            dynamic_model_unload=bool(value.get("dynamic_model_unload", False)),
            dynamic_model_switch=bool(value.get("dynamic_model_switch", False)),
            max_loaded_models=(
                int(value["max_loaded_models"])
                if value.get("max_loaded_models") is not None
                else max(1, len(self.models))
            ),
        )

    def config_state(self) -> dict[str, Any]:
        return {
            "effective": _redact(self.effective_config),
            "desired_revision": self.desired_revision,
            "observed_revision": self.observed_revision,
            "in_sync": not self.drift_fields,
            "drift_fields": list(self.drift_fields),
            "models": dict(self.model_drift),
        }

    def registry_state(self) -> dict[str, Any]:
        if self.registry_reporter is None:
            return {"enabled": False, "status": "disabled", "instance_id": self.instance_id}
        return self.registry_reporter.status()

    def registry_payload(self, settings: ManagementAPIConfig, instance_id: str) -> dict[str, Any]:
        """Build the privacy-safe Registry observation for this engine."""

        identity = settings.registry.instance
        scheme = "https" if settings.tls_certfile else "http"
        advertised_host = identity.host or (socket.gethostname() if settings.host in {"0.0.0.0", "::"} else settings.host)
        management_url = identity.management_url or f"{scheme}://{advertised_host}:{settings.port}"
        config = self.config_state()
        return {
            "instance_id": instance_id,
            "instance_type": "ENGINE",
            "name": identity.name or f"{self.engine}-{advertised_host}",
            "host": advertised_host,
            "management_url": management_url,
            "inference_url": identity.inference_url or self.effective_config.get("inference_url"),
            "pra_version": _package_version("pra-hf"),
            "component_version": _package_version("pra-hf"),
            "engine_kind": self.engine,
            "engine_version": self.engine_version,
            "health": self.instance().health,
            "started_at": self.started_at,
            "capabilities": self.capabilities().model_dump(mode="json"),
            "models": [row.model_dump(mode="json") for row in self.models],
            "runtime_summary": {
                "profiles": [row.name for row in self.profiles],
                "resource_count": len(self._resource_rows()),
                "session_count": len(self._session_rows()),
            },
            "observability": self.observability(settings).model_dump(mode="json"),
            "observed_revision": self.observed_revision,
            "desired_revision": self.desired_revision,
            "in_sync": not self.drift_fields,
            "drift_fields": list(self.drift_fields),
        }

    def apply_registry_desired(self, value: Mapping[str, Any]) -> None:
        """Record desired-state drift without applying destructive changes."""

        desired = value.get("desired") if isinstance(value, Mapping) else None
        desired_revision = value.get("desired_revision") if isinstance(value, Mapping) else None
        differences: list[str] = []
        model_drift: dict[str, dict[str, Any]] = {}
        if isinstance(desired, Mapping):
            desired_models = desired.get("desired_models")
            if (
                not isinstance(desired_models, Sequence)
                or isinstance(desired_models, (str, bytes))
                or not desired_models
            ):
                desired_models = ({
                    "runtime_model_id": "default",
                    "model_id": desired.get("desired_model_id"),
                    "bundle_id": desired.get("desired_bundle_id"),
                    "profile_id": desired.get("desired_profile_id"),
                    "mode": desired.get("desired_mode"),
                },) if desired.get("desired_model_id") is not None else ()
            expected_ids: set[str] = set()
            for item in desired_models:
                if not isinstance(item, Mapping):
                    continue
                runtime_id = str(item.get("runtime_model_id") or "default")
                expected_ids.add(runtime_id)
                model = self.model_runtimes.get(runtime_id)
                comparisons = {
                    "model": (item.get("model_id"), model.model.model_id if model else None),
                    "bundle": (item.get("bundle_id"), model.model.pra_bundle_id if model else None),
                    "profile": (item.get("profile_id"), model.model.profile if model else None),
                    "mode": (item.get("mode"), model.model.execution_mode if model else None),
                }
                fields = [
                    name for name, (expected, actual) in comparisons.items()
                    if expected is not None and expected != actual
                ]
                if model is None:
                    fields.insert(0, "MODEL_NOT_LOADED")
                model_drift[runtime_id] = {
                    "in_sync": not fields,
                    "drift_fields": list(dict.fromkeys(fields)),
                }
                differences.extend(f"models.{runtime_id}.{name}" for name in fields)
            if not bool(desired.get("allow_extra_models", True)):
                for runtime_id in self.model_runtimes.keys() - expected_ids:
                    model_drift[runtime_id] = {
                        "in_sync": False,
                        "drift_fields": ["UNAPPROVED_MODEL_LOADED"],
                    }
                    differences.append(f"models.{runtime_id}.UNAPPROVED_MODEL_LOADED")
        with self._lock:
            self.desired_revision = int(desired_revision) if desired_revision is not None else None
            self.drift_fields = differences
            self.model_drift = model_drift

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

    def model(self, runtime_model_id: str) -> LoadedModel:
        state = self.model_runtimes.get(runtime_model_id)
        if state is not None:
            return state.model
        # Preserve old clients that used the global model ID in the path when
        # that lookup remains unambiguous. Runtime IDs are the canonical key.
        matches = [row for row in self.models if row.model_id == runtime_model_id]
        if len(matches) == 1:
            return matches[0]
        raise ManagementAPIError(404, "MODEL_NOT_FOUND", "Loaded model was not found.")

    def models_by_global_id(self, model_id: str | None = None) -> list[LoadedModel]:
        return [row for row in self.models if model_id is None or row.model_id == model_id]

    def model_runtime(self, runtime_model_id: str | None) -> ModelRuntimeState:
        if runtime_model_id is None:
            if len(self.model_runtimes) == 1:
                return next(iter(self.model_runtimes.values()))
            raise ManagementAPIError(
                422,
                "RUNTIME_MODEL_REQUIRED",
                "runtime_model_id is required when an engine exposes multiple models.",
            )
        try:
            return self.model_runtimes[runtime_model_id]
        except KeyError as error:
            raise ManagementAPIError(404, "MODEL_NOT_FOUND", "Loaded model was not found.") from error

    def profile(self, name: str) -> PRAProfileSummary:
        for row in self.profiles:
            if row.name == name:
                return row
        raise ManagementAPIError(404, "PROFILE_NOT_FOUND", "PRA profile was not found.")

    @staticmethod
    def _safe_id(kind: str, value: str) -> str:
        return hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()[:24]

    def _resource_rows(self, runtime_model_id: str | None = None) -> list[PRAResourceSummary]:
        rows: list[PRAResourceSummary] = []
        states = (
            ((runtime_model_id, self.model_runtime(runtime_model_id)),)
            if runtime_model_id is not None
            else tuple(self.model_runtimes.items())
        )
        seen_source: set[tuple[str, str, str | None]] = set()
        for current_id, state in states:
            manager = state.storage_manager
            if manager is None:
                continue
            for key, entry in sorted(manager.entries.items()):
                tier = getattr(entry.current_tier, "value", str(entry.current_tier))
                source_identity = (
                    str(entry.record_type),
                    str(entry.resource_version),
                    entry.source_sha256 or f"{current_id}:{key}",
                )
                if runtime_model_id is None and tier == "source" and source_identity in seen_source:
                    continue
                if tier == "source":
                    seen_source.add(source_identity)
                rows.append(PRAResourceSummary(
                    resource_id=self._safe_id("resource", f"{current_id}:{key}"),
                    runtime_model_id=current_id,
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

    def _resource_key(self, resource_id: str) -> tuple[ModelRuntimeState, str]:
        for runtime_id, state in self.model_runtimes.items():
            if state.storage_manager is None:
                continue
            for key in state.storage_manager.entries:
                if self._safe_id("resource", f"{runtime_id}:{key}") == resource_id:
                    return state, str(key)
        raise ManagementAPIError(404, "RESOURCE_NOT_FOUND", "PRA resource was not found.")

    def _session_rows(self, runtime_model_id: str | None = None) -> list[SessionSummary]:
        rows: list[SessionSummary] = []
        states = (
            ((runtime_model_id, self.model_runtime(runtime_model_id)),)
            if runtime_model_id is not None
            else tuple(self.model_runtimes.items())
        )
        for current_id, state in states:
            if state.session_source is None:
                continue
            for value in state.session_source.inspect_all():
                raw_id = str(value.get("session_id", "unknown"))
                materializations = value.get("visible_materializations", ())
                rows.append(SessionSummary(
                    session_id=self._safe_id("session", f"{current_id}:{raw_id}"),
                    runtime_model_id=current_id,
                    model_fingerprint=state.model.model_fingerprint,
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

    def storage(self, runtime_model_id: str | None = None) -> StorageState:
        if runtime_model_id is None:
            states = list(self.model_runtimes.items())
            unique = {id(state.storage_manager): state.storage_manager for _, state in states if state.storage_manager is not None}
            breakdown = {
                model_id: self.storage(model_id).model_dump(mode="json", exclude={"models"})
                for model_id, state in states if state.storage_manager is not None
            }
            if not unique:
                return StorageState(
                    tiers={name: {"bytes": 0, "resources": 0} for name in ("hot", "warm", "cold", "source")},
                    quotas={}, retention_policy={}, maintenance_status="not_configured", models={},
                )
            details = [self._storage_state(manager) for manager in unique.values()]
            return StorageState(
                tiers={tier: {
                    "bytes": sum(row.tiers[tier]["bytes"] for row in details),
                    "resources": sum(row.tiers[tier]["resources"] for row in details),
                } for tier in ("hot", "warm", "cold", "source")},
                quotas={"scope": "per-model"},
                evictions=sum(row.evictions for row in details),
                reloads=sum(row.reloads for row in details),
                promotions=sum(row.promotions for row in details),
                reconstructions=sum(row.reconstructions for row in details),
                retention_policy={"scope": "per-model"},
                maintenance_status="running" if any(row.maintenance_status == "running" for row in details) else "manual",
                models=breakdown,
            )
        manager = self.model_runtime(runtime_model_id).storage_manager
        if manager is None:
            return StorageState(
                tiers={name: {"bytes": 0, "resources": 0} for name in ("hot", "warm", "cold", "source")},
                quotas={}, retention_policy={}, maintenance_status="not_configured",
            )
        return self._storage_state(manager)

    @staticmethod
    def _storage_state(manager: Any) -> StorageState:
        inspected = manager.inspect()
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
                if getattr(manager, "_maintenance_thread", None) is not None
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
        if self.registry_reporter is not None:
            try:
                self.registry_reporter.publish_observed()
            except Exception:
                pass
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

        if name in {"prefetch", "promote", "demote"} and any(
            state.storage_manager is not None for state in self.model_runtimes.values()
        ):
            if not request.resource_id:
                raise ManagementAPIError(422, "RESOURCE_REQUIRED", "resource_id is required.")
            state, key = self._resource_key(request.resource_id)
            manager = state.storage_manager
            assert manager is not None
            if name in {"prefetch", "promote"}:
                manager.promote(
                    key,
                    tenant_id=request.tenant_id,
                    authorization_scopes=actor.scopes,
                )
            else:
                manager.demote_hot(key)
            detail: Mapping[str, Any] = {"storage_tier": self.resource(request.resource_id).storage_tier}
        elif name == "maintenance" and any(
            state.storage_manager is not None for state in self.model_runtimes.values()
        ):
            states = (
                (self.model_runtime(request.runtime_model_id),)
                if request.runtime_model_id is not None
                else tuple(self.model_runtimes.values())
            )
            for state in states:
                if state.storage_manager is not None:
                    state.storage_manager.run_maintenance()
            detail = {"storage": self.storage().model_dump(mode="json")}
        elif name == "load-model":
            try:
                detail = self._load_model(request)
            except Exception:
                self.record_audit(
                    "MODEL_LOAD_FAILED", actor, request_id, "failure",
                    {"runtime_model_id": request.runtime_model_id},
                )
                raise
        elif name == "unload-model":
            detail = self._unload_model(request)
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
            "load-model": "MODEL_LOADED",
            "unload-model": "MODEL_UNLOADED",
        }[name]
        self.record_audit(event, actor, request_id, "success", {
            "resource_id": request.resource_id,
            "runtime_model_id": request.runtime_model_id,
        })
        if self.registry_reporter is not None:
            try:
                self.registry_reporter.publish_observed()
            except Exception:
                pass
        return result

    def _load_model(self, request: ActionRequest) -> Mapping[str, Any]:
        capabilities = self.capabilities()
        if not capabilities.dynamic_model_load:
            raise ManagementAPIError(501, "ACTION_NOT_SUPPORTED", "Dynamic model loading is not supported.")
        if not request.runtime_model_id or not request.model_id:
            raise ManagementAPIError(
                422, "MODEL_IDENTITY_REQUIRED", "runtime_model_id and model_id are required."
            )
        if request.runtime_model_id in self.model_runtimes:
            raise ManagementAPIError(409, "MODEL_ALREADY_LOADED", "runtime_model_id is already loaded.")
        if capabilities.max_loaded_models is not None and len(self.models) >= capabilities.max_loaded_models:
            raise ManagementAPIError(409, "MODEL_LIMIT_REACHED", "The engine model residency limit was reached.")
        handler = self.action_handlers.get("load-model")
        if handler is None:
            raise ManagementAPIError(501, "ACTION_NOT_SUPPORTED", "No engine model loader is attached.")
        detail = dict(handler(request) or {})
        model = LoadedModel(
            runtime_model_id=request.runtime_model_id,
            model_id=request.model_id,
            revision=request.revision,
            pra_bundle_id=request.bundle,
            profile=request.profile,
            execution_mode=request.execution_mode,
            loaded_at=time.time(),
            runtime_state=str(detail.get("runtime_state", "loaded")),
        )
        self._add_runtime_state(ModelRuntimeState(
            model=model,
            storage_manager=detail.pop("storage_manager", None),
            session_source=detail.pop("session_source", None),
            capabilities=dict(detail.pop("capabilities", {})),
            effective_config=dict(detail.pop("effective_config", {})),
        ))
        self._sync_models()
        self.observed_revision += 1
        return {"model": model.model_dump(mode="json"), **detail}

    def _unload_model(self, request: ActionRequest) -> Mapping[str, Any]:
        if not self.capabilities().dynamic_model_unload:
            raise ManagementAPIError(501, "ACTION_NOT_SUPPORTED", "Dynamic model unloading is not supported.")
        if not request.runtime_model_id:
            raise ManagementAPIError(422, "RUNTIME_MODEL_REQUIRED", "runtime_model_id is required.")
        state = self.model_runtime(request.runtime_model_id)
        handler = self.action_handlers.get("unload-model")
        if handler is None:
            raise ManagementAPIError(501, "ACTION_NOT_SUPPORTED", "No engine model unloader is attached.")
        detail = dict(handler(request) or {})
        manager = state.storage_manager
        if manager is not None:
            for key, entry in tuple(manager.entries.items()):
                tier = getattr(entry.current_tier, "value", str(entry.current_tier))
                if tier == "hot" and not getattr(entry, "request_pin_count", 0):
                    manager.demote_hot(key)
        sessions = state.session_source
        if sessions is not None and hasattr(sessions, "invalidate_model"):
            sessions.invalidate_model(
                state.model.model_fingerprint or state.model.model_id,
                reason="model_unloaded",
            )
        del self.model_runtimes[request.runtime_model_id]
        self._sync_models()
        self.observed_revision += 1
        return {"runtime_model_id": request.runtime_model_id, "force": request.force, **detail}

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

    @app.get(f"{API_PREFIX}/registry", tags=["registry"])
    def registry_state(actor: Actor = Depends(require("pra:read"))):
        return provider.registry_state()

    @app.post(f"{API_PREFIX}/registry/register", tags=["registry"])
    def registry_register(actor: Actor = Depends(require("pra:admin"))):
        if provider.registry_reporter is None:
            raise ManagementAPIError(409, "REGISTRY_DISABLED", "Registry registration is not configured.")
        return provider.registry_reporter.register_now()

    @app.get(f"{API_PREFIX}/models", response_model=Page, tags=["models"])
    def models(
        model_id: str | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:models")),
    ):
        return _page(provider.models_by_global_id(model_id), offset, limit)

    @app.get(f"{API_PREFIX}/models/{{runtime_model_id}}/storage", response_model=StorageState, tags=["storage"])
    def model_storage(runtime_model_id: str, actor: Actor = Depends(require("pra:storage"))):
        return provider.storage(runtime_model_id)

    @app.get(f"{API_PREFIX}/models/{{runtime_model_id:path}}", response_model=LoadedModel, tags=["models"])
    def model(runtime_model_id: str, actor: Actor = Depends(require("pra:models"))):
        return provider.model(runtime_model_id)

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
        runtime_model_id: str | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:read")),
    ):
        rows = provider._resource_rows(runtime_model_id)
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
        runtime_model_id: str | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
        actor: Actor = Depends(require("pra:sessions")),
    ):
        rows = provider._session_rows(runtime_model_id)
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
        "load-model": "pra:models", "unload-model": "pra:models",
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
    reporter = _configure_registry_reporter(provider, settings)
    if reporter is not None:
        reporter.start()
    try:
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
    finally:
        if reporter is not None:
            reporter.stop()


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
    deadline = time.monotonic() + 5
    while not getattr(server, "started", False) and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    reporter = _configure_registry_reporter(provider, settings)
    if reporter is not None:
        try:
            reporter.start()
        except Exception:
            server.should_exit = True
            thread.join(timeout=5)
            raise
    server.pra_registry_reporter = reporter
    return server


def stop_management_api(server: Any | None, timeout: float = 5.0) -> None:
    if server is None:
        return
    reporter = getattr(server, "pra_registry_reporter", None)
    if reporter is not None:
        reporter.stop(timeout=min(timeout, 3))
    server.should_exit = True
    thread = getattr(server, "pra_thread", None)
    if thread is not None:
        thread.join(timeout=timeout)


def _configure_registry_reporter(
    provider: ManagementProvider, settings: ManagementAPIConfig,
) -> RegistryRegistrationClient | None:
    if not settings.registry.enabled or not settings.registry.url:
        return None
    reporter = RegistryRegistrationClient(
        settings.registry, "ENGINE",
        lambda instance_id: provider.registry_payload(settings, instance_id),
        desired_callback=provider.apply_registry_desired,
    )
    provider.instance_id = reporter.instance_id
    provider.registry_reporter = reporter
    return reporter


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
