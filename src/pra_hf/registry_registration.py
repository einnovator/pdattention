"""Resilient self-registration shared by PRA engines and gateways.

The Registry owns fleet discovery and desired state.  A runtime publishes only
its stable identity and privacy-safe observed state; credentials remain local.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .product_config import pra_home


class RegistryClientAuth(BaseModel):
    """Local credential references for Registry authentication."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="none", pattern="^(none|bearer|oauth2_client_credentials|mtls)$")
    token_env: str | None = None
    token_file: str | None = None
    token_url: str | None = None
    client_id_env: str | None = None
    client_secret_env: str | None = None
    scopes: tuple[str, ...] = ()
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None

    def bearer_token(self) -> str | None:
        if self.token_env and os.environ.get(self.token_env):
            return os.environ[self.token_env]
        if self.token_file:
            return Path(self.token_file).expanduser().read_text(encoding="utf-8").strip()
        return None


class RuntimeInstanceIdentity(BaseModel):
    """Stable fleet identity and placement labels for one managed runtime."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    instance_id: str | None = Field(default=None, alias="id")
    name: str | None = None
    environment: str = "development"
    region: str = "local"
    cluster: str = "default"
    namespace: str = "default"
    host: str | None = None
    management_url: str | None = None
    inference_url: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    identity_file: str | None = None

    @field_validator("management_url", "inference_url")
    @classmethod
    def validate_runtime_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Runtime URL must use http:// or https://")
        return value.rstrip("/")


class RuntimeRegistryConfig(BaseModel):
    """Optional Registry connection and retry policy for a managed runtime."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    url: str | None = None
    required: bool = False
    heartbeat_seconds: float = Field(default=30, ge=0.1, le=3600)
    refresh_seconds: float = Field(default=300, ge=1, le=86_400)
    retry_initial_seconds: float = Field(default=1, ge=0.05, le=300)
    retry_max_seconds: float = Field(default=60, ge=0.1, le=3600)
    request_timeout_seconds: float = Field(default=5, gt=0, le=120)
    auth: RegistryClientAuth = Field(default_factory=RegistryClientAuth)
    instance: RuntimeInstanceIdentity = Field(default_factory=RuntimeInstanceIdentity)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        """Accept the initial flat gateway settings while moving to one schema."""

        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if data.get("url") and "enabled" not in data:
            data["enabled"] = True
        auth = dict(data.get("auth") or {})
        instance = dict(data.get("instance") or {})
        if data.pop("token_env", None):
            auth.update({"type": "bearer", "token_env": value["token_env"]})
        for old, new in (("instance_id", "id"), ("environment", "environment"), ("cluster", "cluster")):
            if old in data:
                instance[new] = data.pop(old)
        # Old deployment/model fields belonged to the deployment workaround and
        # are intentionally ignored during its compatibility window.
        data.pop("deployment_id", None)
        data.pop("model_id", None)
        if auth:
            data["auth"] = auth
        if instance:
            data["instance"] = instance
        return data

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Registry URL must use http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_enabled(self) -> "RuntimeRegistryConfig":
        if self.enabled and not self.url:
            raise ValueError("enabled Registry registration requires url")
        if self.auth.type == "bearer" and not (self.auth.token_env or self.auth.token_file):
            raise ValueError("bearer Registry auth requires token_env or token_file")
        if self.auth.type == "oauth2_client_credentials" and not all((
            self.auth.token_url, self.auth.client_id_env, self.auth.client_secret_env,
        )):
            raise ValueError("OAuth2 Registry auth requires token_url and client credential environment names")
        if self.auth.type == "mtls" and not all((self.auth.cert_file, self.auth.key_file)):
            raise ValueError("mTLS Registry auth requires cert_file and key_file")
        return self


def resolve_instance_id(
    kind: str, configured: str | None = None, identity_file: str | Path | None = None,
) -> str:
    """Resolve explicit, persisted, or newly generated stable runtime identity."""

    if configured:
        return configured
    path = Path(identity_file).expanduser() if identity_file else pra_home() / "instances" / f"{kind.lower()}.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and value.get("instance_id"):
            return str(value["instance_id"])
    instance_id = f"{kind.lower()}-{uuid.uuid4().hex}"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="instance-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump({"instance_id": instance_id, "instance_type": kind.upper()}, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return instance_id


class RegistryRegistrationClient:
    """Register, refresh observed state, heartbeat, and cleanly deregister."""

    def __init__(
        self,
        config: RuntimeRegistryConfig,
        instance_type: str,
        payload_factory: Callable[[str], Mapping[str, Any]],
        *,
        status_callback: Callable[[Mapping[str, Any]], None] | None = None,
        desired_callback: Callable[[Mapping[str, Any]], None] | None = None,
        token_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self.config = config
        self.instance_type = instance_type.upper()
        self.payload_factory = payload_factory
        self.status_callback = status_callback
        self.desired_callback = desired_callback
        self.token_provider = token_provider
        identity = config.instance
        self.instance_id = (
            resolve_instance_id(self.instance_type, identity.instance_id, identity.identity_file)
            if config.enabled and config.url else (identity.instance_id or "")
        )
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.registered = False
        self._last_refresh = 0.0
        self._oauth_token: tuple[str, float] | None = None
        self._lock = threading.RLock()
        self._status: dict[str, Any] = {
            "enabled": bool(config.enabled and config.url),
            "required": config.required,
            "status": "not_started" if config.enabled and config.url else "disabled",
            "instance_id": self.instance_id,
            "registration_success_total": 0,
            "registration_failure_total": 0,
            "heartbeat_success_total": 0,
            "heartbeat_failure_total": 0,
            "metrics": {
                "pra_registry_connected": 0,
                "pra_registry_registration_failures_total": 0,
                "pra_registry_heartbeat_failures_total": 0,
            },
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self) -> None:
        """Start resilient registration; strict mode establishes it first."""

        if not self.config.enabled or not self.config.url:
            return
        if self.thread is not None and self.thread.is_alive():
            return
        if self.config.required:
            self.register_now()
        self.thread = threading.Thread(
            target=self._run, name=f"pra-{self.instance_type.lower()}-registry", daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 3) -> None:
        if not self.config.enabled or not self.config.url:
            return
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=timeout)
        if self.registered:
            try:
                self._request("POST", f"/v1/instances/{self.instance_id}/deregister", {})
                self._set_status(status="offline", last_deregistered=time.time())
            except Exception as error:
                self._set_status(status="deregister_error", detail=str(error))

    def register_now(self) -> Mapping[str, Any]:
        try:
            result = self._request("POST", "/v1/instances/register", self._registration_payload())
        except Exception as error:
            self.registered = False
            self._bump("registration_failure_total", status="error", detail=str(error), last_attempt=time.time())
            raise
        self.registered = True
        self._last_refresh = time.monotonic()
        self._bump("registration_success_total", status="online", detail=None, last_registered=time.time())
        self._pull_desired()
        return result

    def heartbeat_now(self) -> Mapping[str, Any]:
        if not self.registered:
            return self.register_now()
        observed = dict(self.payload_factory(self.instance_id))
        body = {
            "health": observed.get("health", "healthy"),
            "uptime_seconds": max(0, time.time() - self.started_at),
            "observed_revision": int(observed.get("observed_revision", 1)),
            "runtime_summary": dict(observed.get("runtime_summary") or {}),
            "timestamp": time.time(),
        }
        if observed.get("capabilities_changed"):
            body["capabilities"] = observed.get("capabilities", {})
        try:
            result = self._request("POST", f"/v1/instances/{self.instance_id}/heartbeat", body)
        except Exception as error:
            self.registered = False
            self._bump("heartbeat_failure_total", status="error", detail=str(error), last_attempt=time.time())
            raise
        self._bump("heartbeat_success_total", status="online", detail=None, last_heartbeat=time.time())
        return result

    def publish_observed(self) -> Mapping[str, Any]:
        """Publish heavier state after an event or periodic refresh."""

        if not self.registered:
            return self.register_now()
        payload = self._registration_payload()
        allowed = {
            "health", "management_url", "inference_url", "component_version", "engine_version",
            "capabilities", "models", "runtime_summary", "observability", "observed_revision",
            "in_sync", "drift_fields", "labels", "metadata",
        }
        result = self._request(
            "PATCH", f"/v1/instances/{self.instance_id}/observed",
            {key: value for key, value in payload.items() if key in allowed},
        )
        self._last_refresh = time.monotonic()
        self._set_status(last_observed_update=time.time())
        self._pull_desired()
        return result

    def _run(self) -> None:
        retry = self.config.retry_initial_seconds
        while not self.stop_event.is_set():
            try:
                self.heartbeat_now()
                if time.monotonic() - self._last_refresh >= self.config.refresh_seconds:
                    self.publish_observed()
                retry = self.config.retry_initial_seconds
                delay = self.config.heartbeat_seconds
            except Exception:
                delay = retry
                retry = min(self.config.retry_max_seconds, retry * 2)
            self.stop_event.wait(delay)

    def _registration_payload(self) -> dict[str, Any]:
        payload = dict(self.payload_factory(self.instance_id))
        identity = self.config.instance
        payload.update({
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "name": identity.name or payload.get("name") or self.instance_id,
            "environment": identity.environment,
            "region": identity.region,
            "cluster": identity.cluster,
            "namespace": identity.namespace,
            "labels": {**dict(payload.get("labels") or {}), **identity.labels},
            "metadata": {**dict(payload.get("metadata") or {}), **identity.metadata},
            "registration_source": "SELF",
        })
        return payload

    def _pull_desired(self) -> None:
        if self.desired_callback is None:
            return
        try:
            value = self._request("GET", f"/v1/instances/{self.instance_id}/desired")
            self.desired_callback(value)
        except Exception as error:
            self._set_status(desired_pull_error=str(error))

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        assert self.config.url is not None
        headers = {"Accept": "application/json", "User-Agent": "pra-runtime-registry/1"}
        token = self._token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.config.url}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(
            request, timeout=self.config.request_timeout_seconds, context=self._ssl_context(),
        ) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}

    def _token(self) -> str | None:
        if self.token_provider is not None:
            return self.token_provider()
        auth = self.config.auth
        if auth.type == "bearer":
            return auth.bearer_token()
        if auth.type != "oauth2_client_credentials":
            return None
        if self._oauth_token and self._oauth_token[1] > time.time() + 10:
            return self._oauth_token[0]
        client_id = os.environ.get(auth.client_id_env or "")
        client_secret = os.environ.get(auth.client_secret_env or "")
        if not client_id or not client_secret:
            raise RuntimeError("OAuth2 client credentials are unavailable")
        data = urllib.parse.urlencode({
            "grant_type": "client_credentials", "client_id": client_id,
            "client_secret": client_secret, "scope": " ".join(auth.scopes),
        }).encode("utf-8")
        request = urllib.request.Request(
            str(auth.token_url), data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds, context=self._ssl_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
        token = str(payload["access_token"])
        self._oauth_token = (token, time.time() + float(payload.get("expires_in", 300)))
        return token

    def _ssl_context(self) -> ssl.SSLContext | None:
        auth = self.config.auth
        if auth.type != "mtls" and not auth.ca_file:
            return None
        context = ssl.create_default_context(cafile=auth.ca_file)
        if auth.cert_file:
            context.load_cert_chain(auth.cert_file, auth.key_file)
        return context

    def _bump(self, key: str, **values: Any) -> None:
        with self._lock:
            self._status[key] = int(self._status.get(key, 0)) + 1
            metrics = dict(self._status.get("metrics") or {})
            if key == "registration_failure_total":
                metrics["pra_registry_registration_failures_total"] = self._status[key]
            elif key == "heartbeat_failure_total":
                metrics["pra_registry_heartbeat_failures_total"] = self._status[key]
            if values.get("status") == "online":
                metrics["pra_registry_connected"] = 1
            elif values.get("status") in {"error", "offline", "deregister_error"}:
                metrics["pra_registry_connected"] = 0
            self._status["metrics"] = metrics
        self._set_status(**values)

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)
            snapshot = dict(self._status)
        if self.status_callback:
            self.status_callback(snapshot)


def runtime_host() -> str:
    """Return the advertised host default without resolving private addresses."""

    return socket.gethostname()
