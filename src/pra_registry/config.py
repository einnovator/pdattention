"""Registry configuration with YAML, dotenv, and environment overlays."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class RegistryAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = "none"
    static_token: SecretStr | None = None
    static_token_env: str | None = None
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    service_credentials: dict[str, SecretStr] = Field(default_factory=dict)

    def token(self) -> str | None:
        if self.static_token:
            return self.static_token.get_secret_value()
        return os.environ.get(self.static_token_env or "") or None


class RegistryObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    prometheus_enabled: bool = False
    otel_enabled: bool = False
    otel_endpoint: str | None = None


class InstanceRegistrationPolicy(BaseModel):
    """Admission and liveness policy for self-registering runtimes."""

    model_config = ConfigDict(extra="forbid")
    allowed_identities: list[str] = Field(default_factory=list)
    allowed_environments: list[str] = Field(default_factory=list)
    allowed_clusters: list[str] = Field(default_factory=list)
    instance_name_pattern: str | None = None
    offline_after_seconds: int = Field(default=90, ge=15, le=86_400)


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=9200, ge=1, le=65535)
    database_url: str = "sqlite:///./pra-registry.db"
    auth: RegistryAuthConfig = Field(default_factory=RegistryAuthConfig)
    observability: RegistryObservabilityConfig = Field(default_factory=RegistryObservabilityConfig)
    instance_registration: InstanceRegistrationPolicy = Field(default_factory=InstanceRegistrationPolicy)

    def validate_binding(self) -> None:
        if self.auth.mode == "none" and self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Unauthenticated registry may bind only to loopback")
        if self.auth.mode == "static_token" and not self.auth.token():
            raise ValueError("static_token auth requires static_token or static_token_env")
        if self.auth.mode in {"oidc", "jwt", "oidc_jwt"} and not all((
            self.auth.oidc_issuer, self.auth.oidc_audience, self.auth.oidc_jwks_url,
        )):
            raise ValueError("OIDC/JWT auth requires issuer, audience, and JWKS URL")
        if self.auth.mode == "service_credentials" and not self.auth.service_credentials:
            raise ValueError("service_credentials auth requires at least one credential")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RegistryConfig":
        _load_dotenv(Path(path).parent / ".env" if path else Path(".env"))
        raw: dict[str, Any] = {}
        if path:
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            raw = dict(loaded.get("registry", loaded))
        env_map = {
            "PRA_REGISTRY_HOST": ("host", str),
            "PRA_REGISTRY_PORT": ("port", int),
            "PRA_REGISTRY_DATABASE_URL": ("database_url", str),
        }
        for name, (field, cast) in env_map.items():
            if name in os.environ:
                raw[field] = cast(os.environ[name])
        auth = dict(raw.get("auth") or {})
        if os.environ.get("PRA_REGISTRY_AUTH_MODE"):
            auth["mode"] = os.environ["PRA_REGISTRY_AUTH_MODE"]
        if os.environ.get("PRA_REGISTRY_TOKEN"):
            auth["static_token"] = os.environ["PRA_REGISTRY_TOKEN"]
        if auth:
            raw["auth"] = auth
        return cls.model_validate(raw)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
