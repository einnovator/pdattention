"""Control Plane configuration with CLI, environment, YAML, and defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .rbac import Role


class IdentityProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str = Field(pattern="^(local|github|google|oidc|saml)$")
    enabled: bool = True
    client_id: str | None = None
    client_secret_env: str | None = None
    issuer: str | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    userinfo_url: str | None = None
    jwks_url: str | None = None
    metadata_url: str | None = None
    scopes: list[str] = Field(default_factory=lambda: ["openid", "profile", "email"])
    saml_metadata_url: str | None = None
    role_claim: str = "pra_role"
    default_role: Role = Role.VIEWER

    def client_secret(self) -> str | None:
        return os.environ.get(self.client_secret_env or "") or None


class LocalUserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    password_env: str
    role: Role = Role.VIEWER
    display_name: str | None = None

    def password(self) -> str | None:
        return os.environ.get(self.password_env)


class ControlAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    providers: list[IdentityProviderConfig] = Field(default_factory=lambda: [
        IdentityProviderConfig(name="local", kind="local")
    ])
    local_users: list[LocalUserConfig] = Field(default_factory=list)
    cookie_secret_env: str = "PRA_CONTROL_COOKIE_SECRET"
    cookie_secure: bool = True
    allow_local_auth_non_loopback: bool = False
    session_ttl_seconds: int = Field(default=28_800, ge=300, le=604_800)

    def cookie_secret(self) -> str | None:
        return os.environ.get(self.cookie_secret_env)


class EngineTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    management_url: str
    token_env: str | None = None
    environment: str = "development"
    region: str = "local"
    cluster: str = "default"
    namespace: str = "default"
    host: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    def token(self) -> str | None:
        return os.environ.get(self.token_env or "") or None


class FleetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    discovery_mode: str = Field(default="static", pattern="^(static|manual|registry|combined)$")
    engines: list[EngineTargetConfig] = Field(default_factory=list)
    poll_interval_seconds: int = Field(default=15, ge=2, le=300)


class ServiceLinkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str | None = None
    token_env: str | None = None

    def token(self) -> str | None:
        return os.environ.get(self.token_env or "") or None


class ControlPlaneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=9300, ge=1, le=65535)
    public_url: str = "http://127.0.0.1:9300"
    database_url: str = "sqlite:///./pra-control.db"
    auth: ControlAuthConfig = Field(default_factory=ControlAuthConfig)
    registry: ServiceLinkConfig = Field(default_factory=lambda: ServiceLinkConfig(url="http://127.0.0.1:9200"))
    fleet: FleetConfig = Field(default_factory=FleetConfig)
    grafana: ServiceLinkConfig = Field(default_factory=ServiceLinkConfig)
    tempo: ServiceLinkConfig = Field(default_factory=ServiceLinkConfig)
    prometheus: ServiceLinkConfig = Field(default_factory=ServiceLinkConfig)
    agent_enabled: bool = True
    replay_limit: int = Field(default=250, ge=10, le=5000)

    @field_validator("public_url")
    @classmethod
    def public_url_is_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("public_url must be HTTP(S)")
        return value.rstrip("/")

    def validate_security(self) -> None:
        local_only = all(provider.kind == "local" for provider in self.auth.providers)
        if local_only and self.host not in {"127.0.0.1", "localhost", "::1"} and not self.auth.allow_local_auth_non_loopback:
            raise ValueError("local-only authentication may bind only to loopback unless explicitly opted in")
        if not self.auth.cookie_secret():
            raise ValueError(f"set {self.auth.cookie_secret_env} for signed sessions")
        for provider in self.auth.providers:
            if provider.kind in {"github", "google", "oidc"} and not provider.client_secret():
                raise ValueError(f"provider {provider.name!r} requires {provider.client_secret_env}")

    @classmethod
    def load(
        cls, path: str | Path | None = None, *, overrides: Mapping[str, Any] | None = None,
    ) -> "ControlPlaneConfig":
        _load_dotenv(Path(path).parent / ".env" if path else Path(".env"))
        raw: dict[str, Any] = {}
        if path:
            value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            raw = dict(value.get("control_plane", value))
            for section in ("registry", "fleet", "grafana", "tempo", "prometheus"):
                if section in value and section not in raw:
                    raw[section] = value[section]
        environment = {
            "PRA_CONTROL_HOST": ("host", str),
            "PRA_CONTROL_PORT": ("port", int),
            "PRA_CONTROL_PUBLIC_URL": ("public_url", str),
            "PRA_CONTROL_DATABASE_URL": ("database_url", str),
        }
        for name, (field, cast) in environment.items():
            if os.environ.get(name):
                raw[field] = cast(os.environ[name])
        if os.environ.get("PRA_REGISTRY_URL"):
            raw["registry"] = {**dict(raw.get("registry") or {}), "url": os.environ["PRA_REGISTRY_URL"]}
        raw.update(dict(overrides or {}))
        return cls.model_validate(raw)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
