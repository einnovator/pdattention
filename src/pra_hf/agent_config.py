"""Typed, secret-referencing configuration for PRA Agent clients."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .product_config import deep_merge


class SecretReference(BaseModel):
    """Authentication metadata that stores references rather than secret values."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["none", "bearer", "oauth", "oidc", "client_credentials", "headers", "mtls"] = "none"
    token_env: str | None = None
    token_file: str | None = None
    client_id: str | None = None
    client_secret_env: str | None = None
    token_url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    cert_file: str | None = None
    key_file: str | None = None

    def token(self) -> str | None:
        if self.token_env:
            return os.environ.get(self.token_env)
        if self.token_file:
            path = Path(self.token_file).expanduser()
            return path.read_text(encoding="utf-8").strip() if path.is_file() else None
        if self.client_secret_env:
            return os.environ.get(self.client_secret_env)
        return None

    def resolved_headers(self) -> dict[str, str]:
        values = dict(self.headers)
        token = self.token()
        if token and self.type in {"bearer", "oauth", "oidc", "client_credentials"}:
            values["Authorization"] = f"Bearer {token}"
        return values


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "openai"
    base_url: str | None = None
    model: str | None = None
    engine_instance: str | None = None
    runtime_model_id: str = "default"
    api_key_env: str | None = None
    credentials_file: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    required: bool = False
    transport: Literal["http", "stdio"] = "http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    auth: SecretReference = Field(default_factory=SecretReference)
    timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    retries: int = Field(default=1, ge=0, le=10)
    backoff_seconds: float = Field(default=0.25, ge=0, le=30)
    tool_allow: list[str] = Field(default_factory=lambda: ["*"])
    tool_deny: list[str] = Field(default_factory=list)
    resource_allow: list[str] = Field(default_factory=lambda: ["*"])
    resource_deny: list[str] = Field(default_factory=list)
    namespace: str | None = None
    prompts_enabled: bool = False
    annotations: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value

    @model_validator(mode="after")
    def validate_transport_target(self) -> "MCPServerConfig":
        if self.enabled and self.transport == "http" and not self.url:
            raise ValueError("HTTP MCP servers require url.")
        if self.enabled and self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers require command.")
        return self


class MCPAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    cache_ttl_seconds: float = Field(default=60.0, ge=0, le=3600)
    prompts_enabled: bool = False


class ControlPlaneClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    required: bool = False
    url: str | None = None
    auth: SecretReference = Field(default_factory=SecretReference)
    timeout_seconds: float = Field(default=10.0, gt=0, le=300)
    cache_ttl_seconds: float = Field(default=30.0, ge=0, le=3600)
    follow_recommendation: bool = True
    prefer_direct_rest_for: list[str] = Field(default_factory=lambda: [
        "model_discovery", "engine_discovery", "qualification",
    ])

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value


class PasteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_threshold_chars: int = Field(default=2000, ge=1)
    block_threshold_lines: int = Field(default=20, ge=1)


class TUIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history_size: int = Field(default=1000, ge=0, le=100_000)
    history_file: str = "~/.local/share/pra/agent/history"
    suppress_duplicates: bool = True
    autocomplete: bool = True
    theme: Literal["auto", "dark", "light", "none"] = "auto"
    verbose_tools: bool = False
    paste: PasteConfig = Field(default_factory=PasteConfig)


class AgentSessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = ".pra/sessions"
    resume_last: bool = False


class ConfirmationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    high_impact_tools: bool = True
    unknown_mcp_mutations: bool = True


class AgentRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    provider: str | None = None
    system_prompt_file: str | None = None
    user_id: str = "local-user"
    tenant_id: str = "default"
    task_scope: str = "task-adaptive"
    context_records: int = Field(default=12, gt=0)
    tool_candidates: int = Field(default=8, gt=0)
    max_tool_rounds: int = Field(default=1, ge=0)
    allow_writes: bool = False
    allow_destructive: bool = False
    max_new_tokens: int = Field(default=256, gt=0)
    confirmations: ConfirmationConfig = Field(default_factory=ConfirmationConfig)


class PRAAgentSettings(BaseModel):
    """Complete Agent application schema shared by SDK, CLI, and TUI."""

    model_config = ConfigDict(extra="forbid")
    agent: AgentRuntimeSettings = Field(default_factory=AgentRuntimeSettings)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    mcp: MCPAgentConfig = Field(default_factory=MCPAgentConfig)
    control_plane: ControlPlaneClientConfig | None = None
    tui: TUIConfig = Field(default_factory=TUIConfig)
    session: AgentSessionConfig = Field(default_factory=AgentSessionConfig)
    source_file: str | None = Field(default=None, exclude=True)

    @classmethod
    def from_file(cls, path: str | Path) -> "PRAAgentSettings":
        source = Path(path).expanduser().resolve()
        value = _read_config(source)
        value["source_file"] = str(source)
        return cls.model_validate(value)

    @classmethod
    def merge(cls, *values: "PRAAgentSettings | Mapping[str, Any]") -> "PRAAgentSettings":
        merged: dict[str, Any] = {}
        for value in values:
            payload = value.model_dump(exclude_none=True) if isinstance(value, BaseModel) else dict(value)
            merged = deep_merge(merged, payload)
        return cls.model_validate(merged)

    @classmethod
    def compose(
        cls, *, config_file: str | Path | None = None,
        config: "PRAAgentSettings | Mapping[str, Any] | None" = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "PRAAgentSettings":
        """Apply defaults < environment < file < object < explicit overrides."""
        values: list[Mapping[str, Any] | PRAAgentSettings] = [_environment_values()]
        if config_file:
            file_settings = cls.from_file(config_file)
            values.append(file_settings)
        if config is not None:
            values.append(config)
        if overrides:
            values.append(overrides)
        result = cls.merge(*values)
        if config_file:
            result.source_file = str(Path(config_file).expanduser().resolve())
        return result

    def resolve_env(self) -> dict[str, Any]:
        """Return runtime credentials separately from serializable configuration."""
        return {
            "providers": {
                name: os.environ.get(provider.api_key_env or "") or None
                for name, provider in self.providers.items()
            },
            "mcp": {name: server.auth.token() for name, server in self.mcp.servers.items()},
            "control_plane": self.control_plane.auth.token() if self.control_plane else None,
        }

    def redacted(self) -> dict[str, Any]:
        value = self.model_dump(mode="json", exclude_none=True)
        for server in value.get("mcp", {}).get("servers", {}).values():
            auth = server.get("auth", {})
            if auth.get("headers"):
                auth["headers"] = {name: "<redacted>" for name in auth["headers"]}
        control = value.get("control_plane") or {}
        if control.get("auth", {}).get("headers"):
            control["auth"]["headers"] = {
                name: "<redacted>" for name in control["auth"]["headers"]
            }
        return value

    def save(self, path: str | Path | None = None) -> Path:
        target_value = path or self.source_file
        if not target_value:
            raise ValueError("A config path is required for explicit persistence.")
        target = Path(target_value).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude={"source_file"}, exclude_none=True)
        if target.suffix.casefold() == ".json":
            text = json.dumps(payload, indent=2) + "\n"
        elif target.suffix.casefold() == ".toml":
            raise ValueError("Writing TOML is not supported; save as YAML or JSON.")
        else:
            text = yaml.safe_dump(payload, sort_keys=False)
        target.write_text(text, encoding="utf-8")
        self.source_file = str(target)
        return target


def _read_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    if suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    else:
        raise ValueError("Agent config must be YAML, JSON, or TOML.")
    if not isinstance(value, dict):
        raise ValueError("Agent config root must be an object.")
    return dict(value)


def _environment_values() -> dict[str, Any]:
    values: dict[str, Any] = {}
    agent: dict[str, Any] = {}
    if os.environ.get("PRA_AGENT_MODEL"):
        agent["model"] = os.environ["PRA_AGENT_MODEL"]
    if os.environ.get("PRA_AGENT_PROVIDER"):
        agent["provider"] = os.environ["PRA_AGENT_PROVIDER"]
    if os.environ.get("PRA_AGENT_USER_ID"):
        agent["user_id"] = os.environ["PRA_AGENT_USER_ID"]
    if agent:
        values["agent"] = agent
    return values
