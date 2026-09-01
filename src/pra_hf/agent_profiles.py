"""Versioned agent-profile loading and shared agent launch services."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .agent import PRAAgent, PRAAgentConfig
from .agent_transport import (
    ContextTransportMode,
    NegotiatedRemoteBackend,
    resolve_wire_mode,
)
from .product_config import deep_merge, pra_home, read_yaml
from .observability import Observability
from .runtime import PRARuntime, PRARuntimeConfig
from .runtime_providers import RuntimeConfig
from .session_service import LocalSessionService
from .skill_records import Skill, load_skill_directory
from .task_scope import TaskScopePolicy


@dataclass(frozen=True)
class ToolPolicy:
    approval: str = "ask"
    allow_writes: bool = False
    allow_destructive: bool = False
    candidates: int = 8
    max_rounds: int = 1
    external: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.approval not in {"ask", "allow_safe", "allow_all", "deny"}:
            raise ValueError(f"Unknown tool approval policy: {self.approval}")
        if self.candidates <= 0 or self.max_rounds < 0:
            raise ValueError("Tool candidate count must be positive and rounds cannot be negative.")


@dataclass(frozen=True)
class AgentProfile:
    """Product behavior layered above low-level PRA runtime configuration."""

    name: str
    model: str | None = None
    model_revision: str | None = None
    runtime_mode: str = "embedded"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pra: str | None = None
    workspace: str = "."
    sessions_path: str = ".pra/sessions"
    resume_last: bool = False
    credentials_file: str | None = None
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    skill_directories: tuple[str, ...] = ()
    mcp: Mapping[str, Any] = field(default_factory=dict)
    tasks: Mapping[str, Any] = field(default_factory=dict)
    context_records: int = 12
    context_transport: ContextTransportMode | str = ContextTransportMode.AUTO
    allow_text_fallback: bool = True
    required_context_capabilities: tuple[str, ...] = ()
    max_new_tokens: int = 1024
    reserved: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_records <= 0 or self.max_new_tokens <= 0:
            raise ValueError("Agent context and generation budgets must be positive.")
        object.__setattr__(self, "skill_directories", tuple(self.skill_directories))
        object.__setattr__(self, "context_transport", ContextTransportMode(self.context_transport))
        object.__setattr__(
            self, "required_context_capabilities", tuple(self.required_context_capabilities)
        )
        object.__setattr__(self, "mcp", dict(self.mcp))
        object.__setattr__(self, "tasks", dict(self.tasks))
        object.__setattr__(self, "reserved", dict(self.reserved))

    @classmethod
    def from_dict(cls, name: str, values: Mapping[str, Any]) -> "AgentProfile":
        value = dict(values)
        model_value = value.get("model")
        if isinstance(model_value, Mapping):
            model = model_value.get("id")
            revision = model_value.get("revision")
        else:
            model = model_value
            revision = None
        runtime_value = dict(value.get("runtime", {}))
        engine = str(runtime_value.get("engine", "hf"))
        mode = str(runtime_value.get("mode", "embedded"))
        runtime = RuntimeConfig(
            engine=engine,
            model=str(model) if model else None,
            revision=str(revision) if revision else None,
            endpoint=runtime_value.get("endpoint"),
            device=str(runtime_value.get("device", "auto")),
            dtype=str(runtime_value.get("dtype", "auto")),
            engine_options=dict(runtime_value.get("engine_options", {})),
        )
        pra_value = value.get("pra")
        if isinstance(pra_value, Mapping):
            pra_value = pra_value.get("profile") or pra_value.get("config") or pra_value.get("bundle")
        tools_value = dict(value.get("tools", {}))
        tools = ToolPolicy(
            approval=str(tools_value.get("approval", "ask")),
            allow_writes=bool(tools_value.get("allow_writes", False)),
            allow_destructive=bool(tools_value.get("allow_destructive", False)),
            candidates=int(tools_value.get("candidates", 8)),
            max_rounds=int(tools_value.get("max_rounds", 1)),
            external=tuple(tools_value.get("external", ())),
        )
        skills = value.get("skills", {})
        skill_directories = skills.get("directories", ()) if isinstance(skills, Mapping) else skills or ()
        sessions = dict(value.get("sessions", {}))
        credentials = value.get("credentials", {})
        credentials_file = credentials.get("file") if isinstance(credentials, Mapping) else credentials
        context = dict(value.get("context", {}))
        generation = dict(value.get("generation", {}))
        known = {
            "model", "runtime", "pra", "workspace", "sessions", "credentials", "tools",
            "skills", "mcp", "tasks", "context", "generation", "subagents", "memory", "connectors",
        }
        return cls(
            name=name,
            model=str(model) if model else None,
            model_revision=str(revision) if revision else None,
            runtime_mode=mode,
            runtime=runtime,
            pra=str(pra_value) if pra_value else None,
            workspace=str(value.get("workspace", ".")),
            sessions_path=str(sessions.get("path", ".pra/sessions")),
            resume_last=bool(sessions.get("resume_last", False)),
            credentials_file=str(credentials_file) if credentials_file else None,
            tools=tools,
            skill_directories=tuple(str(path) for path in skill_directories),
            mcp=dict(value.get("mcp", {})),
            tasks=dict(value.get("tasks", {})),
            context_records=int(context.get("records", 12)),
            context_transport=str(context.get("transport", "auto")),
            allow_text_fallback=bool(context.get("allow_text_fallback", True)),
            required_context_capabilities=tuple(context.get("require", ())),
            max_new_tokens=int(generation.get("max_new_tokens", 1024)),
            reserved={key: value[key] for key in ("subagents", "memory", "connectors") if key in value},
        )

    def redacted_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.credentials_file:
            value["credentials_file"] = str(Path(self.credentials_file).expanduser())
        value["credentials"] = "referenced, not loaded into profile output" if self.credentials_file else None
        return value


@dataclass(frozen=True)
class AgentProfileDocument:
    version: int
    default_profile: str
    profiles: Mapping[str, AgentProfile]
    sources: tuple[str, ...] = ()

    def resolve(self, name: str | None = None) -> AgentProfile:
        selected = name or self.default_profile
        try:
            return self.profiles[selected]
        except KeyError as error:
            raise KeyError(f"Unknown agent profile '{selected}'. Available: {', '.join(sorted(self.profiles))}") from error


class AgentProfileRegistry:
    """Discover, merge, validate, and resolve user/project agent profiles."""

    DEFAULTS = {
        "version": 1,
        "default_profile": "default",
        "profiles": {
            "default": {
                "runtime": {"mode": "embedded", "engine": "hf", "device": "auto"},
                "pra": {"profile": "REFERENCE_CORRECTNESS"},
                "tools": {"approval": "ask", "allow_writes": False, "allow_destructive": False},
            }
        },
    }

    def discover(self, workspace: str | Path | None = None) -> tuple[Path, ...]:
        root = Path(workspace or ".").expanduser().resolve()
        candidates = (
            Path.home() / ".config" / "pra" / "agents.yaml",
            root / ".pra" / "agents.yaml",
        )
        return tuple(path for path in candidates if path.is_file())

    def load(
        self,
        *,
        workspace: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> AgentProfileDocument:
        value: dict[str, Any] = dict(self.DEFAULTS)
        sources = ["package defaults"]
        for path in self.discover(workspace):
            value = deep_merge(value, read_yaml(path))
            sources.append(str(path))
        if config_path:
            path = Path(config_path).expanduser().resolve()
            value = deep_merge(value, read_yaml(path))
            sources.append(str(path))
        version = int(value.get("version", 0))
        if version != 1:
            raise ValueError(f"Unsupported agent configuration version: {version}")
        profiles = {
            str(name): AgentProfile.from_dict(str(name), profile)
            for name, profile in dict(value.get("profiles", {})).items()
        }
        default = str(value.get("default_profile", "default"))
        if default not in profiles:
            raise ValueError(f"Default agent profile does not exist: {default}")
        return AgentProfileDocument(version, default, profiles, tuple(sources))

    def resolve(
        self,
        *,
        profile_name: str | None = None,
        workspace: str | Path | None = None,
        config_path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> tuple[AgentProfile, tuple[str, ...]]:
        document = self.load(workspace=workspace, config_path=config_path)
        selected = document.resolve(profile_name)
        if overrides:
            raw = selected.redacted_dict()
            raw.pop("credentials", None)
            raw = _profile_to_input(raw)
            selected = AgentProfile.from_dict(selected.name, deep_merge(raw, overrides))
        return selected, document.sources


def _profile_to_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the normalized dataclass form back to profile YAML shape."""

    runtime = dict(value.get("runtime", {}))
    runtime.pop("model", None)
    runtime.pop("revision", None)
    runtime.pop("pra_bundle", None)
    runtime.pop("profile", None)
    return {
        "model": {"id": value.get("model"), "revision": value.get("model_revision")},
        "runtime": {**runtime, "mode": value.get("runtime_mode", "embedded")},
        "pra": {"profile": value.get("pra")},
        "workspace": value.get("workspace"),
        "sessions": {"path": value.get("sessions_path"), "resume_last": value.get("resume_last")},
        "credentials": {"file": value.get("credentials_file")},
        "tools": value.get("tools", {}),
        "skills": {"directories": value.get("skill_directories", ())},
        "mcp": value.get("mcp", {}),
        "tasks": value.get("tasks", {}),
        "context": {
            "records": value.get("context_records"),
            "transport": (
                value.get("context_transport", ContextTransportMode.AUTO).value
                if isinstance(value.get("context_transport", ContextTransportMode.AUTO), ContextTransportMode)
                else value.get("context_transport", "auto")
            ),
            "allow_text_fallback": value.get("allow_text_fallback", True),
            "require": value.get("required_context_capabilities", ()),
        },
        "generation": {"max_new_tokens": value.get("max_new_tokens")},
    }


class RemoteOpenAIBackend(NegotiatedRemoteBackend):
    """Compatibility name for the negotiated OpenAI/PRA remote transport."""

    def __init__(
        self,
        endpoint: str,
        model: str | None,
        credentials_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            endpoint,
            model,
            credentials_file=credentials_file,
            **kwargs,
        )


@dataclass(frozen=True)
class AgentLaunch:
    agent: PRAAgent
    profile: AgentProfile
    summary: Mapping[str, Any]


class AgentLauncher:
    """Build the same PRA agent for CLI, TUI, scripts, and web transports."""

    def launch(
        self, profile: AgentProfile, *, observability: Observability | None = None
    ) -> AgentLaunch:
        workspace = Path(profile.workspace).expanduser()
        sessions = Path(profile.sessions_path).expanduser()
        allow_all = profile.tools.approval == "allow_all"
        agent_config = PRAAgentConfig(
            allow_writes=profile.tools.allow_writes or allow_all,
            allow_destructive=profile.tools.allow_destructive or allow_all,
            context_records=profile.context_records,
            tool_candidates=profile.tools.candidates,
            max_tool_rounds=profile.tools.max_rounds,
            max_new_tokens=profile.max_new_tokens,
            task_scope=profile.tasks.get("scope_policy", TaskScopePolicy.TASK_ADAPTIVE.value),
        )
        skill_values: list[Skill] = []
        for directory in profile.skill_directories:
            path = Path(directory).expanduser()
            if path.is_dir():
                skill_values.extend(load_skill_directory(path))
        mode = _runtime_mode(profile)
        if mode == "embedded":
            if not profile.model:
                raise ValueError("Embedded agent profiles require a model.")
            agent = PRAAgent.from_pretrained(
                profile.model,
                config=agent_config,
                workspace=workspace,
                skills=tuple(skill_values),
                session_service=LocalSessionService(sessions),
                runtime_config={"profile": profile.pra} if profile.pra else None,
                revision=profile.model_revision,
                observability=observability,
            )
        else:
            if not profile.runtime.endpoint:
                raise ValueError(f"Runtime mode '{mode}' requires an endpoint.")
            backend = NegotiatedRemoteBackend(
                profile.runtime.endpoint,
                profile.model,
                transport=profile.context_transport,
                allow_text_fallback=profile.allow_text_fallback,
                required_capabilities=profile.required_context_capabilities,
                credentials_file=profile.credentials_file,
                observability=observability,
            )
            capabilities = backend.capabilities()
            resolved_transport = resolve_wire_mode(
                profile.context_transport,
                capabilities,
                required=profile.required_context_capabilities,
                allow_text_fallback=profile.allow_text_fallback,
            )
            runtime = PRARuntime(
                config=PRARuntimeConfig(profile=profile.pra),
                backend=backend,
                native_result_routing=False,
                session_service=LocalSessionService(sessions),
                observability=observability,
            )
            agent = PRAAgent(runtime, config=agent_config, observability=observability)
        transport_summary = (
            {
                "requested": profile.context_transport.value,
                "resolved": "EMBEDDED_TEXT",
                "capability_source": "python_inspection",
                "typed_records": False,
                "resource_delta": False,
                "native_kv": False,
            }
            if mode == "embedded"
            else {
                "requested": profile.context_transport.value,
                "resolved": resolved_transport.value,
                "capability_source": capabilities.capability_source,
                "typed_records": capabilities.typed_records,
                "resource_delta": capabilities.resource_delta,
                "native_kv": capabilities.native_kv,
                "gateway_mode": capabilities.gateway_mode,
                "integration_level": capabilities.integration_level,
            }
        )
        summary = {
            "agent_profile": profile.name,
            "model": profile.model or "provider-default",
            "revision": profile.model_revision or "provider-default",
            "runtime_mode": mode,
            "engine": profile.runtime.engine,
            "endpoint": profile.runtime.endpoint or "embedded",
            "pra_profile": profile.pra or "REFERENCE_CORRECTNESS",
            "workspace": str(workspace),
            "sessions": str(sessions),
            "tools": profile.tools.approval,
            "skills": len(skill_values),
            "context_transport": transport_summary,
        }
        agent.product_summary = summary
        return AgentLaunch(agent, profile, summary)


def _runtime_mode(profile: AgentProfile) -> str:
    return profile.runtime_mode


def load_mcp_config(profile: AgentProfile) -> dict[str, Any]:
    """Merge external then inline MCP configuration without loading secrets."""

    value: dict[str, Any] = {}
    path = profile.mcp.get("file")
    if path:
        value = read_yaml(Path(str(path)).expanduser()) if str(path).endswith((".yaml", ".yml")) else json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    inline = {key: item for key, item in profile.mcp.items() if key != "file"}
    return deep_merge(value, inline)
