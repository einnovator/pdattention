"""Task-aware agent facade over the product PRA runtime.

The facade is intentionally provider-neutral: model generation proposes text or
a typed tool call, while the host owns capability disclosure, authorization,
task mutation, and durable session state.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .agent_execution import (
    ExecutionAuthorization,
    ToolCall,
    parse_tool_call,
    resource_tool_schema,
)
from .agent_resources import AgentResource, DiscoveryRequest, PersistentResourceIndex, SideEffectClass
from .agent_transport import AgentTurnContext, NegotiatedRemoteBackend, render_text_prompt
from .agent_config import PRAAgentSettings
from .agent_control_plane import ControlPlaneClient, InferenceTarget, InferenceTargetManager
from .agent_mcp import MCPClientManager
from .agent_workspace import AttachmentManager, HistoryManager, export_session
from .agent_execution import SafeToolExecutor
from .capability_sdk import AgentConfig, CapabilitySDK
from .context_records import ContextRecord, RecordType
from .model import GenerationResult
from .runtime import PRARuntime, PRARuntimeConfig, RuntimeToolExecution
from .session_service import AgentSessionState, LocalSessionService, SessionService
from .skill_records import Skill
from .task_context import TaskEvent, TaskEventType
from .task_scope import ScopeSelection, TaskScopePolicy
from .toolsets import Toolset, default_toolset
from .observability import DISABLED_OBSERVABILITY, Observability


@dataclass(frozen=True)
class PRAAgentConfig:
    """Host policy for one task-aware PRA agent."""

    user_id: str = "local-user"
    tenant_id: str = "default"
    task_scope: TaskScopePolicy | str = TaskScopePolicy.TASK_ADAPTIVE
    context_records: int = 12
    tool_candidates: int = 8
    max_tool_rounds: int = 1
    allow_writes: bool = False
    allow_destructive: bool = False
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_scope", TaskScopePolicy(self.task_scope))
        if min(self.context_records, self.tool_candidates, self.max_new_tokens) <= 0:
            raise ValueError("Context, tool, and generation budgets must be positive.")
        if self.max_tool_rounds < 0:
            raise ValueError("max_tool_rounds cannot be negative.")


@dataclass(frozen=True)
class AgentTurn:
    """Auditable result of one user turn."""

    text: str
    session: AgentSessionState
    selected_record_ids: tuple[str, ...] = ()
    disclosed_tool_uris: tuple[str, ...] = ()
    disclosed_skill_uris: tuple[str, ...] = ()
    tool_executions: tuple[RuntimeToolExecution, ...] = ()
    transport: Mapping[str, object] = field(default_factory=dict)


def _generated_text(value: str | GenerationResult) -> str:
    return value.text if isinstance(value, GenerationResult) else str(value)


class PRAAgent:
    """Long-running agent coupling task state, PRA context, and safe tools."""

    def __init__(
        self,
        runtime: PRARuntime,
        *,
        config: PRAAgentConfig | None = None,
        toolset: Toolset | None = None,
        authorization_callback: Callable[[AgentResource, ToolCall], bool] | None = None,
        observability: Observability | None = None,
        settings: PRAAgentSettings | Mapping[str, Any] | None = None,
        config_file: str | Path | None = None,
    ) -> None:
        self.runtime = runtime
        base_settings = PRAAgentSettings.compose(config_file=config_file, config=settings)
        host = base_settings.agent
        self.config = config or PRAAgentConfig(
            user_id=host.user_id,
            tenant_id=host.tenant_id,
            task_scope=host.task_scope,
            context_records=host.context_records,
            tool_candidates=host.tool_candidates,
            max_tool_rounds=host.max_tool_rounds,
            allow_writes=host.allow_writes,
            allow_destructive=host.allow_destructive,
            max_new_tokens=host.max_new_tokens,
        )
        self.toolset = toolset
        self.authorization_callback = authorization_callback
        self.observability = observability or getattr(
            runtime, "observability", DISABLED_OBSERVABILITY
        )
        self.session = None
        self.settings = PRAAgentSettings.merge(
            base_settings,
            {"agent": {
                "user_id": self.config.user_id,
                "tenant_id": self.config.tenant_id,
                "context_records": self.config.context_records,
                "tool_candidates": self.config.tool_candidates,
                "max_tool_rounds": self.config.max_tool_rounds,
                "allow_writes": self.config.allow_writes,
                "allow_destructive": self.config.allow_destructive,
                "max_new_tokens": self.config.max_new_tokens,
            }} if config is not None else {},
        )
        self.settings.source_file = base_settings.source_file
        cp_config = self.settings.control_plane
        self.control_plane = ControlPlaneClient(cp_config) if cp_config and cp_config.enabled else None
        self.mcp = MCPClientManager(self.settings.mcp)
        self.targets = InferenceTargetManager(self.settings.providers, self.control_plane)
        self.sessions = self.runtime.session_service
        self.attachments = AttachmentManager(
            lambda record: self.runtime.append_session_record(self.session, record),
            lambda: self.state.records,
        )
        self.history = HistoryManager(
            self.settings.tui.history_file,
            limit=self.settings.tui.history_size,
            suppress_duplicates=self.settings.tui.suppress_duplicates,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: PRAAgentConfig | None = None,
        runtime_config: PRARuntimeConfig | Mapping[str, object] | None = None,
        tools: Toolset | None = None,
        workspace: str | Path = ".",
        default_tools: bool = True,
        skills: Sequence[Skill] = (),
        skills_path: str | Path | None = None,
        session_service: SessionService | None = None,
        sessions_path: str | Path = ".pra/sessions",
        observability: Observability | None = None,
        settings: PRAAgentSettings | Mapping[str, Any] | None = None,
        config_file: str | Path | None = None,
        **model_kwargs: object,
    ) -> "PRAAgent":
        """Load a model and assemble the complete local agent SDK."""

        agent_config = config or PRAAgentConfig()
        bundles = []
        if default_tools:
            bundles.append(default_toolset(workspace, tenant_id=agent_config.tenant_id))
        if tools is not None:
            bundles.append(tools)
        toolset = Toolset.merge(*bundles, name="pra-agent") if bundles else Toolset(())
        capabilities = CapabilitySDK(
            AgentConfig(
                tools=toolset.records,
                skills=tuple(skills),
                skills_path=skills_path,
                namespace="pra-agent",
                tenant_id=agent_config.tenant_id,
                max_candidates=agent_config.tool_candidates,
            )
        )
        runtime = PRARuntime.from_pretrained(
            model_name_or_path,
            runtime_config=runtime_config,
            capability_sdk=capabilities,
            executor=toolset.executor(),
            session_service=session_service or LocalSessionService(sessions_path),
            observability=observability,
            **model_kwargs,
        )
        return cls(
            runtime,
            config=agent_config,
            toolset=toolset,
            observability=observability,
            settings=settings,
            config_file=config_file,
        )

    @classmethod
    def from_config_file(
        cls, path: str | Path, *, config: PRAAgentSettings | Mapping[str, Any] | None = None,
        runtime_config: PRARuntimeConfig | Mapping[str, object] | None = None,
        observability: Observability | None = None,
    ) -> "PRAAgent":
        """Create a remote Agent from the configured provider without loading a local model."""

        settings = PRAAgentSettings.compose(config_file=path, config=config)
        provider_name = settings.agent.provider or next(iter(settings.providers), None)
        if not provider_name or provider_name not in settings.providers:
            raise ValueError("Agent config must select a configured provider.")
        provider = settings.providers[provider_name]
        if not provider.base_url:
            raise ValueError(f"Provider {provider_name!r} requires base_url.")
        host = settings.agent
        policy = PRAAgentConfig(
            user_id=host.user_id, tenant_id=host.tenant_id, task_scope=host.task_scope,
            context_records=host.context_records, tool_candidates=host.tool_candidates,
            max_tool_rounds=host.max_tool_rounds, allow_writes=host.allow_writes,
            allow_destructive=host.allow_destructive, max_new_tokens=host.max_new_tokens,
        )
        backend = NegotiatedRemoteBackend(provider.base_url, provider.model or settings.agent.model,
                                           credentials_file=provider.credentials_file,
                                           observability=observability)
        runtime = PRARuntime(config=PRARuntimeConfig(**dict(runtime_config or {})), backend=backend,
                             session_service=LocalSessionService(settings.session.path),
                             observability=observability)
        return cls(runtime, config=policy, settings=settings, config_file=path,
                   observability=observability)

    async def start(self) -> None:
        """Connect optional remote services and publish their typed capabilities."""

        await self.mcp.connect_all()
        await self._refresh_mcp_capabilities()
        if self.control_plane:
            await self.control_plane.list_engines()

    async def _refresh_mcp_capabilities(self) -> None:
        records = await self.mcp.tool_records(self.config.tenant_id)
        if not records:
            return
        existing = tuple(self.runtime.capabilities.tools) if self.runtime.capabilities else ()
        capabilities = CapabilitySDK(AgentConfig(
            tools=(*existing, *records), namespace="pra-agent", tenant_id=self.config.tenant_id,
            max_candidates=self.config.tool_candidates,
        ))
        resources = capabilities.resources()
        handlers = dict(getattr(self.runtime.executor, "handlers", {}))
        handlers.update(self.mcp.tool_handlers())
        self.runtime.capabilities = capabilities
        self.runtime.discovery = PersistentResourceIndex(resources)
        self.runtime.executor = SafeToolExecutor(resources, handlers)

    async def switch_target(self, value: str) -> InferenceTarget:
        """Switch providers while retaining logical records and dropping model-native state."""

        target = await self.targets.resolve(value)
        if not target.endpoint:
            raise ValueError(f"Target {target.target_id!r} has no inference endpoint.")
        backend = NegotiatedRemoteBackend(
            target.endpoint, target.model_id,
            credentials_file=target.credentials_ref,
            observability=self.observability,
        )
        state = self.state if self.session is not None else None
        previous = self.targets.active.target_id if self.targets.active else None
        if self.session is not None:
            self.runtime.append_session_record(
                self.session,
                ContextRecord(
                    f"model-switch:{uuid.uuid4().hex}", RecordType.SESSION_RECORD,
                    {"event": "model_switch", "previous_target": previous,
                     "target": target.target_id, "native_state_invalidated": True,
                     "timestamp": time.time()},
                ),
            )
            state = self.state
        if self.session is not None:
            self.runtime.close_session(self.session)
            self.session = None
        self.runtime.backend = backend
        if state is not None:
            self.start_session(state.session_id, resume=True)
        self.targets.active = target
        return target

    async def attach_mcp_resource(self, server: str, uri: str) -> Any:
        """Read one MCP resource and retain its identity/content in this session."""

        result = await self.mcp.read_resource(server, uri)
        contents = getattr(result, "contents", ())
        text = "\n".join(str(getattr(row, "text", "")) for row in contents)
        mime = next((str(getattr(row, "mimeType", "text/plain")) for row in contents), "text/plain")
        return self.attachments.add_mcp_resource(server, uri, text, mime_type=mime)

    def export_session(self, path: str | Path) -> Path:
        return export_session(self.state, path)

    def start_session(
        self,
        session_id: str | None = None,
        *,
        resume: bool = False,
        task_description: str | None = None,
    ) -> AgentSessionState:
        """Create or resume durable state and open its ephemeral model session."""

        self.session = self.runtime.open_session(
            session_id=session_id,
            user_id=self.config.user_id,
            tenant_id=self.config.tenant_id,
            resume=resume,
            task_description=task_description,
        )
        self.observability.set_gauge(
            "pra_agent_active_sessions", len(self.runtime.sessions)
        )
        return self.runtime.logical_session_for(self.session)

    @property
    def state(self) -> AgentSessionState:
        if self.session is None:
            raise RuntimeError("Start or resume a session first.")
        return self.runtime.logical_session_for(self.session)

    def _event(
        self,
        event_type: TaskEventType,
        task_id: str,
        *,
        payload: Mapping[str, object] | None = None,
        expected_version: int | None = None,
    ) -> AgentSessionState:
        state = self.state
        sequence = state.tasks.last_sequence + 1
        return self.runtime.apply_task_event(
            self.session,
            TaskEvent(
                event_id=f"{state.session_id}:{sequence}:{uuid.uuid4().hex[:8]}",
                sequence=sequence,
                event_type=event_type,
                task_id=task_id,
                expected_version=expected_version,
                payload=dict(payload or {}),
            ),
        )

    def create_task(
        self,
        description: str,
        *,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        depends_on: Sequence[str] = (),
        activate: bool = True,
    ) -> AgentSessionState:
        task_id = task_id or f"task-{len(self.state.tasks.tasks) + 1}"
        state = self._event(
            TaskEventType.CREATE,
            task_id,
            payload={
                "description": description,
                "parent_task_id": parent_task_id,
                "depends_on": list(depends_on),
            },
        )
        if activate:
            state = self.activate_task(task_id)
        return state

    def activate_task(self, task_id: str) -> AgentSessionState:
        task = next(row for row in self.state.tasks.tasks if row.task_id == task_id)
        return self._event(TaskEventType.ACTIVATE, task_id, expected_version=task.version)

    def complete_task(self, task_id: str, *, result_ref: str | None = None) -> AgentSessionState:
        task = next(row for row in self.state.tasks.tasks if row.task_id == task_id)
        return self._event(
            TaskEventType.COMPLETE,
            task_id,
            expected_version=task.version,
            payload={"result_ref": result_ref},
        )

    def _append_message(self, role: str, text: str) -> AgentSessionState:
        return self.runtime.append_session_record(
            self.session,
            ContextRecord(
                f"message:{uuid.uuid4().hex}",
                RecordType.GENERIC_TEXT,
                {"role": role, "text": text, "timestamp": time.time()},
            ),
        )

    def _context(self, query: str) -> ScopeSelection | None:
        with self.observability.span("pra.agent.context.prepare"):
            if self.state.active_task_id is None or not self.state.records:
                return None
            return self.runtime.select_task_context(
                self.session,
                query,
                policy=self.config.task_scope,
                max_records=self.config.context_records,
            )

    def _disclosed_capabilities(
        self, query: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.runtime.discovery is None:
            return (), ()
        with self.observability.span("pra.agent.tool.select"):
            trace = self.runtime.discover_resources(
                DiscoveryRequest(
                    query=query,
                    tenant_id=self.config.tenant_id,
                    top_k=self.config.tool_candidates,
                )
            )
        uris = tuple(dict.fromkeys((
            *trace.selected_uris,
            *(row.uri for row in trace.candidates[: self.config.tool_candidates]),
        )))[: self.config.tool_candidates]
        if self.runtime.capabilities is not None and uris:
            self.runtime.activate_capability_candidates(uris)
            for uri in uris:
                self.runtime.activate_capability(uri)
        tool_ids = {
            tool.to_agent_resource().uri
            for tool in (self.runtime.capabilities.tools if self.runtime.capabilities else ())
        }
        return (
            tuple(uri for uri in uris if uri in tool_ids),
            tuple(uri for uri in uris if uri not in tool_ids),
        )

    @staticmethod
    def _is_conversational_record(record: ContextRecord) -> bool:
        """Keep the pretrained chat trajectory separate from detached context."""

        return (
            record.record_type == RecordType.GENERIC_TEXT
            and isinstance(record.payload, Mapping)
            and record.payload.get("role") in {"system", "user", "assistant", "tool"}
            and "text" in record.payload
        )

    def _turn_context(
        self,
        query: str,
        scope: ScopeSelection | None,
        tool_uris: Sequence[str],
        skill_uris: Sequence[str],
        *,
        extra_messages: Sequence[Mapping[str, object]] = (),
    ) -> AgentTurnContext:
        """Build semantic turn state without choosing its wire representation."""

        selected = (
            scope.selected_records
            if scope is not None
            else self.state.records[-self.config.context_records :]
        )
        records = tuple(
            record for record in selected if not self._is_conversational_record(record)
        )
        conversation = tuple(
            {
                "role": str(record.payload["role"]),
                "content": str(record.payload["text"]),
            }
            for record in self.state.records
            if self._is_conversational_record(record)
        )
        system = {
            "role": "system",
            "content": (
                "You are a task-aware PRA agent. Use only disclosed tools. "
                "Emit a tool request as <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>. "
                f"Active task: {self.state.active_task_id or 'none'}."
            ),
        }
        by_uri = {
            resource.uri: resource
            for resource in (
                (tool.to_agent_resource() for tool in self.runtime.capabilities.tools)
                if self.runtime.capabilities
                else ()
            )
        }
        schemas = [resource_tool_schema(by_uri[uri]) for uri in tool_uris if uri in by_uri]
        capability_records = {
            record.record_id: record
            for record in (
                self.runtime.capabilities.records if self.runtime.capabilities else ()
            )
        }
        task = next(
            (
                row for row in self.state.tasks.tasks
                if row.task_id == self.state.active_task_id
            ),
            None,
        )
        return AgentTurnContext(
            messages=(system, *conversation, *tuple(dict(row) for row in extra_messages)),
            records=records,
            tool_records=tuple(
                capability_records[uri] for uri in tool_uris if uri in capability_records
            ),
            skill_records=tuple(
                capability_records[uri] for uri in skill_uris if uri in capability_records
            ),
            tools=tuple(schemas),
            task_id=self.state.active_task_id,
            task_metadata={} if task is None else task.to_dict(),
            selected_record_ids=(
                scope.selected_record_ids if scope is not None
                else tuple(record.record_id for record in records)
            ),
            metadata={
                "user_id": self.config.user_id,
                "tenant_id": self.config.tenant_id,
                "query": query,
            },
        )

    def _generate_turn(self, turn: AgentTurnContext) -> str:
        """Delegate rendering/transport to the configured model backend."""

        operation = getattr(self.runtime.backend, "generate_turn", None)
        if operation is not None:
            return _generated_text(operation(
                turn,
                tenant_id=self.config.tenant_id,
                session_id=self.state.session_id,
                max_new_tokens=self.config.max_new_tokens,
            ))
        return _generated_text(
            self.runtime.generate(
                render_text_prompt(turn), max_new_tokens=self.config.max_new_tokens
            )
        )

    def run_turn(self, query: str) -> AgentTurn:
        """Run one instrumented turn without capturing user content by default."""

        started = time.perf_counter()
        status = "success"
        result: AgentTurn | None = None
        try:
            with self.observability.span(
                "pra.agent.turn",
                lambda: {
                    "pra.tenant.id_hash": self.observability.hash_id(self.config.tenant_id),
                    "pra.session.id_hash": self.observability.hash_id(self.state.session_id),
                    "pra.task.id_hash": self.observability.hash_id(self.state.active_task_id),
                },
            ):
                result = self._run_turn_uninstrumented(query)
                return result
        except BaseException:
            status = "error"
            raise
        finally:
            self.observability.increment(
                "pra_agent_turns_total", labels={"status": status}
            )
            self.observability.observe(
                "pra_agent_turn_duration_seconds",
                time.perf_counter() - started,
                labels={"status": status},
            )
            if result is not None:
                for execution in result.tool_executions:
                    tool_status = "success" if execution.execution.executed else "rejected"
                    self.observability.increment(
                        "pra_agent_tool_calls_total", labels={"status": tool_status}
                    )

    def _run_turn_uninstrumented(self, query: str) -> AgentTurn:
        """Generate one answer and execute at most the configured tool rounds."""

        if not query.strip():
            raise ValueError("A non-empty user message is required.")
        self._append_message("user", query)
        scope = self._context(query)
        tool_uris, skill_uris = self._disclosed_capabilities(query)
        turn_context = self._turn_context(query, scope, tool_uris, skill_uris)
        text = self._generate_turn(turn_context)
        executions = []
        for _ in range(self.config.max_tool_rounds):
            call = parse_tool_call(text)
            if call is None:
                break
            resource = None
            if self.runtime.executor is not None:
                resource = self.runtime.executor.by_name.get(call.name)
            approved = False
            if resource is not None and self.authorization_callback is not None:
                if resource.side_effect_class in {
                    SideEffectClass.WRITE,
                    SideEffectClass.DESTRUCTIVE,
                }:
                    approved = bool(self.authorization_callback(resource, call))
            execution = self.runtime.execute_tool_and_record(
                text,
                session=self.session,
                selected_uris=tool_uris,
                authorization=ExecutionAuthorization(
                    frozenset(tool_uris),
                    allow_writes=self.config.allow_writes or approved,
                    allow_destructive=self.config.allow_destructive or (
                        approved
                        and resource is not None
                        and resource.side_effect_class == SideEffectClass.DESTRUCTIVE
                    ),
                ),
                call_id=f"{self.state.session_id}:{uuid.uuid4().hex[:12]}",
            )
            executions.append(execution)
            if not execution.execution.executed:
                text = f"Tool call rejected: {execution.execution.reason}"
                break
            tool_message = {
                "role": "tool",
                "tool_call_id": execution.execution.resource_uri or "pra-tool-result",
                "content": json.dumps(execution.record.compact_view(), default=str),
            }
            follow_up = self._turn_context(
                query,
                self._context(query),
                tool_uris,
                skill_uris,
                extra_messages=(
                    {"role": "assistant", "content": text},
                    tool_message,
                ),
            )
            text = self._generate_turn(follow_up)
        self._append_message("assistant", text)
        return AgentTurn(
            text=text,
            session=self.state,
            selected_record_ids=scope.selected_record_ids if scope else (),
            disclosed_tool_uris=tool_uris,
            disclosed_skill_uris=skill_uris,
            tool_executions=tuple(executions),
            transport=dict(self.runtime.backend.inspect().get("transport", {})),
        )

    def close(self) -> None:
        """Release ephemeral model state while retaining durable session state."""

        if self.session is not None and not self.session.closed:
            self.runtime.close_session(self.session)
        self.session = None
        self.observability.set_gauge(
            "pra_agent_active_sessions", len(self.runtime.sessions)
        )

    async def aclose(self) -> None:
        """Close MCP transports and ephemeral model state."""

        await self.mcp.disconnect_all()
        self.close()
