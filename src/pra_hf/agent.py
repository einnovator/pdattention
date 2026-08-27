"""Task-aware agent facade over the product PRA runtime.

The facade is intentionally provider-neutral: model generation proposes text or
a typed tool call, while the host owns capability disclosure, authorization,
task mutation, and durable session state.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .agent_execution import (
    ExecutionAuthorization,
    ToolCall,
    parse_tool_call,
    resource_tool_schema,
)
from .agent_resources import AgentResource, DiscoveryRequest, SideEffectClass
from .capability_sdk import AgentConfig, CapabilitySDK
from .context_records import ContextRecord, RecordType, RecordViewName, serialize_record
from .model import GenerationResult
from .runtime import PRARuntime, PRARuntimeConfig, RuntimeToolExecution
from .session_service import AgentSessionState, LocalSessionService, SessionService
from .skill_records import Skill
from .task_context import TaskEvent, TaskEventType
from .task_scope import ScopeSelection, TaskScopePolicy
from .toolsets import Toolset, default_toolset


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
    ) -> None:
        self.runtime = runtime
        self.config = config or PRAAgentConfig()
        self.toolset = toolset
        self.authorization_callback = authorization_callback
        self.session = None

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
            **model_kwargs,
        )
        return cls(runtime, config=agent_config, toolset=toolset)

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

    def _prompt(
        self,
        query: str,
        scope: ScopeSelection | None,
        tool_uris: Sequence[str],
        skill_uris: Sequence[str],
    ) -> str:
        records = scope.selected_records if scope is not None else self.state.records[-self.config.context_records :]
        context = "\n".join(
            serialize_record(record, view=RecordViewName.FULL) for record in records
        )
        by_uri = {
            resource.uri: resource
            for resource in (
                (tool.to_agent_resource() for tool in self.runtime.capabilities.tools)
                if self.runtime.capabilities
                else ()
            )
        }
        schemas = [resource_tool_schema(by_uri[uri]) for uri in tool_uris if uri in by_uri]
        skills = {
            resource.uri: resource.content
            for resource in (
                (skill.to_agent_resource() for skill in self.runtime.capabilities.skills)
                if self.runtime.capabilities
                else ()
            )
        }
        return (
            "You are a task-aware PRA agent. Use only disclosed tools. "
            "Emit a tool request as <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>.\n"
            f"Active task: {self.state.active_task_id or 'none'}\n"
            f"Context:\n{context or '(empty)'}\n"
            f"Disclosed tools:\n{json.dumps(schemas, sort_keys=True)}\n"
            f"Selected skills:\n{json.dumps([skills[uri] for uri in skill_uris if uri in skills])}\n"
            f"User: {query}\nAssistant:"
        )

    def run_turn(self, query: str) -> AgentTurn:
        """Generate one answer and execute at most the configured tool rounds."""

        if not query.strip():
            raise ValueError("A non-empty user message is required.")
        self._append_message("user", query)
        scope = self._context(query)
        tool_uris, skill_uris = self._disclosed_capabilities(query)
        prompt = self._prompt(query, scope, tool_uris, skill_uris)
        text = _generated_text(
            self.runtime.generate(prompt, max_new_tokens=self.config.max_new_tokens)
        )
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
            prompt += (
                f"\nTool result: {json.dumps(execution.record.compact_view(), default=str)}"
                "\nAnswer the user using the result.\nAssistant:"
            )
            text = _generated_text(
                self.runtime.generate(prompt, max_new_tokens=self.config.max_new_tokens)
            )
        self._append_message("assistant", text)
        return AgentTurn(
            text=text,
            session=self.state,
            selected_record_ids=scope.selected_record_ids if scope else (),
            disclosed_tool_uris=tool_uris,
            disclosed_skill_uris=skill_uris,
            tool_executions=tuple(executions),
        )

    def close(self) -> None:
        """Release ephemeral model state while retaining durable session state."""

        if self.session is not None and not self.session.closed:
            self.runtime.close_session(self.session)
        self.session = None
