"""Conventional subagent lifecycle with PRA-aware record reuse hooks.

The harness owns concrete execution callbacks and permission boundaries.  It
passes only normalized effects and resource identities to the generic
cross-agent runtime in :mod:`pra_hf.subagent_context`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Protocol, Sequence

from .context_records import ContextRecord, RecordType
from .subagent_context import (
    AgentContextGraph,
    AgentDescriptor,
    AgentStatus,
    ContextVisibilityPolicy,
    CrossAgentReuseRuntime,
    NativeKVHandle,
    ReuseDecisionRecord,
    SubagentCapabilities,
    ToolEffectDescriptor,
)


class NativeKVReusePort(Protocol):
    """Minimal engine callback needed to share one immutable native record."""

    def encode(self, record: ContextRecord) -> NativeKVHandle | None:
        """Return a native-state receipt after encoding a new result."""

    def materialize(self, handle: NativeKVHandle, *, target_agent_uuid: str) -> bool:
        """Attach one compatible handle exactly once to a target request."""


@dataclass(frozen=True)
class SubagentSpec:
    """Harness-owned choices made for one child execution stream."""

    model_descriptor: str = "same"
    workspace_id: str | None = None
    tool_profile_id: str = "inherited"
    context_policy: ContextVisibilityPolicy = field(default_factory=ContextVisibilityPolicy)
    parent_task_uuid: str | None = None
    spawn_call_record_uuid: str | None = None


@dataclass(frozen=True)
class DeclarativeTool:
    """Concrete harness callback paired with generic runtime metadata."""

    uri: str
    execute: Callable[[Mapping[str, object]], object]
    describe_effect: Callable[[Mapping[str, object]], ToolEffectDescriptor]

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("Tool URI is required.")


@dataclass(frozen=True)
class HarnessExecutionResult:
    """Result returned to an agent loop with its reuse and record receipts."""

    output: object
    record: ContextRecord
    reuse: ReuseDecisionRecord
    executed: bool
    elapsed_ms: float


@dataclass(frozen=True)
class SubagentInspection:
    """Stable public view used by inspect/list commands and remote gateways."""

    descriptor: AgentDescriptor
    child_agent_uuids: tuple[str, ...]
    local_record_uuids: tuple[str, ...]
    visible_record_uuids: tuple[str, ...]


class SubagentHarness:
    """Spawn ordinary child loops while preserving PRA context topology.

    This class deliberately does not implement an LLM loop.  Existing agent
    frameworks can call these lifecycle and tool methods around their own loop,
    which keeps Paper 9 independent of a specific planner or provider.
    """

    capabilities = SubagentCapabilities()

    def __init__(
        self,
        session_uuid: str,
        *,
        root_agent_uuid: str | None = None,
        root_model: str = "same",
        root_workspace_id: str = "shared",
        root_context_policy: ContextVisibilityPolicy | None = None,
        native_port: NativeKVReusePort | None = None,
    ) -> None:
        self.graph = AgentContextGraph(session_uuid)
        self.reuse_runtime = CrossAgentReuseRuntime(self.graph)
        self.native_port = native_port
        self._materialized_native: set[tuple[str, str, str, str, str]] = set()
        self.root = self.graph.start_root(
            root_agent_uuid,
            model_descriptor=root_model,
            workspace_id=root_workspace_id,
            context_policy=root_context_policy,
        )

    def spawn_subagent(
        self,
        parent_agent_uuid: str,
        spec: SubagentSpec | None = None,
        *,
        agent_uuid: str | None = None,
    ) -> AgentDescriptor:
        spec = spec or SubagentSpec()
        return self.graph.spawn(
            parent_agent_uuid,
            agent_uuid=agent_uuid,
            parent_task_uuid=spec.parent_task_uuid,
            spawn_call_record_uuid=spec.spawn_call_record_uuid,
            model_descriptor=spec.model_descriptor,
            workspace_id=spec.workspace_id,
            tool_profile_id=spec.tool_profile_id,
            context_policy=spec.context_policy,
        )

    def resume_subagent(self, agent_uuid: str) -> AgentDescriptor:
        return self.graph.resume(agent_uuid)

    def stop_subagent(
        self,
        agent_uuid: str,
        *,
        status: AgentStatus | str = AgentStatus.STOPPED,
        summary_record_uuid: str | None = None,
        result_record_uuids: Sequence[str] = (),
        artifact_record_uuids: Sequence[str] = (),
    ) -> AgentDescriptor:
        return self.graph.stop(
            agent_uuid,
            status=status,
            summary_record_uuid=summary_record_uuid,
            result_record_uuids=result_record_uuids,
            artifact_record_uuids=artifact_record_uuids,
        )

    def inspect_subagent(self, agent_uuid: str) -> SubagentInspection:
        descriptor = self.graph.descriptor(agent_uuid)
        children = tuple(
            row.agent_uuid for row in self.graph.agents if row.parent_agent_uuid == agent_uuid
        )
        return SubagentInspection(
            descriptor=descriptor,
            child_agent_uuids=children,
            local_record_uuids=tuple(row.record_id for row in self.graph.records(agent_uuid)),
            visible_record_uuids=tuple(row.record_id for row in self.graph.visible_records(agent_uuid)),
        )

    def execute_tool(
        self,
        agent_uuid: str,
        tool: DeclarativeTool,
        arguments: Mapping[str, object],
        *,
        allow_payload_reuse: bool = True,
        allow_native_kv_reuse: bool = True,
        requested_native: NativeKVHandle | None = None,
        external_validator: Callable[[object, str | None], bool] | None = None,
    ) -> HarnessExecutionResult:
        """Execute or safely reuse one exact tool call.

        Unknown/write operations always execute.  Read and pure results may be
        reused only after lineage visibility and the declared consistency check
        both pass.  Every attempt is appended as a typed decision record.
        """

        started = time.perf_counter()
        effect = tool.describe_effect(arguments)
        entry, decision = self.reuse_runtime.lookup(
            request_agent_uuid=agent_uuid,
            tool_uri=tool.uri,
            arguments=arguments,
            requested_effect=effect,
            requested_native=requested_native,
            allow_payload=allow_payload_reuse,
            allow_native_kv=allow_native_kv_reuse,
            external_validator=external_validator,
        )
        if entry is not None:
            if decision.kv_reused and self.native_port is not None and entry.native_kv is not None:
                materialization_key = (
                    agent_uuid,
                    entry.native_kv.record_uuid,
                    entry.native_kv.model_id,
                    entry.native_kv.model_revision,
                    entry.native_kv.encoding_revision,
                )
                attached = materialization_key in self._materialized_native
                if not attached:
                    attached = self.native_port.materialize(
                        entry.native_kv, target_agent_uuid=agent_uuid
                    )
                    if attached:
                        self._materialized_native.add(materialization_key)
                if not attached:
                    decision = replace(
                        decision,
                        kv_reused=False,
                        native_tokens_reused=0,
                        native_bytes_reused=0,
                    )
            tagged_decision = self.graph.append_record(
                agent_uuid,
                decision.to_context_record(self.graph.session_uuid, self.graph.logical_clock + 1),
            )
            return HarnessExecutionResult(
                output=entry.payload,
                record=entry.source_record,
                reuse=decision,
                executed=False,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        output = tool.execute(dict(arguments))
        record = ContextRecord(
            record_id=f"tool-result:{agent_uuid}:{uuid.uuid4().hex}",
            record_type=RecordType.TOOL_RESPONSE,
            payload={
                "tool_uri": tool.uri,
                "arguments": dict(arguments),
                "output": output,
                "effect": effect.effect.value,
                "resources": [row.canonical for row in effect.resources],
            },
        )
        tagged = self.graph.append_record(agent_uuid, record)
        if effect.effect.value == "write":
            self.reuse_runtime.register_write(agent_uuid, effect)
        elif effect.reuse_enabled:
            native = self.native_port.encode(tagged) if self.native_port is not None else None
            self.reuse_runtime.register_result(
                tool_uri=tool.uri,
                arguments=arguments,
                source_record=tagged,
                effect=effect,
                payload=output,
                native_kv=native,
            )
        self.graph.append_record(
            agent_uuid,
            decision.to_context_record(self.graph.session_uuid, self.graph.logical_clock + 1),
        )
        return HarnessExecutionResult(
            output=output,
            record=tagged,
            reuse=decision,
            executed=True,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
