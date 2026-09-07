"""Conventional subagent lifecycle with PRA-aware record reuse hooks.

The harness owns concrete execution callbacks and permission boundaries.  It
passes only normalized effects and resource identities to the generic
cross-agent runtime in :mod:`pra_hf.subagent_context`.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Callable, Generic, Mapping, Protocol, Sequence, TypeVar

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
    additional_parent_agent_uuids: tuple[str, ...] = ()
    peer_agent_uuids: tuple[str, ...] = ()


T = TypeVar("T")


@dataclass(frozen=True)
class SubagentRunResult(Generic[T]):
    """Outcome of one harness-scheduled child callback."""

    agent_uuid: str
    value: T | None
    status: AgentStatus
    elapsed_ms: float
    error: str | None = None


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
        self._state_lock = RLock()
        self._call_locks: dict[str, RLock] = {}
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
        with self._state_lock:
            descriptor = self.graph.spawn(
                parent_agent_uuid,
                agent_uuid=agent_uuid,
                parent_task_uuid=spec.parent_task_uuid,
                spawn_call_record_uuid=spec.spawn_call_record_uuid,
                model_descriptor=spec.model_descriptor,
                workspace_id=spec.workspace_id,
                tool_profile_id=spec.tool_profile_id,
                context_policy=spec.context_policy,
            )
            for parent_uuid in spec.additional_parent_agent_uuids:
                descriptor = self.graph.add_parent(descriptor.agent_uuid, parent_uuid)
            for peer_uuid in spec.peer_agent_uuids:
                self.graph.link_peers(descriptor.agent_uuid, peer_uuid)
            return descriptor

    def join_subagent(self, agent_uuid: str, parent_agent_uuid: str) -> AgentDescriptor:
        """Add an explicit DAG parent to an existing child stream."""

        with self._state_lock:
            return self.graph.add_parent(agent_uuid, parent_agent_uuid)

    def link_subagent_peers(self, left_agent_uuid: str, right_agent_uuid: str) -> None:
        """Link peers while leaving their visibility policies unchanged."""

        with self._state_lock:
            self.graph.link_peers(left_agent_uuid, right_agent_uuid)

    def run_subagents(
        self,
        parent_agent_uuid: str,
        specs: Sequence[SubagentSpec],
        runner: Callable[[AgentDescriptor, "SubagentHarness"], T],
        *,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> tuple[SubagentRunResult[T], ...]:
        """Spawn and run child callbacks sequentially or with bounded fan-out.

        Scheduling belongs to the harness, not the PRA runtime. Each callback
        receives a normal child descriptor and this harness, so integrations
        can wrap OpenHands, mini-SWE-agent, or an internal agent loop without
        changing the context/reuse contract.
        """

        children = [self.spawn_subagent(parent_agent_uuid, spec) for spec in specs]

        def run_one(child: AgentDescriptor) -> SubagentRunResult[T]:
            started = time.perf_counter()
            try:
                value = runner(child, self)
                self.stop_subagent(child.agent_uuid)
                return SubagentRunResult(
                    child.agent_uuid,
                    value,
                    AgentStatus.STOPPED,
                    (time.perf_counter() - started) * 1000.0,
                )
            except Exception as error:  # pragma: no cover - exercised through contract test
                self.stop_subagent(child.agent_uuid, status=AgentStatus.FAILED)
                return SubagentRunResult(
                    child.agent_uuid,
                    None,
                    AgentStatus.FAILED,
                    (time.perf_counter() - started) * 1000.0,
                    f"{type(error).__name__}: {error}",
                )

        if not parallel:
            return tuple(run_one(child) for child in children)
        workers = max_workers or len(children) or 1
        completed: dict[str, SubagentRunResult[T]] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pra-subagent") as pool:
            futures = {pool.submit(run_one, child): child.agent_uuid for child in children}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        return tuple(completed[child.agent_uuid] for child in children)

    def resume_subagent(self, agent_uuid: str) -> AgentDescriptor:
        with self._state_lock:
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
        with self._state_lock:
            return self.graph.stop(
                agent_uuid,
                status=status,
                summary_record_uuid=summary_record_uuid,
                result_record_uuids=result_record_uuids,
                artifact_record_uuids=artifact_record_uuids,
            )

    def inspect_subagent(self, agent_uuid: str) -> SubagentInspection:
        with self._state_lock:
            descriptor = self.graph.descriptor(agent_uuid)
            children = tuple(
                row.agent_uuid for row in self.graph.agents if agent_uuid in row.parent_agent_uuids
            )
            return SubagentInspection(
                descriptor=descriptor,
                child_agent_uuids=children,
                local_record_uuids=tuple(row.record_id for row in self.graph.records(agent_uuid)),
                visible_record_uuids=tuple(
                    row.record_id for row in self.graph.visible_records(agent_uuid)
                ),
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

        effect = tool.describe_effect(arguments)
        signature = self.reuse_runtime.call_signature(tool.uri, arguments)
        with self._state_lock:
            call_lock = self._call_locks.setdefault(signature, RLock())
        # Serialize only equivalent calls. Different tools/resources remain
        # concurrent, while duplicate misses cannot race into two executions.
        with call_lock:
            return self._execute_tool_once(
                agent_uuid,
                tool,
                arguments,
                effect,
                allow_payload_reuse=allow_payload_reuse,
                allow_native_kv_reuse=allow_native_kv_reuse,
                requested_native=requested_native,
                external_validator=external_validator,
            )

    def _execute_tool_once(
        self,
        agent_uuid: str,
        tool: DeclarativeTool,
        arguments: Mapping[str, object],
        effect: ToolEffectDescriptor,
        *,
        allow_payload_reuse: bool,
        allow_native_kv_reuse: bool,
        requested_native: NativeKVHandle | None,
        external_validator: Callable[[object, str | None], bool] | None,
    ) -> HarnessExecutionResult:
        started = time.perf_counter()
        with self._state_lock:
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
                self.graph.append_record(
                    agent_uuid,
                    decision.to_context_record(
                        self.graph.session_uuid, self.graph.logical_clock + 1
                    ),
                )
                return HarnessExecutionResult(
                    output=entry.payload,
                    record=entry.source_record,
                    reuse=decision,
                    executed=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )

        # The concrete callback may be expensive; do not hold the graph lock.
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
        with self._state_lock:
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
                decision.to_context_record(
                    self.graph.session_uuid, self.graph.logical_clock + 1
                ),
            )
        return HarnessExecutionResult(
            output=output,
            record=tagged,
            reuse=decision,
            executed=True,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
