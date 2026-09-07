"""Lineage-aware context and reuse contracts for PRA subagents.

This module is intentionally tool agnostic.  Harness adapters translate files,
HTTP resources, databases, or other concrete systems into the generic effects
and resource identities defined here.  The runtime then decides visibility and
validity without learning tool-specific behavior.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Callable, Mapping, Sequence

from .context_records import ContextRecord, RecordType


class EffectType(str, Enum):
    """Minimal side-effect vocabulary understood by the generic runtime."""

    PURE = "pure"
    READ = "read"
    WRITE = "write"
    UNKNOWN = "unknown"


class ConsistencyMode(str, Enum):
    """Mutation scope used when validating a reusable result."""

    ANCESTRY = "ancestry"
    SESSION_TREE = "session_tree"
    EXTERNAL = "external"


class RecordVisibility(str, Enum):
    """How records across an agent boundary become visible."""

    NONE = "none"
    SELECTED = "selected"
    ROUTABLE = "routable"
    INHERIT_ALL_REFERENCES = "inherit_all_references"
    COMPLETED_ONLY = "completed_only"


class AgentStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class ReuseDecision(str, Enum):
    HIT = "hit"
    MISS = "miss"
    INVALID = "invalid"


class ReuseMissReason(str, Enum):
    NONE = "none"
    NO_MATCH = "no_match"
    NOT_VISIBLE = "not_visible"
    EXPIRED = "expired"
    WRITE_INVALIDATION = "write_invalidation"
    EXTERNAL_VALIDATION_FAILED = "external_validation_failed"
    MODEL_INCOMPATIBLE = "model_incompatible"
    KV_EVICTED = "kv_evicted"
    UNKNOWN_EFFECT = "unknown_effect"
    POLICY_DISABLED = "policy_disabled"


@dataclass(frozen=True, order=True)
class ResourceIdentity:
    """Normalized identity supplied by a harness-owned tool adapter."""

    domain: str
    key: str
    workspace_id: str = "shared"

    def __post_init__(self) -> None:
        if not self.domain or not self.key or not self.workspace_id:
            raise ValueError("Resource domain, key, and workspace_id are required.")

    @property
    def canonical(self) -> str:
        return f"{self.workspace_id}:{self.domain}:{self.key}"


@dataclass(frozen=True)
class ToolEffectDescriptor:
    """Declarative validity contract attached to one concrete tool operation."""

    effect: EffectType | str = EffectType.UNKNOWN
    resources: tuple[ResourceIdentity, ...] = ()
    reuse_enabled: bool = False
    consistency: ConsistencyMode | str = ConsistencyMode.SESSION_TREE
    max_age_seconds: float | None = None
    version_token: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect", EffectType(self.effect))
        object.__setattr__(self, "consistency", ConsistencyMode(self.consistency))
        object.__setattr__(self, "resources", tuple(self.resources))
        if self.max_age_seconds is not None and self.max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative.")
        if self.effect == EffectType.UNKNOWN and self.reuse_enabled:
            raise ValueError("UNKNOWN effects cannot enable cross-agent reuse.")


@dataclass(frozen=True)
class ContextVisibilityPolicy:
    """Cross-stream visibility selected when the harness spawns an agent."""

    ancestor: RecordVisibility | str = RecordVisibility.NONE
    descendant: RecordVisibility | str = RecordVisibility.NONE
    selected_record_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ancestor", RecordVisibility(self.ancestor))
        object.__setattr__(self, "descendant", RecordVisibility(self.descendant))
        object.__setattr__(self, "selected_record_ids", frozenset(self.selected_record_ids))


@dataclass(frozen=True)
class SubagentCapabilities:
    """Explicitly reports implemented and experimental harness semantics."""

    spawn: str = "supported"
    resume: str = "supported"
    parallel: str = "experimental"
    dag: str = "not_implemented"
    ancestor_record_visibility: str = "experimental"
    descendant_record_visibility: str = "experimental"
    cross_agent_result_reuse: str = "experimental"
    cross_agent_kv_reuse: str = "experimental"


@dataclass(frozen=True)
class AgentDescriptor:
    """One execution stream and its position in a session lineage tree."""

    agent_uuid: str
    session_uuid: str
    parent_agent_uuid: str | None = None
    parent_task_uuid: str | None = None
    spawn_call_record_uuid: str | None = None
    model_descriptor: str = "same"
    workspace_id: str = "shared"
    tool_profile_id: str = "inherited"
    context_policy: ContextVisibilityPolicy = field(default_factory=ContextVisibilityPolicy)
    status: AgentStatus | str = AgentStatus.RUNNING
    started_at: float = field(default_factory=time.time)
    stopped_at: float | None = None
    summary_record_uuid: str | None = None
    result_record_uuids: tuple[str, ...] = ()
    artifact_record_uuids: tuple[str, ...] = ()
    resume_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AgentStatus(self.status))
        object.__setattr__(self, "result_record_uuids", tuple(self.result_record_uuids))
        object.__setattr__(self, "artifact_record_uuids", tuple(self.artifact_record_uuids))
        if not self.agent_uuid or not self.session_uuid or not self.workspace_id:
            raise ValueError("Agent, session, and workspace identities are required.")
        if self.parent_agent_uuid == self.agent_uuid:
            raise ValueError("An agent cannot be its own parent.")


@dataclass(frozen=True)
class NativeKVHandle:
    """Engine-neutral receipt for native state that may be shared by reference."""

    record_uuid: str
    model_id: str
    model_revision: str
    encoding_revision: str
    position_contract: str
    token_count: int
    byte_count: int
    resident: bool = True
    storage_uri: str | None = None

    def compatible_with(self, other: "NativeKVHandle") -> bool:
        return (
            self.model_id == other.model_id
            and self.model_revision == other.model_revision
            and self.encoding_revision == other.encoding_revision
            and self.position_contract == other.position_contract
        )


@dataclass(frozen=True)
class ReusableToolResult:
    """Indexed result plus the dependency state captured when it was produced."""

    signature: str
    source_record: ContextRecord
    source_agent_uuid: str
    effect: ToolEffectDescriptor
    payload: object
    resource_epochs: Mapping[str, int]
    created_at: float
    logical_clock: int
    native_kv: NativeKVHandle | None = None


@dataclass(frozen=True)
class ReuseDecisionRecord:
    """Structured audit receipt for one cross-agent reuse attempt."""

    request_agent_uuid: str
    source_record_uuid: str | None
    source_agent_uuid: str | None
    resource_identities: tuple[str, ...]
    decision: ReuseDecision | str
    reason: ReuseMissReason | str
    validation_mode: ConsistencyMode | str
    age_seconds: float
    causal_position: int
    kv_reused: bool = False
    payload_reused: bool = False
    tool_execution_avoided: bool = False
    native_tokens_reused: int = 0
    native_bytes_reused: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", ReuseDecision(self.decision))
        object.__setattr__(self, "reason", ReuseMissReason(self.reason))
        object.__setattr__(self, "validation_mode", ConsistencyMode(self.validation_mode))
        object.__setattr__(self, "resource_identities", tuple(self.resource_identities))

    def to_context_record(self, session_uuid: str, logical_clock: int) -> ContextRecord:
        """Persist the decision in the requesting agent's typed stream."""

        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["reason"] = self.reason.value
        payload["validation_mode"] = self.validation_mode.value
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return ContextRecord(
            record_id=f"reuse:{self.request_agent_uuid}:{logical_clock}:{digest}",
            record_type=RecordType.REUSE_DECISION,
            payload=payload,
            session_uuid=session_uuid,
            agent_uuid=self.request_agent_uuid,
            created_at=time.time(),
            logical_clock=logical_clock,
        )


class AgentContextGraph:
    """In-memory authoritative lineage and typed-record stream index."""

    def __init__(self, session_uuid: str) -> None:
        if not session_uuid:
            raise ValueError("session_uuid is required.")
        self.session_uuid = session_uuid
        self._agents: dict[str, AgentDescriptor] = {}
        self._records: dict[str, list[ContextRecord]] = {}
        self._clock = 0

    @property
    def logical_clock(self) -> int:
        return self._clock

    @property
    def agents(self) -> tuple[AgentDescriptor, ...]:
        return tuple(sorted(self._agents.values(), key=lambda row: (row.started_at, row.agent_uuid)))

    @property
    def all_records(self) -> tuple[ContextRecord, ...]:
        """Return a causally ordered snapshot suitable for session persistence."""

        return tuple(sorted(
            (record for rows in self._records.values() for record in rows),
            key=lambda row: (row.logical_clock or 0, row.record_id),
        ))

    @classmethod
    def from_records(
        cls, session_uuid: str, records: Sequence[ContextRecord]
    ) -> "AgentContextGraph":
        """Rehydrate lineage and streams from persisted Paper 9 records."""

        graph = cls(session_uuid)
        ordered = sorted(records, key=lambda row: (row.logical_clock or 0, row.record_id))
        for record in ordered:
            if record.session_uuid not in {None, session_uuid} or record.agent_uuid is None:
                raise ValueError("Every replayed subagent record must identify its session and agent.")
            if record.record_type == RecordType.AGENT_START:
                descriptor = cls._descriptor_from_payload(record.payload)
                if descriptor.agent_uuid in graph._agents:
                    raise ValueError(f"Duplicate agent start: {descriptor.agent_uuid}")
                graph._agents[descriptor.agent_uuid] = descriptor
                graph._records[descriptor.agent_uuid] = []
            elif record.agent_uuid not in graph._agents:
                raise ValueError(f"Record precedes agent start: {record.record_id}")
            elif record.record_type in {RecordType.AGENT_STOP, RecordType.AGENT_RESUME}:
                graph._agents[record.agent_uuid] = cls._descriptor_from_payload(record.payload)
            graph._records[record.agent_uuid].append(record)
            graph._clock = max(graph._clock, record.logical_clock or 0)
        if graph._agents and sum(row.parent_agent_uuid is None for row in graph._agents.values()) != 1:
            raise ValueError("Replayed context graph must contain exactly one root agent.")
        return graph

    @staticmethod
    def _descriptor_from_payload(payload: Mapping[str, object] | str) -> AgentDescriptor:
        if not isinstance(payload, Mapping):
            raise ValueError("Agent lifecycle payload must be a mapping.")
        values = dict(payload)
        raw_policy = values.get("context_policy", {})
        if not isinstance(raw_policy, Mapping):
            raise ValueError("Agent lifecycle context_policy must be a mapping.")
        values["context_policy"] = ContextVisibilityPolicy(
            ancestor=raw_policy.get("ancestor", RecordVisibility.NONE.value),
            descendant=raw_policy.get("descendant", RecordVisibility.NONE.value),
            selected_record_ids=frozenset(raw_policy.get("selected_record_ids", ())),
        )
        return AgentDescriptor(**values)

    def descriptor(self, agent_uuid: str) -> AgentDescriptor:
        try:
            return self._agents[agent_uuid]
        except KeyError as error:
            raise KeyError(f"Unknown agent: {agent_uuid}") from error

    def start_root(
        self,
        agent_uuid: str | None = None,
        *,
        model_descriptor: str = "same",
        workspace_id: str = "shared",
        context_policy: ContextVisibilityPolicy | None = None,
    ) -> AgentDescriptor:
        if any(row.parent_agent_uuid is None for row in self._agents.values()):
            raise ValueError("A session context graph has exactly one root agent.")
        return self._start(
            agent_uuid or uuid.uuid4().hex,
            None,
            model_descriptor=model_descriptor,
            workspace_id=workspace_id,
            context_policy=context_policy or ContextVisibilityPolicy(),
        )

    def spawn(
        self,
        parent_agent_uuid: str,
        *,
        agent_uuid: str | None = None,
        parent_task_uuid: str | None = None,
        spawn_call_record_uuid: str | None = None,
        model_descriptor: str = "same",
        workspace_id: str | None = None,
        tool_profile_id: str = "inherited",
        context_policy: ContextVisibilityPolicy | None = None,
    ) -> AgentDescriptor:
        parent = self.descriptor(parent_agent_uuid)
        if parent.status != AgentStatus.RUNNING:
            raise ValueError("Only a running agent can spawn a child.")
        return self._start(
            agent_uuid or uuid.uuid4().hex,
            parent_agent_uuid,
            parent_task_uuid=parent_task_uuid,
            spawn_call_record_uuid=spawn_call_record_uuid,
            model_descriptor=model_descriptor,
            workspace_id=workspace_id or parent.workspace_id,
            tool_profile_id=tool_profile_id,
            context_policy=context_policy or ContextVisibilityPolicy(),
        )

    def _start(self, agent_uuid: str, parent_agent_uuid: str | None, **kwargs: object) -> AgentDescriptor:
        if agent_uuid in self._agents:
            raise ValueError(f"Agent already exists: {agent_uuid}")
        descriptor = AgentDescriptor(
            agent_uuid=agent_uuid,
            session_uuid=self.session_uuid,
            parent_agent_uuid=parent_agent_uuid,
            **kwargs,
        )
        self._agents[agent_uuid] = descriptor
        self._records[agent_uuid] = []
        self.append_record(agent_uuid, self._lifecycle_record(descriptor, RecordType.AGENT_START))
        return descriptor

    def stop(
        self,
        agent_uuid: str,
        *,
        status: AgentStatus | str = AgentStatus.STOPPED,
        summary_record_uuid: str | None = None,
        result_record_uuids: Sequence[str] = (),
        artifact_record_uuids: Sequence[str] = (),
    ) -> AgentDescriptor:
        descriptor = self.descriptor(agent_uuid)
        status = AgentStatus(status)
        if status == AgentStatus.RUNNING:
            raise ValueError("Stopping an agent requires a terminal status.")
        updated = replace(
            descriptor,
            status=status,
            stopped_at=time.time(),
            summary_record_uuid=summary_record_uuid,
            result_record_uuids=tuple(result_record_uuids),
            artifact_record_uuids=tuple(artifact_record_uuids),
        )
        self._agents[agent_uuid] = updated
        self.append_record(agent_uuid, self._lifecycle_record(updated, RecordType.AGENT_STOP))
        return updated

    def resume(self, agent_uuid: str) -> AgentDescriptor:
        descriptor = self.descriptor(agent_uuid)
        if descriptor.status == AgentStatus.RUNNING:
            raise ValueError("Agent is already running.")
        updated = replace(
            descriptor,
            status=AgentStatus.RUNNING,
            stopped_at=None,
            resume_count=descriptor.resume_count + 1,
        )
        self._agents[agent_uuid] = updated
        self.append_record(agent_uuid, self._lifecycle_record(updated, RecordType.AGENT_RESUME))
        return updated

    def append_record(self, agent_uuid: str, record: ContextRecord) -> ContextRecord:
        self.descriptor(agent_uuid)
        if record.session_uuid not in {None, self.session_uuid}:
            raise ValueError("Record session_uuid does not match the context graph.")
        if record.agent_uuid not in {None, agent_uuid}:
            raise ValueError("Record agent_uuid does not match its stream.")
        if any(record.record_id == existing.record_id for rows in self._records.values() for existing in rows):
            raise ValueError(f"Record already exists in context graph: {record.record_id}")
        self._clock += 1
        tagged = replace(
            record,
            session_uuid=self.session_uuid,
            agent_uuid=agent_uuid,
            created_at=record.created_at if record.created_at is not None else time.time(),
            logical_clock=record.logical_clock if record.logical_clock is not None else self._clock,
        )
        self._records[agent_uuid].append(tagged)
        return tagged

    def records(self, agent_uuid: str) -> tuple[ContextRecord, ...]:
        self.descriptor(agent_uuid)
        return tuple(self._records[agent_uuid])

    def lineage(self, agent_uuid: str, *, include_self: bool = True) -> tuple[str, ...]:
        self.descriptor(agent_uuid)
        values = [agent_uuid] if include_self else []
        parent = self._agents[agent_uuid].parent_agent_uuid
        while parent is not None:
            values.append(parent)
            parent = self._agents[parent].parent_agent_uuid
        return tuple(values)

    def descendants(self, agent_uuid: str) -> tuple[str, ...]:
        self.descriptor(agent_uuid)
        found: list[str] = []
        frontier = [agent_uuid]
        while frontier:
            parent = frontier.pop(0)
            children = sorted(
                row.agent_uuid for row in self._agents.values() if row.parent_agent_uuid == parent
            )
            found.extend(children)
            frontier.extend(children)
        return tuple(found)

    def is_visible(self, requester_uuid: str, source_uuid: str, record_uuid: str) -> bool:
        if requester_uuid == source_uuid:
            return True
        requester = self.descriptor(requester_uuid)
        if source_uuid in self.lineage(requester_uuid, include_self=False):
            mode = requester.context_policy.ancestor
            return mode in {RecordVisibility.ROUTABLE, RecordVisibility.INHERIT_ALL_REFERENCES} or (
                mode == RecordVisibility.SELECTED
                and record_uuid in requester.context_policy.selected_record_ids
            )
        if source_uuid in self.descendants(requester_uuid):
            source = self.descriptor(source_uuid)
            mode = requester.context_policy.descendant
            return source.status != AgentStatus.RUNNING and mode in {
                RecordVisibility.COMPLETED_ONLY,
                RecordVisibility.ROUTABLE,
                RecordVisibility.INHERIT_ALL_REFERENCES,
            }
        return False

    def visible_records(self, requester_uuid: str) -> tuple[ContextRecord, ...]:
        rows = list(self.records(requester_uuid))
        for source_uuid, source_rows in self._records.items():
            if source_uuid == requester_uuid:
                continue
            rows.extend(
                row for row in source_rows if self.is_visible(requester_uuid, source_uuid, row.record_id)
            )
        return tuple(sorted(rows, key=lambda row: (row.logical_clock or 0, row.record_id)))

    def _lifecycle_record(self, descriptor: AgentDescriptor, record_type: RecordType) -> ContextRecord:
        payload = asdict(descriptor)
        payload["status"] = descriptor.status.value
        payload["context_policy"] = {
            "ancestor": descriptor.context_policy.ancestor.value,
            "descendant": descriptor.context_policy.descendant.value,
            "selected_record_ids": sorted(descriptor.context_policy.selected_record_ids),
        }
        return ContextRecord(
            record_id=f"{record_type.value}:{descriptor.agent_uuid}:{descriptor.resume_count}",
            record_type=record_type,
            payload=payload,
            task_uuid=descriptor.parent_task_uuid,
        )


class CrossAgentReuseRuntime:
    """Exact, validity-aware reuse over records visible in an agent graph."""

    def __init__(self, graph: AgentContextGraph) -> None:
        self.graph = graph
        self._results: dict[str, list[ReusableToolResult]] = {}
        self._resource_epochs: dict[str, int] = {}
        self._mutations: list[tuple[int, str, str]] = []

    @staticmethod
    def call_signature(tool_uri: str, arguments: Mapping[str, object]) -> str:
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{tool_uri}\0{canonical}".encode("utf-8")).hexdigest()

    def register_result(
        self,
        *,
        tool_uri: str,
        arguments: Mapping[str, object],
        source_record: ContextRecord,
        effect: ToolEffectDescriptor,
        payload: object,
        native_kv: NativeKVHandle | None = None,
    ) -> ReusableToolResult:
        if source_record.agent_uuid is None or source_record.logical_clock is None:
            raise ValueError("Reusable records must first be attached to an agent context graph.")
        if effect.effect not in {EffectType.PURE, EffectType.READ}:
            raise ValueError("Only PURE or READ results can enter the reuse index.")
        signature = self.call_signature(tool_uri, arguments)
        entry = ReusableToolResult(
            signature=signature,
            source_record=source_record,
            source_agent_uuid=source_record.agent_uuid,
            effect=effect,
            payload=payload,
            resource_epochs={row.canonical: self._resource_epochs.get(row.canonical, 0) for row in effect.resources},
            created_at=source_record.created_at or time.time(),
            logical_clock=source_record.logical_clock,
            native_kv=native_kv,
        )
        self._results.setdefault(signature, []).append(entry)
        return entry

    def register_write(self, agent_uuid: str, effect: ToolEffectDescriptor) -> None:
        if effect.effect != EffectType.WRITE:
            raise ValueError("register_write requires a WRITE descriptor.")
        self.graph.descriptor(agent_uuid)
        for resource in effect.resources:
            self._resource_epochs[resource.canonical] = self._resource_epochs.get(resource.canonical, 0) + 1
            self._mutations.append((self.graph.logical_clock, agent_uuid, resource.canonical))

    def lookup(
        self,
        *,
        request_agent_uuid: str,
        tool_uri: str,
        arguments: Mapping[str, object],
        requested_effect: ToolEffectDescriptor,
        requested_native: NativeKVHandle | None = None,
        allow_payload: bool = True,
        allow_native_kv: bool = True,
        now: float | None = None,
        external_validator: Callable[[ResourceIdentity, str | None], bool] | None = None,
    ) -> tuple[ReusableToolResult | None, ReuseDecisionRecord]:
        now = time.time() if now is None else now
        if not requested_effect.reuse_enabled:
            return None, self._decision(request_agent_uuid, None, ReuseDecision.MISS, ReuseMissReason.POLICY_DISABLED, requested_effect, 0.0)
        if requested_effect.effect == EffectType.UNKNOWN:
            return None, self._decision(request_agent_uuid, None, ReuseDecision.MISS, ReuseMissReason.UNKNOWN_EFFECT, requested_effect, 0.0)
        entries = reversed(self._results.get(self.call_signature(tool_uri, arguments), ()))
        invisible = False
        invalid_reason: ReuseMissReason | None = None
        for entry in entries:
            if not self.graph.is_visible(request_agent_uuid, entry.source_agent_uuid, entry.source_record.record_id):
                invisible = True
                continue
            age = max(0.0, now - entry.created_at)
            reason = self._validity_reason(request_agent_uuid, entry, age, external_validator)
            if reason is not ReuseMissReason.NONE:
                invalid_reason = reason
                continue
            kv_reused = False
            kv_reason = ReuseMissReason.NONE
            if allow_native_kv and requested_native is not None and entry.native_kv is not None:
                if not entry.native_kv.resident:
                    kv_reason = ReuseMissReason.KV_EVICTED
                elif not entry.native_kv.compatible_with(requested_native):
                    kv_reason = ReuseMissReason.MODEL_INCOMPATIBLE
                else:
                    kv_reused = True
            payload_reused = allow_payload
            if not payload_reused and not kv_reused:
                reason = kv_reason if kv_reason is not ReuseMissReason.NONE else ReuseMissReason.POLICY_DISABLED
                return None, self._decision(request_agent_uuid, entry, ReuseDecision.MISS, reason, requested_effect, age)
            decision = self._decision(
                request_agent_uuid,
                entry,
                ReuseDecision.HIT,
                kv_reason,
                requested_effect,
                age,
                kv_reused=kv_reused,
                payload_reused=payload_reused,
            )
            return entry, decision
        reason = invalid_reason or (ReuseMissReason.NOT_VISIBLE if invisible else ReuseMissReason.NO_MATCH)
        state = ReuseDecision.INVALID if invalid_reason else ReuseDecision.MISS
        return None, self._decision(request_agent_uuid, None, state, reason, requested_effect, 0.0)

    def _validity_reason(
        self,
        requester_uuid: str,
        entry: ReusableToolResult,
        age: float,
        external_validator: Callable[[ResourceIdentity, str | None], bool] | None,
    ) -> ReuseMissReason:
        effect = entry.effect
        if effect.max_age_seconds is not None and age > effect.max_age_seconds:
            return ReuseMissReason.EXPIRED
        if effect.consistency == ConsistencyMode.EXTERNAL:
            if external_validator is None or any(
                not external_validator(resource, effect.version_token) for resource in effect.resources
            ):
                return ReuseMissReason.EXTERNAL_VALIDATION_FAILED
        requester_lineage = set(self.graph.lineage(requester_uuid))
        for resource in effect.resources:
            canonical = resource.canonical
            current = self._resource_epochs.get(canonical, 0)
            captured = entry.resource_epochs.get(canonical, 0)
            if effect.consistency == ConsistencyMode.SESSION_TREE and current != captured:
                return ReuseMissReason.WRITE_INVALIDATION
            if effect.consistency == ConsistencyMode.ANCESTRY and any(
                clock > entry.logical_clock and writer in requester_lineage and target == canonical
                for clock, writer, target in self._mutations
            ):
                return ReuseMissReason.WRITE_INVALIDATION
        return ReuseMissReason.NONE

    def _decision(
        self,
        requester_uuid: str,
        entry: ReusableToolResult | None,
        decision: ReuseDecision,
        reason: ReuseMissReason,
        requested_effect: ToolEffectDescriptor,
        age: float,
        *,
        kv_reused: bool = False,
        payload_reused: bool = False,
    ) -> ReuseDecisionRecord:
        native = entry.native_kv if entry is not None else None
        return ReuseDecisionRecord(
            request_agent_uuid=requester_uuid,
            source_record_uuid=entry.source_record.record_id if entry else None,
            source_agent_uuid=entry.source_agent_uuid if entry else None,
            resource_identities=tuple(row.canonical for row in requested_effect.resources),
            decision=decision,
            reason=reason,
            validation_mode=requested_effect.consistency,
            age_seconds=age,
            causal_position=self.graph.logical_clock,
            kv_reused=kv_reused,
            payload_reused=payload_reused,
            tool_execution_avoided=decision == ReuseDecision.HIT,
            native_tokens_reused=native.token_count if kv_reused and native else 0,
            native_bytes_reused=native.byte_count if kv_reused and native else 0,
        )
