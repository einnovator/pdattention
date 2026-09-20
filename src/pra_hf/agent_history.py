"""Engine-neutral logical history records for agent mediation.

The types in this module deliberately contain no tensors, cache pages, model
handles, or agent-specific command syntax.  Applications may provide explicit
``pra_record`` metadata.  Otherwise :class:`OpenAIRecordizer` understands only
the standard OpenAI system/user/assistant/tool and ``tool_call_id`` relations.
Non-standard agent protocols belong in optional harness adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence


class AgentRecordRole(str, Enum):
    SYSTEM = "system"
    TASK = "task"
    USER_INPUT = "user_input"
    ASSISTANT_ACTION = "assistant_action"
    TOOL_OBSERVATION = "tool_observation"
    SOURCE_VIEW = "source_view"
    MUTATION = "mutation"
    VERIFICATION = "verification"
    PROGRESS = "progress"
    ERROR_OR_REJECTION = "error_or_rejection"
    FINALIZATION = "finalization"


class TransportStatus(str, Enum):
    """Whether an action/result exchange completed at the transport layer."""

    UNKNOWN = "unknown"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class SemanticStatus(str, Enum):
    """Portable operation outcome, independent of provider termination."""

    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AgentRecord:
    record_id: str
    turn_id: str
    causal_group_id: str
    message_index: int
    role: str
    content: str
    primary_role: AgentRecordRole
    semantic_roles: tuple[AgentRecordRole, ...]
    command: str | None = None
    return_code: int | None = None
    resource_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    thread_id: str | None = None
    subagent_id: str | None = None
    parent_thread_id: str | None = None
    transport_status: TransportStatus = TransportStatus.UNKNOWN
    semantic_status: SemanticStatus = SemanticStatus.UNKNOWN
    complete: bool | None = None
    error_kind: str | None = None
    workspace_generation: str | None = None
    depends_on: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_role", AgentRecordRole(self.primary_role))
        object.__setattr__(
            self,
            "semantic_roles",
            tuple(dict.fromkeys(AgentRecordRole(value) for value in self.semantic_roles)),
        )
        object.__setattr__(self, "resource_ids", tuple(dict.fromkeys(self.resource_ids)))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "transport_status", TransportStatus(self.transport_status))
        object.__setattr__(self, "semantic_status", SemanticStatus(self.semantic_status))
        object.__setattr__(self, "depends_on", tuple(dict.fromkeys(self.depends_on)))
        object.__setattr__(self, "supersedes", tuple(dict.fromkeys(self.supersedes)))
        if (
            self.semantic_status == SemanticStatus.SUCCEEDED
            and self.transport_status not in {
                TransportStatus.UNKNOWN,
                TransportStatus.COMPLETED,
            }
        ):
            raise ValueError("semantic success requires a completed or unknown transport")

    def has_role(self, role: AgentRecordRole) -> bool:
        return role == self.primary_role or role in self.semantic_roles


@dataclass(frozen=True)
class AgentTurn:
    turn_id: str
    causal_group_id: str
    record_ids: tuple[str, ...]
    first_message_index: int
    complete: bool


@dataclass(frozen=True)
class CanonicalAgentHistory:
    records: tuple[AgentRecord, ...]
    turns: tuple[AgentTurn, ...]

    @property
    def record_by_id(self) -> dict[str, AgentRecord]:
        return {record.record_id: record for record in self.records}

    @property
    def turn_by_id(self) -> dict[str, AgentTurn]:
        return {turn.turn_id: turn for turn in self.turns}

    @property
    def digest(self) -> str:
        """Stable identity used to reject stale cross-request selection plans."""

        payload = [
            {
                "record_id": row.record_id,
                "turn_id": row.turn_id,
                "causal_group_id": row.causal_group_id,
                "role": row.role,
                "content": row.content,
                "primary_role": row.primary_role.value,
                "semantic_roles": tuple(role.value for role in row.semantic_roles),
                "resource_ids": row.resource_ids,
                "session_id": row.session_id,
                "thread_id": row.thread_id,
                "subagent_id": row.subagent_id,
                "parent_thread_id": row.parent_thread_id,
                "transport_status": row.transport_status.value,
                "semantic_status": row.semantic_status.value,
                "complete": row.complete,
                "error_kind": row.error_kind,
                "workspace_generation": row.workspace_generation,
                "depends_on": row.depends_on,
                "supersedes": row.supersedes,
            }
            for row in self.records
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AgentMemoryExclusion:
    """Auditable logical exclusion; the tombstone is never model-visible."""

    causal_group_id: str
    record_ids: tuple[str, ...]
    rule_id: str
    classification: str
    reason: str
    resource_ids: tuple[str, ...]
    witness_record_ids: tuple[str, ...]
    tombstone: str
    excluded_tokens: int


@dataclass(frozen=True)
class AgentMemoryBudget:
    max_tokens: int
    max_records: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive when provided")


@dataclass(frozen=True)
class AgentMemoryPlan:
    policy: str
    selected_record_ids: tuple[str, ...]
    selected_causal_group_ids: tuple[str, ...]
    selection_reasons: tuple[tuple[str, str], ...]
    full_history_tokens: int
    selected_tokens: int
    requested_budget_tokens: int
    mandatory_tokens: int
    mandatory_overflow_tokens: int
    head_turns: int
    tail_turns: int
    middle_candidate_turns: int
    middle_selected_turns: int
    exclusions: tuple[AgentMemoryExclusion, ...] = ()

    @property
    def realized_retention_fraction(self) -> float:
        return self.selected_tokens / self.full_history_tokens if self.full_history_tokens else 1.0

    @property
    def digest(self) -> str:
        payload = {
            "policy": self.policy,
            "selected_record_ids": self.selected_record_ids,
            "selected_causal_group_ids": self.selected_causal_group_ids,
            "selection_reasons": self.selection_reasons,
            "requested_budget_tokens": self.requested_budget_tokens,
            "head_turns": self.head_turns,
            "tail_turns": self.tail_turns,
            "exclusions": [
                {
                    "causal_group_id": row.causal_group_id,
                    "record_ids": row.record_ids,
                    "rule_id": row.rule_id,
                    "classification": row.classification,
                    "resource_ids": row.resource_ids,
                    "witness_record_ids": row.witness_record_ids,
                    "tombstone": row.tombstone,
                    "excluded_tokens": row.excluded_tokens,
                }
                for row in self.exclusions
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RecordizationResult:
    history: CanonicalAgentHistory
    source: str
    explicit_records: int
    inferred_records: int
    ambiguity_reasons: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return not self.ambiguity_reasons


class AgentRecordizer(Protocol):
    def recordize(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> RecordizationResult: ...


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _stable_id(
    *, session_id: str, index: int, role: str, content: str, suffix: str = "",
) -> str:
    value = f"{session_id}\0{index}\0{role}\0{content}\0{suffix}".encode()
    return "r-" + hashlib.sha256(value).hexdigest()[:20]


def _record_declaration(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = message.get("pra_record")
    if isinstance(direct, Mapping):
        return direct
    metadata = message.get("metadata")
    if isinstance(metadata, Mapping):
        direct = metadata.get("pra_record")
        if isinstance(direct, Mapping):
            return direct
        pra = metadata.get("pra")
        if isinstance(pra, Mapping) and isinstance(pra.get("record"), Mapping):
            return pra["record"]
    return None


def _execution_receipt(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = message.get("pra_execution_receipt")
    if isinstance(direct, Mapping):
        return direct
    metadata = message.get("metadata")
    if isinstance(metadata, Mapping):
        direct = metadata.get("pra_execution_receipt")
        if isinstance(direct, Mapping):
            return direct
        pra = metadata.get("pra")
        if isinstance(pra, Mapping) and isinstance(pra.get("execution_receipt"), Mapping):
            return pra["execution_receipt"]
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item) for item in value if str(item)))


def _receipt_resources(receipt: Mapping[str, Any] | None) -> tuple[str, ...]:
    if receipt is None:
        return ()
    resources: list[str] = []
    for row in receipt.get("resources", ()):
        if isinstance(row, Mapping) and row.get("resource_id"):
            resources.append(str(row["resource_id"]))
    return tuple(dict.fromkeys(resources))


def _outcome_fields(
    declaration: Mapping[str, Any] | None,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if declaration is not None:
        source.update(declaration)
    if receipt is not None:
        source.update(receipt)
    return {
        "transport_status": TransportStatus(str(source.get("transport_status", "unknown"))),
        "semantic_status": SemanticStatus(str(source.get("semantic_status", "unknown"))),
        "complete": (
            bool(source["result_complete"])
            if source.get("result_complete") is not None
            else bool(source["complete"])
            if source.get("complete") is not None else None
        ),
        "error_kind": (
            str(source["error_kind"]) if source.get("error_kind") is not None else None
        ),
        "workspace_generation": (
            str(source["workspace_generation"])
            if source.get("workspace_generation") is not None else None
        ),
        "depends_on": _string_tuple(source.get("depends_on")),
        "supersedes": _string_tuple(source.get("supersedes")),
    }


def _roles(value: Any, primary: AgentRecordRole) -> tuple[AgentRecordRole, ...]:
    if value is None:
        return (primary,)
    if not isinstance(value, (list, tuple)):
        raise ValueError("semantic_roles must be a sequence")
    return tuple(dict.fromkeys((primary, *(AgentRecordRole(item) for item in value))))


def _tool_arguments(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _argument_value(arguments: Mapping[str, Any], path: str) -> Any:
    value: Any = arguments
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _resolve_tool_semantics(
    declaration: Mapping[str, Any], arguments: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Resolve portable semantics from a tool declaration and its arguments.

    A single mixed tool (for example a file editor with ``view`` and
    ``str_replace`` commands) may declare ``operation_argument`` plus an
    ``operation_map``. ``resource_arguments`` extracts stable resource names
    without teaching the runtime an agent-specific schema.
    """

    resolved = dict(declaration)
    parsed = _tool_arguments(arguments)
    operation_argument = resolved.get("operation_argument")
    operation_map = resolved.get("operation_map")
    if isinstance(operation_argument, str) and isinstance(operation_map, Mapping):
        operation_value = _argument_value(parsed, operation_argument)
        mapped = operation_map.get(str(operation_value))
        if mapped is None:
            mapped = resolved.get("default_operation_kind")
        if mapped is not None:
            resolved["operation_kind"] = str(mapped)

    static_resources = resolved.get("resource_ids", ())
    if isinstance(static_resources, str):
        static_resources = (static_resources,)
    resources: list[str] = []
    if isinstance(static_resources, (list, tuple)):
        resources.extend(str(item) for item in static_resources if str(item))

    resource_arguments = resolved.get("resource_arguments", ())
    if isinstance(resource_arguments, str):
        resource_arguments = (resource_arguments,)
    if isinstance(resource_arguments, (list, tuple)):
        for path in resource_arguments:
            if not isinstance(path, str):
                continue
            value = _argument_value(parsed, path)
            if isinstance(value, str) and value:
                resources.append(value)
            elif isinstance(value, (list, tuple)):
                resources.extend(str(item) for item in value if str(item))
    return resolved, tuple(dict.fromkeys(resources))


def _semantic_roles_for_operation(
    primary: AgentRecordRole,
    operation_kind: str | None,
    semantic_status: SemanticStatus,
) -> tuple[AgentRecordRole, ...]:
    roles = [primary]
    if operation_kind == "read":
        roles.append(AgentRecordRole.SOURCE_VIEW)
    elif operation_kind == "write":
        roles.append(AgentRecordRole.MUTATION)
    elif operation_kind == "verify":
        roles.append(AgentRecordRole.VERIFICATION)
    elif operation_kind == "progress":
        roles.append(AgentRecordRole.PROGRESS)
    elif operation_kind == "finalization":
        roles.append(AgentRecordRole.FINALIZATION)
    if semantic_status in {SemanticStatus.FAILED, SemanticStatus.PARTIAL}:
        roles.append(AgentRecordRole.ERROR_OR_REJECTION)
    return tuple(dict.fromkeys(roles))


class OpenAIRecordizer:
    """Recordize explicit PRA metadata or standard OpenAI tool-call traffic.

    The fallback intentionally does not guess that an arbitrary user message is
    a tool result.  A non-standard agent can attach typed metadata or register a
    compatibility adapter outside the PRA engine.
    """

    def recordize(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> RecordizationResult:
        metadata = dict(request_metadata or {})
        semantics_by_name = metadata.get("tool_semantics_by_name")
        semantics_by_name = (
            dict(semantics_by_name) if isinstance(semantics_by_name, Mapping) else {}
        )
        session_id = str(metadata.get("session_id") or "stateless")
        records: list[AgentRecord] = []
        ambiguity: list[str] = []
        explicit_count = inferred_count = 0
        task_seen = False
        pending_groups: dict[str, str] = {}
        pending_calls: dict[str, Mapping[str, Any]] = {}
        open_failure_group: str | None = None

        for index, raw in enumerate(messages):
            message = dict(raw)
            role = str(message.get("role", ""))
            content = _content(message)
            declaration = _record_declaration(message)
            receipt = _execution_receipt(message)
            if declaration is not None:
                explicit_count += 1
                primary = AgentRecordRole(str(declaration["primary_role"]))
                record_id = str(declaration.get("record_id") or _stable_id(
                    session_id=session_id, index=index, role=role, content=content,
                ))
                turn_id = str(declaration.get("turn_id") or f"turn:{record_id}")
                group_id = str(declaration.get("causal_group_id") or turn_id)
                record_metadata = dict(message.get("metadata") or {})
                record_metadata.update(dict(declaration.get("metadata") or {}))
                if receipt is not None:
                    record_metadata["execution_receipt"] = dict(receipt)
                outcome = _outcome_fields(declaration, receipt)
                declared_resources = tuple(
                    str(value) for value in declaration.get("resource_ids", ())
                )
                records.append(AgentRecord(
                    record_id=record_id,
                    turn_id=turn_id,
                    causal_group_id=group_id,
                    message_index=index,
                    role=role,
                    content=content,
                    primary_role=primary,
                    semantic_roles=_roles(declaration.get("semantic_roles"), primary),
                    command=(str(declaration["command"]) if declaration.get("command") is not None else None),
                    return_code=(
                        int(receipt["return_code"])
                        if receipt is not None and receipt.get("return_code") is not None
                        else int(declaration["return_code"])
                        if declaration.get("return_code") is not None else None
                    ),
                    resource_ids=tuple(dict.fromkeys(
                        (*declared_resources, *_receipt_resources(receipt))
                    )),
                    metadata=record_metadata,
                    session_id=str(declaration.get("session_id") or session_id),
                    thread_id=(
                        str(declaration["thread_id"])
                        if declaration.get("thread_id") is not None
                        else str(metadata["thread_id"])
                        if metadata.get("thread_id") is not None else None
                    ),
                    subagent_id=(
                        str(declaration["subagent_id"])
                        if declaration.get("subagent_id") is not None else None
                    ),
                    parent_thread_id=(
                        str(declaration["parent_thread_id"])
                        if declaration.get("parent_thread_id") is not None else None
                    ),
                    **outcome,
                ))
                continue

            inferred_count += 1
            inferred_resource_ids: tuple[str, ...] = ()
            outcome = _outcome_fields(None, receipt)
            operation_kind: str | None = None
            tool_metadata: dict[str, Any] = {}
            record_id = _stable_id(
                session_id=session_id, index=index, role=role, content=content,
            )
            if role == "system":
                primary = AgentRecordRole.SYSTEM
                turn_id = group_id = f"system:{record_id}"
                semantic = (primary,)
            elif role == "user" and not task_seen:
                task_seen = True
                primary = AgentRecordRole.TASK
                turn_id = group_id = f"task:{record_id}"
                semantic = (primary,)
            elif role == "assistant":
                calls = message.get("tool_calls")
                calls = calls if isinstance(calls, (list, tuple)) else ()
                call_ids = tuple(
                    str(call.get("id")) for call in calls
                    if isinstance(call, Mapping) and call.get("id")
                )
                call_declarations = []
                call_resource_ids: list[str] = []
                for call in calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    function = function if isinstance(function, Mapping) else {}
                    name = str(function.get("name") or call.get("name") or "")
                    declaration = semantics_by_name.get(name)
                    arguments = function.get("arguments", call.get("arguments"))
                    resolved_declaration: Mapping[str, Any] | None = None
                    resources: tuple[str, ...] = ()
                    if isinstance(declaration, Mapping):
                        resolved_declaration, resources = _resolve_tool_semantics(
                            declaration, arguments,
                        )
                        call_resource_ids.extend(resources)
                    call_declarations.append({
                        "id": str(call.get("id") or ""),
                        "name": name,
                        "arguments": arguments,
                        **(
                            {"declared_semantics": dict(resolved_declaration)}
                            if resolved_declaration is not None else {}
                        ),
                        **({"resource_ids": list(resources)} if resources else {}),
                    })
                inferred_resource_ids = tuple(dict.fromkeys(call_resource_ids))
                group_seed = ",".join(call_ids) or record_id
                group_id = "turn:" + hashlib.sha256(group_seed.encode()).hexdigest()[:16]
                turn_id = group_id.removeprefix("turn:")
                for call_id in call_ids:
                    pending_groups[call_id] = group_id
                for call_declaration in call_declarations:
                    call_id = str(call_declaration.get("id") or "")
                    if call_id:
                        pending_calls[call_id] = call_declaration
                primary = AgentRecordRole.ASSISTANT_ACTION
                if (
                    len(call_declarations) == 1
                    and isinstance(call_declarations[0].get("declared_semantics"), Mapping)
                ):
                    operation_kind = str(
                        call_declarations[0]["declared_semantics"].get(
                            "operation_kind", "unknown"
                        )
                    )
                semantic = _semantic_roles_for_operation(
                    primary, operation_kind, outcome["semantic_status"]
                )
                if open_failure_group and not outcome["depends_on"]:
                    outcome["depends_on"] = (open_failure_group,)
                if content.strip() and AgentRecordRole.PROGRESS not in semantic:
                    semantic = (*semantic, AgentRecordRole.PROGRESS)
                if not calls:
                    ambiguity.append(f"assistant_without_typed_tool_call:{index}")
            elif role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                group_id = pending_groups.get(call_id, "")
                call_declaration = pending_calls.get(call_id, {})
                if not call_id or not group_id:
                    ambiguity.append(f"unpaired_tool_observation:{index}")
                    group_id = f"unpaired:{record_id}"
                turn_id = group_id.removeprefix("turn:")
                primary = AgentRecordRole.TOOL_OBSERVATION
                declared_semantics = call_declaration.get("declared_semantics")
                if isinstance(declared_semantics, Mapping):
                    operation_kind = str(declared_semantics.get("operation_kind", "unknown"))
                    tool_metadata.update({
                        "operation_kind": operation_kind,
                        "tool_category": str(declared_semantics.get("category", "generic")),
                    })
                if receipt is not None and receipt.get("operation_kind") is not None:
                    operation_kind = str(receipt["operation_kind"])
                    tool_metadata["operation_kind"] = operation_kind
                if receipt is not None:
                    receipt_resources = tuple(
                        row for row in receipt.get("resources", ())
                        if isinstance(row, Mapping) and row.get("resource_id")
                    )
                    resource_accesses = []
                    for resource in receipt_resources:
                        span = resource.get("span")
                        span = span if isinstance(span, Mapping) else {}
                        resource_accesses.append({
                            "resource_id": str(resource["resource_id"]),
                            "version": str(
                                resource.get("version_after")
                                or resource.get("resource_version_fingerprint")
                                or resource.get("version_before")
                                or "unknown"
                            ),
                            "span_kind": str(span.get("kind") or "whole"),
                            **(
                                {"span_start": int(span["start"])}
                                if span.get("start") is not None else {}
                            ),
                            **(
                                {"span_end": int(span["end"])}
                                if span.get("end") is not None else {}
                            ),
                            **(
                                {"signature": str(span["signature"])}
                                if span.get("signature") is not None else {}
                            ),
                        })
                    tool_metadata.update({
                        "resource_accesses": resource_accesses,
                        "output_complete": bool(receipt.get(
                            "result_complete", receipt.get("complete", False)
                        )),
                        "timed_out": (
                            str(receipt.get("transport_status") or "") == "timed_out"
                        ),
                        "effect_trace_complete": bool(receipt.get(
                            "effect_trace_complete", receipt.get("complete", False)
                        )),
                    })
                    if operation_kind == "write":
                        tool_metadata["changed_resource_ids"] = [
                            str(row["resource_id"]) for row in receipt_resources
                        ]
                    elif operation_kind == "search_discovery":
                        tool_metadata["discovered_resource_ids"] = [
                            str(row["resource_id"]) for row in receipt_resources
                        ]
                inferred_resource_ids = tuple(dict.fromkeys((
                    *(str(value) for value in call_declaration.get("resource_ids", ())),
                    *_receipt_resources(receipt),
                )))
                semantic = _semantic_roles_for_operation(
                    primary, operation_kind, outcome["semantic_status"]
                )
            else:
                primary = AgentRecordRole.USER_INPUT
                turn_id = group_id = f"input:{record_id}"
                semantic = (primary,)
                if role == "user" and records and records[-1].role == "assistant":
                    ambiguity.append(f"untyped_user_after_assistant:{index}")

            records.append(AgentRecord(
                record_id=record_id,
                turn_id=turn_id,
                causal_group_id=group_id,
                message_index=index,
                role=role,
                content=content,
                primary_role=primary,
                semantic_roles=semantic,
                resource_ids=inferred_resource_ids,
                metadata={
                    **dict(message.get("metadata") or {}),
                    **tool_metadata,
                    **(
                        {"execution_receipt": dict(receipt)}
                        if receipt is not None else {}
                    ),
                    **(
                        {
                            "tool_calls": call_declarations,
                            **(
                                {
                                    "operation_kind": call_declarations[0]["declared_semantics"].get(
                                        "operation_kind", "unknown"
                                    ),
                                    "tool_category": call_declarations[0]["declared_semantics"].get(
                                        "category", "generic"
                                    ),
                                }
                                if len(call_declarations) == 1
                                and isinstance(call_declarations[0].get("declared_semantics"), Mapping)
                                else {}
                            ),
                        }
                        if role == "assistant" else {}
                    ),
                },
                return_code=(
                    int(receipt["return_code"])
                    if receipt is not None and receipt.get("return_code") is not None
                    else None
                ),
                session_id=session_id,
                thread_id=(
                    str(metadata["thread_id"])
                    if metadata.get("thread_id") is not None else None
                ),
                subagent_id=(
                    str(metadata["subagent_id"])
                    if metadata.get("subagent_id") is not None else None
                ),
                parent_thread_id=(
                    str(metadata["parent_thread_id"])
                    if metadata.get("parent_thread_id") is not None else None
                ),
                **outcome,
            ))
            if role == "tool":
                if outcome["semantic_status"] in {
                    SemanticStatus.FAILED,
                    SemanticStatus.PARTIAL,
                }:
                    open_failure_group = group_id
                elif open_failure_group is not None:
                    open_failure_group = None

        grouped: dict[str, list[AgentRecord]] = {}
        for record in records:
            if record.primary_role in {
                AgentRecordRole.SYSTEM, AgentRecordRole.TASK, AgentRecordRole.USER_INPUT,
            }:
                continue
            grouped.setdefault(record.causal_group_id, []).append(record)
        turns = tuple(
            AgentTurn(
                turn_id=rows[0].turn_id,
                causal_group_id=group_id,
                record_ids=tuple(row.record_id for row in rows),
                first_message_index=min(row.message_index for row in rows),
                complete=(
                    any(row.has_role(AgentRecordRole.ASSISTANT_ACTION) for row in rows)
                    and any(row.has_role(AgentRecordRole.TOOL_OBSERVATION) for row in rows)
                    and all(row.complete is not False for row in rows)
                ),
            )
            for group_id, rows in grouped.items()
        )
        source = (
            "typed"
            if explicit_count == len(records)
            else "openai_standard"
            if explicit_count == 0
            else "mixed"
        )
        return RecordizationResult(
            CanonicalAgentHistory(tuple(records), turns),
            source,
            explicit_count,
            inferred_count,
            tuple(dict.fromkeys(ambiguity)),
        )


__all__ = [
    "AgentMemoryBudget",
    "AgentMemoryExclusion",
    "AgentMemoryPlan",
    "AgentRecord",
    "AgentRecordRole",
    "AgentRecordizer",
    "AgentTurn",
    "CanonicalAgentHistory",
    "OpenAIRecordizer",
    "RecordizationResult",
    "SemanticStatus",
    "TransportStatus",
]
