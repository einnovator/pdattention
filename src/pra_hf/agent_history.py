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

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_role", AgentRecordRole(self.primary_role))
        object.__setattr__(
            self,
            "semantic_roles",
            tuple(dict.fromkeys(AgentRecordRole(value) for value in self.semantic_roles)),
        )
        object.__setattr__(self, "resource_ids", tuple(dict.fromkeys(self.resource_ids)))
        object.__setattr__(self, "metadata", dict(self.metadata))

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
                "role": row.role,
                "content": row.content,
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


def _roles(value: Any, primary: AgentRecordRole) -> tuple[AgentRecordRole, ...]:
    if value is None:
        return (primary,)
    if not isinstance(value, (list, tuple)):
        raise ValueError("semantic_roles must be a sequence")
    return tuple(dict.fromkeys((primary, *(AgentRecordRole(item) for item in value))))


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
        session_id = str(metadata.get("session_id") or "stateless")
        records: list[AgentRecord] = []
        ambiguity: list[str] = []
        explicit_count = inferred_count = 0
        task_seen = False
        pending_groups: dict[str, str] = {}

        for index, raw in enumerate(messages):
            message = dict(raw)
            role = str(message.get("role", ""))
            content = _content(message)
            declaration = _record_declaration(message)
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
                    return_code=(int(declaration["return_code"]) if declaration.get("return_code") is not None else None),
                    resource_ids=tuple(str(value) for value in declaration.get("resource_ids", ())),
                    metadata=record_metadata,
                ))
                continue

            inferred_count += 1
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
                group_seed = ",".join(call_ids) or record_id
                group_id = "turn:" + hashlib.sha256(group_seed.encode()).hexdigest()[:16]
                turn_id = group_id.removeprefix("turn:")
                for call_id in call_ids:
                    pending_groups[call_id] = group_id
                primary = AgentRecordRole.ASSISTANT_ACTION
                semantic = (
                    (primary, AgentRecordRole.PROGRESS)
                    if content.strip() else (primary,)
                )
                if not calls:
                    ambiguity.append(f"assistant_without_typed_tool_call:{index}")
            elif role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                group_id = pending_groups.get(call_id, "")
                if not call_id or not group_id:
                    ambiguity.append(f"unpaired_tool_observation:{index}")
                    group_id = f"unpaired:{record_id}"
                turn_id = group_id.removeprefix("turn:")
                primary = AgentRecordRole.TOOL_OBSERVATION
                semantic = (primary,)
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
                metadata=dict(message.get("metadata") or {}),
            ))

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
]
