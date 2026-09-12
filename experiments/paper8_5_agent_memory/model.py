"""Logical schemas for the Paper 8.5 agent-memory study.

The objects in this module deliberately contain no engine handles, tensors,
pages, K/V arrays, or cache identities.  They describe only the canonical
trajectory and a logical selection decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class AgentRecordRole(str, Enum):
    SYSTEM = "system"
    TASK = "task"
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

    @property
    def realized_retention_fraction(self) -> float:
        if self.full_history_tokens == 0:
            return 1.0
        return self.selected_tokens / self.full_history_tokens

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
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
