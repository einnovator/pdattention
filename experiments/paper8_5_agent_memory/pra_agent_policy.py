"""Engine-neutral Paper 8.5 history policies for typed PRA Agent records.

The adapter operates on :class:`ContextRecord` identities before either the
ordinary-text fallback or a native PRA engine renders them.  It therefore does
not parse an agent prompt and the same selected identity ledger can later be
consumed by Paper 4.5's resident-K/V engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping, Sequence

from pra_hf.agent_history import (
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
)
from pra_hf.context_records import (
    ContextRecord,
    RecordType,
    RecordViewName,
    serialize_record,
)

from .matched_token_tail import (
    MatchedTokenTailConfig,
    matched_token_tail_full_floor_record_ids,
    materialize_matched_token_tail,
)
from .dag import FrontierDagRetirementSelector
from .model import AgentMemoryBudget
from .selectors import (
    PersistentInstructionEpochRetirementConfig,
    PersistentInstructionEpochRetirementSelector,
)


TokenCounter = Callable[[str], int]


def _visible_content(record: ContextRecord) -> str:
    """Return exactly the variable history body exposed by the text fallback."""

    if (
        record.record_type == RecordType.GENERIC_TEXT
        and isinstance(record.payload, Mapping)
        and record.payload.get("role") in {"system", "user", "assistant", "tool"}
        and "text" in record.payload
    ):
        return str(record.payload["text"])
    return serialize_record(record, view=RecordViewName.FULL)


def _tool_roles(record: ContextRecord) -> tuple[AgentRecordRole, ...]:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    uri = str(payload.get("producer_tool_uri") or "").lower()
    compact = payload.get("compact")
    compact_fields = (
        compact.get("fields", {})
        if isinstance(compact, Mapping) else {}
    )
    operation_kind = str(
        compact_fields.get("operation_kind")
        if isinstance(compact_fields, Mapping) else ""
    ).lower()
    roles = [AgentRecordRole.TOOL_OBSERVATION]
    if (
        any(name in uri for name in ("list_files", "read_file", "search_text"))
        or operation_kind in {"read", "search_discovery"}
    ):
        roles.append(AgentRecordRole.SOURCE_VIEW)
    if (
        any(name in uri for name in ("write_file", "replace_text"))
        or operation_kind == "write"
    ):
        roles.append(AgentRecordRole.MUTATION)
    if "git_status" in uri or operation_kind in {"diff", "verify"}:
        roles.append(AgentRecordRole.VERIFICATION)
    return tuple(roles)


def recordize_pra_agent_records(
    records: Sequence[ContextRecord],
) -> CanonicalAgentHistory:
    """Map typed runtime records to portable causal action/observation groups.

    The first ordinary user record is the immutable task statement.  Later
    genuine user records remain immutable user input.  An assistant action and
    its following typed tool response are one indivisible completed turn.
    Product-generated malformed-action recovery is likewise kept with the
    rejected assistant action rather than mistaken for a new user task.
    """

    logical: list[AgentRecord] = []
    turns: list[AgentTurn] = []
    pending: dict[str, object] | None = None
    task_seen = False
    turn_index = 0

    def finish(*, complete: bool) -> None:
        nonlocal pending
        if pending is None:
            return
        turns.append(AgentTurn(
            turn_id=str(pending["turn_id"]),
            causal_group_id=str(pending["group_id"]),
            record_ids=tuple(pending["record_ids"]),
            first_message_index=int(pending["first_index"]),
            complete=complete,
        ))
        pending = None

    for index, record in enumerate(records):
        content = _visible_content(record)
        payload = record.payload if isinstance(record.payload, Mapping) else {}
        role = str(payload.get("role") or "")

        if record.record_type == RecordType.GENERIC_TEXT and role == "assistant":
            finish(complete=False)
            turn_id = f"pra-turn-{turn_index:04d}"
            turn_index += 1
            group_id = f"pra-group-{turn_index:04d}"
            semantic_role = str(payload.get("pra_agent_semantic_role") or "")
            if semantic_role == "finalization":
                logical.append(AgentRecord(
                    record.record_id,
                    turn_id,
                    group_id,
                    index,
                    role,
                    content,
                    AgentRecordRole.FINALIZATION,
                    (AgentRecordRole.FINALIZATION, AgentRecordRole.PROGRESS),
                    session_id=record.session_uuid,
                    complete=True,
                ))
                turns.append(AgentTurn(
                    turn_id=turn_id,
                    causal_group_id=group_id,
                    record_ids=(record.record_id,),
                    first_message_index=index,
                    complete=True,
                ))
                continue
            pending = {
                "turn_id": turn_id,
                "group_id": group_id,
                "record_ids": [record.record_id],
                "first_index": index,
                "has_assistant": True,
            }
            logical.append(AgentRecord(
                record.record_id,
                turn_id,
                group_id,
                index,
                role,
                content,
                AgentRecordRole.ASSISTANT_ACTION,
                (AgentRecordRole.ASSISTANT_ACTION, AgentRecordRole.PROGRESS),
                session_id=record.session_uuid,
            ))
            continue

        if record.record_type == RecordType.TOOL_RESPONSE:
            if pending is None:
                turn_id = f"pra-orphan-{turn_index:04d}"
                turn_index += 1
                group_id = f"pra-orphan-group-{turn_index:04d}"
                pending = {
                    "turn_id": turn_id,
                    "group_id": group_id,
                    "record_ids": [],
                    "first_index": index,
                    "has_assistant": False,
                }
            pending["record_ids"].append(record.record_id)
            roles = _tool_roles(record)
            logical.append(AgentRecord(
                record.record_id,
                str(pending["turn_id"]),
                str(pending["group_id"]),
                index,
                "tool",
                content,
                AgentRecordRole.TOOL_OBSERVATION,
                roles,
                session_id=record.session_uuid,
                complete=True,
            ))
            finish(complete=bool(pending["has_assistant"]))
            continue

        if record.record_type == RecordType.GENERIC_TEXT and role == "user":
            product_recovery = content.startswith((
                "[Tool decision rejected:",
                "[Completion rejected:",
            ))
            if pending is not None and product_recovery:
                pending["record_ids"].append(record.record_id)
                logical.append(AgentRecord(
                    record.record_id,
                    str(pending["turn_id"]),
                    str(pending["group_id"]),
                    index,
                    role,
                    content,
                    AgentRecordRole.ERROR_OR_REJECTION,
                    (
                        AgentRecordRole.TOOL_OBSERVATION,
                        AgentRecordRole.ERROR_OR_REJECTION,
                    ),
                    session_id=record.session_uuid,
                    complete=True,
                ))
                finish(complete=True)
                continue
            finish(complete=False)
            primary = AgentRecordRole.TASK if not task_seen else AgentRecordRole.USER_INPUT
            task_seen = True
            logical.append(AgentRecord(
                record.record_id,
                "task" if primary == AgentRecordRole.TASK else f"user-{index}",
                "task" if primary == AgentRecordRole.TASK else f"user-{index}",
                index,
                role,
                content,
                primary,
                (primary,),
                session_id=record.session_uuid,
            ))
            continue

        # Preserve unfamiliar typed records as immutable input.  This fails
        # toward retention until their producer supplies portable semantics.
        finish(complete=False)
        logical.append(AgentRecord(
            record.record_id,
            f"input-{index}",
            f"input-{index}",
            index,
            role or record.record_type.value,
            content,
            AgentRecordRole.USER_INPUT,
            (AgentRecordRole.USER_INPUT,),
            session_id=record.session_uuid,
        ))

    finish(complete=False)
    return CanonicalAgentHistory(tuple(logical), tuple(turns))


@dataclass
class PRAAgentMatchedTailSelector:
    """Callable H2/T4 selector with an auditable per-request identity ledger."""

    retention_fraction: float
    count_tokens: TokenCounter
    tokenizer_identity: str
    protected_head_turns: int = 2
    protected_tail_turns: int = 4
    traces: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.retention_fraction <= 1:
            raise ValueError("retention_fraction must be in (0, 1]")

    def __call__(
        self, records: Sequence[ContextRecord], query: str,
    ) -> Sequence[ContextRecord]:
        del query  # The matched-recency control is deliberately query independent.
        history = recordize_pra_agent_records(records)
        full_tokens = sum(self.count_tokens(row.content) for row in history.records)
        requested = max(1, math.ceil(full_tokens * self.retention_fraction))
        config = MatchedTokenTailConfig(
            protected_head_turns=self.protected_head_turns,
            protected_tail_turns=self.protected_tail_turns,
            boundary_compaction="declared_safe",
        )
        mandatory_floor_ids = matched_token_tail_full_floor_record_ids(
            history, config
        )
        mandatory_floor_tokens = sum(
            self.count_tokens(row.content)
            for row in history.records
            if row.record_id in mandatory_floor_ids
        )
        # A percentage is a target ceiling, never permission to discard task
        # instructions or configured causal floors.  Early short histories
        # therefore realize 100% retention and report the nominal overflow.
        effective_budget = max(requested, mandatory_floor_tokens)
        plan = materialize_matched_token_tail(
            history,
            max_materialized_tokens=effective_budget,
            config=config,
            count_tokens=self.count_tokens,
        )
        by_id = {row.record_id: row for row in history.records}
        for materialized in plan.records:
            if materialized.content != by_id[materialized.record_id].content:
                raise ValueError(
                    "PRA Agent identity selector cannot rewrite an existing record; "
                    "materialize compact content before K/V creation instead."
                )
        selected_ids = set(plan.logical_plan.selected_record_ids)
        selected = tuple(row for row in records if row.record_id in selected_ids)
        self.traces.append({
            "request_index": len(self.traces) + 1,
            "policy": "matched_token_tail_h2_t4",
            "retention_fraction": self.retention_fraction,
            "tokenizer": self.tokenizer_identity,
            "full_history_tokens": plan.logical_plan.full_history_tokens,
            "selected_history_tokens": plan.logical_plan.selected_tokens,
            "materialized_history_tokens": plan.materialized_tokens,
            "requested_budget_tokens": requested,
            "effective_budget_tokens": effective_budget,
            "mandatory_tokens": mandatory_floor_tokens,
            "mandatory_overflow_tokens": max(0, mandatory_floor_tokens - requested),
            "full_record_ids": [row.record_id for row in records],
            "selected_record_ids": [row.record_id for row in selected],
            "excluded_record_ids": [
                row.record_id for row in records if row.record_id not in selected_ids
            ],
        })
        return selected


@dataclass
class PRAAgentInstructionEpochSelector:
    """Identity-only E1/E2/E3 policy for persistent PRA Agent sessions."""

    count_tokens: TokenCounter
    tokenizer_identity: str
    prior_full_epochs: int = 2
    prior_recent_turns: int = 0
    prior_mutation_turns: int = 0
    prior_verification_turns: int = 0
    prior_protocol_turns: int = 0
    prior_finalization_turns: int = 0
    traces: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._selector = PersistentInstructionEpochRetirementSelector(
            PersistentInstructionEpochRetirementConfig(
                prior_recent_turns=self.prior_recent_turns,
                prior_mutation_turns=self.prior_mutation_turns,
                prior_verification_turns=self.prior_verification_turns,
                prior_protocol_turns=self.prior_protocol_turns,
                prior_finalization_turns=self.prior_finalization_turns,
                prior_full_epochs=self.prior_full_epochs,
                retire_closed_instructions=True,
                keep_completed_task_statements=True,
            )
        )

    def __call__(
        self, records: Sequence[ContextRecord], query: str,
    ) -> Sequence[ContextRecord]:
        history = recordize_pra_agent_records(records)
        full_tokens = sum(self.count_tokens(row.content) for row in history.records)
        plan = self._selector.select(
            history=history,
            query=query,
            budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
            count_tokens=self.count_tokens,
        )
        selected_ids = set(plan.selected_record_ids)
        selected = tuple(row for row in records if row.record_id in selected_ids)
        self.traces.append({
            "request_index": len(self.traces) + 1,
            "policy": f"instruction_epoch_e{self.prior_full_epochs}",
            "tokenizer": self.tokenizer_identity,
            "full_history_tokens": plan.full_history_tokens,
            "selected_history_tokens": plan.selected_tokens,
            "materialized_history_tokens": plan.selected_tokens,
            "requested_budget_tokens": plan.requested_budget_tokens,
            "mandatory_tokens": plan.mandatory_tokens,
            "mandatory_overflow_tokens": plan.mandatory_overflow_tokens,
            "full_record_ids": [row.record_id for row in records],
            "selected_record_ids": [row.record_id for row in selected],
            "excluded_record_ids": [
                row.record_id for row in records
                if row.record_id not in selected_ids
            ],
            "exclusions": [
                {
                    "causal_group_id": row.causal_group_id,
                    "record_ids": list(row.record_ids),
                    "rule_id": row.rule_id,
                    "excluded_tokens": row.excluded_tokens,
                }
                for row in plan.exclusions
            ],
        })
        return selected


@dataclass
class PRAAgentFrontierSelector:
    """Identity-only recent-frontier policy for typed PRA Agent records."""

    count_tokens: TokenCounter
    tokenizer_identity: str
    recent_user_prompts: int = 2
    protocol_exemplars: int = 1
    workflow_exemplars: int = 0
    allow_heuristic: bool = True
    traces: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._selector = FrontierDagRetirementSelector(
            recent_user_prompts=self.recent_user_prompts,
            allow_heuristic=self.allow_heuristic,
            valid_protocol_exemplars=self.protocol_exemplars,
            valid_workflow_exemplars=self.workflow_exemplars,
        )

    def __call__(
        self, records: Sequence[ContextRecord], query: str,
    ) -> Sequence[ContextRecord]:
        history = recordize_pra_agent_records(records)
        full_tokens = sum(self.count_tokens(row.content) for row in history.records)
        plan = self._selector.select(
            history=history,
            query=query,
            budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
            count_tokens=self.count_tokens,
        )
        selected_ids = set(plan.selected_record_ids)
        selected = tuple(row for row in records if row.record_id in selected_ids)
        self.traces.append({
            "request_index": len(self.traces) + 1,
            "policy": plan.policy,
            "tokenizer": self.tokenizer_identity,
            "full_history_tokens": plan.full_history_tokens,
            "selected_history_tokens": plan.selected_tokens,
            "materialized_history_tokens": plan.selected_tokens,
            "requested_budget_tokens": plan.requested_budget_tokens,
            "mandatory_tokens": plan.mandatory_tokens,
            "mandatory_overflow_tokens": plan.mandatory_overflow_tokens,
            "full_record_ids": [row.record_id for row in records],
            "selected_record_ids": [row.record_id for row in selected],
            "excluded_record_ids": [
                row.record_id for row in records
                if row.record_id not in selected_ids
            ],
            "exclusions": [
                {
                    "causal_group_id": row.causal_group_id,
                    "record_ids": list(row.record_ids),
                    "rule_id": row.rule_id,
                    "classification": row.classification,
                    "excluded_tokens": row.excluded_tokens,
                }
                for row in plan.exclusions
            ],
        })
        return selected
