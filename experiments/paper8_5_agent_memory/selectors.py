"""Engine-neutral head/middle/tail agent-memory selectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Callable, Mapping, Protocol

from .model import (
    AgentMemoryBudget,
    AgentMemoryPlan,
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
)


TokenCounter = Callable[[str], int]
_TERM = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")


def whitespace_tokens(text: str) -> int:
    return len(text.split())


class MiddleSelectionStrategy(str, Enum):
    NONE = "none"
    RECENCY = "recency"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    ORACLE = "oracle"


class AgentMemorySelector(Protocol):
    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: AgentMemoryBudget,
        count_tokens: TokenCounter = whitespace_tokens,
    ) -> AgentMemoryPlan: ...


def _record_costs(
    history: CanonicalAgentHistory,
    count_tokens: TokenCounter,
) -> dict[str, int]:
    return {record.record_id: count_tokens(record.content) for record in history.records}


def _plan(
    *,
    policy: str,
    history: CanonicalAgentHistory,
    selected_ids: set[str],
    reasons: Mapping[str, str],
    costs: Mapping[str, int],
    budget: AgentMemoryBudget,
    mandatory_ids: set[str],
    head_turns: int,
    tail_turns: int,
    middle_candidate_turns: int,
    middle_selected_turns: int,
) -> AgentMemoryPlan:
    ordered = tuple(r.record_id for r in history.records if r.record_id in selected_ids)
    groups = tuple(dict.fromkeys(
        r.causal_group_id for r in history.records if r.record_id in selected_ids
    ))
    selected_tokens = sum(costs[rid] for rid in ordered)
    mandatory_tokens = sum(costs[rid] for rid in mandatory_ids)
    return AgentMemoryPlan(
        policy=policy,
        selected_record_ids=ordered,
        selected_causal_group_ids=groups,
        selection_reasons=tuple((rid, reasons[rid]) for rid in ordered),
        full_history_tokens=sum(costs.values()),
        selected_tokens=selected_tokens,
        requested_budget_tokens=budget.max_tokens,
        mandatory_tokens=mandatory_tokens,
        mandatory_overflow_tokens=max(0, mandatory_tokens - budget.max_tokens),
        head_turns=head_turns,
        tail_turns=tail_turns,
        middle_candidate_turns=middle_candidate_turns,
        middle_selected_turns=middle_selected_turns,
    )


class FullHistorySelector:
    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: AgentMemoryBudget,
        count_tokens: TokenCounter = whitespace_tokens,
    ) -> AgentMemoryPlan:
        del query
        costs = _record_costs(history, count_tokens)
        selected = {record.record_id for record in history.records}
        reasons = {record_id: "full_history" for record_id in selected}
        return _plan(
            policy="full",
            history=history,
            selected_ids=selected,
            reasons=reasons,
            costs=costs,
            budget=budget,
            mandatory_ids=selected,
            head_turns=len(history.turns),
            tail_turns=0,
            middle_candidate_turns=0,
            middle_selected_turns=0,
        )


@dataclass(frozen=True)
class HeadMiddleTailConfig:
    """Independent head/tail floors with selection restricted to the middle."""

    head_turns: int = 1
    tail_turns: int = 2
    middle_strategy: MiddleSelectionStrategy = MiddleSelectionStrategy.RECENCY
    source_turns: int = 0
    mutation_turns: int = 0
    verification_turns: int = 0
    progress_turns: int = 0
    error_turns: int = 0
    oracle_utilities: Mapping[str, float] | None = None
    recency_weight: float = 0.15
    round_up_to_budget: bool = False
    """Treat ``max_tokens`` as a retention floor at whole-turn boundaries.

    This mode is used for nominal retention treatments such as PRA-90.  The
    ordinary mode remains a hard ceiling for matched-token controls.
    """

    def __post_init__(self) -> None:
        values = (
            self.head_turns,
            self.tail_turns,
            self.source_turns,
            self.mutation_turns,
            self.verification_turns,
            self.progress_turns,
            self.error_turns,
        )
        if any(value < 0 for value in values):
            raise ValueError("turn floors cannot be negative")
        if self.middle_strategy == MiddleSelectionStrategy.ORACLE and self.oracle_utilities is None:
            raise ValueError("oracle strategy requires frozen oracle_utilities")


class HeadMiddleTailSelector:
    def __init__(self, config: HeadMiddleTailConfig):
        self.config = config

    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: AgentMemoryBudget,
        count_tokens: TokenCounter = whitespace_tokens,
    ) -> AgentMemoryPlan:
        costs = _record_costs(history, count_tokens)
        records = history.record_by_id
        complete = [turn for turn in history.turns if turn.complete]
        head = complete[: self.config.head_turns]
        head_ids = {turn.turn_id for turn in head}
        tail = [
            turn for turn in complete[-self.config.tail_turns :]
            if turn.turn_id not in head_ids
        ] if self.config.tail_turns else []
        mandatory_turn_ids = {turn.turn_id for turn in (*head, *tail)}
        middle = [turn for turn in complete if turn.turn_id not in mandatory_turn_ids]

        mandatory_ids = {
            record.record_id
            for record in history.records
            if record.has_role(AgentRecordRole.SYSTEM)
            or record.has_role(AgentRecordRole.TASK)
        }
        reasons = {record_id: "immutable_prompt" for record_id in mandatory_ids}
        for turn in head:
            mandatory_ids.update(turn.record_ids)
            reasons.update({record_id: "head_turn" for record_id in turn.record_ids})
        for turn in tail:
            mandatory_ids.update(turn.record_ids)
            reasons.update({record_id: "tail_turn" for record_id in turn.record_ids})

        selected_turn_ids: set[str] = set()
        role_floors = (
            (AgentRecordRole.SOURCE_VIEW, self.config.source_turns),
            (AgentRecordRole.MUTATION, self.config.mutation_turns),
            (AgentRecordRole.VERIFICATION, self.config.verification_turns),
            (AgentRecordRole.PROGRESS, self.config.progress_turns),
            (AgentRecordRole.ERROR_OR_REJECTION, self.config.error_turns),
        )
        for role, count in role_floors:
            eligible = [
                turn for turn in middle
                if any(records[rid].has_role(role) for rid in turn.record_ids)
            ]
            for turn in eligible[-count:] if count else ():
                selected_turn_ids.add(turn.turn_id)
                mandatory_ids.update(turn.record_ids)
                reasons.update({rid: f"role_floor:{role.value}" for rid in turn.record_ids})

        selected_ids = set(mandatory_ids)
        for turn in middle:
            if turn.turn_id in selected_turn_ids:
                selected_ids.update(turn.record_ids)

        ranked = self._rank_middle(middle, records, query)
        for turn in ranked:
            if turn.turn_id in selected_turn_ids:
                continue
            added = [rid for rid in turn.record_ids if rid not in selected_ids]
            projected_tokens = sum(costs[rid] for rid in selected_ids) + sum(
                costs[rid] for rid in added
            )
            projected_records = len(selected_ids) + len(added)
            if budget.max_records is not None and projected_records > budget.max_records:
                continue
            if self.config.round_up_to_budget:
                if sum(costs[rid] for rid in selected_ids) >= budget.max_tokens:
                    break
            elif projected_tokens > budget.max_tokens:
                continue
            selected_turn_ids.add(turn.turn_id)
            selected_ids.update(added)
            for rid in added:
                reasons[rid] = f"middle:{self.config.middle_strategy.value}"

        return _plan(
            policy=(
                f"head_middle_tail:{self.config.middle_strategy.value}"
                + (":round_up" if self.config.round_up_to_budget else "")
            ),
            history=history,
            selected_ids=selected_ids,
            reasons=reasons,
            costs=costs,
            budget=budget,
            mandatory_ids=mandatory_ids,
            head_turns=len(head),
            tail_turns=len(tail),
            middle_candidate_turns=len(middle),
            middle_selected_turns=len(selected_turn_ids),
        )

    def _rank_middle(
        self,
        turns: list[AgentTurn],
        records: Mapping[str, AgentRecord],
        query: str,
    ) -> list[AgentTurn]:
        strategy = self.config.middle_strategy
        if strategy == MiddleSelectionStrategy.NONE:
            return []
        if strategy == MiddleSelectionStrategy.RECENCY:
            return list(reversed(turns))
        if strategy == MiddleSelectionStrategy.ORACLE:
            utilities = self.config.oracle_utilities or {}
            return sorted(
                turns,
                key=lambda turn: (utilities.get(turn.causal_group_id, 0.0), turn.first_message_index),
                reverse=True,
            )

        lexical = _lexical_scores(turns, records, query)
        if strategy == MiddleSelectionStrategy.LEXICAL:
            return sorted(
                turns,
                key=lambda turn: (lexical[turn.turn_id], turn.first_message_index),
                reverse=True,
            )
        if strategy == MiddleSelectionStrategy.HYBRID:
            denominator = max(1, len(turns) - 1)
            return sorted(
                turns,
                key=lambda turn: (
                    lexical[turn.turn_id]
                    + self.config.recency_weight * turns.index(turn) / denominator,
                    turn.first_message_index,
                ),
                reverse=True,
            )
        raise AssertionError(f"unsupported middle strategy: {strategy}")


def _terms(text: str) -> list[str]:
    return [term.lower() for term in _TERM.findall(text)]


def _lexical_scores(
    turns: list[AgentTurn],
    records: Mapping[str, AgentRecord],
    query: str,
) -> dict[str, float]:
    query_terms = Counter(_terms(query))
    documents = {
        turn.turn_id: Counter(_terms("\n".join(records[rid].content for rid in turn.record_ids)))
        for turn in turns
    }
    document_frequency = Counter(
        term for document in documents.values() for term in document
    )
    count = max(1, len(documents))
    scores: dict[str, float] = {}
    for turn_id, document in documents.items():
        score = 0.0
        for term, query_frequency in query_terms.items():
            if term not in document:
                continue
            inverse_frequency = math.log1p(count / (1 + document_frequency[term]))
            score += min(document[term], query_frequency) * inverse_frequency
        scores[turn_id] = score
    return scores
