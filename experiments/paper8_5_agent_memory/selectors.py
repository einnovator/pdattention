"""Engine-neutral head/middle/tail agent-memory selectors."""

from __future__ import annotations

from collections import Counter
from bisect import bisect_right
from dataclasses import dataclass, replace
from enum import Enum
import math
import re
from typing import Callable, Mapping, Protocol

from .model import (
    AgentMemoryBudget,
    AgentMemoryExclusion,
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


def immutable_instruction_record_ids(
    history: CanonicalAgentHistory,
) -> frozenset[str]:
    """Return the semantic prompt floor shared by every memory policy.

    ``USER_INPUT`` is provenance-sensitive: later user-authored instructions
    are immutable, whereas tool observations transported with ``role=user``
    remain typed ``TOOL_OBSERVATION`` and are eligible for retirement.
    """

    return frozenset(
        record.record_id
        for record in history.records
        if record.has_role(AgentRecordRole.SYSTEM)
        or record.has_role(AgentRecordRole.TASK)
        or record.has_role(AgentRecordRole.USER_INPUT)
    )


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
class PersistentEpisodeRetirementConfig:
    """Retention floors for completed issues in one developer session."""

    recent_turns: int = 1
    mutation_turns: int = 1
    verification_turns: int = 1
    protocol_turns: int = 0
    # Kept for schema compatibility with early active-only pilots.  User
    # instructions are now an unconditional semantic floor, so False can only
    # select the legacy policy label; it never removes a task statement.
    keep_completed_task_statements: bool = True

    def __post_init__(self) -> None:
        if min(
            self.recent_turns,
            self.mutation_turns,
            self.verification_turns,
            self.protocol_turns,
        ) < 0:
            raise ValueError("completed-episode turn floors cannot be negative")


class PersistentEpisodeRetirementSelector:
    """Keep the active issue whole and retire detail from completed issues.

    This is an issue-boundary treatment, not a token-budget treatment.  It is
    intentionally conservative: every issue statement remains verbatim, while
    each completed issue contributes its newest turns and its latest mutation
    and verification evidence.  Selection remains causal-group atomic.
    """

    def __init__(self, config: PersistentEpisodeRetirementConfig | None = None):
        self.config = config or PersistentEpisodeRetirementConfig()

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
        records = history.record_by_id
        episode_indices = [
            int(record.metadata.get("episode_index", 1)) for record in history.records
        ]
        active_episode = max(episode_indices, default=1)
        mandatory_ids = set(immutable_instruction_record_ids(history))
        mandatory_ids.update(
            record.record_id
            for record in history.records
            if int(record.metadata.get("episode_index", 1)) == active_episode
        )
        reasons = {
            record_id: (
                "active_episode"
                if int(records[record_id].metadata.get("episode_index", 1))
                == active_episode
                and not records[record_id].has_role(AgentRecordRole.SYSTEM)
                and record_id not in immutable_instruction_record_ids(history)
                else "immutable_prompt"
            )
            for record_id in mandatory_ids
        }

        complete_by_episode: dict[int, list[AgentTurn]] = {}
        for turn in history.turns:
            if not turn.complete or not turn.record_ids:
                continue
            episode = int(records[turn.record_ids[0]].metadata.get("episode_index", 1))
            if episode < active_episode:
                complete_by_episode.setdefault(episode, []).append(turn)

        selected_turn_ids: set[str] = set()
        role_floors = (
            (AgentRecordRole.MUTATION, self.config.mutation_turns),
            (AgentRecordRole.VERIFICATION, self.config.verification_turns),
        )
        for turns in complete_by_episode.values():
            if self.config.recent_turns:
                selected_turn_ids.update(
                    turn.turn_id for turn in turns[-self.config.recent_turns :]
                )
            for role, count in role_floors:
                if not count:
                    continue
                eligible = [
                    turn for turn in turns
                    if any(records[rid].has_role(role) for rid in turn.record_ids)
                ]
                selected_turn_ids.update(turn.turn_id for turn in eligible[-count:])
            if self.config.protocol_turns:
                protocol_eligible = []
                disallowed = {
                    AgentRecordRole.ERROR_OR_REJECTION,
                    AgentRecordRole.MUTATION,
                    AgentRecordRole.VERIFICATION,
                    AgentRecordRole.FINALIZATION,
                }
                for turn in turns:
                    roles = {
                        role
                        for record_id in turn.record_ids
                        for role in records[record_id].semantic_roles
                    }
                    if (
                        AgentRecordRole.ASSISTANT_ACTION in roles
                        and AgentRecordRole.TOOL_OBSERVATION in roles
                        and not roles.intersection(disallowed)
                    ):
                        protocol_eligible.append(turn)
                selected_turn_ids.update(
                    turn.turn_id
                    for turn in protocol_eligible[-self.config.protocol_turns :]
                )

        for turn in history.turns:
            if turn.turn_id not in selected_turn_ids:
                continue
            for record_id in turn.record_ids:
                mandatory_ids.add(record_id)
                reasons[record_id] = "completed_episode_progress_spine"

        completed_turns = sum(len(turns) for turns in complete_by_episode.values())
        plan = _plan(
            policy=(
                "persistent_episode_retirement"
                if self.config.keep_completed_task_statements
                or self.config.recent_turns
                or self.config.mutation_turns
                or self.config.verification_turns
                or self.config.protocol_turns
                else "persistent_active_episode"
            ),
            history=history,
            selected_ids=set(mandatory_ids),
            reasons=reasons,
            costs=costs,
            budget=budget,
            mandatory_ids=set(mandatory_ids),
            head_turns=0,
            tail_turns=sum(
                1 for turn in history.turns if turn.turn_id in selected_turn_ids
            ),
            middle_candidate_turns=completed_turns,
            middle_selected_turns=len(selected_turn_ids),
        )
        retired = []
        for turn in history.turns:
            omitted = tuple(
                record_id for record_id in turn.record_ids
                if record_id not in mandatory_ids
            )
            if not omitted:
                continue
            resources = tuple(sorted({
                str(resource)
                for record_id in omitted
                for resource in (
                    records[record_id].metadata.get("resource_ids") or ()
                )
            }))
            retired.append(AgentMemoryExclusion(
                causal_group_id=records[omitted[0]].causal_group_id,
                record_ids=omitted,
                rule_id="completed_episode_retirement",
                classification="policy_retirement",
                reason="completed issue detail outside declared progress-state floors",
                resource_ids=resources,
                witness_record_ids=(),
                tombstone="retired completed-episode causal group",
                excluded_tokens=sum(costs[record_id] for record_id in omitted),
            ))
        return replace(plan, exclusions=tuple(retired))


@dataclass(frozen=True)
class PersistentGlobalRetirementConfig:
    """Retention floors over one continuous, boundary-free session."""

    recent_turns: int = 4
    mutation_turns: int = 2
    verification_turns: int = 2
    protocol_turns: int = 1

    def __post_init__(self) -> None:
        if min(
            self.recent_turns,
            self.mutation_turns,
            self.verification_turns,
            self.protocol_turns,
        ) < 0:
            raise ValueError("global turn floors cannot be negative")


class PersistentGlobalRetirementSelector:
    """Apply progress-state floors without task or workspace boundaries.

    The selector receives a single monotonically ordered transcript.  It keeps
    the system record, every user-authored instruction, incomplete causal
    turns, and global recent/mutation/verification/protocol floors.  Tool
    observations may use the transport role ``user`` in a compatibility
    harness, but remain typed tool records and are selectable.  The selector
    never reads episode IDs, task status, workspace scope, or a current-task
    label.
    """

    def __init__(self, config: PersistentGlobalRetirementConfig | None = None):
        self.config = config or PersistentGlobalRetirementConfig()

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
        records = history.record_by_id
        selected_ids = set(immutable_instruction_record_ids(history))
        reasons = {record_id: "immutable_system" for record_id in selected_ids}

        # User-authored task statements and follow-up instructions are an
        # immutable semantic floor.  This is based on record provenance, not
        # task-boundary knowledge.  Tool observations are intentionally not
        # included even when mini-swe-agent transports them with role=user.
        user_instructions = [
            record for record in history.records
            if record.has_role(AgentRecordRole.TASK)
            or record.has_role(AgentRecordRole.USER_INPUT)
        ]
        for instruction in user_instructions:
            selected_ids.add(instruction.record_id)
            reasons[instruction.record_id] = "immutable_user_instruction"

        complete_turns = [turn for turn in history.turns if turn.complete]
        selected_turn_ids: set[str] = set()
        if self.config.recent_turns:
            selected_turn_ids.update(
                turn.turn_id for turn in complete_turns[-self.config.recent_turns :]
            )

        for role, count in (
            (AgentRecordRole.MUTATION, self.config.mutation_turns),
            (AgentRecordRole.VERIFICATION, self.config.verification_turns),
        ):
            if not count:
                continue
            eligible = [
                turn for turn in complete_turns
                if any(records[record_id].has_role(role) for record_id in turn.record_ids)
            ]
            selected_turn_ids.update(turn.turn_id for turn in eligible[-count:])

        if self.config.protocol_turns:
            disallowed = {
                AgentRecordRole.ERROR_OR_REJECTION,
                AgentRecordRole.MUTATION,
                AgentRecordRole.VERIFICATION,
                AgentRecordRole.FINALIZATION,
            }
            eligible = []
            for turn in complete_turns:
                roles = {
                    role
                    for record_id in turn.record_ids
                    for role in records[record_id].semantic_roles
                }
                if (
                    AgentRecordRole.ASSISTANT_ACTION in roles
                    and AgentRecordRole.TOOL_OBSERVATION in roles
                    and not roles.intersection(disallowed)
                ):
                    eligible.append(turn)
            selected_turn_ids.update(
                turn.turn_id for turn in eligible[-self.config.protocol_turns :]
            )

        for turn in history.turns:
            if not turn.complete:
                for record_id in turn.record_ids:
                    selected_ids.add(record_id)
                    reasons[record_id] = "incomplete_causal_turn"
            elif turn.turn_id in selected_turn_ids:
                for record_id in turn.record_ids:
                    selected_ids.add(record_id)
                    reasons[record_id] = "global_progress_floor"

        plan = _plan(
            policy="persistent_global_retirement",
            history=history,
            selected_ids=set(selected_ids),
            reasons=reasons,
            costs=costs,
            budget=budget,
            mandatory_ids=set(selected_ids),
            head_turns=0,
            tail_turns=sum(
                1 for turn in history.turns if turn.turn_id in selected_turn_ids
            ),
            middle_candidate_turns=len(complete_turns),
            middle_selected_turns=len(selected_turn_ids),
        )

        omitted_by_group: dict[str, list[str]] = {}
        for record in history.records:
            if record.record_id not in selected_ids:
                omitted_by_group.setdefault(record.causal_group_id, []).append(
                    record.record_id
                )
        exclusions = []
        for group_id, omitted in omitted_by_group.items():
            resources = tuple(sorted({
                str(resource)
                for record_id in omitted
                for resource in (records[record_id].metadata.get("resource_ids") or ())
            }))
            exclusions.append(AgentMemoryExclusion(
                causal_group_id=group_id,
                record_ids=tuple(omitted),
                rule_id="global_progress_retirement",
                classification="policy_retirement",
                reason="continuous-session detail outside global progress-state floors",
                resource_ids=resources,
                witness_record_ids=(),
                tombstone="retired boundary-free causal group",
                excluded_tokens=sum(costs[record_id] for record_id in omitted),
            ))
        return replace(plan, exclusions=tuple(exclusions))


@dataclass(frozen=True)
class PersistentInstructionEpochRetirementConfig:
    """Floors for interaction detail preceding the latest user instruction.

    An instruction epoch is an observable transcript interval, not a task
    label.  By default every system/user instruction remains immutable and the
    complete active epoch remains visible.  ``retire_closed_instructions`` is
    a stricter independent-task treatment: an older instruction may be retired
    only when its own epoch contains complete terminal/finalization evidence.
    Optional floors retain a progress spine independently inside each older
    epoch.
    """

    prior_recent_turns: int = 0
    prior_mutation_turns: int = 0
    prior_verification_turns: int = 0
    prior_protocol_turns: int = 0
    prior_finalization_turns: int = 0
    prior_full_epochs: int = 0
    retire_closed_instructions: bool = False

    def __post_init__(self) -> None:
        if min(
            self.prior_recent_turns,
            self.prior_mutation_turns,
            self.prior_verification_turns,
            self.prior_protocol_turns,
            self.prior_finalization_turns,
            self.prior_full_epochs,
        ) < 0:
            raise ValueError("prior instruction-epoch floors cannot be negative")


class PersistentInstructionEpochRetirementSelector:
    """Retire only interaction detail from earlier instruction epochs.

    The policy consumes no evaluator task boundary, episode identifier,
    workspace status, or current-task label.  It recognizes only provenance-
    typed genuine user instructions.  Tool observations transported as
    ``role=user`` remain tool observations and cannot start an epoch.

    This policy is appropriate for the independent-issue stratum.  Related or
    dependent work requires a resource/dependency join before older epochs can
    be retired; absent that evidence a deployment must fail closed.
    """

    def __init__(
        self,
        config: PersistentInstructionEpochRetirementConfig | None = None,
    ):
        self.config = config or PersistentInstructionEpochRetirementConfig()

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
        records = history.record_by_id
        record_positions = {
            record.record_id: index for index, record in enumerate(history.records)
        }
        instruction_positions = [
            index
            for index, record in enumerate(history.records)
            if record.has_role(AgentRecordRole.TASK)
            or record.has_role(AgentRecordRole.USER_INPUT)
        ]
        if not instruction_positions:
            raise ValueError("instruction-epoch policy requires a genuine user instruction")
        active_epoch = len(instruction_positions) - 1
        turn_epochs: dict[str, int] = {}
        closed_epochs: set[int] = set()
        for turn in history.turns:
            if not turn.record_ids:
                continue
            turn_position = max(record_positions[record_id] for record_id in turn.record_ids)
            epoch = bisect_right(instruction_positions, turn_position) - 1
            turn_epochs[turn.turn_id] = epoch
            if (
                epoch >= 0
                and epoch < active_epoch
                and turn.complete
                and any(
                    records[record_id].has_role(AgentRecordRole.FINALIZATION)
                    for record_id in turn.record_ids
                )
            ):
                closed_epochs.add(epoch)

        selected_ids = set(immutable_instruction_record_ids(history))
        if self.config.retire_closed_instructions:
            # Epochs deliberately retained whole remain genuine in-context
            # exemplars, so their instruction must remain paired with their
            # interaction.  Atomic retirement applies only to older closed
            # epochs outside that explicit full-epoch floor.
            oldest_full_epoch = active_epoch - self.config.prior_full_epochs
            selected_ids.difference_update(
                history.records[position].record_id
                for epoch, position in enumerate(instruction_positions)
                if epoch in closed_epochs and epoch < oldest_full_epoch
            )
        reasons = {
            record_id: "immutable_user_instruction"
            for record_id in selected_ids
        }

        complete_by_epoch: dict[int, list[AgentTurn]] = {}
        active_complete_turns = 0
        retained_prior_complete_turns = 0
        for turn in history.turns:
            if not turn.record_ids:
                continue
            epoch = turn_epochs[turn.turn_id]
            if epoch == active_epoch:
                for record_id in turn.record_ids:
                    selected_ids.add(record_id)
                    reasons[record_id] = (
                        "active_instruction_epoch"
                        if turn.complete else "incomplete_causal_turn"
                    )
                if turn.complete:
                    active_complete_turns += 1
            elif (
                epoch >= 0
                and epoch >= active_epoch - self.config.prior_full_epochs
            ):
                for record_id in turn.record_ids:
                    selected_ids.add(record_id)
                    reasons[record_id] = "recent_complete_instruction_epoch"
                if turn.complete:
                    retained_prior_complete_turns += 1
            elif epoch >= 0 and turn.complete:
                complete_by_epoch.setdefault(epoch, []).append(turn)
            elif not turn.complete:
                # A causal group is never split even in malformed or partial
                # replay input.  Failing closed is safer than retiring it.
                for record_id in turn.record_ids:
                    selected_ids.add(record_id)
                    reasons[record_id] = "incomplete_causal_turn"

        selected_prior_turn_ids: set[str] = set()
        for turns in complete_by_epoch.values():
            if self.config.prior_recent_turns:
                selected_prior_turn_ids.update(
                    turn.turn_id
                    for turn in turns[-self.config.prior_recent_turns :]
                )
            for role, count in (
                (AgentRecordRole.MUTATION, self.config.prior_mutation_turns),
                (AgentRecordRole.VERIFICATION, self.config.prior_verification_turns),
                (AgentRecordRole.FINALIZATION, self.config.prior_finalization_turns),
            ):
                if not count:
                    continue
                eligible = [
                    turn for turn in turns
                    if any(records[record_id].has_role(role) for record_id in turn.record_ids)
                ]
                selected_prior_turn_ids.update(
                    turn.turn_id for turn in eligible[-count:]
                )
            if self.config.prior_protocol_turns:
                disallowed = {
                    AgentRecordRole.ERROR_OR_REJECTION,
                    AgentRecordRole.MUTATION,
                    AgentRecordRole.VERIFICATION,
                    AgentRecordRole.FINALIZATION,
                }
                eligible = []
                for turn in turns:
                    roles = {
                        role
                        for record_id in turn.record_ids
                        for role in records[record_id].semantic_roles
                    }
                    if (
                        AgentRecordRole.ASSISTANT_ACTION in roles
                        and AgentRecordRole.TOOL_OBSERVATION in roles
                        and not roles.intersection(disallowed)
                    ):
                        eligible.append(turn)
                selected_prior_turn_ids.update(
                    turn.turn_id
                    for turn in eligible[-self.config.prior_protocol_turns :]
                )

        # A retained spine from an atomically retired epoch needs its natural
        # chat envelope. Otherwise the selected transcript can begin with an
        # assistant action after the system prompt, and the old task appears
        # either missing or unresolved. Restore the epoch's genuine user
        # instruction plus its natural terminal turn; this is transcript
        # control state, not an evaluator-provided boundary.
        spine_epochs = {
            epoch
            for epoch, turns in complete_by_epoch.items()
            if any(turn.turn_id in selected_prior_turn_ids for turn in turns)
        }
        if self.config.retire_closed_instructions:
            for epoch in sorted(spine_epochs.intersection(closed_epochs)):
                instruction_id = history.records[
                    instruction_positions[epoch]
                ].record_id
                selected_ids.add(instruction_id)
                reasons[instruction_id] = "prior_instruction_epoch_envelope"
                terminal_turns = [
                    turn for turn in complete_by_epoch.get(epoch, ())
                    if any(
                        records[record_id].has_role(AgentRecordRole.FINALIZATION)
                        for record_id in turn.record_ids
                    )
                ]
                if not terminal_turns:
                    raise AssertionError("closed instruction epoch lacks terminal turn")
                selected_prior_turn_ids.add(terminal_turns[-1].turn_id)

        for turn in history.turns:
            if turn.turn_id not in selected_prior_turn_ids:
                continue
            for record_id in turn.record_ids:
                selected_ids.add(record_id)
                reasons[record_id] = "prior_instruction_epoch_progress_spine"

        prior_complete_turns = sum(len(turns) for turns in complete_by_epoch.values())
        plan = _plan(
            policy="persistent_instruction_epoch_retirement",
            history=history,
            selected_ids=set(selected_ids),
            reasons=reasons,
            costs=costs,
            budget=budget,
            mandatory_ids=set(selected_ids),
            head_turns=0,
            tail_turns=(
                active_complete_turns
                + retained_prior_complete_turns
                + len(selected_prior_turn_ids)
            ),
            middle_candidate_turns=prior_complete_turns + retained_prior_complete_turns,
            middle_selected_turns=(
                retained_prior_complete_turns + len(selected_prior_turn_ids)
            ),
        )

        omitted_by_group: dict[str, list[str]] = {}
        for record in history.records:
            if record.record_id not in selected_ids:
                omitted_by_group.setdefault(record.causal_group_id, []).append(
                    record.record_id
                )
        exclusions = []
        for group_id, omitted in omitted_by_group.items():
            resources = tuple(sorted({
                str(resource)
                for record_id in omitted
                for resource in (records[record_id].metadata.get("resource_ids") or ())
            }))
            exclusions.append(AgentMemoryExclusion(
                causal_group_id=group_id,
                record_ids=tuple(omitted),
                rule_id="prior_instruction_epoch_retirement",
                classification="policy_retirement",
                reason=(
                    "record belongs to a terminally closed instruction epoch "
                    "or its assistant/tool detail predates the latest genuine "
                    "user instruction and lies outside configured prior-epoch floors"
                ),
                resource_ids=resources,
                witness_record_ids=(),
                tombstone="retired prior instruction-epoch causal group",
                excluded_tokens=sum(costs[record_id] for record_id in omitted),
            ))
        return replace(plan, exclusions=tuple(exclusions))


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

        mandatory_ids = set(immutable_instruction_record_ids(history))
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
