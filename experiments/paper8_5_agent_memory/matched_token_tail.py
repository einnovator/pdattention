"""Strict matched-token tail control for ordinary-text agent replay."""

from __future__ import annotations

from dataclasses import dataclass

from .materialization import (
    MaterializationMode,
    MaterializedMemoryPlan,
    MaterializedRecord,
    materialize_tool_observation_tail_to_ceiling,
)
from .model import AgentMemoryPlan, AgentRecordRole, CanonicalAgentHistory
from .selectors import (
    TokenCounter,
    immutable_instruction_record_ids,
    whitespace_tokens,
)


@dataclass(frozen=True)
class MatchedTokenTailConfig:
    """Configuration for the strict ordinary-text recency control."""

    tool_observation_threshold_tokens: int = 512
    protected_head_turns: int = 0
    protected_tail_turns: int = 0

    def __post_init__(self) -> None:
        if self.tool_observation_threshold_tokens <= 0:
            raise ValueError("tool_observation_threshold_tokens must be positive")
        if self.protected_head_turns < 0:
            raise ValueError("protected_head_turns cannot be negative")
        if self.protected_tail_turns < 0:
            raise ValueError("protected_tail_turns cannot be negative")


def _protected_turn_record_ids(
    history: CanonicalAgentHistory,
    config: MatchedTokenTailConfig,
) -> frozenset[str]:
    """Return whole records belonging to the configured head/tail floors."""

    complete = [turn for turn in history.turns if turn.complete]
    protected = list(complete[: config.protected_head_turns])
    if config.protected_tail_turns:
        protected.extend(complete[-config.protected_tail_turns :])
    return frozenset(
        record_id
        for turn in protected
        for record_id in turn.record_ids
    )


def matched_token_tail_mandatory_record_ids(
    history: CanonicalAgentHistory,
    config: MatchedTokenTailConfig = MatchedTokenTailConfig(),
) -> frozenset[str]:
    """Return prompt, configured floor, and current causal-group records.

    Agent protocols can append a standalone format-error or rejected-action
    recovery record without a preceding assistant action. It is an incomplete
    turn, but it is still the current model input and may never be retired.
    When the last record belongs to a multi-record group, retain the entire
    group so a strict recency ceiling cannot orphan its observation.
    """

    mandatory = set(immutable_instruction_record_ids(history))
    mandatory.update(_protected_turn_record_ids(history, config))
    if history.records:
        current_group = history.records[-1].causal_group_id
        mandatory.update(
            record.record_id
            for record in history.records
            if record.causal_group_id == current_group
        )
    return frozenset(mandatory)


def matched_token_tail_full_floor_record_ids(
    history: CanonicalAgentHistory,
    config: MatchedTokenTailConfig = MatchedTokenTailConfig(),
) -> frozenset[str]:
    """Records that must remain whole when establishing the budget floor."""

    immutable = set(immutable_instruction_record_ids(history))
    immutable.update(_protected_turn_record_ids(history, config))
    if not history.records:
        return frozenset(immutable)
    current_group = history.records[-1].causal_group_id
    current_complete = any(
        turn.causal_group_id == current_group and turn.complete
        for turn in history.turns
    )
    if not current_complete:
        immutable.update(
            record.record_id
            for record in history.records
            if record.causal_group_id == current_group
        )
    return frozenset(immutable)


def materialize_matched_token_tail(
    history: CanonicalAgentHistory,
    *,
    max_materialized_tokens: int,
    config: MatchedTokenTailConfig = MatchedTokenTailConfig(),
    count_tokens: TokenCounter = whitespace_tokens,
) -> MaterializedMemoryPlan:
    """Build a role-valid tail under a strict materialized-token ceiling.

    System and all user-authored instruction records are immutable. Complete
    action--observation turns are admitted newest first. The oldest admitted boundary turn may be made to
    fit only by trimming oversized tool observations; every other record stays
    byte-for-byte whole. Once a boundary turn is partial, no older turn is
    considered, preserving a contiguous recency tail.

    ``AgentMemoryPlan.selected_tokens`` counts the original contents of all
    logically selected records. ``MaterializedMemoryPlan.materialized_tokens``
    counts the text actually sent to the model. They intentionally differ when
    the boundary observation is trimmed.
    """

    if max_materialized_tokens <= 0:
        raise ValueError("max_materialized_tokens must be positive")

    records = history.record_by_id
    costs = {record.record_id: count_tokens(record.content) for record in history.records}
    mandatory_ids = matched_token_tail_mandatory_record_ids(history, config)
    full_floor_ids = matched_token_tail_full_floor_record_ids(history, config)
    mandatory_tokens = sum(costs[record_id] for record_id in full_floor_ids)
    if mandatory_tokens > max_materialized_tokens:
        raise ValueError(
            "matched token ceiling is smaller than mandatory system/task/current "
            f"records: {max_materialized_tokens} < {mandatory_tokens}"
        )

    selected_ids = set(full_floor_ids)
    immutable_ids = immutable_instruction_record_ids(history)
    protected_ids = _protected_turn_record_ids(history, config)
    reasons = {
        record_id: (
            "immutable_prompt"
            if record_id in immutable_ids
            else "protected_head_or_tail_turn"
            if record_id in protected_ids
            else "current_causal_group"
        )
        for record_id in full_floor_ids
    }
    materialized_by_id = {
        record_id: _whole_record(records[record_id], costs[record_id])
        for record_id in full_floor_ids
    }
    used = mandatory_tokens

    complete_turns = [turn for turn in history.turns if turn.complete]
    selected_turns = sum(
        all(record_id in full_floor_ids for record_id in turn.record_ids)
        for turn in complete_turns
    )
    for turn in reversed(complete_turns):
        if all(record_id in selected_ids for record_id in turn.record_ids):
            continue
        turn_rows = [records[record_id] for record_id in turn.record_ids]
        turn_tokens = sum(costs[row.record_id] for row in turn_rows)
        if used + turn_tokens <= max_materialized_tokens:
            selected_turns += 1
            used += turn_tokens
            for row in turn_rows:
                selected_ids.add(row.record_id)
                reasons[row.record_id] = "token_tail_whole_turn"
                materialized_by_id[row.record_id] = _whole_record(row, costs[row.record_id])
            continue

        remaining = max_materialized_tokens - used
        eligible = [
            row for row in turn_rows
            if row.has_role(AgentRecordRole.TOOL_OBSERVATION)
            and costs[row.record_id] > config.tool_observation_threshold_tokens
        ]
        fixed = [row for row in turn_rows if row not in eligible]
        fixed_tokens = sum(costs[row.record_id] for row in fixed)
        if not eligible or fixed_tokens >= remaining:
            break

        # Give every eligible observation a minimal role-preserving shell, then
        # spend the remaining allowance on the newest observations first.
        compacted: dict[str, MaterializedRecord] = {}
        compacted_tokens = 0
        for row in eligible:
            shell = _minimal_representation(
                row,
                remaining - fixed_tokens - compacted_tokens,
                count_tokens,
            )
            if shell is None:
                compacted = {}
                break
            compacted[row.record_id] = shell
            compacted_tokens += shell.materialized_tokens
        if not compacted or fixed_tokens + compacted_tokens > remaining:
            break

        spare = remaining - fixed_tokens - compacted_tokens
        for row in reversed(eligible):
            current = compacted[row.record_id]
            expanded = materialize_tool_observation_tail_to_ceiling(
                row,
                max_tokens=current.materialized_tokens + spare,
                count_tokens=count_tokens,
            )
            if expanded is None:
                continue
            spare -= expanded.materialized_tokens - current.materialized_tokens
            compacted[row.record_id] = expanded

        boundary_tokens = fixed_tokens + sum(
            row.materialized_tokens for row in compacted.values()
        )
        if used + boundary_tokens > max_materialized_tokens:
            raise AssertionError("matched token-tail materialization exceeded its ceiling")
        selected_turns += 1
        used += boundary_tokens
        for row in turn_rows:
            selected_ids.add(row.record_id)
            reasons[row.record_id] = (
                "token_tail_boundary_tool_observation"
                if row.record_id in compacted else "token_tail_boundary_whole_record"
            )
            materialized_by_id[row.record_id] = compacted.get(
                row.record_id, _whole_record(row, costs[row.record_id])
            )
        break

    # A small current observation can be ineligible for boundary compaction,
    # while a highly compressed boundary representation can also be impossible.
    # The current causal group is nevertheless live control state.  Preserve it
    # whole and classify the excess as mandatory overflow instead of rejecting
    # the request or silently retiring the newest action/observation pair.
    missing_mandatory = mandatory_ids - selected_ids
    if missing_mandatory:
        current_group = history.records[-1].causal_group_id
        current_ids = {
            row.record_id for row in history.records
            if row.causal_group_id == current_group
        }
        if not missing_mandatory.issubset(current_ids):
            raise ValueError("matched token tail lost immutable mandatory state")
        for record_id in current_ids:
            selected_ids.add(record_id)
            reasons[record_id] = "current_causal_group_mandatory_overflow"
            materialized_by_id[record_id] = _whole_record(
                records[record_id], costs[record_id]
            )
        selected_turns += 1

    mandatory_tokens = sum(
        materialized_by_id[record_id].materialized_tokens
        for record_id in mandatory_ids
    )
    effective_ceiling = max(max_materialized_tokens, mandatory_tokens)

    ordered_ids = tuple(
        record.record_id for record in history.records if record.record_id in selected_ids
    )
    logical_tokens = sum(costs[record_id] for record_id in ordered_ids)
    plan = AgentMemoryPlan(
        policy="matched_token_tail",
        selected_record_ids=ordered_ids,
        selected_causal_group_ids=tuple(dict.fromkeys(
            records[record_id].causal_group_id for record_id in ordered_ids
        )),
        selection_reasons=tuple((record_id, reasons[record_id]) for record_id in ordered_ids),
        full_history_tokens=sum(costs.values()),
        selected_tokens=logical_tokens,
        requested_budget_tokens=max_materialized_tokens,
        mandatory_tokens=mandatory_tokens,
        mandatory_overflow_tokens=max(0, mandatory_tokens - max_materialized_tokens),
        head_turns=min(config.protected_head_turns, len(complete_turns)),
        tail_turns=selected_turns,
        middle_candidate_turns=max(0, len(complete_turns) - selected_turns),
        middle_selected_turns=0,
    )
    materialized_rows = tuple(materialized_by_id[record_id] for record_id in ordered_ids)
    result = MaterializedMemoryPlan(
        logical_plan=plan,
        records=materialized_rows,
        full_selected_tokens=logical_tokens,
        materialized_tokens=sum(row.materialized_tokens for row in materialized_rows),
    )
    if result.materialized_tokens > effective_ceiling:
        raise AssertionError("matched token-tail result exceeded its materialized ceiling")
    return result


def _whole_record(record, tokens: int) -> MaterializedRecord:
    return MaterializedRecord(
        record.record_id,
        record.content,
        MaterializationMode.WHOLE_RECORD,
        tokens,
        tokens,
    )


def _minimal_representation(
    record,
    upper_bound: int,
    count_tokens: TokenCounter,
) -> MaterializedRecord | None:
    """Find the smallest accepted structural representation for allocation."""

    low, high = 1, upper_bound
    best: MaterializedRecord | None = None
    while low <= high:
        allowance = (low + high) // 2
        candidate = materialize_tool_observation_tail_to_ceiling(
            record,
            max_tokens=allowance,
            count_tokens=count_tokens,
        )
        if candidate is None:
            low = allowance + 1
        else:
            best = candidate
            high = allowance - 1
    return best
