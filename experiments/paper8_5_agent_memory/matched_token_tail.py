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
from .selectors import TokenCounter, whitespace_tokens


@dataclass(frozen=True)
class MatchedTokenTailConfig:
    """Configuration for the strict ordinary-text recency control."""

    tool_observation_threshold_tokens: int = 512

    def __post_init__(self) -> None:
        if self.tool_observation_threshold_tokens <= 0:
            raise ValueError("tool_observation_threshold_tokens must be positive")


def materialize_matched_token_tail(
    history: CanonicalAgentHistory,
    *,
    max_materialized_tokens: int,
    config: MatchedTokenTailConfig = MatchedTokenTailConfig(),
    count_tokens: TokenCounter = whitespace_tokens,
) -> MaterializedMemoryPlan:
    """Build a role-valid tail under a strict materialized-token ceiling.

    System and task records are immutable. Complete action--observation turns
    are admitted newest first. The oldest admitted boundary turn may be made to
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
    immutable_ids = {
        record.record_id
        for record in history.records
        if record.has_role(AgentRecordRole.SYSTEM) or record.has_role(AgentRecordRole.TASK)
    }
    immutable_tokens = sum(costs[record_id] for record_id in immutable_ids)
    if immutable_tokens > max_materialized_tokens:
        raise ValueError(
            "matched token ceiling is smaller than immutable system/task records: "
            f"{max_materialized_tokens} < {immutable_tokens}"
        )

    selected_ids = set(immutable_ids)
    reasons = {record_id: "immutable_prompt" for record_id in immutable_ids}
    materialized_by_id = {
        record_id: _whole_record(records[record_id], costs[record_id])
        for record_id in immutable_ids
    }
    used = immutable_tokens
    selected_turns = 0

    complete_turns = [turn for turn in history.turns if turn.complete]
    for turn in reversed(complete_turns):
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
        mandatory_tokens=immutable_tokens,
        mandatory_overflow_tokens=0,
        head_turns=0,
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
    if result.materialized_tokens > max_materialized_tokens:
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
