"""Frozen-trajectory oracle scheduling without future-context leakage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from .model import AgentMemoryPlan, CanonicalAgentHistory


def restore_causal_groups(
    *,
    history: CanonicalAgentHistory,
    plan: AgentMemoryPlan,
    causal_group_ids: Sequence[str],
    count_tokens: Callable[[str], int],
) -> tuple[AgentMemoryPlan, dict[str, Any]]:
    """Restore complete excluded groups for an offline diagnostic oracle.

    This helper deliberately operates on a frozen logical plan.  It does not
    score, rank, or promote the restored groups into a deployment policy.
    """

    requested = tuple(dict.fromkeys(str(value) for value in causal_group_ids))
    records_by_group: dict[str, list[Any]] = {}
    for record in history.records:
        records_by_group.setdefault(record.causal_group_id, []).append(record)
    selected_before = set(plan.selected_record_ids)
    restored_groups: list[str] = []
    already_selected: list[str] = []
    absent: list[str] = []
    restored_records: list[str] = []
    for group_id in requested:
        group_records = records_by_group.get(group_id)
        if not group_records:
            absent.append(group_id)
        elif all(row.record_id in selected_before for row in group_records):
            already_selected.append(group_id)
        else:
            restored_groups.append(group_id)
            restored_records.extend(
                row.record_id for row in group_records
                if row.record_id not in selected_before
            )
    selected = selected_before | set(restored_records)
    selected_record_ids = tuple(
        row.record_id for row in history.records if row.record_id in selected
    )
    selected_causal_group_ids = tuple(dict.fromkeys(
        row.causal_group_id
        for row in history.records
        if row.record_id in selected
    ))
    reasons = dict(plan.selection_reasons)
    for record_id in restored_records:
        group_id = history.record_by_id[record_id].causal_group_id
        reasons[record_id] = f"oracle_addback:{group_id}"
    selected_tokens = sum(
        count_tokens(history.record_by_id[record_id].content)
        for record_id in selected_record_ids
    )
    restored_group_set = set(restored_groups)
    updated = replace(
        plan,
        selected_record_ids=selected_record_ids,
        selected_causal_group_ids=selected_causal_group_ids,
        selection_reasons=tuple(
            (record_id, reasons[record_id]) for record_id in selected_record_ids
        ),
        selected_tokens=selected_tokens,
        middle_selected_turns=plan.middle_selected_turns + len(restored_groups),
        exclusions=tuple(
            row for row in plan.exclusions
            if row.causal_group_id not in restored_group_set
        ),
    )
    return updated, {
        "requested_causal_group_ids": requested,
        "restored_causal_group_ids": tuple(restored_groups),
        "already_selected_causal_group_ids": tuple(already_selected),
        "not_yet_present_causal_group_ids": tuple(absent),
        "restored_record_ids": tuple(restored_records),
        "restored_tokens": selected_tokens - plan.selected_tokens,
    }


@dataclass(frozen=True)
class OracleAblationCase:
    decision_turn: int
    omitted_causal_group_ids: tuple[str, ...]
    selected_record_ids: tuple[str, ...]
    reference_action_digest: str
    scoring_contract: str = "exact_or_semantically_equivalent_next_action"


def leave_one_bundle_out_cases(
    *,
    history: CanonicalAgentHistory,
    full_plan: AgentMemoryPlan,
    decision_turn: int,
    reference_action_digest: str,
) -> tuple[OracleAblationCase, ...]:
    """Generate bundle-level counterfactuals for an offline future-utility oracle.

    Future action data appears only in the scoring label.  It is never included
    in selected_record_ids and therefore cannot leak into the model request.
    TASK/SYSTEM groups are excluded because their removal is a separate explicit
    semantic-role ablation, not part of the deployment-policy oracle.
    """

    records = history.record_by_id
    immutable_groups = {
        record.causal_group_id
        for record in history.records
        if record.turn_id in {"system", "task"}
    }
    groups = [
        group for group in full_plan.selected_causal_group_ids
        if group not in immutable_groups
    ]
    cases = []
    for group in groups:
        selected = tuple(
            record_id for record_id in full_plan.selected_record_ids
            if records[record_id].causal_group_id != group
        )
        cases.append(OracleAblationCase(
            decision_turn=decision_turn,
            omitted_causal_group_ids=(group,),
            selected_record_ids=selected,
            reference_action_digest=reference_action_digest,
        ))
    return tuple(cases)
