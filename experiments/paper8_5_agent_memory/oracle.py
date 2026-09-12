"""Frozen-trajectory oracle scheduling without future-context leakage."""

from __future__ import annotations

from dataclasses import dataclass

from .model import AgentMemoryPlan, CanonicalAgentHistory


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
