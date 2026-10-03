"""Distance-controlled agent-history ablations.

This module is diagnostic rather than a production selector.  It removes
either complete closed instruction components at a declared prompt distance,
or complete action--observation groups at a declared completed-tool-turn
distance.  Every target must already be disconnected from the current
instruction frontier under the generic information-flow DAG.  Missing closure
or lineage evidence fails closed unless the experiment explicitly admits the
heuristic no-path class.

Distances are one based:

* task distance 1 is the immediately preceding genuine-user instruction
  interval; the active interval is distance 0;
* tool-call distance 1 is the most recent complete action--observation turn
  before the next model decision.

The distinction lets Paper 8.5 measure whether behavioral disturbance decays
with chronology without confusing that question with orphaned prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .dag import (
    FrontierRetirementConfidence,
    build_frontier_information_flow_dag,
)
from .model import (
    AgentMemoryExclusion,
    AgentMemoryPlan,
    CanonicalAgentHistory,
)


@dataclass(frozen=True)
class DistanceConditioningConfig:
    """Frozen targets for one distance-controlled diagnostic arm."""

    task_distances: tuple[int, ...] = ()
    task_distance_at_least: int | None = None
    tool_call_distances: tuple[int, ...] = ()
    tool_call_window_radius: int = 0
    allow_heuristic: bool = False
    allow_superseded_unfinished_after: int | None = None

    def __post_init__(self) -> None:
        if any(value < 1 for value in self.task_distances):
            raise ValueError("task distances must be positive")
        if self.task_distance_at_least is not None and self.task_distance_at_least < 1:
            raise ValueError("task_distance_at_least must be positive")
        if any(value < 1 for value in self.tool_call_distances):
            raise ValueError("tool-call distances must be positive")
        if self.tool_call_window_radius < 0:
            raise ValueError("tool_call_window_radius cannot be negative")
        if (
            self.allow_superseded_unfinished_after is not None
            and self.allow_superseded_unfinished_after < 1
        ):
            raise ValueError("allow_superseded_unfinished_after must be positive")
        if not (
            self.task_distances
            or self.task_distance_at_least is not None
            or self.tool_call_distances
        ):
            raise ValueError("at least one distance target is required")


class DistanceConditioningSelector:
    """Remove only declared-distance, closed, no-path history units."""

    def __init__(self, config: DistanceConditioningConfig) -> None:
        self.config = config

    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: object,
        count_tokens: Callable[[str], int],
    ) -> AgentMemoryPlan:
        del query
        dag = build_frontier_information_flow_dag(
            history,
            recent_user_prompts=1,
        )
        eligible_rows = tuple(
            row for row in dag.retirement_candidates
            if self.config.allow_heuristic
            or row.confidence == FrontierRetirementConfidence.CERTIFIED
        )
        eligible_by_group = {
            row.causal_group_id: row for row in eligible_rows
        }
        complete_turn_by_group = {
            turn.causal_group_id: turn
            for turn in history.turns
            if turn.complete
        }
        excluded_ids: set[str] = set()
        exclusions: list[AgentMemoryExclusion] = []

        def add_exclusion(
            *,
            unit_id: str,
            record_ids: tuple[str, ...],
            rule_id: str,
            reason: str,
            group_ids: tuple[str, ...],
        ) -> None:
            new_ids = tuple(
                record_id for record_id in record_ids
                if record_id not in excluded_ids
            )
            if not new_ids:
                return
            rows = tuple(
                eligible_by_group[group_id]
                for group_id in group_ids
                if group_id in eligible_by_group
            )
            confidence = (
                FrontierRetirementConfidence.CERTIFIED
                if rows and all(
                    row.confidence == FrontierRetirementConfidence.CERTIFIED
                    for row in rows
                )
                else FrontierRetirementConfidence.HEURISTIC
            )
            resources = tuple(dict.fromkeys(
                resource for row in rows for resource in row.resource_ids
            ))
            excluded_ids.update(new_ids)
            exclusions.append(AgentMemoryExclusion(
                causal_group_id=unit_id,
                record_ids=new_ids,
                rule_id=rule_id,
                classification=confidence.value,
                reason=reason,
                resource_ids=resources,
                witness_record_ids=dag.frontier_record_ids,
                tombstone=(
                    f"INACTIVE unit={unit_id} rule={rule_id} "
                    f"confidence={confidence.value}"
                ),
                excluded_tokens=sum(
                    count_tokens(history.record_by_id[record_id].content)
                    for record_id in new_ids
                ),
            ))

        # Instruction distance is evaluated over observable prompt intervals,
        # never evaluator task IDs.  A target is retired atomically only when
        # it is terminally closed and all complete turns are no-path eligible.
        latest_epoch = len(dag.epochs) - 1
        exact_task_distances = set(self.config.task_distances)
        for epoch in dag.epochs[:-1]:
            distance = latest_epoch - epoch.epoch_index
            targeted = (
                distance in exact_task_distances
                or (
                    self.config.task_distance_at_least is not None
                    and distance >= self.config.task_distance_at_least
                )
            )
            if not targeted:
                continue
            interval_groups = tuple(
                group_id for group_id in epoch.causal_group_ids
                if group_id in complete_turn_by_group
            )
            closed = any(
                bool(history.record_by_id[record_id].metadata.get(
                    "protocol_completion_valid"
                ))
                for record_id in epoch.record_ids
            )
            superseded_unfinished = bool(
                not closed
                and self.config.allow_superseded_unfinished_after is not None
                and distance >= self.config.allow_superseded_unfinished_after
            )
            if (
                not (closed or superseded_unfinished)
                or not interval_groups
                or not set(interval_groups).issubset(eligible_by_group)
            ):
                continue
            add_exclusion(
                unit_id=f"task-distance:{distance}",
                record_ids=epoch.record_ids,
                rule_id=(
                    f"DISTANCE_SUPERSEDED_UNFINISHED_D{distance}_V1"
                    if superseded_unfinished else
                    f"DISTANCE_CLOSED_COMPONENT_D{distance}_V1"
                ),
                reason=(
                    (
                        "superseded unfinished instruction component"
                        if superseded_unfinished else
                        "closed instruction component"
                    )
                    + " at the declared prompt distance has no information-flow "
                    "path to the active prompt"
                ),
                group_ids=interval_groups,
            )

        # Tool distance is evaluated over complete causal turns.  Removing one
        # whole action--observation group leaves its instruction component and
        # terminal closure intact, isolating behavioral conditioning from the
        # orphaned-prompt failure mode.
        ordered_turns = sorted(
            complete_turn_by_group.values(),
            key=lambda turn: turn.first_message_index,
            reverse=True,
        )
        target_tool_distances: set[int] = set()
        for center in self.config.tool_call_distances:
            target_tool_distances.update(range(
                max(1, center - self.config.tool_call_window_radius),
                center + self.config.tool_call_window_radius + 1,
            ))
        for distance, turn in enumerate(ordered_turns, start=1):
            if distance not in target_tool_distances:
                continue
            if turn.causal_group_id not in eligible_by_group:
                continue
            add_exclusion(
                unit_id=f"tool-distance:{distance}",
                record_ids=turn.record_ids,
                rule_id=f"DISTANCE_CAUSAL_GROUP_D{distance}_V1",
                reason=(
                    "complete no-path action--observation group at the declared "
                    "completed-tool-turn distance"
                ),
                group_ids=(turn.causal_group_id,),
            )

        ordered_records = tuple(sorted(
            history.records, key=lambda row: row.message_index
        ))
        selected = tuple(
            row for row in ordered_records if row.record_id not in excluded_ids
        )
        selected_ids = tuple(row.record_id for row in selected)
        selected_groups = tuple(dict.fromkeys(
            row.causal_group_id for row in selected
        ))
        full_tokens = sum(count_tokens(row.content) for row in ordered_records)
        selected_tokens = sum(count_tokens(row.content) for row in selected)
        requested = int(getattr(budget, "max_tokens", full_tokens))
        return AgentMemoryPlan(
            policy="distance_conditioning_ablation_v1",
            selected_record_ids=selected_ids,
            selected_causal_group_ids=selected_groups,
            selection_reasons=tuple(
                (record_id, "distance_ablation_control_or_live_state")
                for record_id in selected_ids
            ),
            full_history_tokens=full_tokens,
            selected_tokens=selected_tokens,
            requested_budget_tokens=requested,
            mandatory_tokens=selected_tokens,
            mandatory_overflow_tokens=max(0, selected_tokens - requested),
            head_turns=0,
            tail_turns=1,
            middle_candidate_turns=len(history.turns),
            middle_selected_turns=sum(
                not set(turn.record_ids).issubset(excluded_ids)
                for turn in history.turns
            ),
            exclusions=tuple(exclusions),
        )
