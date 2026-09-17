import json
import hashlib
import subprocess
import tarfile
from dataclasses import replace
import pytest
import experiments.paper8_5_agent_memory.run_frozen_replay as frozen_replay

from experiments.paper8_5_agent_memory import (
    AgentMemoryBudget,
    AgentRecord,
    BashOperation,
    DagEdgeKind,
    DagCertifiedExclusionSelector,
    ExclusionClass,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
    FullHistorySelector,
    HeadMiddleTailConfig,
    HeadMiddleTailSelector,
    MaterializationMode,
    FrontierRetirementConfidence,
    FrontierDagRetirementSelector,
    FrontierSimplificationMode,
    MatchedTokenTailConfig,
    MiddleSelectionStrategy,
    NegativeHeuristicSelector,
    NegativeRealizationMode,
    NegativeRule,
    NegativeSelectionConfig,
    ToolObservationMaterializer,
    build_resource_effect_dag,
    build_frontier_information_flow_dag,
    build_negative_exclusions,
    classify_bash_operation,
    exclude_certified_groups,
    leave_one_bundle_out_cases,
    materialize_matched_token_tail,
    materialize_plan,
    recordize_minisweagent_messages,
    reacquired_excluded_resources,
    realize_negative_receipts,
    serialize_materialized_messages,
    simplify_disconnected_frontier,
    validate_minisweagent_chat,
)
from experiments.paper8_5_agent_memory.observation_instrumentation import (
    build_bash_observation_metadata,
)
from experiments.paper8_5_agent_memory.export_review_history import (
    export_trajectory,
)
from experiments.paper8_5_agent_memory.run_structural_screen import (
    STRUCTURAL_POLICIES,
    structural_screen,
)
from experiments.paper8_5_agent_memory.run_negative_heuristic_screen import (
    DEFAULT_POLICIES,
    negative_structural_screen,
)
from experiments.paper8_5_agent_memory.run_synthetic_diagnostics import (
    run_synthetic_diagnostics,
)
from experiments.paper8_5_agent_memory.selectors import whitespace_tokens
from experiments.paper8_5_agent_memory.workspace_checkpoint import (
    repository_state,
    restore_repository_checkpoint,
)


def _messages(turns: int = 5):
    rows = [
        {"role": "system", "content": "Use one bash command."},
        {"role": "user", "content": "Fix foo.py and preserve the API."},
    ]
    for turn in range(turns):
        command = (
            f"sed -n '{turn + 1},{turn + 3}p' foo.py"
            if turn != 2
            else "python -c \"from pathlib import Path; p=Path('foo.py'); p.write_text(p.read_text().replace('a','b'))\""
        )
        rows.extend((
            {
                "role": "assistant",
                "content": f"THOUGHT: inspect turn {turn}\n\n```mswea_bash_command\n{command}\n```",
            },
            {
                "role": "user",
                "content": f"<returncode>0</returncode>\n<output>foo.py:{turn}: value</output>",
            },
        ))
    return rows


def _information_flow_chain(
    resources,
    *,
    workspace_scopes=None,
):
    records = [AgentRecord(
        "system", "system", "system", 0, "system", "agent protocol",
        AgentRecordRole.SYSTEM, (AgentRecordRole.SYSTEM,),
    )]
    turns = []
    message_index = 1
    scopes = workspace_scopes or [None] * len(resources)
    for epoch, (resource, scope) in enumerate(zip(resources, scopes)):
        instruction_id = f"instruction-{epoch}"
        instruction_role = (
            AgentRecordRole.TASK if epoch == 0 else AgentRecordRole.USER_INPUT
        )
        metadata = {"workspace_lineage_id": scope} if scope else {}
        records.append(AgentRecord(
            instruction_id,
            instruction_id,
            instruction_id,
            message_index,
            "user",
            f"Work on {resource}",
            instruction_role,
            (instruction_role,),
            resource_ids=(resource,),
            metadata=metadata,
        ))
        message_index += 1
        group = f"turn-{epoch}"
        action_id = f"action-{epoch}"
        observation_id = f"observation-{epoch}"
        records.extend((
            AgentRecord(
                action_id,
                group,
                group,
                message_index,
                "assistant",
                "THOUGHT inspect relevant state then decide carefully " * 8
                + f"\n```mswea_bash_command\ncat {resource}\n```",
                AgentRecordRole.ASSISTANT_ACTION,
                (AgentRecordRole.ASSISTANT_ACTION,),
                command=f"cat {resource}",
                resource_ids=(resource,),
                metadata={**metadata, "operation_kind": "read"},
            ),
            AgentRecord(
                observation_id,
                group,
                group,
                message_index + 1,
                "user",
                "<returncode>0</returncode>\n<output>" + "evidence " * 40 + "</output>",
                AgentRecordRole.TOOL_OBSERVATION,
                (AgentRecordRole.TOOL_OBSERVATION,),
                command=f"cat {resource}",
                return_code=0,
                resource_ids=(resource,),
                metadata={**metadata, "operation_kind": "read", "output_complete": True},
            ),
        ))
        turns.append(AgentTurn(group, group, (action_id, observation_id), message_index, True))
        message_index += 2
    return CanonicalAgentHistory(tuple(records), tuple(turns))


def test_frontier_dag_retires_only_old_disconnected_epochs():
    history = _information_flow_chain(
        [f"src/task_{index}.py" for index in range(6)],
        workspace_scopes=[f"workspace-{index}" for index in range(6)],
    )
    dag = build_frontier_information_flow_dag(history, recent_user_prompts=2)

    assert dag.frontier_epoch_indices == (4, 5)
    assert {row.epoch_index for row in dag.retirement_candidates} == {0, 1, 2, 3}
    assert all(
        row.confidence == FrontierRetirementConfidence.CERTIFIED
        for row in dag.retirement_candidates
    )
    retired = {row.causal_group_id for row in dag.retirement_candidates}
    assert retired == {"turn-0", "turn-1", "turn-2", "turn-3"}


def test_frontier_dag_preserves_old_resource_lineage_reaching_live_prompt():
    history = _information_flow_chain(
        [
            "src/shared.py",
            "src/task_1.py",
            "src/task_2.py",
            "src/task_3.py",
            "src/task_4.py",
            "src/shared.py",
        ],
        workspace_scopes=["workspace-live"] * 6,
    )
    dag = build_frontier_information_flow_dag(history, recent_user_prompts=2)

    retired = {row.causal_group_id for row in dag.retirement_candidates}
    assert "turn-0" not in retired
    assert retired == {"turn-1", "turn-2", "turn-3"}
    assert {"action-0", "observation-0"}.issubset(dag.live_ancestor_record_ids)


def test_frontier_dag_does_not_treat_environment_as_cross_epoch_lineage():
    history = _information_flow_chain(["/testbed/common.py"] * 4)
    records = tuple(
        replace(
            row,
            metadata={**row.metadata, "environment_fingerprint": "shared-image"},
        )
        for row in history.records
    )
    history = CanonicalAgentHistory(records, history.turns)

    dag = build_frontier_information_flow_dag(history, recent_user_prompts=2)

    assert {row.causal_group_id for row in dag.retirement_candidates} == {
        "turn-0", "turn-1"
    }
    assert not any(
        edge.kind == DagEdgeKind.RESOURCE_FLOW
        and edge.source_id in {"instruction-0", "action-0", "observation-0"}
        and edge.target_id in {"instruction-3", "action-3", "observation-3"}
        for edge in dag.edges
    )


def test_frontier_dag_does_not_alias_same_path_across_declared_workspaces():
    history = _information_flow_chain(
        ["src/common.py"] * 4,
        workspace_scopes=[f"workspace-{index}" for index in range(4)],
    )
    dag = build_frontier_information_flow_dag(history, recent_user_prompts=2)

    assert {row.causal_group_id for row in dag.retirement_candidates} == {
        "turn-0", "turn-1"
    }


def test_frontier_simplification_preserves_prompts_and_orders_ablation_savings():
    history = _information_flow_chain([f"src/task_{index}.py" for index in range(6)])
    dag = build_frontier_information_flow_dag(history, recent_user_prompts=2)
    plans = {
        mode: simplify_disconnected_frontier(
            history,
            dag,
            mode=mode,
            count_tokens=whitespace_tokens,
        )
        for mode in FrontierSimplificationMode
    }
    instruction_ids = {
        row.record_id for row in history.records
        if row.has_role(AgentRecordRole.TASK)
        or row.has_role(AgentRecordRole.USER_INPUT)
    }
    assert all(
        instruction_ids.issubset(plan.selected_record_ids)
        for plan in plans.values()
    )
    assert (
        plans[FrontierSimplificationMode.WHOLE_CAUSAL_GROUP].saving_fraction
        > plans[FrontierSimplificationMode.ACTION_PARAMETERS_AND_OBSERVATION].saving_fraction
        > plans[FrontierSimplificationMode.OBSERVATION_PAYLOAD].saving_fraction
        > 0
    )


def test_frontier_dag_selector_emits_auditable_atomic_exclusions():
    history = _information_flow_chain(
        [f"src/task_{index}.py" for index in range(6)],
        workspace_scopes=[f"workspace-{index}" for index in range(6)],
    )
    full_tokens = sum(whitespace_tokens(row.content) for row in history.records)
    plan = FrontierDagRetirementSelector(
        recent_user_prompts=2,
        allow_heuristic=False,
    ).select(
        history=history,
        query="",
        budget=AgentMemoryBudget(max_tokens=full_tokens),
        count_tokens=whitespace_tokens,
    )

    assert {row.causal_group_id for row in plan.exclusions} == {
        "turn-0", "turn-1", "turn-2", "turn-3"
    }
    assert all(row.rule_id == "FRONTIER_NO_PATH_M2_V1" for row in plan.exclusions)
    assert all(
        set(row.record_ids).isdisjoint(plan.selected_record_ids)
        for row in plan.exclusions
    )


def test_frontier_dag_p1_retains_latest_valid_not_latest_malformed_completion():
    history = _information_flow_chain(
        [f"src/task_{index}.py" for index in range(5)],
        workspace_scopes=[f"workspace-{index}" for index in range(5)],
    )
    records = []
    for row in history.records:
        if row.record_id == "observation-0":
            row = replace(
                row,
                semantic_roles=(*row.semantic_roles, AgentRecordRole.FINALIZATION),
                metadata={**row.metadata, "protocol_completion_valid": True},
            )
        elif row.record_id == "observation-1":
            row = replace(
                row,
                semantic_roles=(*row.semantic_roles, AgentRecordRole.FINALIZATION),
                metadata={**row.metadata, "protocol_completion_valid": False},
            )
        records.append(row)
    history = CanonicalAgentHistory(tuple(records), history.turns)

    dag = build_frontier_information_flow_dag(
        history,
        recent_user_prompts=2,
        valid_protocol_exemplars=1,
    )
    retired = {row.causal_group_id for row in dag.retirement_candidates}

    assert "turn-0" not in retired
    assert "turn-1" in retired
    assert "observation-0" in dag.live_ancestor_record_ids
    assert "observation-1" not in dag.live_ancestor_record_ids


def test_frontier_dag_p1_is_a_barrier_not_a_whole_epoch_reachability_flood():
    base = _information_flow_chain(
        ["src/old.py", "src/current.py"],
        workspace_scopes=["workspace-old", "workspace-current"],
    )
    records = list(base.records)
    old_action = records[1]
    old_observation = records[2]
    extra_action = replace(
        old_action,
        record_id="action-0-final",
        turn_id="turn-0-final",
        causal_group_id="turn-0-final",
        message_index=3,
        content="```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n```",
        command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        semantic_roles=(AgentRecordRole.ASSISTANT_ACTION, AgentRecordRole.FINALIZATION),
        resource_ids=(),
    )
    extra_observation = replace(
        old_observation,
        record_id="observation-0-final",
        turn_id="turn-0-final",
        causal_group_id="turn-0-final",
        message_index=4,
        content="diff --git a/src/old.py b/src/old.py",
        semantic_roles=(AgentRecordRole.TOOL_OBSERVATION, AgentRecordRole.FINALIZATION),
        resource_ids=(),
        metadata={
            **old_observation.metadata,
            "protocol_completion_valid": True,
        },
    )
    shifted = [
        replace(row, message_index=row.message_index + 2)
        if row.message_index >= 3 else row
        for row in records[3:]
    ]
    history = CanonicalAgentHistory(
        tuple((*records[:3], extra_action, extra_observation, *shifted)),
        (
            base.turns[0],
            AgentTurn(
                "turn-0-final",
                "turn-0-final",
                ("action-0-final", "observation-0-final"),
                3,
                True,
            ),
            replace(base.turns[1], first_message_index=base.turns[1].first_message_index + 2),
        ),
    )

    dag = build_frontier_information_flow_dag(
        history,
        recent_user_prompts=1,
        valid_protocol_exemplars=1,
    )
    retired = {row.causal_group_id for row in dag.retirement_candidates}

    assert "turn-0-final" not in retired
    assert "turn-0" in retired
    assert "observation-0-final" in dag.live_ancestor_record_ids
    assert "observation-0" not in dag.live_ancestor_record_ids


def test_frontier_dag_w1_pins_minimal_successful_workflow_spine():
    base = _information_flow_chain(
        ["src/old.py", "src/current.py"],
        workspace_scopes=["workspace-old", "workspace-current"],
    )
    records = list(base.records)
    template_action = records[2]
    template_observation = records[3]

    def action(record_id, group, index, command, roles):
        return replace(
            template_action,
            record_id=record_id,
            turn_id=group,
            causal_group_id=group,
            message_index=index,
            content=f"```mswea_bash_command\n{command}\n```",
            command=command,
            semantic_roles=(AgentRecordRole.ASSISTANT_ACTION, *roles),
            resource_ids=("src/old.py",),
        )

    def observation(record_id, group, index, command, roles, metadata):
        return replace(
            template_observation,
            record_id=record_id,
            turn_id=group,
            causal_group_id=group,
            message_index=index,
            command=command,
            semantic_roles=(AgentRecordRole.TOOL_OBSERVATION, *roles),
            resource_ids=("src/old.py",),
            metadata={**template_observation.metadata, **metadata},
        )

    mutation_action = action(
        "action-mutation", "turn-mutation", 4, "edit src/old.py",
        (AgentRecordRole.MUTATION,),
    )
    mutation_observation = observation(
        "observation-mutation", "turn-mutation", 5, "edit src/old.py",
        (AgentRecordRole.MUTATION,),
        {"changed_resource_ids": ["src/old.py"]},
    )
    verification_action = action(
        "action-verification", "turn-verification", 6, "cat src/old.py",
        (AgentRecordRole.VERIFICATION,),
    )
    verification_observation = observation(
        "observation-verification", "turn-verification", 7, "cat src/old.py",
        (AgentRecordRole.VERIFICATION,),
        {"changed_resource_ids": []},
    )
    final_action = action(
        "action-final", "turn-final", 8, "submit patch",
        (AgentRecordRole.FINALIZATION,),
    )
    final_observation = observation(
        "observation-final", "turn-final", 9, "submit patch",
        (AgentRecordRole.FINALIZATION,),
        {"protocol_completion_valid": True, "changed_resource_ids": []},
    )
    shifted = [
        replace(row, message_index=row.message_index + 6)
        for row in records[4:]
    ]
    history = CanonicalAgentHistory(
        tuple((
            *records[:4],
            mutation_action, mutation_observation,
            verification_action, verification_observation,
            final_action, final_observation,
            *shifted,
        )),
        (
            base.turns[0],
            AgentTurn("turn-mutation", "turn-mutation", (
                "action-mutation", "observation-mutation"
            ), 4, True),
            AgentTurn("turn-verification", "turn-verification", (
                "action-verification", "observation-verification"
            ), 6, True),
            AgentTurn("turn-final", "turn-final", (
                "action-final", "observation-final"
            ), 8, True),
            replace(base.turns[1], first_message_index=base.turns[1].first_message_index + 6),
        ),
    )

    dag = build_frontier_information_flow_dag(
        history,
        recent_user_prompts=1,
        valid_protocol_exemplars=1,
        valid_workflow_exemplars=1,
    )
    retired = {row.causal_group_id for row in dag.retirement_candidates}

    assert "turn-0" in retired
    assert {"turn-mutation", "turn-verification", "turn-final"}.isdisjoint(retired)
    assert {
        "observation-mutation", "observation-verification", "observation-final"
    }.issubset(dag.live_ancestor_record_ids)
    assert sum(
        edge.kind == DagEdgeKind.WORKFLOW_CONTROL for edge in dag.edges
    ) == 3


def test_recordizer_preserves_action_observation_causal_bundles():
    history = recordize_minisweagent_messages(_messages(3))
    assert len(history.turns) == 3
    assert all(turn.complete and len(turn.record_ids) == 2 for turn in history.turns)
    mutation = history.record_by_id["m6"]
    observation = history.record_by_id["m7"]
    assert mutation.has_role(AgentRecordRole.MUTATION)
    assert observation.has_role(AgentRecordRole.MUTATION)
    assert mutation.causal_group_id == observation.causal_group_id


def test_recordizer_prefers_the_executable_miniswe_tag_over_explanatory_fences():
    rows = _messages(1)
    rows[2]["content"] = (
        "Example only:\n```python\nprint('do not execute')\n```\n"
        "```mswea_bash_command\ncat foo.py\n```"
    )
    history = recordize_minisweagent_messages(rows)
    assert history.record_by_id["m2"].command == "cat foo.py"


def test_recordizer_never_treats_an_explanatory_closing_fence_as_an_action():
    rows = _messages(1)
    rows[2]["content"] = (
        "THOUGHT: example\n```python\nprint('not an action')\n```\n"
        "now act\n```mswea_bash_command\ngit diff > patch.txt\n```"
    )
    history = recordize_minisweagent_messages(rows)

    assert history.record_by_id["m2"].command == "git diff > patch.txt"


def test_head_and_tail_are_independent_and_selection_is_middle_only():
    history = recordize_minisweagent_messages(_messages(6))
    selector = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=2,
        tail_turns=1,
        middle_strategy=MiddleSelectionStrategy.NONE,
    ))
    plan = selector.select(
        history=history,
        query="foo",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    selected_groups = set(plan.selected_causal_group_ids)
    assert {"system", "task", "turn:t0000", "turn:t0001", "turn:t0005"} <= selected_groups
    assert "turn:t0002" not in selected_groups
    assert plan.head_turns == 2
    assert plan.tail_turns == 1
    assert plan.middle_candidate_turns == 3


def test_retention_floor_rounds_up_at_whole_causal_turn_boundary():
    history = recordize_minisweagent_messages(_messages(4))
    full_tokens = sum(whitespace_tokens(row.content) for row in history.records)
    target = int(full_tokens * 0.9)
    ceiling = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=1,
        tail_turns=1,
        middle_strategy=MiddleSelectionStrategy.RECENCY,
    )).select(
        history=history,
        query="foo.py",
        budget=AgentMemoryBudget(max_tokens=target),
        count_tokens=whitespace_tokens,
    )
    floor = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=1,
        tail_turns=1,
        middle_strategy=MiddleSelectionStrategy.RECENCY,
        round_up_to_budget=True,
    )).select(
        history=history,
        query="foo.py",
        budget=AgentMemoryBudget(max_tokens=target),
        count_tokens=whitespace_tokens,
    )

    assert ceiling.selected_tokens <= target
    assert floor.selected_tokens >= target
    assert floor.selected_tokens > ceiling.selected_tokens
    assert floor.policy.endswith(":round_up")


def test_role_floor_rounds_up_the_whole_middle_causal_bundle():
    history = recordize_minisweagent_messages(_messages(5))
    selector = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=1,
        tail_turns=1,
        middle_strategy=MiddleSelectionStrategy.NONE,
        mutation_turns=1,
    ))
    plan = selector.select(
        history=history,
        query="foo",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    assert "turn:t0002" in plan.selected_causal_group_ids
    mutation_turn = history.turn_by_id["t0002"]
    assert set(mutation_turn.record_ids) <= set(plan.selected_record_ids)
    expected_mandatory = sum(
        len(history.record_by_id[record_id].content.split())
        for record_id in plan.selected_record_ids
    )
    assert plan.mandatory_tokens == expected_mandatory


def test_lexical_middle_selection_uses_budget_without_touching_head_or_tail():
    history = recordize_minisweagent_messages(_messages(5))
    selector = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=1,
        tail_turns=1,
        middle_strategy=MiddleSelectionStrategy.LEXICAL,
    ))
    mandatory = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=1,
        tail_turns=1,
        middle_strategy=MiddleSelectionStrategy.NONE,
    )).select(history=history, query="", budget=AgentMemoryBudget(max_tokens=10_000))
    target_turn = history.turn_by_id["t0003"]
    target_cost = sum(
        len(history.record_by_id[rid].content.split()) for rid in target_turn.record_ids
    )
    plan = selector.select(
        history=history,
        query="foo.py:3",
        budget=AgentMemoryBudget(max_tokens=mandatory.selected_tokens + target_cost),
    )
    assert "turn:t0003" in plan.selected_causal_group_ids
    assert "turn:t0000" in plan.selected_causal_group_ids
    assert "turn:t0004" in plan.selected_causal_group_ids


def test_large_tool_response_can_be_compacted_without_dropping_its_action():
    messages = _messages(1)
    messages[-1]["content"] = (
        "<returncode>1</returncode>\n<output>\n"
        + "\n".join(f"frame {index} foo.py" for index in range(200))
        + "\n</output>"
    )
    history = recordize_minisweagent_messages(messages)
    plan = FullHistorySelector().select(
        history=history,
        query="frame 100",
        budget=AgentMemoryBudget(max_tokens=100_000),
    )
    materialized = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(
            mode=MaterializationMode.TOOL_MATCHED_SPAN,
            threshold_tokens=20,
            head_lines=2,
            tail_lines=2,
            match_context_lines=1,
        ),
        query="frame 100",
    )
    observation = materialized.records[-1]
    assert observation.materialized_tokens < observation.original_tokens
    assert "<returncode>1</returncode>" in observation.content
    assert "frame 100 foo.py" in observation.content
    assert plan.selected_record_ids[-2] in tuple(row.record_id for row in materialized.records)
    serialized = serialize_materialized_messages(history, materialized)
    assert serialized[-2]["role"] == "assistant"
    assert serialized[-1]["role"] == "user"


def test_matched_token_tail_is_role_valid_and_never_exceeds_materialized_ceiling():
    messages = _messages(3)
    messages[5]["content"] = (
        "<returncode>0</returncode>\n<output>\n"
        + "\n".join(f"old output line {index}" for index in range(80))
        + "\n</output>"
    )
    history = recordize_minisweagent_messages(messages)
    costs = {
        record.record_id: whitespace_tokens(record.content) for record in history.records
    }
    prompt = costs["m0"] + costs["m1"]
    newest = costs["m6"] + costs["m7"]
    boundary_action = costs["m4"]
    ceiling = prompt + newest + boundary_action + 18

    materialized = materialize_matched_token_tail(
        history,
        max_materialized_tokens=ceiling,
        config=MatchedTokenTailConfig(tool_observation_threshold_tokens=20),
        count_tokens=whitespace_tokens,
    )

    assert materialized.materialized_tokens <= ceiling
    assert materialized.logical_plan.selected_tokens > materialized.materialized_tokens
    assert materialized.logical_plan.selected_record_ids == (
        "m0", "m1", "m4", "m5", "m6", "m7"
    )
    assert "m2" not in materialized.logical_plan.selected_record_ids
    boundary_observation = next(row for row in materialized.records if row.record_id == "m5")
    assert boundary_observation.mode == MaterializationMode.MATCHED_TOKEN_TAIL
    assert "<returncode>0</returncode>" in boundary_observation.content
    assert "old output line 79" in boundary_observation.content
    assert boundary_observation.omitted_prefix_lines > 0
    serialized = serialize_materialized_messages(history, materialized)
    validate_minisweagent_chat(serialized)
    assert [row["role"] for row in serialized[-4:]] == [
        "assistant", "user", "assistant", "user"
    ]


def test_matched_token_tail_does_not_trim_small_or_non_tool_records():
    history = recordize_minisweagent_messages(_messages(1))
    mandatory_tokens = sum(
        whitespace_tokens(history.record_by_id[record_id].content)
        for record_id in ("m0", "m1", "m2", "m3")
    )
    result = materialize_matched_token_tail(
        history,
        max_materialized_tokens=mandatory_tokens,
        config=MatchedTokenTailConfig(tool_observation_threshold_tokens=100),
    )
    assert result.logical_plan.selected_record_ids == ("m0", "m1", "m2", "m3")
    assert all(row.mode == MaterializationMode.WHOLE_RECORD for row in result.records)


def test_matched_token_tail_uses_measured_fallback_for_one_oversized_line():
    messages = _messages(1)
    messages[3]["content"] = (
        "<returncode>0</returncode>\n<output>\n"
        + " ".join(f"token{index}" for index in range(200))
        + "\n</output>"
    )
    history = recordize_minisweagent_messages(messages)
    prompt_and_action = sum(
        whitespace_tokens(history.record_by_id[record_id].content)
        for record_id in ("m0", "m1", "m2")
    )
    result = materialize_matched_token_tail(
        history,
        max_materialized_tokens=prompt_and_action + 12,
        config=MatchedTokenTailConfig(tool_observation_threshold_tokens=10),
    )
    observation = next(row for row in result.records if row.record_id == "m3")
    assert observation.token_fallback_used
    assert observation.materialized_tokens <= 12
    assert result.materialized_tokens <= prompt_and_action + 12


def test_matched_token_tail_fails_closed_when_prompt_exceeds_ceiling():
    history = recordize_minisweagent_messages(_messages(1))
    with pytest.raises(ValueError, match="mandatory system/task/current"):
        materialize_matched_token_tail(history, max_materialized_tokens=1)


def test_chat_validation_rejects_an_orphaned_assistant_action():
    try:
        validate_minisweagent_chat([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "action"},
        ])
    except ValueError as error:
        assert "lacking its observation" in str(error)
    else:
        raise AssertionError("orphaned assistant action was accepted")


def test_dag_abstains_without_runtime_identity_even_for_exact_duplicates():
    messages = _messages(1)
    messages.extend((dict(messages[2]), dict(messages[3])))
    history = recordize_minisweagent_messages(messages)
    dag = build_resource_effect_dag(history)
    safe = [
        row for row in dag.exclusion_candidates
        if row.classification == ExclusionClass.HEURISTIC_EXACT_DUPLICATE
    ]
    assert len(safe) == 1
    assert safe[0].causal_group_id == "turn:t0000"
    filtered = exclude_certified_groups(history, dag)
    assert "turn:t0000" in {turn.causal_group_id for turn in filtered.turns}
    assert "turn:t0001" in {turn.causal_group_id for turn in filtered.turns}


def test_dag_certifies_operational_duplicate_with_complete_runtime_identity():
    messages = _messages(1)
    runtime_identity = {
        "returncode": 0,
        "cwd": "/workspace/repo",
        "environment_fingerprint": "env-sha256",
        "resource_version_fingerprints": {"foo.py": "file-sha256"},
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
        "tool_semantics": {
            "category": "filesystem_read",
            "provenance": "runtime_traced",
            "complete": True,
            "effects": [{
                "kind": "read",
                "resource_id": "foo.py",
                "resource_version_fingerprint": "file-sha256",
            }],
        },
    }
    messages[3]["extra"] = runtime_identity
    messages.extend((dict(messages[2]), dict(messages[3])))
    history = recordize_minisweagent_messages(messages)
    dag = build_resource_effect_dag(history)
    certified = [
        row for row in dag.exclusion_candidates
        if row.classification == ExclusionClass.CERTIFIED_OPERATIONAL_DUPLICATE
    ]
    assert len(certified) == 1
    assert certified[0].certificate is not None
    filtered = exclude_certified_groups(history, dag)
    assert "turn:t0000" not in {turn.causal_group_id for turn in filtered.turns}
    protected_plan = DagCertifiedExclusionSelector(
        protected_head_turns=1,
        protected_tail_turns=1,
    ).select(
        history=history,
        query="foo.py",
        budget=AgentMemoryBudget(max_tokens=100_000),
        count_tokens=whitespace_tokens,
    )
    assert "turn:t0000" in protected_plan.selected_causal_group_ids

    exclusion_plan = DagCertifiedExclusionSelector(
        protected_head_turns=0,
        protected_tail_turns=0,
    ).select(
        history=history,
        query="foo.py",
        budget=AgentMemoryBudget(max_tokens=100_000),
        count_tokens=whitespace_tokens,
    )
    assert [row.causal_group_id for row in exclusion_plan.exclusions] == [
        "turn:t0000"
    ]
    assert exclusion_plan.exclusions[0].rule_id == (
        "TRACE_EXACT_OPERATION_RESULT_V1"
    )


def test_dag_operational_duplicate_allows_different_assistant_reasoning():
    messages = _messages(1)
    runtime_identity = {
        "returncode": 0,
        "cwd": "/workspace/repo",
        "environment_fingerprint": "env-sha256",
        "resource_version_fingerprints": {"foo.py": "file-sha256"},
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
        "tool_semantics": {
            "category": "filesystem_read",
            "provenance": "runtime_traced",
            "complete": True,
            "effects": [{
                "kind": "read",
                "resource_id": "foo.py",
                "resource_version_fingerprint": "file-sha256",
            }],
        },
    }
    messages[3]["extra"] = runtime_identity
    later_action = dict(messages[2])
    later_action["content"] = later_action["content"].replace(
        "inspect turn 0", "verify the unchanged file again"
    )
    messages.extend((later_action, dict(messages[3])))

    dag = build_resource_effect_dag(recordize_minisweagent_messages(messages))
    certified = [
        row for row in dag.exclusion_candidates
        if row.classification == ExclusionClass.CERTIFIED_OPERATIONAL_DUPLICATE
    ]

    assert len(certified) == 1
    assert certified[0].causal_group_id == "turn:t0000"
    assert certified[0].certificate.rule_id == "TRACE_EXACT_OPERATION_RESULT_V1"


def test_runtime_observation_metadata_certifies_only_simple_complete_reads():
    state = {
        "complete": True,
        "workspace_version_fingerprint": "workspace-sha",
        "resource_version_fingerprints": {"foo.py": "file-sha"},
    }
    metadata = build_bash_observation_metadata(
        command="sed -n '1,20p' foo.py",
        cwd="/workspace/repo",
        raw_output="line\n",
        return_code=0,
        exception_info="",
        pre_state=state,
        post_state=state,
        environment_fingerprint="env-sha",
    )
    assert metadata["tool_semantics"]["provenance"] == "runtime_traced"
    assert metadata["tool_semantics"]["complete"] is True
    assert metadata["return_code"] == 0
    assert metadata["command_sha256"] == hashlib.sha256(
        b"sed -n '1,20p' foo.py"
    ).hexdigest()
    assert metadata["tool_semantics"]["effects"] == [{
        "kind": "read",
        "resource_id": "foo.py",
        "resource_version_fingerprint": "file-sha",
    }]


def test_runtime_observation_metadata_fails_closed_for_compound_bash():
    state = {
        "complete": True,
        "workspace_version_fingerprint": "workspace-sha",
        "resource_version_fingerprints": {"foo.py": "file-sha"},
    }
    metadata = build_bash_observation_metadata(
        command="cat foo.py | grep bug",
        cwd="/workspace/repo",
        raw_output="bug\n",
        return_code=0,
        exception_info="",
        pre_state=state,
        post_state=state,
        environment_fingerprint="env-sha",
    )
    assert metadata["tool_semantics"]["complete"] is False
    assert metadata["tool_semantics"]["unknown_barrier"] is True
    assert metadata["tool_semantics"]["provenance"] == "static_heuristic"


def test_runtime_observation_metadata_fails_closed_for_truncated_output():
    state = {
        "complete": True,
        "workspace_version_fingerprint": "workspace-sha",
        "resource_version_fingerprints": {"foo.py": "file-sha"},
    }
    metadata = build_bash_observation_metadata(
        command="cat foo.py",
        cwd="/workspace/repo",
        raw_output="x" * 10_000,
        return_code=0,
        exception_info="",
        pre_state=state,
        post_state=state,
        environment_fingerprint="env-sha",
    )
    assert metadata["output_truncated"] is True
    assert metadata["tool_semantics"]["complete"] is False


def test_repository_state_fingerprints_untracked_file_contents(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "paper8.5@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Paper 8.5 Test"),
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=repository, check=True)
    untracked = repository / "observation.txt"
    untracked.write_text("first\n", encoding="utf-8")
    first = repository_state(repository)
    untracked.write_text("second\n", encoding="utf-8")
    second = repository_state(repository)
    assert first["workspace_version_fingerprint"] != second["workspace_version_fingerprint"]


def test_checkpoint_restore_preserves_index_worktree_and_untracked_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=source, check=True)
    subprocess.run(
        ("git", "config", "user.email", "paper8.5@example.invalid"),
        cwd=source,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Paper 8.5 Test"),
        cwd=source,
        check=True,
    )
    tracked = source / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=source, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=source, check=True)
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=source).decode().strip()
    tracked.write_text("staged\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=source, check=True)
    tracked.write_text("worktree\n", encoding="utf-8")
    (source / "new.txt").write_text("untracked\n", encoding="utf-8")

    index_path = tmp_path / "decision.index.patch"
    worktree_path = tmp_path / "decision.worktree.patch"
    archive_path = tmp_path / "decision.untracked.tar.gz"
    index_path.write_bytes(subprocess.check_output(
        ("git", "diff", "--cached", "--binary", "--full-index"), cwd=source
    ))
    worktree_path.write_bytes(subprocess.check_output(
        ("git", "diff", "--binary", "--full-index"), cwd=source
    ))
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source / "new.txt", arcname="new.txt")
    expected = repository_state(source)
    receipt_path = tmp_path / "decision.json"
    receipt_path.write_text(json.dumps({
        "checkpoint_complete": True,
        "head": head,
        "workspace_version_fingerprint": expected["workspace_version_fingerprint"],
        "index_patch": index_path.name,
        "index_patch_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "worktree_patch": worktree_path.name,
        "worktree_patch_sha256": hashlib.sha256(worktree_path.read_bytes()).hexdigest(),
        "untracked_archive": archive_path.name,
        "untracked_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }), encoding="utf-8")

    target = tmp_path / "target"
    subprocess.run(("git", "clone", "-q", str(source), str(target)), check=True)
    restored = restore_repository_checkpoint(receipt_path, target)
    assert restored == expected
    assert subprocess.check_output(
        ("git", "diff", "--cached", "--name-only"), cwd=target
    ).decode().strip() == "tracked.txt"
    assert (target / "tracked.txt").read_text(encoding="utf-8") == "worktree\n"
    assert (target / "new.txt").read_text(encoding="utf-8") == "untracked\n"


def test_dag_marks_write_invalidation_as_diagnostic_not_safe_exclusion():
    messages = _messages(1)
    messages.extend((
        {
            "role": "assistant",
            "content": (
                "THOUGHT: mutate\n\n```mswea_bash_command\n"
                "sed -i 's/a/b/' foo.py\n```"
            ),
        },
        {"role": "user", "content": "<returncode>0</returncode>\n<output></output>"},
    ))
    history = recordize_minisweagent_messages(messages)
    dag = build_resource_effect_dag(history)
    invalidations = [
        row for row in dag.exclusion_candidates
        if row.classification == ExclusionClass.INVALIDATED_BY_WRITE
    ]
    assert invalidations
    assert not any(row.default_exclusion_eligible for row in invalidations)
    assert not dag.certified_excluded_groups


def _heuristic_messages(*turns):
    messages = [
        {"role": "system", "content": "Use one bash command."},
        {"role": "user", "content": "Fix the reported issue."},
    ]
    for command, output, extra in turns:
        messages.extend((
            {
                "role": "assistant",
                "content": f"THOUGHT: act\n```mswea_bash_command\n{command}\n```",
            },
            {
                "role": "user",
                "content": f"<returncode>0</returncode>\n<output>{output}</output>",
                "extra": {"returncode": 0, **(extra or {})},
            },
        ))
    return messages


def _negative_config(*rules, **overrides):
    values = {
        "rules": rules,
        "protected_head_turns": 0,
        "protected_tail_turns": 0,
    }
    values.update(overrides)
    return NegativeSelectionConfig(**values)


def test_negative_operation_parser_separates_discovery_from_source_grep():
    assert classify_bash_operation("find . -name '*.py'") == BashOperation.SEARCH_DISCOVERY
    assert classify_bash_operation("rg --files src") == BashOperation.SEARCH_DISCOVERY
    assert classify_bash_operation("grep -Rl needle src") == BashOperation.SEARCH_DISCOVERY
    assert classify_bash_operation("grep -n needle src/foo.py") == BashOperation.READ
    assert classify_bash_operation("rg -n needle src/foo.py") == BashOperation.READ


def test_h1_retires_consumed_search_only_after_kf_delay():
    messages = _heuristic_messages(
        ("find . -name foo.py", "./src/foo.py", None),
        ("cat src/foo.py", "source", None),
        ("pwd", "/workspace", None),
        ("echo done", "done", None),
    )
    history = recordize_minisweagent_messages(messages)
    immediate = build_negative_exclusions(
        history, _negative_config(NegativeRule.H1_SEARCH_CONSUMED, search_delay_turns=0)
    )
    delayed = build_negative_exclusions(
        history, _negative_config(NegativeRule.H1_SEARCH_CONSUMED, search_delay_turns=1)
    )
    too_late = build_negative_exclusions(
        history, _negative_config(NegativeRule.H1_SEARCH_CONSUMED, search_delay_turns=2)
    )
    assert [row.causal_group_id for row in immediate] == ["turn:t0000"]
    assert [row.causal_group_id for row in delayed] == ["turn:t0000"]
    assert not too_late


def test_strict_h1_abstains_when_any_discovered_branch_is_unresolved():
    complete = {
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    history = recordize_minisweagent_messages(_heuristic_messages(
        ("find . -name '*.py'", "./a.py\n./b.py", complete),
        ("cat a.py", "a", complete),
        ("pwd", "/workspace", complete),
    ))
    rows = build_negative_exclusions(
        history,
        _negative_config(NegativeRule.H1_ALL_BRANCHES_CONSUMED),
    )
    assert rows == ()


def test_strict_h1_fires_only_after_all_reads_and_non_read_transition():
    assert "h1_all_branches_consumed_strict" in DEFAULT_POLICIES
    complete = {
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    without_transition = recordize_minisweagent_messages(_heuristic_messages(
        ("find . -name '*.py'", "./a.py\n./b.py", complete),
        ("cat a.py", "a", complete),
        ("git diff -- b.py", "b diff", complete),
    ))
    assert build_negative_exclusions(
        without_transition,
        _negative_config(NegativeRule.H1_ALL_BRANCHES_CONSUMED),
    ) == ()

    history = recordize_minisweagent_messages(_heuristic_messages(
        ("find . -name '*.py'", "./a.py\n./b.py", complete),
        ("cat a.py", "a", complete),
        ("git diff -- b.py", "b diff", complete),
        ("pytest", "1 passed", complete),
    ))
    rows = build_negative_exclusions(
        history,
        _negative_config(
            NegativeRule.H1_ALL_BRANCHES_CONSUMED,
            search_delay_turns=0,
        ),
    )
    assert [row.causal_group_id for row in rows] == ["turn:t0000"]
    assert rows[0].classification == "strict_all_branch_supersession_not_certificate"
    assert set(rows[0].resource_ids) >= {"a.py", "b.py"}
    assert set(rows[0].witness_record_ids) >= {"m2", "m4", "m6", "m8"}

    delayed = build_negative_exclusions(
        history,
        _negative_config(
            NegativeRule.H1_ALL_BRANCHES_CONSUMED,
            search_delay_turns=1,
        ),
    )
    assert delayed == ()

    still_exploring = recordize_minisweagent_messages(_heuristic_messages(
        ("find . -name foo.py", "./src/foo.py", None),
        ("cat src/foo.py", "source", None),
        ("grep -n needle src/foo.py", "1:needle", None),
    ))
    assert not build_negative_exclusions(
        still_exploring,
        _negative_config(NegativeRule.H1_SEARCH_CONSUMED, search_delay_turns=0),
    )

    truncated_search = recordize_minisweagent_messages(_heuristic_messages(
        (
            "find . -name foo.py",
            "./src/foo.py",
            {"output_complete": False, "output_truncated": True},
        ),
        ("cat src/foo.py", "source", None),
        ("pwd", "/workspace", None),
    ))
    assert not build_negative_exclusions(
        truncated_search,
        _negative_config(NegativeRule.H1_SEARCH_CONSUMED, search_delay_turns=0),
    )


def test_h2a_requires_current_version_read_and_h2b_explicit_verification_dependency():
    write_extra = {
        "post_resource_version_fingerprints": {"foo.py": "v2"},
    }
    read_extra = {"resource_version_fingerprints": {"foo.py": "v2"}}
    verify_extra = {"verification_resource_ids": ["foo.py"]}
    history = recordize_minisweagent_messages(_heuristic_messages(
        (
            "python -c \"from pathlib import Path; Path('foo.py').write_text('fixed')\"",
            "", write_extra,
        ),
        ("cat foo.py", "fixed", read_extra),
        ("pytest tests/test_foo.py", "1 passed", verify_extra),
    ))
    h2a = build_negative_exclusions(
        history, _negative_config(NegativeRule.H2A_WRITE_CURRENT_READ)
    )
    h2b = build_negative_exclusions(
        history, _negative_config(NegativeRule.H2B_VERIFIED_WRITE)
    )
    assert h2a[0].rule_id == "H2A_WRITE_CURRENT_READ"
    assert h2b[0].rule_id == "H2B_VERIFIED_WRITE"
    assert h2a[0].causal_group_id == h2b[0].causal_group_id == "turn:t0000"

    mismatched = recordize_minisweagent_messages(_heuristic_messages(
        (
            "python -c \"from pathlib import Path; Path('foo.py').write_text('fixed')\"",
            "", write_extra,
        ),
        ("cat foo.py", "stale", {"resource_version_fingerprints": {"foo.py": "v1"}}),
    ))
    assert not build_negative_exclusions(
        mismatched, _negative_config(NegativeRule.H2A_WRITE_CURRENT_READ)
    )


def test_h2_bare_is_explicitly_aggressive_and_obeys_kw():
    history = recordize_minisweagent_messages(_heuristic_messages(
        ("echo fixed > foo.py", "", None),
        ("pwd", "/workspace", None),
    ))
    rows = build_negative_exclusions(
        history,
        _negative_config(NegativeRule.H2_BARE_AGGRESSIVE, write_delay_turns=1),
    )
    assert rows[0].classification == "aggressive_unsafe_ablation"
    assert "does not imply" in rows[0].reason


def test_h1_observation_receipt_keeps_action_and_replaces_only_discovery_output():
    complete = {
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
        "cwd": "/workspace",
    }
    noise = " ".join(f"irrelevant_{index}" for index in range(200))
    history = recordize_minisweagent_messages(_heuristic_messages(
        ("find . -name foo.py", f"./src/foo.py\n{noise}", complete),
        (
            "cat src/foo.py",
            "source",
            {**complete, "resource_version_fingerprints": {"src/foo.py": "v1"}},
        ),
        ("pwd", "/workspace", complete),
    ))
    plan = NegativeHeuristicSelector(_negative_config(
        NegativeRule.H1_SEARCH_CONSUMED,
    )).select(
        history=history,
        query="fix",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    base = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(),
        query="fix",
    )
    realized = realize_negative_receipts(history, plan, base)
    messages = serialize_materialized_messages(history, realized.materialized)

    assert len(realized.receipts) == 1
    assert realized.receipts[0].kind.value == "resource_map"
    assert realized.receipts[0].observation_token_saving > 0
    assert realized.materialized.logical_plan.selected_record_ids == tuple(
        row.record_id for row in history.records
    )
    assert realized.materialized.logical_plan.selected_tokens == sum(
        whitespace_tokens(row.content) for row in history.records
    )
    assert messages[2]["content"] == history.record_by_id["m2"].content
    assert "[PRA memory] search consumed" in messages[3]["content"]
    assert "mapped resources: src/foo.py" in messages[3]["content"]
    assert "output complete" in messages[3]["content"]
    assert "cwd: /workspace" in messages[3]["content"]
    assert "irrelevant_199" not in messages[3]["content"]
    validate_minisweagent_chat(messages)


def test_protocol_stub_keeps_semantic_evidence_out_of_model_visible_text():
    complete = {"output_complete": True, "timed_out": False, "output_truncated": False}
    history = recordize_minisweagent_messages(_heuristic_messages(
        ("cat foo.py", "old " * 100, {
            **complete, "resource_version_fingerprints": {"foo.py": "v1"},
        }),
        ("cat foo.py", "new", {
            **complete, "resource_version_fingerprints": {"foo.py": "v1"},
        }),
    ))
    plan = NegativeHeuristicSelector(_negative_config(
        NegativeRule.H3_READ_SUPERSEDED, protected_head_turns=0,
    )).select(
        history=history, query="fix", budget=AgentMemoryBudget(max_tokens=10_000),
    )
    base = materialize_plan(history, plan, ToolObservationMaterializer(), query="fix")

    realized = realize_negative_receipts(
        history, plan, base, include_semantic_evidence=False,
    )
    messages = serialize_materialized_messages(history, realized.materialized)

    assert messages[3]["content"] == "<returncode>0</returncode>\n<output></output>"
    assert realized.receipts[0].kind.value == "superseded_read"
    assert realized.receipts[0].witness_record_ids
    validate_minisweagent_chat(messages)


def test_h2_observation_receipt_exposes_version_evidence_without_mutation_output():
    complete = {
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    mutation_noise = " ".join(f"write_detail_{index}" for index in range(120))
    history = recordize_minisweagent_messages(_heuristic_messages(
        (
            "python -c \"from pathlib import Path; Path('foo.py').write_text('fixed')\"",
            mutation_noise,
            {
                **complete,
                "post_resource_version_fingerprints": {"foo.py": "v2"},
            },
        ),
        (
            "cat foo.py",
            "fixed",
            {
                **complete,
                "resource_version_fingerprints": {"foo.py": "v2"},
            },
        ),
    ))
    plan = NegativeHeuristicSelector(_negative_config(
        NegativeRule.H2A_WRITE_CURRENT_READ,
    )).select(
        history=history,
        query="fix foo.py",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    base = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(),
        query="fix foo.py",
    )
    realized = realize_negative_receipts(history, plan, base)
    messages = serialize_materialized_messages(history, realized.materialized)

    receipt = realized.receipts[0]
    assert receipt.kind.value == "mutation"
    assert receipt.observation_token_saving > 0
    assert "write_detail_119" not in receipt.content
    assert "current read confirms: foo.py@v2" in receipt.content
    assert "output complete" in receipt.content
    assert messages[2]["content"] == history.record_by_id["m2"].content
    validate_minisweagent_chat(messages)


def test_h2b_observation_receipt_names_successful_verification_evidence():
    complete = {
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    history = recordize_minisweagent_messages(_heuristic_messages(
        (
            "echo fixed > foo.py",
            " ".join(f"write_detail_{index}" for index in range(100)),
            {
                **complete,
                "post_resource_version_fingerprints": {"foo.py": "v2"},
            },
        ),
        (
            "pytest tests/test_foo.py",
            "1 passed",
            {**complete, "verification_resource_ids": ["foo.py"]},
        ),
    ))
    plan = NegativeHeuristicSelector(_negative_config(
        NegativeRule.H2B_VERIFIED_WRITE,
    )).select(
        history=history,
        query="fix foo.py",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    base = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(),
        query="fix foo.py",
    )
    realized = realize_negative_receipts(history, plan, base)

    assert len(realized.receipts) == 1
    assert "verification passed for: foo.py@v2" in realized.receipts[0].content
    assert set(realized.receipts[0].witness_record_ids) >= {"m2", "m4"}


def test_observation_receipt_fails_closed_on_nonstandard_multi_observation_group():
    complete = {
        "returncode": 0,
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    messages = [
        {"role": "system", "content": "Use one bash command."},
        {"role": "user", "content": "Fix the reported issue."},
        {
            "role": "assistant",
            "content": (
                "THOUGHT: locate\n```mswea_bash_command\n"
                "find . -name foo.py\n```"
            ),
        },
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>./src/foo.py</output>",
            "extra": complete,
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode>\n<output>map complete</output>",
            "extra": complete,
        },
        {
            "role": "assistant",
            "content": "THOUGHT: read\n```mswea_bash_command\ncat src/foo.py\n```",
        },
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>source</output>",
            "extra": complete,
        },
        {
            "role": "assistant",
            "content": "THOUGHT: move on\n```mswea_bash_command\npwd\n```",
        },
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>/workspace</output>",
            "extra": complete,
        },
    ]
    history = recordize_minisweagent_messages(messages)
    plan = NegativeHeuristicSelector(_negative_config(
        NegativeRule.H1_SEARCH_CONSUMED,
    )).select(
        history=history,
        query="fix",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    base = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(),
        query="fix",
    )
    realized = realize_negative_receipts(history, plan, base)

    assert realized.receipts == ()
    assert realized.fail_closed_group_ids == ("turn:t0000",)
    assert realized.abstentions[0].reason.value == "malformed_or_prefix_invalid"
    assert realized.abstentions[0].source_observation_tokens is None
    serialized = serialize_materialized_messages(history, realized.materialized)
    assert serialized[2:5] == [
        {"role": message["role"], "content": message["content"]}
        for message in messages[2:5]
    ]
    validate_minisweagent_chat(serialized)


@pytest.mark.parametrize(
    ("rule", "turns"),
    (
        (
            NegativeRule.H1_SEARCH_CONSUMED,
            (
                ("find . -name foo.py", "./src/foo.py", None),
                ("cat src/foo.py", "source", None),
                ("pwd", "/workspace", None),
            ),
        ),
        (
            NegativeRule.H2B_VERIFIED_WRITE,
            (
                (
                    "echo fixed > foo.py",
                    "saved",
                    {"post_resource_version_fingerprints": {"foo.py": "v2"}},
                ),
                (
                    "pytest tests/test_foo.py",
                    "1 passed",
                    {"verification_resource_ids": ["foo.py"]},
                ),
            ),
        ),
    ),
)
def test_observation_receipt_abstains_when_not_strictly_smaller(rule, turns):
    history = recordize_minisweagent_messages(_heuristic_messages(*turns))
    plan = NegativeHeuristicSelector(_negative_config(rule)).select(
        history=history,
        query="fix foo.py",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    base = materialize_plan(
        history,
        plan,
        ToolObservationMaterializer(),
        query="fix foo.py",
    )
    realized = realize_negative_receipts(history, plan, base)

    assert realized.receipts == ()
    assert len(realized.abstentions) == 1
    abstention = realized.abstentions[0]
    assert abstention.reason.value == "not_smaller_than_source_observation"
    assert abstention.candidate_receipt_tokens >= abstention.source_observation_tokens
    assert realized.abstained_candidate_receipt_tokens == (
        abstention.candidate_receipt_tokens
    )
    assert realized.abstained_source_observation_tokens == (
        abstention.source_observation_tokens
    )
    assert realized.materialized.materialized_tokens == sum(
        whitespace_tokens(row.content) for row in history.records
    )
    assert all(
        row.mode == MaterializationMode.WHOLE_RECORD
        for row in realized.materialized.records
    )


def test_h3_keeps_kr_newest_reads_for_the_same_resource_version_and_span():
    extra = {"resource_version_fingerprints": {"foo.py": "v1"}}
    history = recordize_minisweagent_messages(_heuristic_messages(
        ("sed -n '1,10p' foo.py", "first", extra),
        ("sed -n '1,10p' foo.py", "second", extra),
        ("sed -n '1,10p' foo.py", "third", extra),
    ))
    rows = build_negative_exclusions(
        history,
        _negative_config(NegativeRule.H3_READ_SUPERSEDED, same_span_reads_to_keep=2),
    )
    assert [row.causal_group_id for row in rows] == ["turn:t0000"]

    different_span = recordize_minisweagent_messages(_heuristic_messages(
        ("sed -n '1,10p' foo.py", "first", extra),
        ("sed -n '20,30p' foo.py", "second", extra),
    ))
    assert not build_negative_exclusions(
        different_span,
        _negative_config(NegativeRule.H3_READ_SUPERSEDED, same_span_reads_to_keep=1),
    )


def test_h3_scopes_identical_paths_by_runtime_environment():
    common = {
        "cwd": "/testbed",
        "output_complete": True,
        "resource_version_fingerprints": {"setup.cfg": "same-bytes"},
    }
    cross_workspace = recordize_minisweagent_messages(_heuristic_messages(
        ("cat setup.cfg", "first", {**common, "environment_fingerprint": "env-a"}),
        ("cat setup.cfg", "second", {**common, "environment_fingerprint": "env-b"}),
    ))
    assert not build_negative_exclusions(
        cross_workspace,
        _negative_config(NegativeRule.H3_READ_SUPERSEDED, same_span_reads_to_keep=1),
    )

    same_workspace = recordize_minisweagent_messages(_heuristic_messages(
        ("cat setup.cfg", "first", {**common, "environment_fingerprint": "env-a"}),
        ("cat setup.cfg", "second", {**common, "environment_fingerprint": "env-a"}),
    ))
    rows = build_negative_exclusions(
        same_workspace,
        _negative_config(NegativeRule.H3_READ_SUPERSEDED, same_span_reads_to_keep=1),
    )
    assert [row.causal_group_id for row in rows] == ["turn:t0000"]
    assert rows[0].resource_ids == (
        "env=env-a|cwd=/testbed::setup.cfg",
    )

    incomplete = recordize_minisweagent_messages(_heuristic_messages(
        ("cat foo.py", "partial", {
            "resource_version_fingerprints": {"foo.py": "v1"},
            "output_complete": False,
            "output_truncated": True,
        }),
        ("cat foo.py", "complete", {
            "resource_version_fingerprints": {"foo.py": "v1"},
            "output_complete": True,
        }),
    ))
    assert not build_negative_exclusions(
        incomplete,
        _negative_config(NegativeRule.H3_READ_SUPERSEDED, same_span_reads_to_keep=1),
    )


def test_h4_is_dependency_pinned_and_reacquisition_proxy_compares_with_full():
    history = recordize_minisweagent_messages(_heuristic_messages(
        ("cat a.py", "a", None),
        ("cat b.py", "b", None),
        ("cat c.py", "c", None),
        ("cat d.py", "d", None),
    ))
    selector = NegativeHeuristicSelector(_negative_config(
        NegativeRule.H4_WORKING_SET, working_set_resources=2
    ))
    plan = selector.select(
        history=history,
        query="fix",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    assert {row.causal_group_id for row in plan.exclusions} == {
        "turn:t0000", "turn:t0001"
    }
    assert plan.selected_tokens + sum(row.excluded_tokens for row in plan.exclusions) == (
        plan.full_history_tokens
    )
    assert plan.mandatory_tokens < plan.selected_tokens
    assert plan.head_turns == plan.tail_turns == 0
    assert reacquired_excluded_resources("cat a.py", plan.exclusions) == ("a.py",)
    assert reacquired_excluded_resources("pytest", plan.exclusions) == ()
    assert all("INACTIVE group=" in row.tombstone for row in plan.exclusions)
    serialized = serialize_materialized_messages(
        history,
        materialize_plan(
            history,
            plan,
            ToolObservationMaterializer(),
            query="fix",
        ),
    )
    validate_minisweagent_chat(serialized)

    pinned_messages = _heuristic_messages(
        ("cat a.py", "a", None),
        ("cat b.py", "b", None),
        ("cat c.py", "c", None),
        ("cat d.py", "d", None),
    )
    pinned_messages[1]["content"] = "Fix a.py."
    pinned = build_negative_exclusions(
        recordize_minisweagent_messages(pinned_messages),
        _negative_config(NegativeRule.H4_WORKING_SET, working_set_resources=2),
    )
    assert {row.causal_group_id for row in pinned} == {"turn:t0001"}


def test_oracle_cases_ablate_whole_bundles_without_task_leakage():
    history = recordize_minisweagent_messages(_messages(3))
    full = FullHistorySelector().select(
        history=history,
        query="foo",
        budget=AgentMemoryBudget(max_tokens=100_000),
    )
    cases = leave_one_bundle_out_cases(
        history=history,
        full_plan=full,
        decision_turn=3,
        reference_action_digest="abc",
    )
    assert len(cases) == 3
    for case in cases:
        assert "m0" in case.selected_record_ids
        assert "m1" in case.selected_record_ids
        omitted = case.omitted_causal_group_ids[0]
        assert all(
            history.record_by_id[rid].causal_group_id != omitted
            for rid in case.selected_record_ids
        )


def test_structural_screen_aggregates_without_claiming_task_quality(tmp_path):
    trajectory = tmp_path / "task.traj.json"
    trajectory.write_text(json.dumps({
        "instance_id": "task-1",
        "messages": _messages(2),
        "info": {"exit_status": "Submitted", "submission": "diff --git a/foo.py"},
    }), encoding="utf-8")
    result = structural_screen(
        [trajectory],
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        heads=(1,),
        tails=(1,),
        budgets=(0.9,),
        decision_suffixes=(1, 2),
    )
    assert result["evidence_class"] == "structural_only_not_task_quality"
    assert result["decision_row_count"] == 2 * len(STRUCTURAL_POLICIES)
    assert len(result["summary_rows"]) == len(STRUCTURAL_POLICIES)
    assert {row["decision_suffix_start"] for row in result["suffix_summary_rows"]} == {
        1, 2
    }
    assert "decision_rows" not in result


def test_structural_screen_can_match_autonomous_retention_floor(tmp_path):
    trajectory = tmp_path / "task.traj.json"
    trajectory.write_text(json.dumps({
        "instance_id": "task-1",
        "messages": _messages(2),
        "info": {"exit_status": "Submitted", "submission": "diff --git a/foo.py"},
    }), encoding="utf-8")
    result = structural_screen(
        [trajectory],
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        heads=(1,),
        tails=(1,),
        budgets=(0.9,),
        round_up_to_budget=True,
        policies=("middle_recency",),
    )
    assert result["whole_turn_budget_interpretation"] == "retention_floor_round_up"
    assert result["policies"] == ["middle_recency"]
    assert len(result["summary_rows"]) == 1
    recency = next(
        row for row in result["summary_rows"]
        if row["policy"] == "middle_recency"
    )
    assert recency["aggregate_realized_retention_fraction"] >= 0.9


def test_negative_structural_screen_reports_opportunity_without_quality_claim(tmp_path):
    trajectory = tmp_path / "task.traj.json"
    trajectory.write_text(json.dumps({
        "instance_id": "task-1",
        "messages": _heuristic_messages(
            ("cat a.py", "a", None),
            ("cat b.py", "b", None),
            ("cat c.py", "c", None),
            ("cat d.py", "d", None),
        ),
    }), encoding="utf-8")
    result = negative_structural_screen(
        [trajectory],
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        policies=("h4_working_set",),
        working_sets=(2,),
        protected_head_turns=0,
        protected_tail_turns=0,
        decision_suffixes=(1, 3),
        include_decision_rows=True,
    )
    assert result["evidence_class"] == (
        "structural_opportunity_only_not_behavior_or_task_quality"
    )
    assert result["reacquisition_metrics_status"].startswith("not_measured")
    assert result["summary_rows"][0]["excluded_tokens"] > 0
    assert result["decision_rows"][-1]["inactive_tombstones"]
    assert result["schema_version"] == 2
    assert {row["decision_suffix_start"] for row in result["suffix_summary_rows"]} == {
        1, 3
    }
    late = next(
        row for row in result["suffix_summary_rows"]
        if row["decision_suffix_start"] == 3
    )
    assert late["decision_count"] == 2


def test_negative_combination_sweeps_all_relevant_large_k_parameters(tmp_path):
    trajectory = tmp_path / "task.traj.json"
    trajectory.write_text(json.dumps({
        "instance_id": "task-1",
        "messages": _heuristic_messages(
            ("find . -name '*.py'", "./a.py", None),
            ("cat a.py", "a", None),
            ("cat b.py", "b", None),
            ("cat c.py", "c", None),
        ),
    }), encoding="utf-8")
    result = negative_structural_screen(
        [trajectory],
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        policies=("all_h1_h2a_h2b_h3_h4",),
        search_delays=(0, 8),
        read_keeps=(1, 6),
        working_sets=(2, 8),
        protected_head_turns=0,
        protected_tail_turns=0,
    )
    assert len(result["summary_rows"]) == 8


def test_frozen_replay_uses_reference_only_after_selected_request(monkeypatch):
    messages = _messages(2)
    references = [messages[2]["content"], messages[4]["content"]]
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload["messages"])
        content = references[len(calls) - 1]
        return {"choices": [{"message": {"content": content}}], "usage": {}}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(
        trajectory={"instance_id": "task-1", "messages": messages},
        model="test-model",
        base_url="http://example.invalid",
        policy="full",
        head=1,
        tail=1,
        budget_fraction=1.0,
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        materialization_threshold_tokens=20,
        max_decisions=None,
        seed=0,
        max_output_tokens=128,
        api_key=None,
        timeout=1,
    )
    assert result["exact_command_rate"] == 1.0
    assert len(calls) == 2
    assert references[0] not in [row["content"] for row in calls[0]]
    assert calls[1][-1]["role"] == "user"
    assert result["rows"][0]["request_messages_sha256"] == frozen_replay._json_digest(
        calls[0]
    )


def test_frozen_replay_can_target_a_late_decision_suffix(monkeypatch):
    messages = _messages(4)
    references = [messages[6]["content"], messages[8]["content"]]
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload["messages"])
        return {"choices": [{"message": {"content": references[len(calls) - 1]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(
        trajectory={"instance_id": "task-1", "messages": messages},
        model="test-model",
        base_url="http://example.invalid",
        policy="full",
        head=1,
        tail=1,
        budget_fraction=1.0,
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        materialization_threshold_tokens=20,
        max_decisions=None,
        seed=0,
        max_output_tokens=128,
        api_key=None,
        timeout=1,
        min_decision=3,
    )
    assert [row["decision"] for row in result["rows"]] == [3, 4]
    assert result["completed_decisions"] == 2
    assert len(calls) == 2


def test_frozen_replay_limits_after_selecting_late_decision_suffix(monkeypatch):
    messages = _messages(4)
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload["messages"])
        return {"choices": [{"message": {"content": messages[6]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(
        trajectory={"instance_id": "task-1", "messages": messages},
        model="test-model",
        base_url="http://example.invalid",
        policy="full",
        head=1,
        tail=1,
        budget_fraction=1.0,
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        materialization_threshold_tokens=20,
        max_decisions=1,
        seed=0,
        max_output_tokens=128,
        api_key=None,
        timeout=1,
        min_decision=3,
    )
    assert [row["decision"] for row in result["rows"]] == [3]
    assert len(calls) == 1


def test_frozen_replay_records_policy_excess_immediate_reacquisition(monkeypatch):
    messages = _heuristic_messages(
        ("cat a.py", "a", None),
        ("cat b.py", "b", None),
        ("cat c.py", "c", None),
        ("cat d.py", "d", None),
    )
    calls = 0

    def fake_post(url, payload, *, api_key, timeout):
        nonlocal calls
        calls += 1
        content = (
            messages[calls * 2]["content"]
            if calls < 4 else
            "THOUGHT: reacquire\n```mswea_bash_command\ncat a.py\n```"
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(**_replay_arguments(
        messages,
        progress_path=None,
        policy="h4_working_set",
        head=0,
        tail=0,
        working_set_resources=2,
    ))
    row = result["rows"][-1]
    assert row["excluded_group_count"] == 1
    assert row["generated_reacquired_excluded_resource_ids"] == ("a.py",)
    assert row["full_reference_reacquired_excluded_resource_ids"] == ()
    assert row["false_exclusion_immediate_reacquisition_proxy"] is True
    assert row["exclusions"][0]["tombstone_in_model_request"] is False


def test_frozen_replay_oracle_addback_restores_complete_causal_group(monkeypatch):
    messages = _heuristic_messages(
        ("cat a.py", "a", None),
        ("cat b.py", "b", None),
        ("cat c.py", "c", None),
        ("cat d.py", "d", None),
    )
    requests = []

    def fake_post(url, payload, *, api_key, timeout):
        requests.append(payload["messages"])
        return {"choices": [{"message": {"content": messages[8]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(**_replay_arguments(
        messages,
        progress_path=None,
        policy="h4_working_set",
        head=0,
        tail=0,
        working_set_resources=2,
        min_decision=4,
        max_decisions=1,
        oracle_addback_causal_group_ids=("turn:t0000",),
    ))

    row = result["rows"][0]
    audit = row["oracle_addback"]
    assert audit["restored_causal_group_ids"] == ("turn:t0000",)
    assert audit["restored_record_ids"] == ("m2", "m3")
    assert audit["restored_tokens"] > 0
    assert row["excluded_group_count"] == 0
    prompt = "\n".join(message["content"] for message in requests[0])
    assert "cat a.py" in prompt
    assert result["run_configuration"]["oracle_addback_causal_group_ids"] == (
        "turn:t0000",
    )


def test_oracle_addback_rejects_matched_token_tail():
    with pytest.raises(ValueError, match="incompatible with matched_token_tail"):
        frozen_replay.replay(**_replay_arguments(
            _messages(1),
            progress_path=None,
            policy="matched_token_tail",
            matched_budget_replay={"rows": [{
                "decision": 1,
                "materialized_tokens": 20,
            }]},
            oracle_addback_causal_group_ids=("turn:t0000",),
        ))


def test_frozen_replay_oracle_omits_one_complete_middle_group(monkeypatch):
    messages = _heuristic_messages(
        ("cat a.py", "a", None),
        ("cat b.py", "b", None),
        ("cat c.py", "c", None),
        ("cat d.py", "d", None),
    )
    requests = []

    def fake_post(url, payload, *, api_key, timeout):
        requests.append(payload["messages"])
        return {"choices": [{"message": {"content": messages[8]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(**_replay_arguments(
        messages,
        progress_path=None,
        policy="full",
        head=0,
        tail=1,
        min_decision=4,
        max_decisions=1,
        oracle_omit_causal_group_ids=("turn:t0000",),
    ))

    row = result["rows"][0]
    audit = row["oracle_omission"]
    assert audit["omitted_causal_group_ids"] == ("turn:t0000",)
    assert audit["omitted_record_ids"] == ("m2", "m3")
    assert audit["omitted_tokens"] > 0
    prompt = "\n".join(message["content"] for message in requests[0])
    assert "cat a.py" not in prompt
    assert "cat c.py" in prompt
    assert result["run_configuration"]["oracle_omit_causal_group_ids"] == (
        "turn:t0000",
    )


def test_oracle_omission_rejects_a_protected_tail_group():
    with pytest.raises(ValueError, match="intersects immutable/protected"):
        frozen_replay.replay(**_replay_arguments(
            _messages(3),
            progress_path=None,
            policy="full",
            head=0,
            tail=1,
            min_decision=3,
            max_decisions=1,
            oracle_omit_causal_group_ids=("turn:t0001",),
        ))


def test_frozen_replay_observation_receipt_is_prefix_only_and_reports_separate_tokens(
    monkeypatch,
):
    complete = {
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    messages = _heuristic_messages(
        (
            "find . -name foo.py",
            "./src/foo.py\n" + " ".join(f"noise_{index}" for index in range(150)),
            complete,
        ),
        ("cat src/foo.py", "source", complete),
        ("pwd", "/workspace", complete),
    )
    future_reference = (
        "THOUGHT: FUTURE_REFERENCE_SECRET\n"
        "```mswea_bash_command\necho done\n```"
    )
    messages.append({"role": "assistant", "content": future_reference})
    requests = []

    def fake_post(url, payload, *, api_key, timeout):
        requests.append(payload["messages"])
        return {"choices": [{"message": {"content": future_reference}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(**_replay_arguments(
        messages,
        progress_path=None,
        policy="h1_search_consumed",
        head=0,
        tail=0,
        min_decision=4,
        max_decisions=1,
        negative_realization=NegativeRealizationMode.OBSERVATION_RECEIPT,
    ))

    row = result["rows"][0]
    prompt = "\n".join(message["content"] for message in requests[0])
    assert "FUTURE_REFERENCE_SECRET" not in prompt
    assert "[PRA memory] search consumed" in prompt
    assert "noise_149" not in prompt
    assert row["negative_realization"] == "observation_receipt"
    assert row["negative_candidate_group_count"] == 1
    assert row["excluded_group_count"] == 0
    assert row["receipt_count"] == 1
    assert row["receipt_tokens"] > 0
    assert row["receipt_source_observation_tokens"] > row["receipt_tokens"]
    assert row["selected_logical_tokens"] == row["full_history_tokens"]
    assert row["materialized_tokens"] < row["selected_logical_tokens"]
    assert row["model_visible_receipts"][0]["kind"] == "resource_map"
    assert row["exclusions"][0]["realization"] == "observation_receipt"
    assert result["run_configuration"]["negative_selection"]["realization"] == (
        "observation_receipt"
    )


def test_frozen_replay_observation_receipt_reports_size_gate_abstention(monkeypatch):
    messages = _heuristic_messages(
        ("find . -name foo.py", "./src/foo.py", None),
        ("cat src/foo.py", "source", None),
        ("pwd", "/workspace", None),
    )
    reference = "THOUGHT: done\n```mswea_bash_command\necho done\n```"
    messages.append({"role": "assistant", "content": reference})
    requests = []

    def fake_post(url, payload, *, api_key, timeout):
        requests.append(payload["messages"])
        return {"choices": [{"message": {"content": reference}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(**_replay_arguments(
        messages,
        progress_path=None,
        policy="h1_search_consumed",
        head=0,
        tail=0,
        min_decision=4,
        max_decisions=1,
        negative_realization=NegativeRealizationMode.OBSERVATION_RECEIPT,
    ))

    row = result["rows"][0]
    prompt = "\n".join(message["content"] for message in requests[0])
    assert "[PRA memory]" not in prompt
    assert "./src/foo.py" in prompt
    assert row["receipt_count"] == 0
    assert row["receipt_fail_closed_group_ids"] == ("turn:t0000",)
    assert row["receipt_abstentions"] == [{
        "causal_group_id": "turn:t0000",
        "source_record_ids": ("m2", "m3"),
        "rule_id": "H1_SEARCH_CONSUMED",
        "reason": "not_smaller_than_source_observation",
        "source_observation_tokens": row[
            "receipt_abstained_source_observation_tokens"
        ],
        "candidate_receipt_tokens": row["receipt_abstained_candidate_tokens"],
    }]
    assert row["receipt_abstained_candidate_tokens"] >= (
        row["receipt_abstained_source_observation_tokens"]
    )
    assert row["selected_logical_tokens"] == row["full_history_tokens"]
    assert row["materialized_tokens"] == row["full_history_tokens"]
    assert row["excluded_group_count"] == 0
    assert row["exclusions"][0]["realization"] == (
        "fail_closed:not_smaller_than_source_observation"
    )


def test_frozen_replay_rejects_receipt_mode_for_nonnegative_policy():
    with pytest.raises(ValueError, match="requires a negative-selection policy"):
        frozen_replay.replay(**_replay_arguments(
            _messages(1),
            progress_path=None,
            negative_realization=NegativeRealizationMode.OBSERVATION_RECEIPT,
        ))


@pytest.mark.parametrize(
    "base_url",
    (
        "http://example.invalid",
        "http://example.invalid/",
        "http://example.invalid/v1/chat/completions",
        "http://example.invalid/v1/chat/completions/",
    ),
)
def test_frozen_replay_normalizes_root_or_complete_chat_endpoint(
    monkeypatch, base_url
):
    messages = _messages(1)
    endpoints = []

    def fake_post(url, payload, *, api_key, timeout):
        endpoints.append(url)
        return {"choices": [{"message": {"content": messages[2]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    frozen_replay.replay(**_replay_arguments(
        messages,
        progress_path=None,
        base_url=base_url,
    ))

    assert endpoints == ["http://example.invalid/v1/chat/completions"]


def test_frozen_replay_can_use_contemporaneous_full_output_as_reference(monkeypatch):
    messages = _messages(1)
    generated = messages[2]["content"].replace("inspect turn 0", "new reasoning")

    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": generated}}], "usage": {}}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    full_reference = {"rows": [{"decision": 1, "generated_content": generated}]}
    result = frozen_replay.replay(
        trajectory={"instance_id": "task-1", "messages": messages},
        model="test-model",
        base_url="http://example.invalid",
        policy="full",
        head=1,
        tail=1,
        budget_fraction=1.0,
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        materialization_threshold_tokens=20,
        max_decisions=None,
        seed=0,
        max_output_tokens=128,
        api_key=None,
        timeout=1,
        reference_replay=full_reference,
    )
    assert result["comparison_reference"] == "contemporaneous_full_replay"
    assert result["rows"][0]["exact_content"] is True
    assert result["rows"][0]["historical_reference_content_sha256"] != result["rows"][0]["generated_content_sha256"]


def test_frozen_replay_accepts_per_decision_matched_token_ceiling(monkeypatch):
    messages = _messages(1)

    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": messages[2]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    matched = {"rows": [{"decision": 1, "selected_whole_record_tokens": 7}]}
    result = frozen_replay.replay(
        trajectory={"instance_id": "task-1", "messages": messages},
        model="test-model",
        base_url="http://example.invalid",
        policy="middle_recency",
        head=0,
        tail=0,
        budget_fraction=1.0,
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        materialization_threshold_tokens=20,
        max_decisions=None,
        seed=0,
        max_output_tokens=128,
        api_key=None,
        timeout=1,
        matched_budget_replay=matched,
    )
    assert result["rows"][0]["requested_budget_tokens"] == 7
    assert result["matched_budget_source_digest"] is not None


def test_frozen_replay_matched_token_tail_uses_materialized_source_ceiling(monkeypatch):
    messages = _messages(1)

    def fake_post(url, payload, *, api_key, timeout):
        validate_minisweagent_chat(payload["messages"])
        return {"choices": [{"message": {"content": messages[2]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    matched = {"rows": [{
        "decision": 1,
        "selected_whole_record_tokens": 999,
        "materialized_tokens": 20,
    }]}
    result = frozen_replay.replay(
        trajectory={"instance_id": "task-1", "messages": messages},
        model="test-model",
        base_url="http://example.invalid",
        policy="matched_token_tail",
        head=0,
        tail=0,
        budget_fraction=1.0,
        count_tokens=whitespace_tokens,
        tokenizer_identity="test",
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        materialization_threshold_tokens=20,
        max_decisions=None,
        seed=0,
        max_output_tokens=128,
        api_key=None,
        timeout=1,
        matched_budget_replay=matched,
    )
    row = result["rows"][0]
    assert result["materialization_mode"] == "matched_token_tail"
    assert result["run_configuration"]["matched_budget_field"] == "materialized_tokens"
    assert row["requested_budget_tokens"] == 20
    assert row["selected_logical_tokens"] == row["selected_whole_record_tokens"]
    assert row["materialized_tokens"] <= 20
    assert row["materialized_token_ceiling"] == 20
    assert row["materialized_budget_overshoot_tokens"] == 0


def test_frozen_replay_matched_token_tail_compacts_boundary_observation(monkeypatch):
    messages = _messages(2)
    messages[3]["content"] = (
        "<returncode>0</returncode>\n<output>\n"
        + "\n".join(f"source line {index}" for index in range(100))
        + "\n</output>"
    )
    prefix = recordize_minisweagent_messages(messages[:4])
    fixed = sum(
        whitespace_tokens(prefix.record_by_id[record_id].content)
        for record_id in ("m0", "m1", "m2")
    )
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload["messages"])
        return {
            "choices": [{"message": {"content": messages[len(calls) * 2]["content"]}}]
        }

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(
        **_replay_arguments(
            messages,
            progress_path=None,
            policy="matched_token_tail",
            materialization_threshold_tokens=10,
            matched_budget_replay={"rows": [
                {"decision": 1, "materialized_tokens": 20},
                {"decision": 2, "materialized_tokens": fixed + 15},
            ]},
        )
    )

    assert len(calls) == 2
    assert "<elided_prefix_lines" in calls[1][-1]["content"]
    assert result["rows"][1]["selected_logical_tokens"] > result["rows"][1][
        "materialized_tokens"
    ]
    assert result["rows"][1]["materialized_budget_overshoot_tokens"] == 0


def test_frozen_replay_matched_token_tail_requires_materialized_source_counts():
    with pytest.raises(ValueError, match="lacks materialized_tokens"):
        frozen_replay.replay(**_replay_arguments(
            _messages(1),
            progress_path=None,
            policy="matched_token_tail",
            matched_budget_replay={
                "rows": [{"decision": 1, "selected_whole_record_tokens": 20}]
            },
        ))


def _replay_arguments(messages, progress_path, **overrides):
    arguments = {
        "trajectory": {"instance_id": "task-1", "messages": messages},
        "model": "test-model",
        "base_url": "http://example.invalid",
        "policy": "full",
        "head": 1,
        "tail": 1,
        "budget_fraction": 1.0,
        "count_tokens": whitespace_tokens,
        "tokenizer_identity": "test-tokenizer",
        "materialization_mode": MaterializationMode.WHOLE_RECORD,
        "materialization_threshold_tokens": 20,
        "max_decisions": None,
        "seed": 0,
        "max_output_tokens": 128,
        "api_key": None,
        "timeout": 1,
        "progress_path": progress_path,
    }
    arguments.update(overrides)
    return arguments


def test_frozen_replay_resumes_without_requerying_completed_decisions(
    monkeypatch, tmp_path
):
    messages = _messages(2)
    progress = tmp_path / "full.json"
    queried_prefix_lengths = []

    def interrupted_post(url, payload, *, api_key, timeout):
        queried_prefix_lengths.append(len(payload["messages"]))
        if len(queried_prefix_lengths) == 2:
            raise ConnectionError("endpoint disappeared")
        return {"choices": [{"message": {"content": messages[2]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", interrupted_post)
    with pytest.raises(ConnectionError, match="endpoint disappeared"):
        frozen_replay.replay(**_replay_arguments(messages, progress))

    saved = json.loads(progress.read_text(encoding="utf-8"))
    assert saved["attempted_decisions"] == saved["completed_decisions"] == 1
    assert saved["rows"][0]["decision_status"] == "completed"
    assert not list(tmp_path.glob(".full.json.*.tmp"))

    def resumed_post(url, payload, *, api_key, timeout):
        queried_prefix_lengths.append(len(payload["messages"]))
        return {"choices": [{"message": {"content": messages[4]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", resumed_post)
    result = frozen_replay.replay(**_replay_arguments(messages, progress))
    assert queried_prefix_lengths == [2, 4, 4]
    assert result["completed_decisions"] == 2
    assert result["run_configuration_digest"]


def test_frozen_replay_resume_rejects_configuration_and_reference_changes(
    monkeypatch, tmp_path
):
    messages = _messages(1)
    progress = tmp_path / "full.json"
    calls = 0

    def fake_post(url, payload, *, api_key, timeout):
        nonlocal calls
        calls += 1
        return {"choices": [{"message": {"content": messages[2]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    frozen_replay.replay(**_replay_arguments(messages, progress))

    with pytest.raises(ValueError, match="run configuration does not match"):
        frozen_replay.replay(**_replay_arguments(
            messages, progress, model="different-model"
        ))
    with pytest.raises(ValueError, match="reference_replay_digest does not match"):
        frozen_replay.replay(**_replay_arguments(
            messages,
            progress,
            reference_replay={
                "rows": [{"decision": 1, "generated_content": messages[2]["content"]}]
            },
        ))
    assert calls == 1


def test_frozen_replay_records_format_error_and_continues(
    monkeypatch, tmp_path
):
    messages = _messages(2)
    progress = tmp_path / "full.json"
    calls = 0

    def truncated_post(url, payload, *, api_key, timeout):
        nonlocal calls
        calls += 1
        content = messages[2]["content"] if calls == 1 else "THOUGHT: truncated"
        return {
            "id": f"response-{calls}",
            "model": "test-model",
            "choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }

    monkeypatch.setattr(frozen_replay, "_post", truncated_post)
    result = frozen_replay.replay(**_replay_arguments(messages, progress))

    saved = json.loads(progress.read_text(encoding="utf-8"))
    assert saved["attempted_decisions"] == 2
    assert saved["completed_decisions"] == 2
    assert saved["terminal_failure"] is None
    failed = saved["rows"][1]
    assert failed["decision_status"] == "completed"
    assert failed["action_valid"] is False
    assert failed["action_validation_reason"] == "missing_generated_command"
    assert failed["response_diagnostics"]["finish_reason"] == "length"
    assert failed["response_diagnostics"]["response_id"] == "response-2"
    assert failed["response_diagnostics"]["message_keys"] == ["content", "role"]
    assert result["exact_command_rate"] == 0.5

    resumed = frozen_replay.replay(**_replay_arguments(messages, progress))
    assert resumed["completed_decisions"] == 2
    assert calls == 2


def test_frozen_replay_command_rate_excludes_invalid_reference_actions(monkeypatch):
    messages = _messages(2)
    reference = {
        "rows": [
            {"decision": 1, "generated_content": messages[2]["content"]},
            {"decision": 2, "generated_content": "THOUGHT: no command"},
        ]
    }
    generated = iter((messages[2]["content"], "THOUGHT: also no command"))

    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": next(generated)}}]}

    monkeypatch.setattr(frozen_replay, "_post", fake_post)
    result = frozen_replay.replay(
        **_replay_arguments(messages, None), reference_replay=reference
    )
    assert result["reference_command_decisions"] == 1
    assert result["exact_command_decisions"] == 1
    assert result["exact_command_rate"] == 1.0
    assert result["first_command_divergence"] is None


def test_frozen_replay_stops_on_empty_transport_generation_and_requires_restart(
    monkeypatch, tmp_path
):
    messages = _messages(2)
    progress = tmp_path / "full.json"
    calls = 0

    def empty_post(url, payload, *, api_key, timeout):
        nonlocal calls
        calls += 1
        content = messages[2]["content"] if calls == 1 else ""
        return {
            "id": f"response-{calls}",
            "model": "test-model" if content else "",
            "choices": [{
                "finish_reason": "stop" if content else None,
                "message": {"role": "assistant" if content else "", "content": content},
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3} if content else {
                "prompt_tokens": 0, "completion_tokens": 0,
            },
        }

    monkeypatch.setattr(frozen_replay, "_post", empty_post)
    with pytest.raises(frozen_replay.ReplayGenerationError, match="decision 2 stopped"):
        frozen_replay.replay(**_replay_arguments(messages, progress))

    saved = json.loads(progress.read_text(encoding="utf-8"))
    assert saved["attempted_decisions"] == 2
    assert saved["completed_decisions"] == 1
    assert saved["terminal_failure"] == {
        "decision": 2,
        "reason": "empty_generated_content",
    }
    failed = saved["rows"][1]
    assert failed["decision_status"] == "failed_transport_generation"
    assert failed["action_valid"] is False

    with pytest.raises(ValueError, match="terminal failure at decision 2"):
        frozen_replay.replay(**_replay_arguments(messages, progress))
    assert calls == 2

    def healthy_post(url, payload, *, api_key, timeout):
        decision = len(payload["messages"]) // 2
        return {"choices": [{"message": {"content": messages[decision * 2]["content"]}}]}

    monkeypatch.setattr(frozen_replay, "_post", healthy_post)
    restarted = frozen_replay.replay(**_replay_arguments(
        messages, progress, restart=True
    ))
    assert restarted["completed_decisions"] == 2
    assert restarted["terminal_failure"] is None


def test_review_export_keeps_full_json_and_compacts_markdown(tmp_path):
    messages = _messages(1)
    messages[3]["content"] = (
        "<returncode>0</returncode>\n<output>\n"
        + "\n".join(f"unique-output-line-{index}" for index in range(30))
        + "\n</output>"
    )
    trajectory = tmp_path / "task.traj.json"
    trajectory.write_text(json.dumps({
        "instance_id": "owner__task-1",
        "messages": messages,
        "info": {"exit_status": "Submitted"},
    }), encoding="utf-8")

    json_path, markdown_path = export_trajectory(
        trajectory,
        tmp_path / "review",
        head_lines=2,
        tail_lines=2,
        max_line_chars=80,
    )

    exported = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    observation = next(
        row for row in exported["records"]
        if "tool_observation" in row["semantic_roles"]
    )
    assert "unique-output-line-15" in observation["content"]
    assert "unique-output-line-15" not in markdown
    assert "lines omitted" in markdown
    assert observation["content_sha256"] in markdown
    assert f"[{json_path.name}]({json_path.name})" in markdown


def test_structured_evidence_materializer_keeps_middle_failure_not_noise():
    noise = "\n".join(f"unrelated passing test {index}" for index in range(40))
    marker = "E   AssertionError: calculate_total expected 19.80 but got 19.79"
    messages = _heuristic_messages((
        "pytest -q tests/test_cart.py -vv",
        f"{noise}\n{marker}\nsrc/cart.py:87: calculate_total\n{noise}",
        {
            "output_complete": True,
            "verification_resource_ids": ["src/cart.py"],
        },
    ))
    messages[1]["content"] = "Fix calculate_total rounding in src/cart.py."
    history = recordize_minisweagent_messages(messages)
    full = FullHistorySelector().select(
        history=history,
        query=messages[1]["content"],
        budget=AgentMemoryBudget(max_tokens=100_000),
    )

    structured = materialize_plan(
        history,
        full,
        ToolObservationMaterializer(
            mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
            threshold_tokens=1,
            head_lines=2,
            tail_lines=2,
            match_context_lines=1,
            max_matched_lines=6,
        ),
        query=messages[1]["content"],
    )
    compact = next(
        row.content for row in structured.records if row.record_id == "m3"
    )
    assert marker in compact
    assert "src/cart.py:87" in compact
    assert "unrelated passing test 20" not in compact
    assert structured.materialized_tokens < structured.full_selected_tokens


def test_structured_evidence_materializer_fails_closed_without_anchors():
    messages = _heuristic_messages((
        "printf opaque",
        "alpha\nbeta\ngamma\ndelta",
        {"output_complete": True},
    ))
    history = recordize_minisweagent_messages(messages)
    full = FullHistorySelector().select(
        history=history,
        query="unrelated query",
        budget=AgentMemoryBudget(max_tokens=100_000),
    )
    materialized = materialize_plan(
        history,
        full,
        ToolObservationMaterializer(
            mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
            threshold_tokens=1,
        ),
        query="unrelated query",
    )
    observation = next(
        row for row in materialized.records if row.record_id == "m3"
    )
    assert observation.mode == MaterializationMode.WHOLE_RECORD
    assert observation.content == history.record_by_id["m3"].content


def test_synthetic_diagnostics_separate_rule_activation_from_model_quality():
    result = run_synthetic_diagnostics()
    assert result["evidence_class"] == (
        "synthetic_structural_only_not_model_or_task_quality"
    )
    assert result["task_count"] == 3
    assert result["all_mechanism_gates_passed"] is True
    structured = next(
        row for row in result["tasks"]
        if row["task_id"] == "synthetic_structured_failure_evidence"
    )
    modes = {row["mode"]: row for row in structured["materialization_rows"]}
    assert modes["tool_head_tail"]["marker_retained"] is False
    assert modes["tool_structured_evidence"]["marker_retained"] is True
