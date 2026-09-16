import copy

import pytest

from experiments.paper8_5_agent_memory.multi_issue_session import (
    BoundaryMode,
    SessionMode,
    compose_issue_session_schedule,
    compose_multi_issue_session,
)
from experiments.paper8_5_agent_memory.audit_policy_oracle_alignment import (
    audit_prefix,
)
from experiments.paper8_5_agent_memory.recordizer import (
    active_task_content,
    recordize_replay_messages,
)
from experiments.paper8_5_agent_memory.oracle import leave_one_bundle_out_cases
from experiments.paper8_5_agent_memory.run_structural_screen import _query
from experiments.paper8_5_agent_memory.model import AgentMemoryBudget, AgentRecordRole
from experiments.paper8_5_agent_memory.matched_token_tail import (
    matched_token_tail_mandatory_record_ids,
    materialize_matched_token_tail,
)
from experiments.paper8_5_agent_memory.selectors import (
    FullHistorySelector,
    HeadMiddleTailConfig,
    HeadMiddleTailSelector,
    MiddleSelectionStrategy,
    PersistentEpisodeRetirementSelector,
    PersistentEpisodeRetirementConfig,
    PersistentGlobalRetirementConfig,
    PersistentGlobalRetirementSelector,
    PersistentInstructionEpochRetirementConfig,
    PersistentInstructionEpochRetirementSelector,
    immutable_instruction_record_ids,
    whitespace_tokens,
)


def _trajectory(instance_id: str, command: str) -> dict:
    return {
        "instance_id": instance_id,
        "info": {"exit_status": "Submitted", "submission": "diff --git a/a b/a"},
        "messages": [
            {"role": "system", "content": "Use bash."},
            {"role": "user", "content": f"Fix {instance_id}."},
            {"role": "assistant", "content": f"```mswea_bash_command\n{command}\n```"},
            {"role": "user", "content": "<returncode>0</returncode>\n<output>ok</output>"},
            {"role": "assistant", "content": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"},
            {"role": "exit", "content": "diff --git a/a b/a"},
        ],
    }


def test_composer_builds_typed_multi_issue_session_with_unique_identities():
    first = _trajectory("repo__issue-1", "cat a.py")
    second = _trajectory("repo__issue-2", "cat b.py")
    before = copy.deepcopy((first, second))

    result = compose_multi_issue_session((first, second))
    history = recordize_replay_messages(result["messages"])

    assert (first, second) == before
    assert result["issue_count"] == 2
    assert result["reference_success"] is True
    assert [row["role"] for row in result["messages"]].count("system") == 1
    assert all(row["role"] != "exit" for row in result["messages"])
    terminal_observations = [
        row for row in result["messages"]
        if "finalization" in row["metadata"]["pra_record"]["semantic_roles"]
        and row["role"] == "user"
    ]
    assert len(terminal_observations) == 2
    assert len(history.records) == len(result["messages"])
    assert len({row.record_id for row in history.records}) == len(history.records)
    tasks = [row for row in history.records if row.primary_role.value == "task"]
    assert len(tasks) == 2
    assert tasks[0].metadata["episode_status"] == "completed"
    assert tasks[1].metadata["episode_status"] == "active"
    assert "pra_episode_boundary" in tasks[1].content
    assert result["episodes"][1]["first_assistant_decision"] == 3
    assert result["episodes"][1]["last_assistant_decision"] == 4
    assert active_task_content(result["messages"]).endswith("Fix repo__issue-2.")
    assert "repo__issue-2" in _query(result["messages"])
    assert "repo__issue-1" not in _query(result["messages"])


def test_composer_rejects_duplicate_issue_identity():
    trajectory = _trajectory("repo__issue-1", "true")
    with pytest.raises(ValueError, match="distinct non-empty"):
        compose_multi_issue_session((trajectory, trajectory))


def test_boundary_free_composer_hides_episode_identity_from_history():
    result = compose_multi_issue_session(
        (
            _trajectory("repo__issue-1", "cat a.py"),
            _trajectory("repo__issue-2", "cat b.py"),
        ),
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    history = recordize_replay_messages(result["messages"])

    assert result["boundary_mode"] == "boundary_free"
    assert len(result["episodes"]) == 2  # private evaluator ledger
    assert "pra_episode_boundary" not in "\n".join(
        str(row.get("content") or "") for row in result["messages"]
    )
    assert all("pra_episode" not in (row.get("metadata") or {}) for row in result["messages"])
    assert all("episode-" not in row.record_id for row in history.records)
    assert all(
        not {"episode_id", "episode_index", "episode_status", "workspace_scope"}
        .intersection(row.metadata)
        for row in history.records
    )
    assert [row.primary_role for row in history.records].count(AgentRecordRole.TASK) == 1
    assert [row.primary_role for row in history.records].count(AgentRecordRole.USER_INPUT) == 1


def test_boundary_free_global_policy_uses_only_continuous_stream_floors():
    result = compose_multi_issue_session(
        (
            _trajectory("repo__issue-1", "cat a.py"),
            _trajectory("repo__issue-2", "cat b.py"),
        ),
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    history = recordize_replay_messages(result["messages"])
    selector = PersistentGlobalRetirementSelector(
        PersistentGlobalRetirementConfig(
            recent_turns=1,
            mutation_turns=0,
            verification_turns=0,
            protocol_turns=0,
        )
    )
    selected = selector.select(
        history=history, query="ignored", budget=AgentMemoryBudget(max_tokens=100_000)
    )
    rows = [history.record_by_id[row] for row in selected.selected_record_ids]

    assert selected.policy == "persistent_global_retirement"
    assert sum(row.has_role(AgentRecordRole.USER_INPUT) for row in rows) == 1
    assert sum(row.has_role(AgentRecordRole.TASK) for row in rows) == 1
    assert {
        row.record_id for row in rows if row.has_role(AgentRecordRole.ASSISTANT_ACTION)
    } == {history.turns[-1].record_ids[0]}
    assert all("episode" not in reason for _, reason in selected.selection_reasons)
    assert sum(
        reason == "immutable_user_instruction"
        for _, reason in selected.selection_reasons
    ) == 2


def test_instruction_epoch_policy_keeps_active_epoch_and_retires_only_prior_detail():
    result = compose_multi_issue_session(
        (
            _trajectory("repo__issue-1", "cat a.py"),
            _trajectory("repo__issue-2", "cat b.py"),
            _trajectory("repo__issue-3", "cat c.py"),
        ),
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    history = recordize_replay_messages(result["messages"])
    selector = PersistentInstructionEpochRetirementSelector(
        PersistentInstructionEpochRetirementConfig()
    )
    selected = selector.select(
        history=history,
        query="ignored",
        budget=AgentMemoryBudget(max_tokens=100_000),
    )
    selected_ids = set(selected.selected_record_ids)
    records = history.record_by_id

    # System plus all three genuine prompts are immutable.
    assert immutable_instruction_record_ids(history) <= selected_ids
    # The complete active epoch is retained; assistant/tool groups in both
    # older epochs are retired atomically.
    active_instruction = max(
        index for index, record in enumerate(history.records)
        if record.has_role(AgentRecordRole.TASK)
        or record.has_role(AgentRecordRole.USER_INPUT)
    )
    assert all(
        record.record_id in selected_ids
        for record in history.records[active_instruction:]
    )
    assert all(
        all(record_id not in selected_ids for record_id in turn.record_ids)
        for turn in history.turns[:-2]
    )
    assert all(
        all(record_id in selected_ids for record_id in turn.record_ids)
        for turn in history.turns[-2:]
    )
    assert selected.policy == "persistent_instruction_epoch_retirement"
    assert selected.middle_candidate_turns == 4
    assert selected.middle_selected_turns == 0
    assert len(selected.exclusions) == 4
    assert all(
        "episode" not in reason
        for _, reason in selected.selection_reasons
    )
    # The old tool observations use transport role=user but do not become
    # instruction boundaries.
    assert any(
        records[record_id].has_role(AgentRecordRole.TOOL_OBSERVATION)
        for exclusion in selected.exclusions
        for record_id in exclusion.record_ids
    )


def test_instruction_epoch_policy_applies_prior_floors_per_epoch():
    result = compose_multi_issue_session(
        (
            _trajectory("repo__issue-1", "cat a.py"),
            _trajectory("repo__issue-2", "cat b.py"),
            _trajectory("repo__issue-3", "cat c.py"),
        ),
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    history = recordize_replay_messages(result["messages"])
    selected = PersistentInstructionEpochRetirementSelector(
        PersistentInstructionEpochRetirementConfig(prior_recent_turns=1)
    ).select(
        history=history,
        query="ignored",
        budget=AgentMemoryBudget(max_tokens=100_000),
    )
    selected_ids = set(selected.selected_record_ids)

    # One final causal group is retained from each earlier instruction epoch.
    assert all(
        all(record_id in selected_ids for record_id in history.turns[index].record_ids)
        for index in (1, 3)
    )
    assert all(
        all(record_id not in selected_ids for record_id in history.turns[index].record_ids)
        for index in (0, 2)
    )
    assert selected.middle_candidate_turns == 4
    assert selected.middle_selected_turns == 2


def test_instruction_epoch_finalization_floor_closes_each_pinned_prior_prompt():
    trajectories = (
        _trajectory("repo__issue-1", "cat a.py"),
        _trajectory("repo__issue-2", "cat b.py"),
        _trajectory("repo__issue-3", "cat c.py"),
    )
    result = compose_multi_issue_session(
        trajectories,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    history = recordize_replay_messages(result["messages"])
    selected = PersistentInstructionEpochRetirementSelector(
        PersistentInstructionEpochRetirementConfig(prior_finalization_turns=1)
    ).select(
        history=history,
        query="ignored",
        budget=AgentMemoryBudget(max_tokens=100_000),
    )
    selected_ids = set(selected.selected_record_ids)

    prior_finalization_turns = [
        turn for turn in history.turns[:-2]
        if any(
            history.record_by_id[record_id].has_role(AgentRecordRole.FINALIZATION)
            for record_id in turn.record_ids
        )
    ]
    assert len(prior_finalization_turns) == 2
    assert all(
        set(turn.record_ids).issubset(selected_ids)
        for turn in prior_finalization_turns
    )
    assert all(
        set(turn.record_ids).isdisjoint(selected_ids)
        for turn in history.turns[:-2]
        if turn not in prior_finalization_turns
    )


def test_oracle_alignment_audit_exposes_orphaned_e0_and_exact_e0_f1():
    trajectories = (
        _trajectory("repo__issue-1", "cat a.py"),
        _trajectory("repo__issue-2", "cat b.py"),
        _trajectory("repo__issue-3", "cat c.py"),
    )
    result = audit_prefix({
        "schema_version": 1,
        "session_id": "audit-test",
        "episodes": [{"trajectory": row} for row in trajectories],
    })
    policies = {row["policy"]: row for row in result["policies"]}

    assert policies["E0"]["active_interaction_excluded_tokens"] == 0
    assert policies["E0"]["orphaned_prior_instruction_epochs"] == 2
    assert policies["E0_F1"]["active_interaction_excluded_tokens"] == 0
    assert policies["E0_F1"]["orphaned_prior_instruction_epochs"] == 0
    assert policies["E0_F1"]["oracle_exclusion_precision"] == 1.0
    assert policies["E0_F1"]["oracle_exclusion_recall"] == 1.0


def test_every_selector_family_preserves_all_user_instructions():
    result = compose_multi_issue_session(
        (
            _trajectory("repo__issue-1", "cat a.py"),
            _trajectory("repo__issue-2", "cat b.py"),
        ),
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    history = recordize_replay_messages(result["messages"])
    immutable = immutable_instruction_record_ids(history)
    assert len(immutable) == 3  # system, initial task, and later user prompt

    selectors = (
        PersistentEpisodeRetirementSelector(PersistentEpisodeRetirementConfig(
            recent_turns=0,
            mutation_turns=0,
            verification_turns=0,
            keep_completed_task_statements=False,
        )),
        HeadMiddleTailSelector(HeadMiddleTailConfig(
            head_turns=0,
            tail_turns=1,
            middle_strategy=MiddleSelectionStrategy.NONE,
        )),
    )
    for selector in selectors:
        plan = selector.select(
            history=history,
            query="ignored",
            budget=AgentMemoryBudget(max_tokens=10_000),
        )
        assert immutable <= set(plan.selected_record_ids)

    mandatory = matched_token_tail_mandatory_record_ids(history)
    ceiling = sum(
        whitespace_tokens(history.record_by_id[record_id].content)
        for record_id in mandatory
    )
    matched = materialize_matched_token_tail(
        history,
        max_materialized_tokens=ceiling,
    )
    assert immutable <= set(matched.logical_plan.selected_record_ids)

    full = FullHistorySelector().select(
        history=history,
        query="ignored",
        budget=AgentMemoryBudget(max_tokens=10_000),
    )
    cases = leave_one_bundle_out_cases(
        history=history,
        full_plan=full,
        decision_turn=2,
        reference_action_digest="reference",
    )
    assert cases
    assert all(immutable <= set(case.selected_record_ids) for case in cases)


def test_replay_recordizer_rejects_mixed_typed_and_inferred_records():
    result = compose_multi_issue_session((
        _trajectory("repo__issue-1", "true"),
        _trajectory("repo__issue-2", "true"),
    ))
    del result["messages"][-1]["metadata"]
    with pytest.raises(ValueError, match="cannot mix typed and inferred"):
        recordize_replay_messages(result["messages"])


def test_episode_retirement_is_exact_for_one_issue_and_retires_old_detail():
    first = _trajectory("repo__issue-1", "cat a.py")
    first["messages"].insert(
        4,
        {"role": "assistant", "content": "```mswea_bash_command\nprintf x > a.py\n```"},
    )
    first["messages"].insert(
        5, {"role": "user", "content": "<returncode>0</returncode>\n<output>wrote</output>"}
    )
    first["messages"][4]["extra"] = {"operation_kind": "write"}
    one = compose_multi_issue_session((first,))
    one_history = recordize_replay_messages(one["messages"])
    budget = AgentMemoryBudget(max_tokens=100_000)
    full = FullHistorySelector().select(history=one_history, query="", budget=budget)
    selected = PersistentEpisodeRetirementSelector().select(
        history=one_history, query="", budget=budget
    )
    assert selected.selected_record_ids == full.selected_record_ids

    two = compose_multi_issue_session((first, _trajectory("repo__issue-2", "cat b.py")))
    history = recordize_replay_messages(two["messages"])
    selected = PersistentEpisodeRetirementSelector().select(
        history=history, query="", budget=budget
    )
    selected_ids = set(selected.selected_record_ids)
    active_ids = {
        row.record_id for row in history.records if row.metadata.get("episode_index") == 2
    }
    assert active_ids <= selected_ids
    assert "episode-01:m2" not in selected_ids
    assert any(
        reason == "completed_episode_progress_spine"
        for _, reason in selected.selection_reasons
    )


def test_same_locked_order_can_be_scheduled_fresh_or_persistent():
    trajectories = (
        _trajectory("repo__issue-1", "cat a.py"),
        _trajectory("repo__issue-2", "cat b.py"),
    )
    fresh = compose_issue_session_schedule(
        trajectories, mode=SessionMode.FRESH_PER_ISSUE
    )
    persistent = compose_issue_session_schedule(
        trajectories, mode=SessionMode.PERSISTENT
    )

    assert fresh["ordered_instance_ids"] == persistent["ordered_instance_ids"]
    assert fresh["session_count"] == 2
    assert persistent["session_count"] == 1
    assert [row["issue_count"] for row in fresh["sessions"]] == [1, 1]
    assert persistent["sessions"][0]["issue_count"] == 2
    assert len({row["session_id"] for row in fresh["sessions"]}) == 2


def test_active_episode_policy_retires_completed_tools_but_keeps_instructions():
    composed = compose_multi_issue_session((
        _trajectory("repo__issue-1", "cat a.py"),
        _trajectory("repo__issue-2", "cat b.py"),
    ))
    history = recordize_replay_messages(composed["messages"])
    selector = PersistentEpisodeRetirementSelector(
        PersistentEpisodeRetirementConfig(
            recent_turns=0,
            mutation_turns=0,
            verification_turns=0,
            keep_completed_task_statements=False,
        )
    )
    selected = selector.select(
        history=history, query="", budget=AgentMemoryBudget(max_tokens=100_000)
    )
    selected_records = [history.record_by_id[row] for row in selected.selected_record_ids]
    assert selected.policy == "persistent_active_episode"
    assert sum(row.has_role(AgentRecordRole.TASK) for row in selected_records) == 2
    assert all(
        row.has_role(AgentRecordRole.SYSTEM)
        or row.has_role(AgentRecordRole.TASK)
        or row.has_role(AgentRecordRole.USER_INPUT)
        or row.metadata.get("episode_index") == 2
        for row in selected_records
    )
    assert not any(
        row.metadata.get("episode_index") == 1
        and row.has_role(AgentRecordRole.ASSISTANT_ACTION)
        for row in selected_records
    )
