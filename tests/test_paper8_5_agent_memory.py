import json
import experiments.paper8_5_agent_memory.run_frozen_replay as frozen_replay

from experiments.paper8_5_agent_memory import (
    AgentMemoryBudget,
    DagCertifiedExclusionSelector,
    ExclusionClass,
    AgentRecordRole,
    FullHistorySelector,
    HeadMiddleTailConfig,
    HeadMiddleTailSelector,
    MaterializationMode,
    MiddleSelectionStrategy,
    ToolObservationMaterializer,
    build_resource_effect_dag,
    exclude_certified_groups,
    leave_one_bundle_out_cases,
    materialize_plan,
    recordize_minisweagent_messages,
    serialize_materialized_messages,
    validate_minisweagent_chat,
)
from experiments.paper8_5_agent_memory.run_structural_screen import structural_screen
from experiments.paper8_5_agent_memory.selectors import whitespace_tokens


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


def test_recordizer_preserves_action_observation_causal_bundles():
    history = recordize_minisweagent_messages(_messages(3))
    assert len(history.turns) == 3
    assert all(turn.complete and len(turn.record_ids) == 2 for turn in history.turns)
    mutation = history.record_by_id["m6"]
    observation = history.record_by_id["m7"]
    assert mutation.has_role(AgentRecordRole.MUTATION)
    assert observation.has_role(AgentRecordRole.MUTATION)
    assert mutation.causal_group_id == observation.causal_group_id


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
    )
    assert result["evidence_class"] == "structural_only_not_task_quality"
    assert result["decision_row_count"] == 14
    assert len(result["summary_rows"]) == 7
    assert "decision_rows" not in result


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
