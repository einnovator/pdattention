import json
import hashlib
import subprocess
import tarfile
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
from experiments.paper8_5_agent_memory.observation_instrumentation import (
    build_bash_observation_metadata,
)
from experiments.paper8_5_agent_memory.run_structural_screen import structural_screen
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
        "index_patch": str(index_path),
        "index_patch_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "worktree_patch": str(worktree_path),
        "worktree_patch_sha256": hashlib.sha256(worktree_path.read_bytes()).hexdigest(),
        "untracked_archive": str(archive_path),
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
