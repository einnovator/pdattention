import json
import hashlib
import subprocess
import tarfile
import pytest
import experiments.paper8_5_agent_memory.run_frozen_replay as frozen_replay

from experiments.paper8_5_agent_memory import (
    AgentMemoryBudget,
    BashOperation,
    DagCertifiedExclusionSelector,
    ExclusionClass,
    AgentRecordRole,
    FullHistorySelector,
    HeadMiddleTailConfig,
    HeadMiddleTailSelector,
    MaterializationMode,
    MatchedTokenTailConfig,
    MiddleSelectionStrategy,
    NegativeHeuristicSelector,
    NegativeRule,
    NegativeSelectionConfig,
    ToolObservationMaterializer,
    build_resource_effect_dag,
    build_negative_exclusions,
    classify_bash_operation,
    exclude_certified_groups,
    leave_one_bundle_out_cases,
    materialize_matched_token_tail,
    materialize_plan,
    recordize_minisweagent_messages,
    reacquired_excluded_resources,
    serialize_materialized_messages,
    validate_minisweagent_chat,
)
from experiments.paper8_5_agent_memory.observation_instrumentation import (
    build_bash_observation_metadata,
)
from experiments.paper8_5_agent_memory.export_review_history import (
    export_trajectory,
)
from experiments.paper8_5_agent_memory.run_structural_screen import structural_screen
from experiments.paper8_5_agent_memory.run_negative_heuristic_screen import (
    negative_structural_screen,
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
    prompt_tokens = sum(
        whitespace_tokens(history.record_by_id[record_id].content)
        for record_id in ("m0", "m1")
    )
    result = materialize_matched_token_tail(
        history,
        max_materialized_tokens=prompt_tokens + 1,
        config=MatchedTokenTailConfig(tool_observation_threshold_tokens=100),
    )
    assert result.logical_plan.selected_record_ids == ("m0", "m1")
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
    with pytest.raises(ValueError, match="immutable system/task"):
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
    )
    assert result["evidence_class"] == "structural_only_not_task_quality"
    assert result["decision_row_count"] == 14
    assert len(result["summary_rows"]) == 7
    assert "decision_rows" not in result


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
        include_decision_rows=True,
    )
    assert result["evidence_class"] == (
        "structural_opportunity_only_not_behavior_or_task_quality"
    )
    assert result["reacquisition_metrics_status"].startswith("not_measured")
    assert result["summary_rows"][0]["excluded_tokens"] > 0
    assert result["decision_rows"][-1]["inactive_tombstones"]


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
