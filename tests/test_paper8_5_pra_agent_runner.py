from __future__ import annotations

import pytest

from experiments.paper8_5_agent_memory.run_pra_agent_swebench import (
    DockerWorkspaceTools,
    build_parser,
)
from experiments.paper8_5_agent_memory.run_pra_agent_persistent_swebench import (
    _task_spec,
    build_parser as build_persistent_parser,
)
from experiments.paper8_5_agent_memory.pra_agent_policy import (
    PRAAgentInstructionEpochSelector,
    PRAAgentMatchedTailSelector,
    recordize_pra_agent_records,
)
from pra_hf.context_records import ContextRecord, RecordType


def test_docker_workspace_rejects_absolute_and_parent_paths() -> None:
    assert DockerWorkspaceTools._relative_path("django/db/models.py") == (
        "django/db/models.py"
    )
    assert DockerWorkspaceTools._relative_path("/testbed") == "."
    assert DockerWorkspaceTools._relative_path(
        "/testbed/django/db/models.py"
    ) == "django/db/models.py"
    with pytest.raises(PermissionError):
        DockerWorkspaceTools._relative_path("/etc/passwd")
    with pytest.raises(PermissionError):
        DockerWorkspaceTools._relative_path("../outside")


def test_docker_workspace_exposes_typed_portable_tools() -> None:
    workspace = DockerWorkspaceTools("docker", "task-container")
    toolset = workspace.toolset("paper8-5")

    assert {resource.name for resource in toolset.resources} == {
        "list_files",
        "read_file",
        "search_text",
        "git_status",
        "write_file",
        "replace_text",
        "run_command",
    }
    by_name = {resource.name: resource for resource in toolset.resources}
    assert by_name["read_file"].side_effect_class.value == "read"
    assert by_name["replace_text"].side_effect_class.value == "write"
    workspace.bind_container("next-task-container")
    assert workspace.container == "next-task-container"


def _message(record_id: str, role: str, text: str) -> ContextRecord:
    return ContextRecord(
        record_id,
        RecordType.GENERIC_TEXT,
        {"role": role, "text": text},
    )


def _observation(record_id: str, text: str) -> ContextRecord:
    return ContextRecord(
        record_id,
        RecordType.TOOL_RESPONSE,
        {"producer_tool_uri": "pra://tool/read_file", "compact": text},
    )


def test_recordize_pra_agent_records_preserves_typed_causal_pairs() -> None:
    rows = (
        _message("task", "user", "fix the issue"),
        _message("a1", "assistant", "read file"),
        _observation("o1", "source"),
        _message("a2", "assistant", "verify"),
        _observation("o2", "tests pass"),
    )

    history = recordize_pra_agent_records(rows)

    assert history.records[0].primary_role.value == "task"
    assert [turn.record_ids for turn in history.turns] == [
        ("a1", "o1"),
        ("a2", "o2"),
    ]
    assert all(turn.complete for turn in history.turns)


def test_recordize_completion_recovery_is_not_an_immutable_user_prompt() -> None:
    rows = (
        _message("task", "user", "fix the issue"),
        _message("premature", "assistant", "I will implement the fix."),
        _message(
            "guard",
            "user",
            "[Completion rejected: No tracked source patch exists.]",
        ),
    )

    history = recordize_pra_agent_records(rows)

    assert history.turns[0].record_ids == ("premature", "guard")
    assert history.turns[0].complete is True
    assert history.records[-1].primary_role.value == "error_or_rejection"


def test_instruction_epoch_selector_retires_only_old_closed_interactions() -> None:
    def final(record_id: str, text: str) -> ContextRecord:
        return ContextRecord(
            record_id,
            RecordType.GENERIC_TEXT,
            {
                "role": "assistant",
                "text": text,
                "pra_agent_semantic_role": "finalization",
            },
        )

    rows = (
        _message("task1", "user", "fix repository one"),
        _message("a1", "assistant", "inspect one"),
        _observation("o1", "one result"),
        final("f1", "task one complete"),
        _message("task2", "user", "fix repository two"),
        _message("a2", "assistant", "inspect two"),
        _observation("o2", "two result"),
        final("f2", "task two complete"),
        _message("task3", "user", "fix repository three"),
        _message("a3", "assistant", "inspect three"),
        _observation("o3", "three result"),
    )
    selector = PRAAgentInstructionEpochSelector(
        count_tokens=lambda text: len(text.split()),
        tokenizer_identity="unit-whitespace",
        prior_full_epochs=1,
    )

    selected = selector(rows, "continue task three")
    selected_ids = {row.record_id for row in selected}

    assert {"task1", "task2", "task3"}.issubset(selected_ids)
    assert {"a2", "o2", "f2", "a3", "o3"}.issubset(selected_ids)
    assert {"a1", "o1", "f1"}.isdisjoint(selected_ids)
    assert selector.traces[0]["policy"] == "instruction_epoch_e1"


def test_pra_agent_h2_t4_selector_drops_only_an_unprotected_whole_turn() -> None:
    rows = [_message("task", "user", "fix the issue")]
    for index in range(1, 8):
        rows.extend((
            _message(f"a{index}", "assistant", f"action {index}"),
            _observation(
                f"o{index}",
                ("large middle evidence " * 300) if index == 3 else f"result {index}",
            ),
        ))
    selector = PRAAgentMatchedTailSelector(
        retention_fraction=0.9,
        count_tokens=lambda text: len(text.split()),
        tokenizer_identity="unit-whitespace",
    )

    selected = selector(tuple(rows), "ignored")
    selected_ids = {row.record_id for row in selected}

    assert "task" in selected_ids
    assert {"a1", "o1", "a2", "o2"}.issubset(selected_ids)
    assert {"a4", "o4", "a5", "o5", "a6", "o6", "a7", "o7"}.issubset(
        selected_ids
    )
    assert {"a3", "o3"}.isdisjoint(selected_ids)
    assert selector.traces[0]["excluded_record_ids"] == ["a3", "o3"]


def test_pra_agent_percentage_never_overrides_short_history_floor() -> None:
    rows = (
        _message("task", "user", "long immutable task definition " * 100),
        _message("a1", "assistant", "action"),
        _observation("o1", "result"),
    )
    selector = PRAAgentMatchedTailSelector(
        retention_fraction=0.9,
        count_tokens=lambda text: len(text.split()),
        tokenizer_identity="unit-whitespace",
    )

    selected = selector(rows, "ignored")

    assert tuple(row.record_id for row in selected) == ("task", "a1", "o1")
    assert selector.traces[0]["mandatory_overflow_tokens"] > 0
    assert selector.traces[0]["materialized_history_tokens"] == (
        selector.traces[0]["full_history_tokens"]
    )


def test_pra_agent_runner_exposes_auditable_history_policy_controls() -> None:
    args = build_parser().parse_args([
        "--instance-id", "django__django-15277",
        "--reference-trajectory", "trajectory.json",
        "--output", "out",
        "--endpoint", "http://127.0.0.1:8000",
        "--history-policy", "matched_token_tail",
        "--retention-fraction", "0.9",
        "--protected-head-turns", "2",
        "--protected-tail-turns", "4",
        "--tokenizer", "tokenizer-path",
        "--tokenizer-revision", "frozen-revision",
    ])

    assert args.history_policy == "matched_token_tail"
    assert args.retention_fraction == 0.9
    assert (args.protected_head_turns, args.protected_tail_turns) == (2, 4)
    assert args.tokenizer_revision == "frozen-revision"


def test_persistent_runner_accepts_ordered_boundary_free_tasks() -> None:
    args = build_persistent_parser().parse_args([
        "--task", "django__django-15277=task1.json",
        "--task", "scikit-learn__scikit-learn-14087=task2.json",
        "--output", "out",
        "--endpoint", "http://127.0.0.1:8000",
        "--history-policy", "matched_token_tail",
    ])

    assert [row[0] for row in args.task] == [
        "django__django-15277",
        "scikit-learn__scikit-learn-14087",
    ]
    assert args.context_records == 4096
    assert _task_spec("django__django-15277=task.json")[0] == (
        "django__django-15277"
    )

    instruction = build_persistent_parser().parse_args([
        "--task", "django__django-15277=task1.json",
        "--output", "out",
        "--endpoint", "http://127.0.0.1:8000",
        "--history-policy", "instruction_epoch",
        "--prior-full-epochs", "2",
    ])
    assert instruction.prior_full_epochs == 2
