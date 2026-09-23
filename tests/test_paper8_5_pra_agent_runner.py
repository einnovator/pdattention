from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from pra_hf.agent_execution import ToolCall
from experiments.paper8_5_agent_memory.run_pra_agent_swebench import (
    DockerWorkspaceTools,
    PRA_SWE_AGENT_BEHAVIOR,
    SWEInspectionBudgetGuard,
    _durable_tool_events,
    build_parser,
)
from experiments.paper8_5_agent_memory.run_pra_agent_persistent_swebench import (
    _load_task_registry,
    _saving_fraction,
    _initial_task_description,
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
    assert "Do not repeat an identical tool call" in PRA_SWE_AGENT_BEHAVIOR


def test_workspace_file_listing_prunes_git_and_caps_payload(monkeypatch) -> None:
    workspace = DockerWorkspaceTools("docker", "task-container")
    observed: list[str] = []

    def fake_shell(command: str, **_kwargs):
        observed.append(command)
        payload = "".join(f"file-{index}\n" for index in range(501)).encode()
        return subprocess.CompletedProcess(("bash",), 0, payload, b"")

    monkeypatch.setattr(workspace, "_shell", fake_shell)

    result = workspace.list_files(".")

    assert "-path '*/.git' -prune" in observed[0]
    assert len(result["files"]) == 500
    assert result["truncated"] is True


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("sed -i 's/x/y/' source.py", "write"),
        ("pytest -q tests/test_source.py", "verify"),
        ("sed -n '1,20p' source.py", "read"),
    ),
)
def test_run_command_emits_durable_operation_kind(
    monkeypatch, command: str, expected: str,
) -> None:
    workspace = DockerWorkspaceTools("docker", "task-container")
    monkeypatch.setattr(
        workspace,
        "_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ("bash",), 0, b"ok\n", b""
        ),
    )

    result = workspace.run_command(command)

    assert result["operation_kind"] == expected


def test_inspection_budget_allows_mutation_and_rejects_more_discovery(
    monkeypatch,
) -> None:
    workspace = DockerWorkspaceTools("docker", "task-container")
    monkeypatch.setattr(workspace, "has_tracked_source_patch", lambda: False)
    guard = SWEInspectionBudgetGuard(workspace, max_inspections=2)
    executed = (object(), object())

    read_decision = guard(
        None, ToolCall("read_file", {"path": "a.py"}), None, executed
    )
    command_decision = guard(
        None,
        ToolCall("run_command", {"command": "git log --oneline"}),
        None,
        executed,
    )
    assert read_decision is not None
    assert "inspection_budget_exhausted" in read_decision.reason
    assert set(read_decision.suppress_tool_names) == {
        "list_files", "read_file", "search_text", "git_status",
    }
    assert command_decision is not None
    assert "inspection_budget_exhausted" in command_decision.reason
    assert guard(
        None,
        ToolCall("replace_text", {"path": "a.py", "old": "x", "new": "y"}),
        None,
        executed,
    ) is None
    assert guard(
        None,
        ToolCall("run_command", {"command": "sed -i 's/x/y/' a.py"}),
        None,
        executed,
    ) is None


@pytest.mark.parametrize(
    "path",
    (
        "test_lazy_format.py",
        "/testbed/tests/test_formats.py",
        "package-lock.json",
        "benchmarks/easy14.json",
    ),
)
def test_inspection_guard_rejects_protected_structured_writes(path: str) -> None:
    workspace = DockerWorkspaceTools("docker", "task-container")
    guard = SWEInspectionBudgetGuard(workspace, max_inspections=8)

    rejection = guard(
        None,
        ToolCall("write_file", {"path": path, "content": "scratch"}),
        None,
        (),
    )

    assert rejection is not None
    assert "protected_path" in rejection


def test_inspection_guard_allows_structured_source_write_before_budget() -> None:
    workspace = DockerWorkspaceTools("docker", "task-container")
    guard = SWEInspectionBudgetGuard(workspace, max_inspections=8)

    assert guard(
        None,
        ToolCall(
            "replace_text",
            {"path": "django/utils/formats.py", "old": "x", "new": "y"},
        ),
        None,
        (),
    ) is None


def test_durable_tool_events_survive_a_later_turn_failure() -> None:
    records = (
        _message("a1", "assistant", "inspect"),
        ContextRecord(
            "o1",
            RecordType.TOOL_RESPONSE,
            {
                "producer_tool_uri": "pra://tool/read_file",
                "call_id": "call-1",
                "compact": {"text": "source"},
            },
        ),
    )

    events = _durable_tool_events(records)

    assert len(events) == 1
    assert events[0]["record_id"] == "o1"
    assert events[0]["call_id"] == "call-1"


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


@pytest.mark.parametrize(
    ("operation_kind", "expected_role"),
    (
        ("read", "source_view"),
        ("search_discovery", "source_view"),
        ("write", "mutation"),
        ("verify", "verification"),
        ("diff", "verification"),
    ),
)
def test_recordize_uses_typed_shell_operation_semantics(
    operation_kind: str, expected_role: str,
) -> None:
    rows = (
        _message("task", "user", "fix the issue"),
        _message("action", "assistant", "run shell action"),
        ContextRecord(
            "observation",
            RecordType.TOOL_RESPONSE,
            {
                "producer_tool_uri": "pra://tool/run_command",
                "compact": {
                    "fields": {
                        "command": "opaque to policy",
                        "operation_kind": operation_kind,
                    }
                },
            },
        ),
    )

    history = recordize_pra_agent_records(rows)

    observation = next(
        row for row in history.records if row.record_id == "observation"
    )
    assert expected_role in {role.value for role in observation.semantic_roles}


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
    assert args.boundary_mode == "boundary_free"
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


def test_boundary_free_session_does_not_seed_stale_task_graph_state() -> None:
    assert _initial_task_description("boundary_free", "first issue") is None
    assert _initial_task_description("explicit_task", "first issue") == (
        "first issue"
    )
    with pytest.raises(ValueError, match="unknown boundary mode"):
        _initial_task_description("unknown", "first issue")


def test_persistent_runner_keeps_large_record_capacity_for_multi_task_runs() -> None:
    args = build_persistent_parser().parse_args([
        "--task", "django__django-15277=task1.json",
        "--output", "out",
        "--endpoint", "http://127.0.0.1:8000",
    ])

    assert args.context_records == 4096
    assert args.max_tool_rounds == 80


def test_persistent_prefix_saving_uses_cumulative_token_sums() -> None:
    assert _saving_fraction(1000 + 3000, 1000 + 1800) == pytest.approx(0.3)
    assert _saving_fraction(0, 0) == 0.0
    with pytest.raises(ValueError, match="negative"):
        _saving_fraction(-1, 0)


def test_persistent_task_registry_supports_canonical_n_prefix(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    registry = tmp_path / "tasks.json"
    registry.write_text(
        '{"tasks":['
        '{"instance_id":"task-1","reference_artifact":"one.json"},'
        '{"instance_id":"task-2","reference_artifact":"two.json"}'
        ']}',
        encoding="utf-8",
    )

    tasks = _load_task_registry(
        registry, repository=repository, task_count=1
    )

    assert tasks == [("task-1", (repository / "one.json").resolve())]


def test_easy14_v2_campaign_matches_locked_task_registry() -> None:
    root = Path(__file__).parents[1]
    config_root = root / "experiments" / "paper8_5_agent_memory" / "configs"
    campaign = json.loads(
        (config_root / "agent_transfer_easy14_v2.json").read_text(encoding="utf-8")
    )
    registry_path = config_root / campaign["task_source_registry"]
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)

    assert hashlib.sha256(registry_bytes).hexdigest() == (
        campaign["task_source_registry_sha256"]
    )
    assert campaign["ordered_instance_ids"] == [
        row["instance_id"] for row in registry["tasks"]
    ]
    profiles = campaign["model_identity"]["serving_profiles"]
    assert profiles["single_task"]["loaded_context_tokens"] == 65536
    assert profiles["persistent_n3"]["loaded_context_tokens"] == 131072
    assert campaign["persistent_session_contract"][
        "task_boundary_signal_to_selector"
    ] is False
