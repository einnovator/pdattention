from __future__ import annotations

import json
from pathlib import Path

from experiments.paper8_5_agent_memory.run_openhands_swebench import (
    _event_summary,
    _prepare_build_context,
)
from experiments.paper8_5_agent_memory.openhands_receipts import (
    build_openhands_receipts,
    summarize_receipts,
)


def test_openhands_event_summary_separates_actions_and_semantic_errors(
    tmp_path: Path,
):
    events = tmp_path / "events.jsonl"
    rows = [
        {
            "type": "ActionEvent",
            "event": {
                "llm_response_id": "response-1",
                "action": {"kind": "TerminalAction"},
            },
        },
        {
            "type": "ObservationEvent",
            "event": {
                "observation": {
                    "kind": "TerminalObservation",
                    "is_error": False,
                    "exit_code": 2,
                },
            },
        },
        {
            "type": "ActionEvent",
            "event": {
                "llm_response_id": "response-2",
                "action": {"kind": "FinishAction"},
            },
        },
        {
            "type": "paper85_run_summary",
            "event_count": 3,
            "execution_status": "finished",
        },
    ]
    events.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = _event_summary(events)

    assert summary["action_count"] == 2
    assert summary["observation_count"] == 1
    assert summary["distinct_llm_response_count"] == 2
    assert summary["semantic_failure_observations"] == 1
    assert summary["action_counts"] == {
        "FinishAction": 1,
        "TerminalAction": 1,
    }
    assert summary["run_summary"]["execution_status"] == "finished"


def test_openhands_receipt_adapter_preserves_failed_outcome_and_fails_closed_shell(
    tmp_path: Path,
):
    events = tmp_path / "events.jsonl"
    rows = [
        {
            "type": "ActionEvent",
            "event": {
                "id": "action-shell",
                "tool_call_id": "call-shell",
                "action": {"kind": "TerminalAction", "command": "pytest"},
            },
        },
        {
            "type": "ObservationEvent",
            "event": {
                "id": "observation-shell",
                "action_id": "action-shell",
                "observation": {
                    "kind": "TerminalObservation",
                    "is_error": False,
                    "exit_code": 2,
                    "timeout": False,
                    "content": [{"text": "tests failed"}],
                    "metadata": {"working_dir": "/testbed"},
                },
            },
        },
        {
            "type": "ActionEvent",
            "event": {
                "id": "action-editor",
                "tool_call_id": "call-editor",
                "action": {
                    "kind": "FileEditorAction",
                    "command": "view",
                    "path": "/testbed/a.py",
                },
            },
        },
        {
            "type": "ObservationEvent",
            "event": {
                "id": "observation-editor",
                "action_id": "action-editor",
                "observation": {
                    "kind": "FileEditorObservation",
                    "is_error": False,
                    "content": [{"text": "source"}],
                },
            },
        },
    ]
    events.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    receipts = build_openhands_receipts(events)
    shell, editor = receipts
    assert shell["transport_status"] == "completed"
    assert shell["semantic_status"] == "failed"
    assert shell["error_kind"] == "nonzero_exit"
    assert shell["result_complete"] is True
    assert shell["effect_trace_complete"] is False
    assert editor["operation_kind"] == "read"
    assert editor["resources"] == [{
        "resource_id": "file:///testbed/a.py", "kind": "read",
    }]
    assert editor["effect_trace_complete"] is False
    assert summarize_receipts(receipts)["unknown_effect_barrier_count"] == 2


def test_openhands_build_context_contains_only_declared_inputs(tmp_path: Path):
    repository = tmp_path / "repository"
    output = tmp_path / "output"
    output.mkdir()
    docker = repository / "experiments" / "paper8_5_agent_memory" / "docker"
    docker.mkdir(parents=True)
    (repository / "src" / "pra_hf").mkdir(parents=True)
    inputs = {
        docker / "openhands-swebench.Dockerfile": "FROM task\n",
        docker / "openhands-runtime.Dockerfile": "FROM runtime\n",
        repository
        / "experiments"
        / "paper8_5_agent_memory"
        / "openhands_swebench_entry.py": "print('entry')\n",
        repository / "src" / "pra_hf" / "execution_receipts.py": "SCHEMA = 1\n",
    }
    for path, content in inputs.items():
        path.write_text(content, encoding="utf-8")
    (repository / "large-unrelated-file.bin").write_bytes(b"x" * 1024)

    context = _prepare_build_context(repository, output)

    assert sorted(path.name for path in context.iterdir()) == [
        "execution_receipts.py",
        "openhands-runtime.Dockerfile",
        "openhands-swebench.Dockerfile",
        "openhands_swebench_entry.py",
    ]
    assert (context / "execution_receipts.py").read_text(encoding="utf-8") == "SCHEMA = 1\n"
