import json
from pathlib import Path

from experiments.paper8_5_agent_memory.reduce_native_tool_events import (
    reduce_native_tool_events,
)


def test_native_tool_reducer_separates_transport_and_semantic_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text("\n".join([
        json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
        json.dumps({
            "type": "tool_use",
            "part": {
                "id": "read-1",
                "tool": "read",
                "state": {
                    "status": "completed",
                    "input": {"filePath": "/testbed/a.py"},
                    "output": "source",
                },
            },
        }),
        json.dumps({
            "type": "tool_use",
            "part": {
                "id": "bash-1",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "python test.py"},
                    "output": "Traceback (most recent call last): boom",
                },
            },
        }),
        json.dumps({
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "input": 100,
                    "output": 10,
                    "cache": {"read": 40, "write": 0},
                },
            },
        }),
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "reduced"
    summary = reduce_native_tool_events(
        source,
        output,
        agent="fixture",
        tool_semantics={
            "read": {
                "category": "filesystem",
                "operation_kind": "read",
                "resource_arguments": ["filePath"],
            },
            "bash": {"category": "shell", "operation_kind": "unknown"},
        },
        token_note="fixture",
    )
    assert summary["assistant_model_calls"] == 1
    assert summary["native_tool_calls"] == 2
    assert summary["cumulative_logical_input_tokens"] == 140
    assert summary["tool_status_errors"] == 0
    assert summary["tool_failure_signal_results"] == 1
    canonical = [
        json.loads(line)
        for line in (output / "canonical_tool_events.jsonl").read_text().splitlines()
    ]
    assert canonical[0]["resource_ids"] == ["/testbed/a.py"]
    assert canonical[0]["semantic_success"] is True
    assert canonical[1]["transport_status"] == "completed"
    assert canonical[1]["semantic_success"] is False


def test_nested_agent_use_marks_root_trace_incomplete(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps({
        "type": "tool_use",
        "part": {
            "id": "task-1",
            "tool": "task",
            "state": {"status": "completed", "input": {}, "output": "done"},
        },
    }) + "\n", encoding="utf-8")
    summary = reduce_native_tool_events(
        source,
        tmp_path / "reduced",
        agent="fixture",
        tool_semantics={"task": {"category": "agent_delegation"}},
        token_note="fixture",
    )
    assert summary["nested_agent_tool_calls"] == 1
    assert summary["root_event_trace_complete"] is False
