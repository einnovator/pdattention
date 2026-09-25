from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.paper8_5_agent_memory.export_paper4_5_frozen_plan import (
    _selection_digest,
    export_captured_request_fixture,
    export_fixture,
)
from pra_hf.agent_history import OpenAIRecordizer


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _content_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_captured_native_request_preserves_tools_and_stable_records(
    tmp_path: Path,
) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [{
                "id": "call-old",
                "type": "function",
                "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-old", "content": "/testbed"},
        {"role": "user", "content": [{"type": "text", "text": "new task"}]},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run command",
            "parameters": {"type": "object"},
        },
    }]
    payload = {
        "model": "qwen",
        "messages": messages,
        "tools": tools,
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
    }
    session_id = "native-session"
    recordization = OpenAIRecordizer().recordize(
        messages, request_metadata={"session_id": session_id}
    )
    assert recordization.exact
    record_ids = [row.record_id for row in recordization.history.records]
    trace = {
        "request_index": 2,
        "native_request_index": 2,
        "session_id": session_id,
        "policy": "full",
        "plan_policy": "full",
        "plan_digest": "logical",
        "wire_plan_digest": "wire",
        "request_input_sha256": _digest(messages),
        "request_message_roles": [row["role"] for row in messages],
        "request_message_content_sha256": [
            _content_digest(str(row.get("content", ""))) for row in messages
        ],
        "selected_message_content_sha256": [
            _content_digest(str(row.get("content", ""))) for row in messages
        ],
        "selected_messages_sha256": _digest(messages),
        "wire_plan": {
            "selected_record_ids": record_ids,
            "record_replacements": {},
        },
    }
    native_events = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "old answer"}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "toolCall",
                        "id": "call-new",
                        "name": "bash",
                        "arguments": {"command": "git diff"},
                    },
                ],
            },
        },
    ]
    captured = tmp_path / "request.json"
    selection = tmp_path / "selection.jsonl"
    events = tmp_path / "events.jsonl"
    output = tmp_path / "fixture.jsonl"
    replay = tmp_path / "replay.jsonl"
    captured.write_text(json.dumps(payload), encoding="utf-8")
    selection.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    events.write_text(
        "".join(json.dumps(row) + "\n" for row in native_events),
        encoding="utf-8",
    )

    manifest = export_captured_request_fixture(
        captured_request=captured,
        request_selection=selection,
        native_events=events,
        output=output,
        request_replay_output=replay,
    )

    fixture = json.loads(output.read_text(encoding="utf-8"))
    assert fixture["selected_message_indices"] == [0, 1, 2, 3, 4]
    assert fixture["mandatory_message_indices"] == [0, 2, 3, 4]
    request = json.loads(replay.read_text(encoding="utf-8"))
    assert request["logical_payload"]["messages"] == messages
    assert request["logical_payload"]["tools"] == tools
    assert request["logical_payload"]["stream"] is False
    assert "stream_options" not in request["logical_payload"]
    assert "store" not in request["logical_payload"]
    assert request["expected_assistant_message"] == {
        "role": "assistant",
        "content": "inspect",
        "tool_calls": [{
            "id": "call-new",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": "{\"command\":\"git diff\"}",
            },
        }],
    }
    assert manifest["source_request_index"] == 2


def test_exporter_reconstructs_and_validates_frozen_request(tmp_path: Path) -> None:
    prior = {
        "schema_version": 1,
        "session_id": "session",
        "episodes": [{
            "trajectory": {
                "instance_id": "old-task",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "old task"},
                    {"role": "assistant", "content": "old action"},
                    {"role": "exit", "content": "old result"},
                ],
                "info": {"exit_status": "Submitted", "submission": "diff --git a/x b/x"},
            }
        }],
    }
    current = {
        "instance_id": "new-task",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "new task"},
            {"role": "assistant", "content": "new action"},
            {"role": "user", "content": "new result"},
        ],
        "info": {},
    }
    # Boundary-free composition drops the second system message.
    request = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old result"},
        {"role": "user", "content": "new task"},
    ]
    selected = [request[0], request[1], request[4]]
    trace = {
        "request_index": 1,
        "session_id": "session",
        "policy": "frontier_dag_retirement",
        "plan_policy": "frontier_dag_m2_heuristic_p1",
        "plan_digest": "logical",
        "wire_plan_digest": "wire",
        "request_input_sha256": _digest(request),
        "request_message_content_sha256": [_content_digest(row["content"]) for row in request],
        "selected_message_content_sha256": [_content_digest(row["content"]) for row in selected],
        "selected_messages_sha256": _digest(selected),
    }
    prefix = tmp_path / "prefix.json"
    trajectory = tmp_path / "trajectory.json"
    selection = tmp_path / "selection.jsonl"
    output = tmp_path / "fixture.jsonl"
    replay = tmp_path / "requests.jsonl"
    prefix.write_text(json.dumps(prior), encoding="utf-8")
    trajectory.write_text(json.dumps(current), encoding="utf-8")
    selection.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    manifest = export_fixture(
        persistent_prefix=prefix,
        trajectory=trajectory,
        request_selection=selection,
        output=output,
        request_replay_output=replay,
        model="model",
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["requests"] == 1
    assert row["request_input_sha256"] == trace["request_input_sha256"]
    assert row["selected_message_indices"] == [0, 1, 4]
    assert row["mandatory_message_indices"] == [0, 4]
    assert row["resources"] == [{"resource_id": "m1-0-user", "text": "old task"}]
    assert row["selected_resource_digest"] == _selection_digest([
        ("m1-0-user", "old task")
    ])
    replay_row = json.loads(replay.read_text(encoding="utf-8"))
    assert replay_row["request_input_sha256"] == trace["request_input_sha256"]
    assert replay_row["logical_payload"]["messages"] == request
    assert replay_row["logical_payload"]["model"] == "model"
    assert replay_row["expected_assistant_content"] == "new action"
    assert replay_row["expected_assistant_content_sha256"] == _content_digest(
        "new action"
    )
    assert manifest["request_replay_sha256"] == hashlib.sha256(
        replay.read_bytes()
    ).hexdigest()


def test_exporter_accepts_immutable_campaign_trajectory_envelope(
    tmp_path: Path,
) -> None:
    prior = {
        "schema_version": 1,
        "session_id": "session",
        "episodes": [{
            "trajectory": {
                "instance_id": "old-task",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "old task"},
                    {"role": "assistant", "content": "old action"},
                    {"role": "exit", "content": "old result"},
                ],
                "info": {"exit_status": "Submitted", "submission": "diff"},
            }
        }],
    }
    current = {
        "instance_id": "new-task",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "new task"},
            {"role": "assistant", "content": "new action"},
        ],
        "info": {},
    }
    request = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old result"},
        {"role": "user", "content": "new task"},
    ]
    trace = {
        "request_index": 1,
        "session_id": "session",
        "policy": "frontier_dag_retirement",
        "plan_policy": "frontier_dag_retirement",
        "plan_digest": "logical",
        "wire_plan_digest": "wire",
        "request_input_sha256": _digest(request),
        "request_message_content_sha256": [
            _content_digest(row["content"]) for row in request
        ],
        "selected_message_content_sha256": [
            _content_digest(row["content"]) for row in request
        ],
        "selected_messages_sha256": _digest(request),
    }
    prefix = tmp_path / "prefix.json"
    trajectory = tmp_path / "persistent_episode_export.json"
    selection = tmp_path / "selection.jsonl"
    output = tmp_path / "fixture.jsonl"
    prefix.write_text(json.dumps(prior), encoding="utf-8")
    trajectory.write_text(
        json.dumps({"schema_version": 1, "trajectory": current}),
        encoding="utf-8",
    )
    selection.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    manifest = export_fixture(
        persistent_prefix=prefix,
        trajectory=trajectory,
        request_selection=selection,
        output=output,
    )

    assert manifest["requests"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))[
        "mandatory_message_indices"
    ] == [0, 4]


def test_exporter_preserves_wire_materialized_receipts(tmp_path: Path) -> None:
    prior = {
        "schema_version": 1,
        "session_id": "session",
        "episodes": [{
            "trajectory": {
                "instance_id": "old-task",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "old task"},
                    {"role": "assistant", "content": "large old action"},
                    {"role": "exit", "content": "large old final result"},
                ],
                "info": {"exit_status": "Submitted", "submission": "diff"},
            }
        }],
    }
    current = {
        "instance_id": "new-task",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "new task"},
            {"role": "assistant", "content": "new action"},
        ],
        "info": {},
    }
    request = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "large old action"},
        {"role": "user", "content": "large old final result"},
        {"role": "user", "content": "new task"},
    ]
    replacements = {
        "record-000002": "[PRA memory] Prior instruction completed.",
        "record-000003": "[PRA memory] Closure recorded.",
    }
    selected = [
        request[0],
        request[1],
        {"role": "assistant", "content": replacements["record-000002"]},
        {"role": "user", "content": replacements["record-000003"]},
        request[4],
    ]
    trace = {
        "request_index": 1,
        "session_id": "session",
        "policy": "persistent_instruction_epoch_retirement",
        "plan_policy": "persistent_instruction_epoch_retirement",
        "plan_digest": "logical",
        "wire_plan_digest": "wire",
        "request_input_sha256": _digest(request),
        "request_message_content_sha256": [
            _content_digest(row["content"]) for row in request
        ],
        "selected_message_content_sha256": [
            _content_digest(row["content"]) for row in selected
        ],
        "selected_messages_sha256": _digest(selected),
        "wire_plan": {
            "selected_record_ids": [
                "record-000000",
                "record-000001",
                "record-000002",
                "record-000003",
                "record-000004",
            ],
            "record_replacements": replacements,
        },
    }
    prefix = tmp_path / "prefix.json"
    trajectory = tmp_path / "trajectory.json"
    selection = tmp_path / "selection.jsonl"
    output = tmp_path / "fixture.jsonl"
    prefix.write_text(json.dumps(prior), encoding="utf-8")
    trajectory.write_text(json.dumps(current), encoding="utf-8")
    selection.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    export_fixture(
        persistent_prefix=prefix,
        trajectory=trajectory,
        request_selection=selection,
        output=output,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["selected_message_indices"] == [0, 1, 2, 3, 4]
    assert row["resources"] == [
        {"resource_id": "m1-0-user", "text": "old task"},
        {
            "resource_id": "m2-0-assistant",
            "text": "[PRA memory] Prior instruction completed.",
        },
        {
            "resource_id": "m3-0-user",
            "text": "[PRA memory] Closure recorded.",
        },
    ]
    assert row["materialized_message_replacements"] == [
        {
            "record_id": "record-000002",
            "message_index": 2,
            "role": "assistant",
            "content": "[PRA memory] Prior instruction completed.",
            "content_sha256": _content_digest(
                "[PRA memory] Prior instruction completed."
            ),
        },
        {
            "record_id": "record-000003",
            "message_index": 3,
            "role": "user",
            "content": "[PRA memory] Closure recorded.",
            "content_sha256": _content_digest("[PRA memory] Closure recorded."),
        },
    ]
