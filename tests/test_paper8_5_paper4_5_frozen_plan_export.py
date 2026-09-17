from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.paper8_5_agent_memory.export_paper4_5_frozen_plan import (
    _selection_digest,
    export_fixture,
)


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _content_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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
