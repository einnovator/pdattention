"""Long-running in-memory and local session service contracts."""

import json

import pytest

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.session_service import (
    InMemorySessionService,
    LocalSessionService,
    SessionConflict,
)
from pra_hf.task_context import TaskEvent, TaskEventType, TaskStatus


def _exercise(service):
    state = service.create_session("user-1", "session-a", task_description="Implement the feature")
    state = service.append_record(
        "user-1", "session-a", ContextRecord("turn:1", RecordType.GENERIC_TEXT, "hello")
    )
    state = service.apply_task_event(
        "user-1", "session-a",
        TaskEvent("complete:root", 3, TaskEventType.COMPLETE, "task-1", expected_version=2, payload={"result_ref": "turn:1"}),
    )
    resolved = service.resolve_session("user-1", "session-a")
    assert resolved.version == state.version
    assert resolved.records[0].payload == "hello"
    assert resolved.tasks.tasks[0].status == TaskStatus.COMPLETED
    assert resolved.tasks.tasks[0].result_ref == "turn:1"
    return resolved


def test_in_memory_service_resolves_by_user_and_session_and_checks_versions() -> None:
    service = InMemorySessionService()
    current = _exercise(service)
    service.create_session("user-1", "session-b")

    assert service.resolve_session("user-1").session_id == "session-b"
    with pytest.raises(SessionConflict):
        service.save_session(current._next(), expected_version=1)


def test_local_service_round_trips_typed_records_and_task_descriptor(tmp_path) -> None:
    service = LocalSessionService(tmp_path)
    state = _exercise(service)
    reopened = LocalSessionService(tmp_path).get_session("user-1", "session-a")

    assert reopened == state
    manifests = list(tmp_path.rglob("*.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["user_id"] == "user-1"
    assert payload["tasks"]["active_task_id"] is None

