"""Paper 8 authoritative task graph and provenance contracts."""

import pytest

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.task_context import (
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskProvenance,
    TaskStatus,
    attach_task_provenance,
)


def _create(sequence, task_id, *, depends_on=(), parent=None):
    return TaskEvent(
        f"create:{task_id}", sequence, TaskEventType.CREATE, task_id,
        payload={
            "description": f"Execute {task_id}",
            "depends_on": depends_on,
            "parent_task_id": parent,
        },
    )


def test_task_events_are_versioned_idempotent_and_replayable() -> None:
    graph = TaskGraph()
    created = graph.apply(_create(1, "a"))
    active = graph.apply(TaskEvent("activate:a", 2, "task_activate", "a", expected_version=1))
    duplicate = graph.apply(TaskEvent("activate:a", 2, "task_activate", "a", expected_version=1))

    assert created.version == 1
    assert active.version == duplicate.version == 2
    assert active.status == TaskStatus.ACTIVE
    assert graph.snapshot().last_sequence == 2

    replayed = TaskGraph().replay((_create(1, "a"), TaskEvent("activate:a", 2, "task_activate", "a", expected_version=1)))
    assert replayed == graph.snapshot()


def test_dependency_validation_rejects_missing_edges_and_cycles() -> None:
    graph = TaskGraph()
    with pytest.raises(ValueError, match="unknown tasks"):
        graph.apply(_create(1, "join", depends_on=("missing",)))

    graph.apply(_create(1, "a"))
    graph.apply(_create(2, "b", depends_on=("a",)))
    with pytest.raises(ValueError, match="acyclic"):
        graph.apply(TaskEvent(
            "link:a", 3, TaskEventType.LINK, "a", expected_version=1,
            payload={"depends_on": ("b",)},
        ))


def test_structural_closure_includes_join_inputs_but_excludes_siblings() -> None:
    graph = TaskGraph()
    graph.apply(_create(1, "root"))
    graph.apply(_create(2, "left", parent="root"))
    graph.apply(_create(3, "right", parent="root"))
    graph.apply(_create(4, "join", depends_on=("left", "right")))

    assert graph.structural_closure("left") == ("left", "root")
    assert "right" not in graph.structural_closure("left")
    assert set(graph.structural_closure("join")) == {"join", "left", "right", "root"}


def test_result_reference_owners_enter_structural_closure() -> None:
    graph = TaskGraph()
    graph.apply(_create(1, "producer"))
    graph.apply(TaskEvent(
        "complete:producer", 2, TaskEventType.COMPLETE, "producer", expected_version=1,
        payload={"output_refs": ("result:alpha",), "result_ref": "result:alpha"},
    ))
    graph.apply(TaskEvent(
        "create:consumer", 3, TaskEventType.CREATE, "consumer",
        payload={"description": "Use alpha", "evidence_refs": ("result:alpha",)},
    ))

    assert graph.structural_closure("consumer") == ("consumer", "producer")


def test_task_provenance_preserves_typed_record_views() -> None:
    source = ContextRecord("record:1", RecordType.GENERIC_TEXT, "evidence")
    tagged = attach_task_provenance(source, TaskProvenance("task-a", event_sequence=7))

    assert TaskProvenance.from_record(tagged).task_id == "task-a"
    assert tagged.payload == source.payload
    assert tagged.views == source.views
