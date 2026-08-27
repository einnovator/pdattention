"""Task-aware scope selection and cache lifecycle tests."""

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.task_context import TaskEvent, TaskEventType, TaskGraph, TaskProvenance, attach_task_provenance
from pra_hf.task_scope import ResidencyState, TaskScopePolicy, TaskScopeSelector, TaskWorkingSet


def _graph():
    graph = TaskGraph()
    graph.apply(TaskEvent("create:a", 1, TaskEventType.CREATE, "a", payload={"description": "Find alpha"}))
    graph.apply(TaskEvent("create:b", 2, TaskEventType.CREATE, "b", payload={"description": "Find beta"}))
    graph.apply(TaskEvent("create:join", 3, TaskEventType.CREATE, "join", payload={"description": "Join alpha and beta", "depends_on": ("a", "b")}))
    return graph


def _record(record_id, task_id, text, sequence):
    return attach_task_provenance(
        ContextRecord(record_id, RecordType.GENERIC_TEXT, text),
        TaskProvenance(task_id, event_sequence=sequence),
    )


def test_local_scope_excludes_wrong_task_and_structural_scope_recovers_join() -> None:
    records = (
        _record("a:evidence", "a", "alpha evidence answer red", 1),
        _record("b:evidence", "b", "beta evidence answer blue", 2),
        _record("join:query", "join", "combine alpha beta", 3),
    )
    selector = TaskScopeSelector(_graph(), records)

    local = selector.select("join", "alpha beta answer", policy="task_local", max_records=3)
    structural = selector.select("join", "alpha beta answer", policy="task_structural", max_records=3)

    assert local.selected_record_ids == ("join:query",)
    assert {record.record_id for record in structural.selected_records} == {
        "a:evidence", "b:evidence", "join:query"
    }


def test_session_scope_budget_can_select_hot_semantic_distractor() -> None:
    records = (
        _record("a:old", "a", "release alpha stable evidence", 1),
        _record("b:hot", "b", "release alpha unstable distractor", 99),
    )
    selector = TaskScopeSelector(_graph(), records)
    session = selector.select("a", "release alpha", policy=TaskScopePolicy.SESSION, max_records=1)
    local = selector.select("a", "release alpha", policy=TaskScopePolicy.TASK_LOCAL, max_records=1)

    assert session.selected_record_ids == ("b:hot",)
    assert local.selected_record_ids == ("a:old",)


def test_working_set_distinguishes_cold_warm_hot_and_demotes_wrong_task() -> None:
    working = TaskWorkingSet()
    working.register_backing("a", backing_bytes=4096)
    first = working.activate("a", native_tokens=64)
    switched = working.activate("b", native_tokens=32)
    resumed = working.activate("a", native_tokens=64)
    completed = working.complete("a")

    assert first.kv_promoted == 64
    assert switched.kv_demoted == 64
    assert working.residency("b").state == ResidencyState.WARM
    assert resumed.kv_reused == 64
    assert completed.new_state == ResidencyState.WARM

