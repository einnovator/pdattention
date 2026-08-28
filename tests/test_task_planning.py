"""Task complexity gate and deterministic preflight parsers."""

import pytest

from pra_hf.task_context import TaskGraph
from pra_hf.task_planning import (
    ComplexityGate,
    apply_task_operations,
    parse_json_plan,
    parse_markdown_plan,
    parse_model_markdown_plan,
    parse_task_operations,
    plan_events,
)


def test_complexity_gate_skips_atomic_request_and_detects_ordered_deliverables() -> None:
    gate = ComplexityGate()
    assert not gate.evaluate("Summarize this file.").needs_decomposition
    decision = gate.evaluate("1. Implement the API.\n2. Then test it.\n3. Finally build the docs.")
    assert decision.needs_decomposition
    assert "multiple_deliverables" in decision.reasons


def test_json_and_markdown_plans_produce_same_valid_graph() -> None:
    json_tasks = parse_json_plan({"tasks": [
        {"task_id": "a", "description": "Collect evidence"},
        {"task_id": "b", "description": "Write result", "depends_on": ["a"]},
    ]})
    markdown_tasks = parse_markdown_plan(
        "## Task a\nDescription: Collect evidence\nDepends on: none\n\n"
        "## Task b\nDescription: Write result\nDepends on: a\n"
    )

    assert json_tasks == markdown_tasks
    assert [event.task_id for event in plan_events(json_tasks)] == ["a", "b"]


def test_plan_validation_rejects_missing_dependency_and_cycle() -> None:
    with pytest.raises(ValueError, match="unknown dependencies"):
        parse_json_plan({"tasks": [{"task_id": "a", "description": "A", "depends_on": ["x"]}]})
    with pytest.raises(ValueError, match="cycle"):
        parse_json_plan({"tasks": [
            {"task_id": "a", "description": "A", "depends_on": ["b"]},
            {"task_id": "b", "description": "B", "depends_on": ["a"]},
        ]})


def test_model_markdown_bullets_and_online_operations_remain_harness_validated() -> None:
    tasks = parse_model_markdown_plan(
        "- a: Collect evidence independently.\n- b: Write result after a."
    )
    graph = TaskGraph()
    graph.replay(plan_events(tasks))
    operations = parse_task_operations({
        "operations": [{"action": "link", "task_id": "b", "depends_on": ["a"]}]
    })
    events = apply_task_operations(graph, operations, sequence_start=2)
    assert events[0].event_type.value == "task_link"
    assert graph.tasks["b"].depends_on == ("a",)


def test_online_link_adds_dependency_without_erasing_existing_join_parents() -> None:
    tasks = parse_json_plan({"tasks": [
        {"task_id": "a", "description": "A"},
        {"task_id": "b", "description": "B"},
        {"task_id": "c", "description": "C", "depends_on": ["a"]},
    ]})
    graph = TaskGraph()
    graph.replay(plan_events(tasks))
    apply_task_operations(
        graph,
        parse_task_operations({
            "operations": [{"action": "link", "task_id": "c", "depends_on": ["b"]}]
        }),
        sequence_start=3,
    )
    assert graph.tasks["c"].depends_on == ("a", "b")
