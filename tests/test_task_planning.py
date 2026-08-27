"""Task complexity gate and deterministic preflight parsers."""

import pytest

from pra_hf.task_planning import ComplexityGate, parse_json_plan, parse_markdown_plan, plan_events


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
