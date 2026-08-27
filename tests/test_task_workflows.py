"""Controlled Paper 8 workflow generator invariants."""

import pytest

from data.task_workflows import WorkflowFamily, generate_task_workflow
from pra_hf.task_context import TaskGraph, TaskProvenance


@pytest.mark.parametrize("family", tuple(WorkflowFamily))
def test_workflow_generator_is_deterministic_and_replayable(family) -> None:
    first = generate_task_workflow(family, task_count=6, records_per_task=3, seed=23)
    second = generate_task_workflow(family, task_count=6, records_per_task=3, seed=23)

    assert first == second
    assert TaskGraph(first.graph).structural_closure(first.active_task_id) == first.relevant_task_ids
    assert all(TaskProvenance.from_record(record) is not None for record in first.records)
    assert len(first.records) == first.complexity.total_records


def test_join_requires_every_predecessor_and_parallel_scope_remains_local() -> None:
    join = generate_task_workflow("join", task_count=5, seed=11)
    parallel = generate_task_workflow("parallel", task_count=5, seed=11)

    assert set(join.relevant_task_ids) == {"t0", "t1", "t2", "t3", "t4"}
    assert parallel.relevant_task_ids == ("t4",)
    assert join.complexity.join_count == 1
    assert parallel.complexity.edge_count == 0
