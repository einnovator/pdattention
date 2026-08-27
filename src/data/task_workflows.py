"""Controlled interleaved task workflows for Paper 8 mechanism experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.task_context import (
    TaskDescriptor,
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskProvenance,
    attach_task_provenance,
)


class WorkflowFamily(str, Enum):
    ATOMIC = "atomic"
    LINEAR = "linear"
    PARALLEL = "parallel"
    FORK = "fork"
    JOIN = "join"
    DAG = "dag"


@dataclass(frozen=True)
class WorkflowComplexity:
    task_count: int
    edge_count: int
    critical_path_depth: int
    maximum_width: int
    join_count: int
    fork_count: int
    records_per_task: int
    total_records: int


@dataclass(frozen=True)
class TaskWorkflowCase:
    case_id: str
    family: WorkflowFamily
    seed: int
    graph: TaskDescriptor
    records: tuple[ContextRecord, ...]
    active_task_id: str
    query: str
    relevant_record_ids: tuple[str, ...]
    relevant_task_ids: tuple[str, ...]
    hot_distractor_task_id: str | None
    complexity: WorkflowComplexity


def _dependencies(family: WorkflowFamily, count: int) -> dict[str, tuple[str, ...]]:
    ids = [f"t{index}" for index in range(count)]
    values = {task_id: () for task_id in ids}
    if family == WorkflowFamily.LINEAR:
        for index in range(1, count):
            values[ids[index]] = (ids[index - 1],)
    elif family == WorkflowFamily.FORK and count > 1:
        for index in range(1, count):
            values[ids[index]] = (ids[0],)
    elif family == WorkflowFamily.JOIN and count > 1:
        values[ids[-1]] = tuple(ids[:-1])
    elif family == WorkflowFamily.DAG:
        for index in range(1, count):
            parents = [ids[index - 1]]
            if index > 2 and index % 3 == 0:
                parents.append(ids[index - 3])
            values[ids[index]] = tuple(dict.fromkeys(parents))
    return values


def _depths(dependencies: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    values: dict[str, int] = {}
    for task_id in dependencies:
        parents = dependencies[task_id]
        values[task_id] = 1 + max((values[parent] for parent in parents), default=0)
    return values


def generate_task_workflow(
    family: WorkflowFamily | str,
    *,
    task_count: int,
    records_per_task: int = 6,
    seed: int = 11,
) -> TaskWorkflowCase:
    """Generate one semantically confusable, physically interleaved workflow."""

    family = WorkflowFamily(family)
    if task_count <= 0 or records_per_task <= 0:
        raise ValueError("Task and record counts must be positive.")
    if family == WorkflowFamily.ATOMIC:
        task_count = 1
    dependencies = _dependencies(family, task_count)
    graph = TaskGraph()
    sequence = 0
    for task_id, parents in dependencies.items():
        sequence += 1
        graph.apply(TaskEvent(
            f"{seed}:create:{task_id}", sequence, TaskEventType.CREATE, task_id,
            payload={
                "description": f"Resolve shared project status for scope {task_id}",
                "depends_on": parents,
            },
        ))
    active = f"t{task_count - 1}"
    sequence += 1
    graph.apply(TaskEvent(
        f"{seed}:activate:{active}", sequence, TaskEventType.ACTIVATE, active,
        expected_version=1,
    ))

    rng = random.Random(seed)
    records = []
    event_sequence = sequence
    # Round-robin physical order makes records from distinct tasks interleave.
    for row_index in range(records_per_task):
        task_order = list(dependencies)
        rng.shuffle(task_order)
        for task_id in task_order:
            event_sequence += 1
            is_evidence = row_index == 0
            kind = "evidence answer" if is_evidence else "working note"
            payload = (
                f"shared project status {kind}; owner={task_id}; "
                f"value=value-{task_id}; step={row_index}"
            )
            record = ContextRecord(
                f"{task_id}:record:{row_index}",
                RecordType.GENERIC_TEXT,
                payload,
            )
            records.append(attach_task_provenance(
                record,
                TaskProvenance(task_id, event_sequence=event_sequence),
            ))

    relevant_tasks = graph.structural_closure(active)
    relevant_ids = tuple(f"{task_id}:record:0" for task_id in relevant_tasks)
    unrelated = [task_id for task_id in dependencies if task_id not in relevant_tasks]
    hot_distractor = unrelated[-1] if unrelated else None
    # Make one semantically equivalent wrong-task record physically hottest.
    if hot_distractor is not None:
        event_sequence += 100
        target = f"{hot_distractor}:record:0"
        records = [
            attach_task_provenance(
                record,
                TaskProvenance(hot_distractor, event_sequence=event_sequence),
            ) if record.record_id == target else record
            for record in records
        ]

    depths = _depths(dependencies)
    children = {task_id: 0 for task_id in dependencies}
    for parents in dependencies.values():
        for parent in parents:
            children[parent] += 1
    width_by_depth: dict[int, int] = {}
    for depth in depths.values():
        width_by_depth[depth] = width_by_depth.get(depth, 0) + 1
    complexity = WorkflowComplexity(
        task_count=task_count,
        edge_count=sum(len(value) for value in dependencies.values()),
        critical_path_depth=max(depths.values(), default=1),
        maximum_width=max(width_by_depth.values(), default=1),
        join_count=sum(len(value) > 1 for value in dependencies.values()),
        fork_count=sum(value > 1 for value in children.values()),
        records_per_task=records_per_task,
        total_records=len(records),
    )
    return TaskWorkflowCase(
        case_id=f"{family.value}-n{task_count}-r{records_per_task}-s{seed}",
        family=family,
        seed=seed,
        graph=graph.snapshot(),
        records=tuple(records),
        active_task_id=active,
        query="shared project status evidence answer",
        relevant_record_ids=relevant_ids,
        relevant_task_ids=relevant_tasks,
        hot_distractor_task_id=hot_distractor,
        complexity=complexity,
    )
