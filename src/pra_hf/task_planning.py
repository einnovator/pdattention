"""Deterministic task-acquisition controls for Paper 8.

These helpers validate model-produced plans; they do not make model output
authoritative.  The harness converts accepted rows into versioned task events.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .task_context import TaskEvent, TaskEventType, TaskGraph


@dataclass(frozen=True)
class ComplexityDecision:
    needs_decomposition: bool
    score: int
    reasons: tuple[str, ...]


class ComplexityGate:
    """Cheap preflight gate that avoids a second call for atomic requests."""

    def __init__(self, *, threshold: int = 2) -> None:
        if threshold <= 0:
            raise ValueError("Complexity threshold must be positive.")
        self.threshold = threshold

    def evaluate(self, request: str) -> ComplexityDecision:
        text = request.strip()
        reasons = []
        if len(text.split()) >= 80:
            reasons.append("long_request")
        if len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text)) >= 2:
            reasons.append("multiple_deliverables")
        if re.search(r"\b(?:then|after|before|depends on|once|finally)\b", text, re.I):
            reasons.append("sequencing")
        if len(re.findall(r"\b(?:and|also|plus)\b", text, re.I)) >= 2:
            reasons.append("conjunctions")
        if len(re.findall(r"\b(?:create|implement|test|build|update|analyze|compare)\b", text, re.I)) >= 3:
            reasons.append("multiple_actions")
        return ComplexityDecision(len(reasons) >= self.threshold, len(reasons), tuple(reasons))


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    description: str
    depends_on: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


def validate_plan(tasks: Sequence[PlannedTask]) -> tuple[PlannedTask, ...]:
    """Validate uniqueness, references, and acyclicity through ``TaskGraph``."""

    tasks = tuple(tasks)
    ids = [task.task_id for task in tasks]
    if not tasks or len(ids) != len(set(ids)):
        raise ValueError("Task plan must contain unique task IDs.")
    graph = TaskGraph()
    pending = list(tasks)
    sequence = 0
    while pending:
        progress = False
        for task in tuple(pending):
            if not set(task.depends_on).issubset(graph.tasks):
                continue
            sequence += 1
            graph.apply(TaskEvent(
                f"plan:create:{task.task_id}", sequence, TaskEventType.CREATE, task.task_id,
                payload={
                    "description": task.description,
                    "depends_on": task.depends_on,
                    "constraints": task.constraints,
                },
            ))
            pending.remove(task)
            progress = True
        if not progress:
            missing = sorted({dep for task in pending for dep in task.depends_on} - set(ids))
            if missing:
                raise ValueError(f"Task plan references unknown dependencies: {missing}")
            raise ValueError("Task plan contains a dependency cycle.")
    return tasks


def parse_json_plan(value: str | Mapping[str, object]) -> tuple[PlannedTask, ...]:
    """Parse a schema-constrained ``{"tasks": [...]}`` plan."""

    payload = json.loads(value) if isinstance(value, str) else dict(value)
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("JSON plan requires a tasks array.")
    tasks = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every task must be an object.")
        task_id = str(row.get("task_id", "")).strip()
        description = str(row.get("description", "")).strip()
        if not task_id or not description:
            raise ValueError("Every task requires task_id and description.")
        tasks.append(PlannedTask(
            task_id,
            description,
            tuple(str(value) for value in row.get("depends_on", ())),
            tuple(str(value) for value in row.get("constraints", ())),
        ))
    return validate_plan(tasks)


_TASK_HEADER = re.compile(r"^##\s+Task\s+([^\s]+)\s*$", re.I)


def parse_markdown_plan(value: str) -> tuple[PlannedTask, ...]:
    """Parse the deliberately small Paper-8 Markdown task grammar."""

    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in value.splitlines():
        line = raw_line.strip()
        header = _TASK_HEADER.match(line)
        if header:
            if current:
                rows.append(current)
            current = {"task_id": header.group(1), "depends_on": (), "constraints": ()}
            continue
        if current is None or not line:
            continue
        name, separator, content = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid task-plan line: {line}")
        key = name.strip().lower()
        content = content.strip()
        if key == "description":
            current["description"] = content
        elif key == "depends on":
            current["depends_on"] = () if content.lower() == "none" else tuple(
                value.strip() for value in content.split(",") if value.strip()
            )
        elif key == "constraints":
            current["constraints"] = () if content.lower() == "none" else tuple(
                value.strip() for value in content.split(";") if value.strip()
            )
        else:
            raise ValueError(f"Unknown task-plan field: {name}")
    if current:
        rows.append(current)
    return validate_plan(PlannedTask(**row) for row in rows)


def plan_events(tasks: Sequence[PlannedTask], *, sequence_start: int = 0) -> tuple[TaskEvent, ...]:
    """Convert a validated preflight plan to replayable creation events."""

    tasks = validate_plan(tasks)
    return tuple(
        TaskEvent(
            f"preflight:create:{task.task_id}", sequence_start + index,
            TaskEventType.CREATE, task.task_id,
            payload={
                "description": task.description,
                "depends_on": task.depends_on,
                "constraints": task.constraints,
            },
        )
        for index, task in enumerate(tasks, start=1)
    )
