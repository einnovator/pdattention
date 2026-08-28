"""Deterministic task-acquisition controls for Paper 8.

These helpers validate model-produced plans; they do not make model output
authoritative.  The harness converts accepted rows into versioned task events.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .task_context import TaskEvent, TaskEventType, TaskGraph


class TaskOperationKind(str, Enum):
    """Harness-validated mutation proposed by a model-managed task tool."""

    CREATE = "create"
    UPDATE = "update"
    LINK = "link"
    ACTIVATE = "activate"
    COMPLETE = "complete"
    BLOCK = "block"
    CANCEL = "cancel"


@dataclass(frozen=True)
class TaskOperation:
    """Untrusted task mutation proposal awaiting harness validation."""

    kind: TaskOperationKind | str
    task_id: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TaskOperationKind(self.kind))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not self.task_id:
            raise ValueError("Task operations require a task_id.")
        if self.kind == TaskOperationKind.CREATE and not self.description:
            raise ValueError("Create operations require a task description.")


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


def extract_json_payload(value: str) -> Mapping[str, object]:
    """Extract one JSON object from plain or fenced model output."""

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model output contains no JSON object.")
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("Model output must contain a JSON object.")
    return payload


def parse_model_json_plan(value: str) -> tuple[PlannedTask, ...]:
    """Parse a model response while tolerating one Markdown JSON fence."""

    return parse_json_plan(extract_json_payload(value))


def parse_task_operations(value: str | Mapping[str, object]) -> tuple[TaskOperation, ...]:
    """Parse task-tool proposals without granting them mutation authority."""

    payload = extract_json_payload(value) if isinstance(value, str) else dict(value)
    rows = payload.get("operations")
    if not isinstance(rows, list):
        raise ValueError("Task-tool output requires an operations array.")
    operations = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every task operation must be an object.")
        operations.append(TaskOperation(
            kind=str(row.get("action", row.get("kind", ""))).lower(),
            task_id=str(row.get("task_id", "")).strip(),
            description=str(row.get("description", "")).strip(),
            depends_on=tuple(str(value) for value in row.get("depends_on", ())),
            constraints=tuple(str(value) for value in row.get("constraints", ())),
        ))
    return tuple(operations)


def apply_task_operations(
    graph: TaskGraph,
    operations: Sequence[TaskOperation],
    *,
    sequence_start: int = 0,
) -> tuple[TaskEvent, ...]:
    """Validate and commit model proposals through the authoritative task graph."""

    events = []
    for offset, operation in enumerate(operations, start=1):
        event_type = {
            TaskOperationKind.CREATE: TaskEventType.CREATE,
            TaskOperationKind.UPDATE: TaskEventType.UPDATE,
            TaskOperationKind.LINK: TaskEventType.LINK,
            TaskOperationKind.ACTIVATE: TaskEventType.ACTIVATE,
            TaskOperationKind.COMPLETE: TaskEventType.COMPLETE,
            TaskOperationKind.BLOCK: TaskEventType.BLOCK,
            TaskOperationKind.CANCEL: TaskEventType.CANCEL,
        }[operation.kind]
        payload: dict[str, object] = {}
        if operation.kind == TaskOperationKind.CREATE:
            payload = {
                "description": operation.description,
                "depends_on": operation.depends_on,
                "constraints": operation.constraints,
            }
        elif operation.kind == TaskOperationKind.UPDATE:
            payload = {
                "description": operation.description,
                "constraints": operation.constraints,
            }
        elif operation.kind == TaskOperationKind.LINK:
            current = graph.tasks.get(operation.task_id)
            if current is None:
                raise ValueError(f"Cannot link unknown task {operation.task_id!r}.")
            payload = {
                "depends_on": tuple(dict.fromkeys((*current.depends_on, *operation.depends_on)))
            }
        event = TaskEvent(
            f"model-task-tool:{sequence_start + offset}:{operation.kind.value}:{operation.task_id}",
            sequence_start + offset,
            event_type,
            operation.task_id,
            payload=payload,
        )
        graph.apply(event)
        events.append(event)
    return tuple(events)


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


_BULLET_TASK = re.compile(
    r"^-\s*([^:]+):\s*(.*?)\s+(independently|after\s+(.+?))\.?\s*$",
    re.I,
)


def parse_model_markdown_plan(value: str) -> tuple[PlannedTask, ...]:
    """Parse the documented heading grammar or a bounded model bullet variant."""

    start = value.lower().find("## task")
    if start >= 0:
        text = value[start:]
        if "```" in text:
            text = text.split("```", 1)[0]
        return parse_markdown_plan(text)
    tasks = []
    for raw_line in value.splitlines():
        match = _BULLET_TASK.match(raw_line.strip())
        if not match:
            continue
        dependencies = ()
        if match.group(3).lower() != "independently" and match.group(4):
            dependencies = tuple(
                row.strip() for row in match.group(4).split(",") if row.strip()
            )
        tasks.append(PlannedTask(match.group(1).strip(), match.group(2).strip(), dependencies))
    if not tasks:
        raise ValueError("Model output contains no recognized Markdown task rows.")
    return validate_plan(tasks)


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
