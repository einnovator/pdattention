"""Authoritative task state and replay for interleaved single-session work.

Paper 8 treats task structure as execution state supplied by a harness.  The
model may propose mutations, but only validated :class:`TaskEvent` objects are
committed here.  Events are versioned and idempotent so a model runtime can
reconstruct the same task graph after process or cache loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .context_records import ContextRecord, RecordType


class TaskStatus(str, Enum):
    """Lifecycle status owned by the session harness."""

    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskEventType(str, Enum):
    """Validated mutations accepted by :class:`TaskGraph`."""

    CREATE = "task_create"
    UPDATE = "task_update"
    LINK = "task_link"
    ACTIVATE = "task_activate"
    BLOCK = "task_block"
    RESUME = "task_resume"
    COMPLETE = "task_complete"
    CANCEL = "task_cancel"


class TaskRelation(str, Enum):
    """Typed relation represented in task state or record provenance."""

    LOCAL = "local"
    PARENT = "parent"
    DEPENDENCY = "dependency"
    BLOCKER = "blocker"
    EVIDENCE = "evidence"
    OUTPUT = "output"
    RELATED = "related"


@dataclass(frozen=True)
class TaskState:
    """Canonical projected state for one logical task.

    Large outputs are represented by ``result_ref`` and ``output_refs``.  The
    referenced typed records retain compact views and exact Paper-7 backing.
    """

    task_id: str
    version: int
    status: TaskStatus | str
    description: str
    constraints: tuple[str, ...] = ()
    parent_task_id: str | None = None
    depends_on: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    blocker_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    result_ref: str | None = None
    completion_condition: str | None = None
    created_seq: int = 0
    updated_seq: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", TaskStatus(self.status))
        for name in (
            "constraints", "depends_on", "after", "blocker_refs",
            "evidence_refs", "output_refs",
        ):
            object.__setattr__(self, name, tuple(dict.fromkeys(getattr(self, name))))
        if not self.task_id or not self.description.strip():
            raise ValueError("Task ID and description are required.")
        if self.version <= 0:
            raise ValueError("Task version must be positive.")
        if self.task_id in self.depends_on or self.task_id == self.parent_task_id:
            raise ValueError("A task cannot depend on or parent itself.")
        if self.created_seq < 0 or self.updated_seq < self.created_seq:
            raise ValueError("Task sequence values are invalid.")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskState":
        return cls(**dict(value))


@dataclass(frozen=True)
class TaskEvent:
    """One monotonic, idempotent task mutation committed by the harness."""

    event_id: str
    sequence: int
    event_type: TaskEventType | str
    task_id: str
    expected_version: int | None = None
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", TaskEventType(self.event_type))
        object.__setattr__(self, "payload", dict(self.payload))
        if not self.event_id or not self.task_id or self.sequence <= 0:
            raise ValueError("Task events require an ID, task ID, and positive sequence.")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskEvent":
        return cls(**dict(value))


@dataclass(frozen=True)
class TaskDescriptor:
    """Serializable snapshot paired with the physical session record stream."""

    tasks: tuple[TaskState, ...] = ()
    active_task_id: str | None = None
    last_sequence: int = 0
    applied_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "applied_event_ids", tuple(self.applied_event_ids))
        ids = {task.task_id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("Task descriptor contains duplicate task IDs.")
        if self.active_task_id is not None and self.active_task_id not in ids:
            raise ValueError("Active task is absent from the task descriptor.")

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "active_task_id": self.active_task_id,
            "last_sequence": self.last_sequence,
            "applied_event_ids": list(self.applied_event_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskDescriptor":
        return cls(
            tasks=tuple(TaskState.from_dict(row) for row in value.get("tasks", ())),
            active_task_id=value.get("active_task_id"),
            last_sequence=int(value.get("last_sequence", 0)),
            applied_event_ids=tuple(value.get("applied_event_ids", ())),
        )


@dataclass(frozen=True)
class TaskProvenance:
    """Task ownership and relation metadata attached to a typed record."""

    task_id: str
    producing_task_id: str | None = None
    consuming_task_ids: tuple[str, ...] = ()
    relation: TaskRelation | str = TaskRelation.LOCAL
    event_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", TaskRelation(self.relation))
        object.__setattr__(self, "consuming_task_ids", tuple(dict.fromkeys(self.consuming_task_ids)))
        if not self.task_id or self.event_sequence < 0:
            raise ValueError("Task provenance requires a task ID and nonnegative sequence.")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["relation"] = self.relation.value
        return value

    @classmethod
    def from_record(cls, record: ContextRecord) -> "TaskProvenance | None":
        value = record.selection_provenance.get("task")
        return cls(**dict(value)) if isinstance(value, Mapping) else None


def attach_task_provenance(record: ContextRecord, provenance: TaskProvenance) -> ContextRecord:
    """Return ``record`` with task metadata while preserving its typed views."""

    return replace(
        record,
        selection_provenance={**record.selection_provenance, "task": provenance.to_dict()},
    )


def task_state_record(task: TaskState) -> ContextRecord:
    """Expose compact authoritative task state as a typed context record."""

    return ContextRecord(
        record_id=f"task:{task.task_id}:v{task.version}",
        record_type=RecordType.TASK_STATE,
        payload=task.to_dict(),
        selection_provenance={"task": TaskProvenance(task.task_id, event_sequence=task.updated_seq).to_dict()},
        version=f"v{task.version}",
    )


class TaskGraph:
    """Validated DAG and replay engine for one physical session."""

    def __init__(self, descriptor: TaskDescriptor | None = None) -> None:
        descriptor = descriptor or TaskDescriptor()
        self._tasks = {task.task_id: task for task in descriptor.tasks}
        self.active_task_id = descriptor.active_task_id
        self.last_sequence = descriptor.last_sequence
        self._event_ids = set(descriptor.applied_event_ids)
        self._validate_graph(self._tasks)

    @property
    def tasks(self) -> Mapping[str, TaskState]:
        return dict(self._tasks)

    def snapshot(self) -> TaskDescriptor:
        return TaskDescriptor(
            tasks=tuple(sorted(self._tasks.values(), key=lambda task: (task.created_seq, task.task_id))),
            active_task_id=self.active_task_id,
            last_sequence=self.last_sequence,
            applied_event_ids=tuple(sorted(self._event_ids)),
        )

    @staticmethod
    def _validate_graph(tasks: Mapping[str, TaskState]) -> None:
        ids = set(tasks)
        for task in tasks.values():
            references = set(task.depends_on) | set(task.after)
            if task.parent_task_id:
                references.add(task.parent_task_id)
            missing = references - ids
            if missing:
                raise ValueError(f"Task {task.task_id!r} references unknown tasks: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("Task dependency graph must be acyclic.")
            if task_id in visited:
                return
            visiting.add(task_id)
            task = tasks[task_id]
            for dependency in (*task.depends_on, *task.after):
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(tasks):
            visit(task_id)

    def _replace(self, task: TaskState, event: TaskEvent, **changes: object) -> TaskState:
        if event.expected_version is not None and event.expected_version != task.version:
            raise ValueError(
                f"Task {task.task_id!r} version is {task.version}, expected {event.expected_version}."
            )
        return replace(task, version=task.version + 1, updated_seq=event.sequence, **changes)

    def apply(self, event: TaskEvent) -> TaskState:
        """Commit an event once, preserving sequence and graph invariants."""

        if event.event_id in self._event_ids:
            task = self._tasks.get(event.task_id)
            if task is None:
                raise ValueError("Idempotent event references a missing task.")
            return task
        if event.sequence <= self.last_sequence:
            raise ValueError("Task event sequence must increase monotonically.")
        tasks = dict(self._tasks)
        kind = event.event_type
        if kind == TaskEventType.CREATE:
            if event.task_id in tasks:
                raise ValueError(f"Task already exists: {event.task_id}")
            task = TaskState(
                task_id=event.task_id,
                version=1,
                status=event.payload.get("status", TaskStatus.PENDING.value),
                description=str(event.payload.get("description", "")).strip(),
                constraints=tuple(event.payload.get("constraints", ())),
                parent_task_id=event.payload.get("parent_task_id"),
                depends_on=tuple(event.payload.get("depends_on", ())),
                after=tuple(event.payload.get("after", ())),
                blocker_refs=tuple(event.payload.get("blocker_refs", ())),
                evidence_refs=tuple(event.payload.get("evidence_refs", ())),
                output_refs=tuple(event.payload.get("output_refs", ())),
                result_ref=event.payload.get("result_ref"),
                completion_condition=event.payload.get("completion_condition"),
                created_seq=event.sequence,
                updated_seq=event.sequence,
            )
        else:
            if event.task_id not in tasks:
                raise ValueError(f"Unknown task: {event.task_id}")
            current = tasks[event.task_id]
            if kind == TaskEventType.UPDATE:
                allowed = {
                    "description", "constraints", "blocker_refs", "evidence_refs",
                    "output_refs", "result_ref", "completion_condition",
                }
                changes = {key: value for key, value in event.payload.items() if key in allowed}
                task = self._replace(current, event, **changes)
            elif kind == TaskEventType.LINK:
                task = self._replace(
                    current,
                    event,
                    parent_task_id=event.payload.get("parent_task_id", current.parent_task_id),
                    depends_on=tuple(event.payload.get("depends_on", current.depends_on)),
                    after=tuple(event.payload.get("after", current.after)),
                )
            else:
                transitions = {
                    TaskEventType.ACTIVATE: TaskStatus.ACTIVE,
                    TaskEventType.RESUME: TaskStatus.ACTIVE,
                    TaskEventType.BLOCK: TaskStatus.BLOCKED,
                    TaskEventType.COMPLETE: TaskStatus.COMPLETED,
                    TaskEventType.CANCEL: TaskStatus.CANCELLED,
                }
                status = transitions[kind]
                task = self._replace(
                    current,
                    event,
                    status=status,
                    result_ref=event.payload.get("result_ref", current.result_ref),
                    output_refs=tuple(event.payload.get("output_refs", current.output_refs)),
                    blocker_refs=tuple(event.payload.get("blocker_refs", current.blocker_refs)),
                )
        tasks[event.task_id] = task
        self._validate_graph(tasks)
        if kind in {TaskEventType.ACTIVATE, TaskEventType.RESUME}:
            for task_id, row in tuple(tasks.items()):
                if task_id != event.task_id and row.status == TaskStatus.ACTIVE:
                    tasks[task_id] = replace(row, status=TaskStatus.PENDING)
            self.active_task_id = event.task_id
        elif kind in {TaskEventType.COMPLETE, TaskEventType.CANCEL, TaskEventType.BLOCK} and self.active_task_id == event.task_id:
            self.active_task_id = None
        self._tasks = tasks
        self.last_sequence = event.sequence
        self._event_ids.add(event.event_id)
        return task

    def replay(self, events: Iterable[TaskEvent]) -> TaskDescriptor:
        for event in sorted(events, key=lambda row: row.sequence):
            self.apply(event)
        return self.snapshot()

    def structural_closure(self, task_id: str, *, max_depth: int = 8) -> tuple[str, ...]:
        """Return task, parent, dependencies, and tasks owning required references."""

        if task_id not in self._tasks:
            raise KeyError(task_id)
        admitted: list[str] = []
        frontier = [(task_id, 0)]
        while frontier:
            current_id, depth = frontier.pop(0)
            if current_id in admitted or depth > max_depth:
                continue
            admitted.append(current_id)
            current = self._tasks[current_id]
            related = list(current.depends_on)
            if current.parent_task_id:
                related.append(current.parent_task_id)
            frontier.extend((value, depth + 1) for value in related)
        required_refs = set(self._tasks[task_id].blocker_refs) | set(self._tasks[task_id].evidence_refs)
        for other in self._tasks.values():
            if required_refs.intersection((*other.output_refs, other.result_ref)):
                if other.task_id not in admitted:
                    admitted.append(other.task_id)
        return tuple(admitted)

    def related_tasks(self, task_id: str) -> tuple[str, ...]:
        """Return direct structural neighbors without treating all siblings as closure."""

        closure = set(self.structural_closure(task_id))
        related = set(closure)
        for other in self._tasks.values():
            if task_id in other.depends_on or other.parent_task_id == task_id:
                related.add(other.task_id)
        return tuple(sorted(related))

