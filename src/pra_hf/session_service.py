"""Session persistence for long-running PRA agents.

A session is resolved by ``user_id`` and ``session_id`` and owns one ordered
typed-record stream plus a versioned task descriptor.  Services persist only
authoritative logical state; model-native K/V remains a reconstructible cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .context_records import (
    ContextRecord,
    RecordBoundary,
    RecordPolicy,
    RecordView,
    RecordViewName,
)
from .task_context import TaskDescriptor, TaskEvent, TaskEventType, TaskGraph


@dataclass(frozen=True)
class AgentSessionState:
    """Immutable logical state of one long-running physical session."""

    user_id: str
    session_id: str
    tenant_id: str = "default"
    version: int = 1
    records: tuple[ContextRecord, ...] = ()
    tasks: TaskDescriptor = field(default_factory=TaskDescriptor)
    task_events: tuple[TaskEvent, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "task_events", tuple(self.task_events))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.user_id or not self.session_id or not self.tenant_id:
            raise ValueError("Session user_id, session_id, and tenant_id are required.")
        if self.version <= 0 or self.updated_at < self.created_at:
            raise ValueError("Session version or timestamps are invalid.")
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Session record IDs must be unique.")

    @property
    def active_task_id(self) -> str | None:
        return self.tasks.active_task_id

    def with_record(self, record: ContextRecord) -> "AgentSessionState":
        if any(existing.record_id == record.record_id for existing in self.records):
            raise ValueError(f"Session record already exists: {record.record_id}")
        return self._next(records=(*self.records, record))

    def with_task_event(self, event: TaskEvent) -> "AgentSessionState":
        graph = TaskGraph(self.tasks)
        descriptor = graph.replay((*self.task_events, event)) if not self.task_events else None
        if descriptor is None:
            graph = TaskGraph(self.tasks)
            graph.apply(event)
            descriptor = graph.snapshot()
        return self._next(tasks=descriptor, task_events=(*self.task_events, event))

    def _next(self, **changes: object) -> "AgentSessionState":
        values = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "version": self.version + 1,
            "records": self.records,
            "tasks": self.tasks,
            "task_events": self.task_events,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "metadata": self.metadata,
        }
        values.update(changes)
        return AgentSessionState(**values)


class SessionNotFound(KeyError):
    """Raised when a user/session pair cannot be resolved."""


class SessionConflict(RuntimeError):
    """Raised when optimistic session version validation fails."""


class SessionService(ABC):
    """Abstract authoritative store for logical PRA agent sessions."""

    @abstractmethod
    def create_session(
        self,
        user_id: str,
        session_id: str | None = None,
        *,
        tenant_id: str = "default",
        task_description: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AgentSessionState:
        """Create and persist a session, optionally with one active root task."""

    @abstractmethod
    def get_session(self, user_id: str, session_id: str) -> AgentSessionState:
        """Resolve one exact user/session pair."""

    @abstractmethod
    def resolve_session(self, user_id: str, session_id: str | None = None) -> AgentSessionState:
        """Resolve an exact session or the user's most recently updated session."""

    @abstractmethod
    def save_session(
        self, state: AgentSessionState, *, expected_version: int | None = None
    ) -> AgentSessionState:
        """Persist state with optional optimistic version validation."""

    @abstractmethod
    def list_sessions(self, user_id: str) -> tuple[AgentSessionState, ...]:
        """List a user's sessions newest first."""

    @abstractmethod
    def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete one logical session."""

    def append_record(
        self, user_id: str, session_id: str, record: ContextRecord
    ) -> AgentSessionState:
        current = self.get_session(user_id, session_id)
        return self.save_session(current.with_record(record), expected_version=current.version)

    def apply_task_event(
        self, user_id: str, session_id: str, event: TaskEvent
    ) -> AgentSessionState:
        current = self.get_session(user_id, session_id)
        return self.save_session(current.with_task_event(event), expected_version=current.version)


def _initial_state(
    user_id: str,
    session_id: str | None,
    tenant_id: str,
    task_description: str | None,
    metadata: Mapping[str, object] | None,
) -> AgentSessionState:
    session_id = session_id or uuid.uuid4().hex
    state = AgentSessionState(
        user_id=user_id,
        session_id=session_id,
        tenant_id=tenant_id,
        metadata=dict(metadata or {}),
    )
    if task_description:
        root_id = "task-1"
        create = TaskEvent(
            f"{session_id}:task:create:1", 1, TaskEventType.CREATE, root_id,
            payload={"description": task_description},
        )
        activate = TaskEvent(
            f"{session_id}:task:activate:2", 2, TaskEventType.ACTIVATE, root_id,
            expected_version=1,
        )
        state = state.with_task_event(create).with_task_event(activate)
    return state


class InMemorySessionService(SessionService):
    """Thread-safe process-local implementation for tests and ephemeral agents."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], AgentSessionState] = {}
        self._lock = threading.RLock()

    def create_session(self, user_id: str, session_id: str | None = None, **kwargs: object) -> AgentSessionState:
        state = _initial_state(
            user_id, session_id, str(kwargs.get("tenant_id", "default")),
            kwargs.get("task_description"), kwargs.get("metadata"),
        )
        key = (state.user_id, state.session_id)
        with self._lock:
            if key in self._states:
                raise SessionConflict(f"Session already exists: {key}")
            self._states[key] = state
        return state

    def get_session(self, user_id: str, session_id: str) -> AgentSessionState:
        with self._lock:
            try:
                return self._states[(user_id, session_id)]
            except KeyError as error:
                raise SessionNotFound((user_id, session_id)) from error

    def resolve_session(self, user_id: str, session_id: str | None = None) -> AgentSessionState:
        if session_id is not None:
            return self.get_session(user_id, session_id)
        values = self.list_sessions(user_id)
        if not values:
            raise SessionNotFound(user_id)
        return values[0]

    def save_session(self, state: AgentSessionState, *, expected_version: int | None = None) -> AgentSessionState:
        key = (state.user_id, state.session_id)
        with self._lock:
            current = self._states.get(key)
            if current is None:
                raise SessionNotFound(key)
            if expected_version is not None and current.version != expected_version:
                raise SessionConflict(
                    f"Session version is {current.version}, expected {expected_version}."
                )
            if state.version <= current.version:
                raise SessionConflict("Saved session version must increase.")
            self._states[key] = state
        return state

    def list_sessions(self, user_id: str) -> tuple[AgentSessionState, ...]:
        with self._lock:
            values = [state for (owner, _), state in self._states.items() if owner == user_id]
        return tuple(sorted(values, key=lambda state: (state.updated_at, state.session_id), reverse=True))

    def delete_session(self, user_id: str, session_id: str) -> None:
        with self._lock:
            if self._states.pop((user_id, session_id), None) is None:
                raise SessionNotFound((user_id, session_id))


def _policy_to_dict(policy: RecordPolicy) -> dict[str, object]:
    return {
        "record_type": policy.record_type.value,
        "selection": policy.selection.value,
        "authority": policy.authority.value,
        "atomicity": policy.atomicity.value,
        "materialization": policy.materialization.value,
        "allow_partial_tools": policy.allow_partial_tools,
        "initial_view": policy.initial_view.value,
        "selected_view": policy.selected_view.value,
    }


def _record_to_dict(record: ContextRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "record_type": record.record_type.value,
        "payload": record.payload,
        "parent_id": record.parent_id,
        "child_ids": list(record.child_ids),
        "boundaries": [vars(row) for row in record.boundaries],
        "selection_provenance": record.selection_provenance,
        "policy": _policy_to_dict(record.policy),
        "version": record.version,
        "source_fingerprint": record.source_fingerprint,
        "views": {
            name.value: {
                "name": view.name.value,
                "payload": view.payload,
                "fields": list(view.fields),
                "token_count": view.token_count,
            }
            for name, view in record.views.items()
        },
    }


def _record_from_dict(value: Mapping[str, object]) -> ContextRecord:
    return ContextRecord(
        record_id=str(value["record_id"]),
        record_type=str(value["record_type"]),
        payload=value["payload"],
        parent_id=value.get("parent_id"),
        child_ids=tuple(value.get("child_ids", ())),
        boundaries=tuple(RecordBoundary(**row) for row in value.get("boundaries", ())),
        selection_provenance=dict(value.get("selection_provenance", {})),
        policy=RecordPolicy(**dict(value["policy"])),
        version=str(value.get("version", "v1")),
        source_fingerprint=str(value.get("source_fingerprint", "")),
        views={
            RecordViewName(name): RecordView(**row)
            for name, row in dict(value.get("views", {})).items()
        },
    )


def session_to_dict(state: AgentSessionState) -> dict[str, object]:
    """Encode an authoritative session without model-cache state."""

    return {
        "schema_version": 1,
        "user_id": state.user_id,
        "session_id": state.session_id,
        "tenant_id": state.tenant_id,
        "version": state.version,
        "records": [_record_to_dict(record) for record in state.records],
        "tasks": state.tasks.to_dict(),
        "task_events": [event.to_dict() for event in state.task_events],
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "metadata": state.metadata,
    }


def session_from_dict(value: Mapping[str, object]) -> AgentSessionState:
    if int(value.get("schema_version", 1)) != 1:
        raise ValueError("Unsupported session schema version.")
    return AgentSessionState(
        user_id=str(value["user_id"]),
        session_id=str(value["session_id"]),
        tenant_id=str(value.get("tenant_id", "default")),
        version=int(value.get("version", 1)),
        records=tuple(_record_from_dict(row) for row in value.get("records", ())),
        tasks=TaskDescriptor.from_dict(dict(value.get("tasks", {}))),
        task_events=tuple(TaskEvent.from_dict(row) for row in value.get("task_events", ())),
        created_at=float(value.get("created_at", time.time())),
        updated_at=float(value.get("updated_at", time.time())),
        metadata=dict(value.get("metadata", {})),
    )


class LocalSessionService(SessionService):
    """Atomic JSON persistence below a user-hashed local directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    def _path(self, user_id: str, session_id: str) -> Path:
        return self.root / self._hash(user_id) / f"{self._hash(session_id)}.json"

    def _write(self, state: AgentSessionState) -> None:
        path = self._path(state.user_id, state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(session_to_dict(state), indent=2, sort_keys=True, ensure_ascii=True)
        handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def create_session(self, user_id: str, session_id: str | None = None, **kwargs: object) -> AgentSessionState:
        state = _initial_state(
            user_id, session_id, str(kwargs.get("tenant_id", "default")),
            kwargs.get("task_description"), kwargs.get("metadata"),
        )
        with self._lock:
            if self._path(state.user_id, state.session_id).exists():
                raise SessionConflict(f"Session already exists: {(state.user_id, state.session_id)}")
            self._write(state)
        return state

    def get_session(self, user_id: str, session_id: str) -> AgentSessionState:
        path = self._path(user_id, session_id)
        with self._lock:
            if not path.exists():
                raise SessionNotFound((user_id, session_id))
            state = session_from_dict(json.loads(path.read_text(encoding="utf-8")))
        if state.user_id != user_id or state.session_id != session_id:
            raise SessionConflict("Session manifest identity does not match its lookup key.")
        return state

    def resolve_session(self, user_id: str, session_id: str | None = None) -> AgentSessionState:
        if session_id is not None:
            return self.get_session(user_id, session_id)
        values = self.list_sessions(user_id)
        if not values:
            raise SessionNotFound(user_id)
        return values[0]

    def save_session(self, state: AgentSessionState, *, expected_version: int | None = None) -> AgentSessionState:
        with self._lock:
            current = self.get_session(state.user_id, state.session_id)
            if expected_version is not None and current.version != expected_version:
                raise SessionConflict(
                    f"Session version is {current.version}, expected {expected_version}."
                )
            if state.version <= current.version:
                raise SessionConflict("Saved session version must increase.")
            self._write(state)
        return state

    def list_sessions(self, user_id: str) -> tuple[AgentSessionState, ...]:
        directory = self.root / self._hash(user_id)
        if not directory.exists():
            return ()
        values = []
        with self._lock:
            for path in directory.glob("*.json"):
                try:
                    state = session_from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if state.user_id == user_id:
                    values.append(state)
        return tuple(sorted(values, key=lambda state: (state.updated_at, state.session_id), reverse=True))

    def delete_session(self, user_id: str, session_id: str) -> None:
        path = self._path(user_id, session_id)
        with self._lock:
            if not path.exists():
                raise SessionNotFound((user_id, session_id))
            path.unlink()
