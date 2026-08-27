"""Task-aware candidate scoping and working-set lifecycle for Paper 8."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .context_records import ContextRecord, RecordViewName, serialize_record
from .task_context import TaskGraph, TaskProvenance, TaskStatus


class TaskScopePolicy(str, Enum):
    """Scope breadth applied before ordinary PRA discovery."""

    SESSION = "session"
    TASK_LOCAL = "task_local"
    TASK_STRUCTURAL = "task_structural"
    TASK_ADAPTIVE = "task_adaptive"


class DetailDepth(str, Enum):
    """Paper-7 materialization depth, orthogonal to task scope."""

    INDEX = "index"
    COMPACT = "compact"
    SELECTED = "selected"
    FULL = "full"
    NATIVE_KV = "native_kv"


class ResidencyState(str, Enum):
    """Availability of one task working set across host and device tiers."""

    COLD = "cold"
    ENCODED_HOST = "encoded_host"
    DEVICE_HOT = "device_hot"
    # Source-compatible aliases for the original logical terminology.
    WARM = "encoded_host"
    HOT = "device_hot"


@dataclass(frozen=True)
class ScopePartition:
    """Task-derived routing restriction before any lexical or model ranking."""

    policy: TaskScopePolicy
    active_task_id: str
    admitted_task_ids: tuple[str, ...]
    candidate_records: tuple[ContextRecord, ...]
    excluded_record_ids: tuple[str, ...]
    widened: bool
    scope_seconds: float

    @property
    def admitted_record_ids(self) -> tuple[str, ...]:
        return tuple(record.record_id for record in self.candidate_records)


@dataclass(frozen=True)
class ScopeSelection:
    """Candidate admission result before ordinary record/chunk ranking."""

    policy: TaskScopePolicy
    active_task_id: str
    admitted_task_ids: tuple[str, ...]
    candidate_records: tuple[ContextRecord, ...]
    selected_records: tuple[ContextRecord, ...]
    excluded_record_ids: tuple[str, ...]
    widened: bool
    scope_seconds: float

    @property
    def selected_record_ids(self) -> tuple[str, ...]:
        return tuple(record.record_id for record in self.selected_records)

    @property
    def active_tokens(self) -> int:
        return sum(
            len(serialize_record(record, view=_visible_view(record)).split())
            for record in self.selected_records
        )


def _visible_view(record: ContextRecord) -> RecordViewName:
    for name in (RecordViewName.COMPACT, record.policy.initial_view, RecordViewName.FULL):
        if name in record.views:
            return name
    return RecordViewName.FULL


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _record_task_id(record: ContextRecord) -> str | None:
    provenance = TaskProvenance.from_record(record)
    return provenance.task_id if provenance else None


class TaskScopeSelector:
    """Apply task structure first, then a deterministic PRA-like ranker.

    The ranker is deliberately simple and frozen.  Paper 8 varies scope while
    keeping within-scope discovery fixed, preserving the causal decomposition.
    """

    def __init__(self, graph: TaskGraph, records: Sequence[ContextRecord]) -> None:
        self.graph = graph
        self.records = tuple(records)

    def _tasks_for(self, task_id: str, policy: TaskScopePolicy) -> tuple[str, ...]:
        if policy == TaskScopePolicy.SESSION:
            return tuple(sorted(self.graph.tasks))
        if policy == TaskScopePolicy.TASK_LOCAL:
            return (task_id,)
        if policy == TaskScopePolicy.TASK_STRUCTURAL:
            return self.graph.structural_closure(task_id)
        return self.graph.related_tasks(task_id)

    def partition(
        self,
        task_id: str,
        *,
        policy: TaskScopePolicy | str,
        minimum_records: int = 1,
        metadata_complete: bool = True,
    ) -> ScopePartition:
        """Return a reusable routing restriction without ranking candidates.

        The result contains only stable task and record identities, so a local
        PRA model, deployment gateway, or remote runtime can apply the same
        scope before its own production discovery implementation.
        """

        if minimum_records <= 0:
            raise ValueError("minimum_records must be positive.")
        started = time.perf_counter()
        policy = TaskScopePolicy(policy)
        admitted_tasks = list(self._tasks_for(task_id, policy))
        candidates = [
            record for record in self.records
            if _record_task_id(record) in admitted_tasks or _record_task_id(record) is None
        ]
        widened = False
        if policy == TaskScopePolicy.TASK_ADAPTIVE and (
            not metadata_complete or len(candidates) < minimum_records
        ):
            admitted_tasks = list(sorted(self.graph.tasks))
            candidates = list(self.records)
            widened = True
        candidate_ids = {record.record_id for record in candidates}
        return ScopePartition(
            policy=policy,
            active_task_id=task_id,
            admitted_task_ids=tuple(admitted_tasks),
            candidate_records=tuple(candidates),
            excluded_record_ids=tuple(
                record.record_id for record in self.records
                if record.record_id not in candidate_ids
            ),
            widened=widened,
            scope_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _score(record: ContextRecord, query: str, position: int) -> tuple[float, int, str]:
        view = _visible_view(record)
        text = serialize_record(record, view=view)
        overlap = len(_terms(query) & _terms(text))
        provenance = TaskProvenance.from_record(record)
        recency = provenance.event_sequence if provenance else position
        return float(overlap), recency, record.record_id

    def select(
        self,
        task_id: str,
        query: str,
        *,
        policy: TaskScopePolicy | str,
        max_records: int,
        minimum_records: int = 1,
    ) -> ScopeSelection:
        """Return at most ``max_records`` after task admission and fixed ranking."""

        if max_records <= 0 or minimum_records <= 0:
            raise ValueError("Record budgets must be positive.")
        partition = self.partition(
            task_id,
            policy=policy,
            minimum_records=minimum_records,
        )
        candidates = list(partition.candidate_records)
        ranked = sorted(
            enumerate(candidates),
            key=lambda row: self._score(row[1], query, row[0]),
            reverse=True,
        )
        selected = tuple(record for _, record in ranked[:max_records])
        selected_ids = {record.record_id for record in selected}
        return ScopeSelection(
            policy=partition.policy,
            active_task_id=task_id,
            admitted_task_ids=partition.admitted_task_ids,
            candidate_records=tuple(candidates),
            selected_records=selected,
            excluded_record_ids=tuple(
                record.record_id for record in self.records if record.record_id not in selected_ids
            ),
            widened=partition.widened,
            scope_seconds=partition.scope_seconds,
        )


@dataclass(frozen=True)
class TaskResidency:
    """Current cache/materialization state for one task working set."""

    task_id: str
    state: ResidencyState = ResidencyState.COLD
    encoded_native_tokens: int = 0
    device_active_tokens: int = 0
    backing_bytes: int = 0
    transitions: int = 0

    @property
    def native_tokens(self) -> int:
        """Compatibility alias for encoded native tokens."""

        return self.encoded_native_tokens


@dataclass(frozen=True)
class TaskSwitchResult:
    """Accounting for one deterministic task activation or completion."""

    task_id: str
    previous_task_id: str | None
    previous_state: ResidencyState
    new_state: ResidencyState
    encoded_tokens_reused: int
    device_tokens_promoted: int
    device_tokens_demoted: int
    seconds: float

    @property
    def kv_reused(self) -> int:
        return self.encoded_tokens_reused

    @property
    def kv_promoted(self) -> int:
        return self.device_tokens_promoted

    @property
    def kv_demoted(self) -> int:
        return self.device_tokens_demoted


class TaskWorkingSet:
    """Account for cold, host-encoded, and device-active task state."""

    def __init__(self) -> None:
        self._residency: dict[str, TaskResidency] = {}
        self.active_task_id: str | None = None

    def residency(self, task_id: str) -> TaskResidency:
        return self._residency.get(task_id, TaskResidency(task_id))

    def register_backing(self, task_id: str, *, backing_bytes: int) -> TaskResidency:
        """Register exact backing bytes without claiming model encoding exists."""

        current = self.residency(task_id)
        updated = TaskResidency(
            task_id,
            current.state,
            current.encoded_native_tokens,
            current.device_active_tokens,
            max(current.backing_bytes, int(backing_bytes)),
            current.transitions + 1,
        )
        self._residency[task_id] = updated
        return updated

    def register_encoded(
        self,
        task_id: str,
        *,
        native_tokens: int,
        backing_bytes: int = 0,
    ) -> TaskResidency:
        """Record host-cached native state without treating it as device-active."""

        current = self.residency(task_id)
        state = (
            ResidencyState.DEVICE_HOT
            if current.state == ResidencyState.DEVICE_HOT
            else ResidencyState.ENCODED_HOST
        )
        updated = TaskResidency(
            task_id,
            state,
            max(current.encoded_native_tokens, int(native_tokens)),
            current.device_active_tokens,
            max(current.backing_bytes, int(backing_bytes)),
            current.transitions + 1,
        )
        self._residency[task_id] = updated
        return updated

    def activate(self, task_id: str, *, native_tokens: int) -> TaskSwitchResult:
        started = time.perf_counter()
        previous_id = self.active_task_id
        previous = self.residency(previous_id) if previous_id else TaskResidency("")
        demoted = 0
        if previous_id and previous_id != task_id and previous.state == ResidencyState.DEVICE_HOT:
            demoted = previous.device_active_tokens
            self._residency[previous_id] = TaskResidency(
                previous_id,
                ResidencyState.ENCODED_HOST,
                previous.encoded_native_tokens,
                0,
                previous.backing_bytes,
                previous.transitions + 1,
            )
        current = self.residency(task_id)
        requested = int(native_tokens)
        if current.state == ResidencyState.DEVICE_HOT:
            reused = min(current.device_active_tokens, requested)
            promoted = max(0, requested - current.device_active_tokens)
        else:
            reused = min(current.encoded_native_tokens, requested)
            promoted = requested
        self._residency[task_id] = TaskResidency(
            task_id,
            ResidencyState.DEVICE_HOT,
            max(current.encoded_native_tokens, requested),
            requested,
            current.backing_bytes,
            current.transitions + 1,
        )
        self.active_task_id = task_id
        return TaskSwitchResult(
            task_id, previous_id, previous.state, ResidencyState.DEVICE_HOT,
            reused, promoted, demoted, time.perf_counter() - started,
        )

    def complete(self, task_id: str) -> TaskSwitchResult:
        started = time.perf_counter()
        current = self.residency(task_id)
        demoted = (
            current.device_active_tokens
            if current.state == ResidencyState.DEVICE_HOT
            else 0
        )
        new_state = (
            ResidencyState.ENCODED_HOST
            if current.encoded_native_tokens
            else ResidencyState.COLD
        )
        self._residency[task_id] = TaskResidency(
            task_id,
            new_state,
            current.encoded_native_tokens,
            0,
            current.backing_bytes,
            current.transitions + 1,
        )
        if self.active_task_id == task_id:
            self.active_task_id = None
        return TaskSwitchResult(
            task_id, task_id, current.state, new_state, 0, 0, demoted,
            time.perf_counter() - started,
        )

    def snapshot(self) -> Mapping[str, TaskResidency]:
        return dict(self._residency)
