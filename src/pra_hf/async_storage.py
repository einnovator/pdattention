"""Event-loop-owned prefetch and HOT admission for PRA storage.

The lifecycle manager intentionally exposes synchronous promotion because an
engine may restore device objects, mmap segments, or remote blobs.  Serving
schedulers should not perform that work on their request loop.  This module
owns the asynchronous request-side coordination while the caller's event loop
owns the executor and task lifetime.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from functools import partial
import time

from .storage_lifecycle import PRAStorageManager, PRAStorageTier


@dataclass(frozen=True)
class PRAHotAdmissionCandidate:
    """One reusable logical object considered for proactive HOT promotion."""

    logical_key: str
    expected_reuse: float = 1.0
    priority: float = 0.0


@dataclass(frozen=True)
class PRAHotAdmissionDecision:
    """Explain whether one candidate entered the bounded prefetch set."""

    logical_key: str
    admitted: bool
    detail_bytes: int
    expected_reuse: float
    priority: float
    reason: str


@dataclass
class PRAAsyncPromotionMetrics:
    """Disjoint scheduling, admission, transfer, and demand counters."""

    scheduled: int = 0
    coalesced: int = 0
    already_hot: int = 0
    completed: int = 0
    failed: int = 0
    admission_rejected: int = 0
    ready_at_demand: int = 0
    late_demands: int = 0
    wasted_prefetches: int = 0
    bytes_scheduled: int = 0
    bytes_promoted: int = 0
    demand_stall_ns: int = 0
    promotion_latency_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class PRAAsyncPromotionScheduler:
    """Coordinate nonblocking lifecycle promotion on a serving event loop.

    ``prefetch`` returns immediately with an ``asyncio.Task``.  Blocking WARM,
    COLD, or SOURCE work runs through the event loop's executor, and duplicate
    requests share the same task.  ``resolve`` is the demand boundary: it
    records whether prefetch completed in time and waits only for the remaining
    promotion.  Engine code should pin the returned HOT value separately for
    the exact request lifetime.
    """

    def __init__(
        self,
        storage: PRAStorageManager,
        *,
        max_inflight: int = 2,
        executor: Executor | None = None,
    ) -> None:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive.")
        self.storage = storage
        self.executor = executor
        self._semaphore = asyncio.Semaphore(max_inflight)
        self._tasks: dict[str, asyncio.Task[object]] = {}
        self._consumed: set[str] = set()
        self._metrics = PRAAsyncPromotionMetrics()
        self._closed = False

    def _entry_bytes(self, key: str) -> int:
        try:
            return int(self.storage.entries[key].detail_bytes)
        except KeyError as error:
            raise KeyError(f"Unknown PRA storage key: {key}") from error

    async def _promote(
        self,
        key: str,
        *,
        tenant_id: str | None,
        authorization_scopes: tuple[str, ...],
    ) -> object:
        async with self._semaphore:
            started = time.monotonic_ns()
            loop = asyncio.get_running_loop()
            operation = partial(
                self.storage.promote,
                key,
                tenant_id=tenant_id,
                authorization_scopes=authorization_scopes,
            )
            try:
                value = await loop.run_in_executor(self.executor, operation)
            except BaseException:
                self._metrics.failed += 1
                raise
            elapsed = time.monotonic_ns() - started
            self._metrics.completed += 1
            self._metrics.bytes_promoted += self._entry_bytes(key)
            self._metrics.promotion_latency_ns += elapsed
            return value

    def prefetch(
        self,
        key: str,
        *,
        tenant_id: str | None = None,
        authorization_scopes: Iterable[str] = (),
    ) -> asyncio.Task[object]:
        """Schedule one deduplicated promotion without blocking the caller."""

        if self._closed:
            raise RuntimeError("PRA async promotion scheduler is closed.")
        pending = self._tasks.get(key)
        if pending is not None:
            self._metrics.coalesced += 1
            return pending
        if self.storage.entries[key].current_tier is PRAStorageTier.HOT:
            self._metrics.already_hot += 1

        self._metrics.scheduled += 1
        self._metrics.bytes_scheduled += self._entry_bytes(key)
        task = asyncio.create_task(
            self._promote(
                key,
                tenant_id=tenant_id,
                authorization_scopes=tuple(authorization_scopes),
            ),
            name=f"pra-promote:{key}",
        )
        self._tasks[key] = task
        return task

    async def resolve(
        self,
        key: str,
        *,
        tenant_id: str | None = None,
        authorization_scopes: Iterable[str] = (),
    ) -> object:
        """Return HOT detail and report whether demand outran prefetch."""

        task = self._tasks.get(key)
        if task is None:
            task = self.prefetch(
                key,
                tenant_id=tenant_id,
                authorization_scopes=authorization_scopes,
            )
        ready = task.done()
        if ready:
            self._metrics.ready_at_demand += 1
        else:
            self._metrics.late_demands += 1
        wait_started = time.monotonic_ns()
        value = await task
        self._metrics.demand_stall_ns += time.monotonic_ns() - wait_started
        self._consumed.add(key)
        return value

    def admit_hot_set(
        self,
        candidates: Iterable[PRAHotAdmissionCandidate],
        *,
        max_prefetch_bytes: int | None = None,
        min_expected_reuse: float = 1.0,
        tenant_id: str | None = None,
        authorization_scopes: Iterable[str] = (),
    ) -> tuple[PRAHotAdmissionDecision, ...]:
        """Schedule the highest-value candidates within a proactive byte cap."""

        if max_prefetch_bytes is None:
            max_prefetch_bytes = self.storage.policy.hot.max_bytes
        remaining = float("inf") if max_prefetch_bytes is None else max_prefetch_bytes
        scopes = tuple(authorization_scopes)
        ranked = sorted(
            candidates,
            key=lambda item: (-item.expected_reuse, -item.priority, item.logical_key),
        )
        decisions = []
        for candidate in ranked:
            size = self._entry_bytes(candidate.logical_key)
            if candidate.expected_reuse < min_expected_reuse:
                admitted, reason = False, "reuse_below_threshold"
            elif size > remaining:
                admitted, reason = False, "prefetch_byte_budget"
            else:
                admitted, reason = True, "admitted"
                remaining -= size
                self.prefetch(
                    candidate.logical_key,
                    tenant_id=tenant_id,
                    authorization_scopes=scopes,
                )
            if not admitted:
                self._metrics.admission_rejected += 1
            decisions.append(
                PRAHotAdmissionDecision(
                    logical_key=candidate.logical_key,
                    admitted=admitted,
                    detail_bytes=size,
                    expected_reuse=candidate.expected_reuse,
                    priority=candidate.priority,
                    reason=reason,
                )
            )
        return tuple(decisions)

    async def close(self) -> None:
        """Drain submitted work and account for prefetched-but-unused objects."""

        if self._closed:
            return
        self._closed = True
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._metrics.wasted_prefetches += sum(
            task.done()
            and not task.cancelled()
            and task.exception() is None
            and key not in self._consumed
            for key, task in self._tasks.items()
        )

    def metrics(self) -> PRAAsyncPromotionMetrics:
        """Return a detached scheduler-metrics snapshot."""

        return PRAAsyncPromotionMetrics(**self._metrics.to_dict())

    def pending(self) -> Mapping[str, asyncio.Task[object]]:
        """Expose a read-only copy for serving-loop diagnostics."""

        return dict(self._tasks)
