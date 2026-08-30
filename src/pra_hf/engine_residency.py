"""Physical residency and request-lifetime control for logical PRA blocks.

The logical block store owns identity and authorization.  This module owns the
temporary mapping from those identities to engine-native payloads.  Payloads
remain opaque so MLX arrays, vLLM block handles, and SGLang cache objects can
share one lifecycle implementation without sharing tensor code.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterable, Iterator, Mapping

from .engine_memory import LogicalPRABlockStore, PRAResidencyState


class PRAEvictionPolicy(str, Enum):
    """Deterministic policies used before any learned eviction experiment."""

    LRU = "lru"
    SIZE_AWARE_LRU = "size_aware_lru"
    REUSE_COUNT = "reuse_count"
    RELOAD_COST = "reload_cost"


@dataclass(frozen=True)
class PRAResidencyEvent:
    """One auditable physical-memory transition or request action."""

    action: str
    logical_key: str
    timestamp_ns: int
    bytes: int = 0
    duration_ns: int = 0
    request_id: str | None = None
    reused: bool = False
    wasted: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _PhysicalEntry:
    payload: object
    handle: str
    byte_count: int
    loaded_ns: int
    last_access_ns: int
    load_duration_ns: int
    access_count: int = 0
    pin_count: int = 0
    prefetched: bool = False
    consumed_after_prefetch: bool = False


@dataclass(frozen=True)
class PRAResidencyMetrics:
    """Disjoint counters for placement, transfer, sharing, and prefetch."""

    resident_bytes: int
    resident_blocks: int
    peak_resident_bytes: int
    loads: int
    evictions: int
    reloads: int
    bytes_loaded: int
    bytes_evicted: int
    duplicate_transfer_bytes_avoided: int
    prefetches: int
    prefetch_hits: int
    wasted_prefetches: int
    late_block_stall_ns: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class EnginePRAResidencyManager:
    """Map authorized logical blocks to bounded engine-native payloads.

    A materializer is called outside the manager lock.  The resulting payload
    is installed atomically and may be shared by concurrent requests with the
    same immutable logical identity.  Request pinning prevents eviction until
    decode cleanup completes.
    """

    def __init__(
        self,
        block_store: LogicalPRABlockStore,
        *,
        max_resident_bytes: int,
        policy: PRAEvictionPolicy | str = PRAEvictionPolicy.LRU,
        prefetch_workers: int = 1,
        payload_disposer: Callable[[object], None] | None = None,
    ) -> None:
        if max_resident_bytes <= 0:
            raise ValueError("PRA residency budget must be positive.")
        if prefetch_workers <= 0:
            raise ValueError("PRA prefetch worker count must be positive.")
        self.block_store = block_store
        self.max_resident_bytes = int(max_resident_bytes)
        self.policy = PRAEvictionPolicy(policy)
        self._payload_disposer = payload_disposer or (lambda _payload: None)
        self._entries: dict[str, _PhysicalEntry] = {}
        self._futures: dict[str, Future[object]] = {}
        self._load_counts: dict[str, int] = {}
        self._events: list[PRAResidencyEvent] = []
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=prefetch_workers, thread_name_prefix="pra-prefetch"
        )
        self._peak_bytes = 0
        self._loads = 0
        self._evictions = 0
        self._reloads = 0
        self._bytes_loaded = 0
        self._bytes_evicted = 0
        self._duplicate_bytes_avoided = 0
        self._prefetches = 0
        self._prefetch_hits = 0
        self._wasted_prefetches = 0
        self._late_stall_ns = 0

    def close(self) -> None:
        """Stop background work and account for unused prefetched payloads."""

        self._pool.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            if any(entry.pin_count for entry in self._entries.values()):
                raise RuntimeError("Cannot close with request-pinned PRA blocks.")
            for key, entry in self._entries.items():
                if entry.prefetched and not entry.consumed_after_prefetch:
                    self._wasted_prefetches += 1
                    self._events.append(
                        PRAResidencyEvent(
                            "prefetch_wasted",
                            key,
                            time.monotonic_ns(),
                            bytes=entry.byte_count,
                            wasted=True,
                        )
                    )
                self._payload_disposer(entry.payload)
                current = self.block_store.get(key)
                if current.state != PRAResidencyState.INVALID:
                    self.block_store.transition(key, PRAResidencyState.OFF_DEVICE)
            self._entries.clear()

    def _resident_bytes(self) -> int:
        return sum(entry.byte_count for entry in self._entries.values())

    def _eviction_order(self) -> list[tuple[str, _PhysicalEntry]]:
        candidates = [row for row in self._entries.items() if row[1].pin_count == 0]
        if self.policy == PRAEvictionPolicy.LRU:
            key_fn = lambda row: (row[1].last_access_ns, row[0])
        elif self.policy == PRAEvictionPolicy.SIZE_AWARE_LRU:
            key_fn = lambda row: (
                row[1].last_access_ns / max(row[1].byte_count, 1),
                row[0],
            )
        elif self.policy == PRAEvictionPolicy.REUSE_COUNT:
            key_fn = lambda row: (row[1].access_count, row[1].last_access_ns, row[0])
        else:
            key_fn = lambda row: (
                row[1].access_count * max(row[1].load_duration_ns, 1),
                row[1].last_access_ns,
                row[0],
            )
        return sorted(candidates, key=key_fn)

    def release(self, key: str) -> int:
        """Release one unpinned HOT payload without invalidating its identity."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            if entry.pin_count:
                raise RuntimeError("Cannot release a request-pinned PRA block.")
            self._payload_disposer(entry.payload)
            del self._entries[key]
            current = self.block_store.get(key)
            if current.state != PRAResidencyState.INVALID:
                self.block_store.transition(key, PRAResidencyState.OFF_DEVICE)
            self._evictions += 1
            self._bytes_evicted += entry.byte_count
            self._events.append(
                PRAResidencyEvent(
                    "release_hot", key, time.monotonic_ns(), bytes=entry.byte_count
                )
            )
            return entry.byte_count

    def hot_bytes(self, key: str) -> int:
        """Return physical bytes for one HOT logical block, or zero."""

        with self._lock:
            entry = self._entries.get(key)
            return 0 if entry is None else entry.byte_count

    def get(self, key: str) -> object:
        """Return one resident engine payload without changing its pin state."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(key)
            entry.access_count += 1
            entry.last_access_ns = time.monotonic_ns()
            return entry.payload

    def pin(self, key: str, request_id: str) -> None:
        """Pin one resolved block until the matching request cleanup."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise RuntimeError("PRA blocks must be resolved before request pinning.")
            entry.pin_count += 1
            if entry.pin_count == 1:
                self.block_store.transition(key, PRAResidencyState.PINNED)
            self._events.append(
                PRAResidencyEvent("pin", key, time.monotonic_ns(), request_id=request_id)
            )

    def unpin(self, key: str, request_id: str) -> None:
        """Release one request pin while retaining the HOT payload."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.pin_count <= 0:
                raise RuntimeError("PRA block has no matching request pin.")
            entry.pin_count -= 1
            if entry.pin_count == 0:
                self.block_store.transition(key, PRAResidencyState.RESIDENT)
            self._events.append(
                PRAResidencyEvent("unpin", key, time.monotonic_ns(), request_id=request_id)
            )

    def _make_room(self, incoming_bytes: int) -> None:
        if incoming_bytes > self.max_resident_bytes:
            raise MemoryError("One PRA block exceeds the complete residency budget.")
        while self._resident_bytes() + incoming_bytes > self.max_resident_bytes:
            order = self._eviction_order()
            if not order:
                raise MemoryError("Pinned PRA blocks exhaust the residency budget.")
            key, entry = order[0]
            self._payload_disposer(entry.payload)
            del self._entries[key]
            self.block_store.transition(key, PRAResidencyState.OFF_DEVICE)
            self._evictions += 1
            self._bytes_evicted += entry.byte_count
            if entry.prefetched and not entry.consumed_after_prefetch:
                self._wasted_prefetches += 1
            self._events.append(
                PRAResidencyEvent(
                    "evict",
                    key,
                    time.monotonic_ns(),
                    bytes=entry.byte_count,
                    wasted=entry.prefetched and not entry.consumed_after_prefetch,
                )
            )

    def _load(
        self,
        key: str,
        materializer: Callable[[], tuple[object, int]],
        *,
        prefetched: bool,
    ) -> object:
        started = time.monotonic_ns()
        payload, byte_count = materializer()
        finished = time.monotonic_ns()
        byte_count = int(byte_count)
        if byte_count < 0:
            raise ValueError("Materialized PRA byte count cannot be negative.")
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._duplicate_bytes_avoided += byte_count
                return existing.payload
            self._make_room(byte_count)
            self.block_store.record_detail_bytes(key, byte_count)
            loads = self._load_counts.get(key, 0)
            handle = f"pra:{key}:{loads + 1}"
            entry = _PhysicalEntry(
                payload=payload,
                handle=handle,
                byte_count=byte_count,
                loaded_ns=finished,
                last_access_ns=finished,
                load_duration_ns=finished - started,
                prefetched=prefetched,
            )
            self._entries[key] = entry
            self._load_counts[key] = loads + 1
            self._loads += 1
            self._reloads += int(loads > 0)
            self._bytes_loaded += byte_count
            self._peak_bytes = max(self._peak_bytes, self._resident_bytes())
            current = self.block_store.get(key)
            if current.state == PRAResidencyState.PREFETCHING:
                self.block_store.transition(
                    key,
                    PRAResidencyState.RESIDENT,
                    physical_handles=(handle,),
                    storage_tier="engine",
                )
            elif current.state != PRAResidencyState.RESIDENT:
                self.block_store.transition(
                    key,
                    PRAResidencyState.RESIDENT,
                    physical_handles=(handle,),
                    storage_tier="engine",
                )
            self._events.append(
                PRAResidencyEvent(
                    "prefetch_complete" if prefetched else "load",
                    key,
                    finished,
                    bytes=byte_count,
                    duration_ns=finished - started,
                )
            )
            return payload

    def prefetch(
        self, key: str, materializer: Callable[[], tuple[object, int]]
    ) -> Future[object]:
        """Begin one deduplicated background materialization."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                future: Future[object] = Future()
                future.set_result(entry.payload)
                self._duplicate_bytes_avoided += entry.byte_count
                return future
            pending = self._futures.get(key)
            if pending is not None:
                return pending
            state = self.block_store.get(key).state
            if state in {PRAResidencyState.INDEXED_ONLY, PRAResidencyState.OFF_DEVICE}:
                self.block_store.transition(key, PRAResidencyState.PREFETCHING)
            self._prefetches += 1
            self._events.append(PRAResidencyEvent("prefetch_start", key, time.monotonic_ns()))
            future = self._pool.submit(self._load, key, materializer, prefetched=True)
            self._futures[key] = future

            def clear(done: Future[object]) -> None:
                with self._lock:
                    self._futures.pop(key, None)
                    if done.cancelled() or done.exception() is not None:
                        current = self.block_store.get(key)
                        if current.state == PRAResidencyState.PREFETCHING:
                            self.block_store.transition(key, PRAResidencyState.OFF_DEVICE)

            future.add_done_callback(clear)
            return future

    def resolve(
        self,
        key: str,
        materializer: Callable[[], tuple[object, int]],
        *,
        request_id: str | None = None,
    ) -> object:
        """Return a resident payload, waiting for or performing one load."""

        with self._lock:
            entry = self._entries.get(key)
            pending = self._futures.get(key)
            if entry is not None:
                entry.access_count += 1
                entry.last_access_ns = time.monotonic_ns()
                if entry.prefetched and not entry.consumed_after_prefetch:
                    entry.consumed_after_prefetch = True
                    self._prefetch_hits += 1
                self._events.append(
                    PRAResidencyEvent(
                        "reuse", key, entry.last_access_ns, request_id=request_id, reused=True
                    )
                )
                return entry.payload
        wait_started = time.monotonic_ns()
        payload = (
            pending.result()
            if pending is not None
            else self._load(key, materializer, prefetched=False)
        )
        waited = time.monotonic_ns() - wait_started
        with self._lock:
            self._late_stall_ns += waited
            entry = self._entries[key]
            entry.access_count += 1
            entry.last_access_ns = time.monotonic_ns()
            if entry.prefetched and not entry.consumed_after_prefetch:
                entry.consumed_after_prefetch = True
                self._prefetch_hits += 1
            self._events.append(
                PRAResidencyEvent(
                    "resolve",
                    key,
                    entry.last_access_ns,
                    bytes=entry.byte_count,
                    duration_ns=waited,
                    request_id=request_id,
                )
            )
        return payload

    @contextmanager
    def pin_request(self, request_id: str, keys: Iterable[str]) -> Iterator[None]:
        """Protect a fully resolved block set until request cleanup."""

        unique = tuple(dict.fromkeys(keys))
        with self._lock:
            missing = [key for key in unique if key not in self._entries]
            if missing:
                raise RuntimeError("PRA blocks must be resolved before request pinning.")
            for key in unique:
                self.pin(key, request_id)
        try:
            yield
        finally:
            with self._lock:
                for key in unique:
                    if key in self._entries:
                        self.unpin(key, request_id)

    def invalidate(self, keys: Iterable[str]) -> None:
        """Drop physical payloads after their logical versions are invalidated."""

        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and entry.pin_count:
                    raise RuntimeError("Cannot invalidate a request-pinned PRA block.")
                if entry is not None:
                    self._payload_disposer(entry.payload)
                    del self._entries[key]
                    self._bytes_evicted += entry.byte_count
                self._events.append(PRAResidencyEvent("invalidate", key, time.monotonic_ns()))

    def metrics(self) -> PRAResidencyMetrics:
        with self._lock:
            resident = self._resident_bytes()
            return PRAResidencyMetrics(
                resident_bytes=resident,
                resident_blocks=len(self._entries),
                peak_resident_bytes=self._peak_bytes,
                loads=self._loads,
                evictions=self._evictions,
                reloads=self._reloads,
                bytes_loaded=self._bytes_loaded,
                bytes_evicted=self._bytes_evicted,
                duplicate_transfer_bytes_avoided=self._duplicate_bytes_avoided,
                prefetches=self._prefetches,
                prefetch_hits=self._prefetch_hits,
                wasted_prefetches=self._wasted_prefetches,
                late_block_stall_ns=self._late_stall_ns,
            )

    def events(self) -> tuple[Mapping[str, object], ...]:
        with self._lock:
            return tuple(event.to_dict() for event in self._events)
