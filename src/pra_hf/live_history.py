"""Engine-neutral contracts for selecting records from resident agent K/V.

The selection plan describes logical token coordinates in the canonical live
agent trajectory.  Engines may expose those cells as views, sequence
memberships, or page references, but must not reconstruct them from text.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, order=True)
class LiveKVInterval:
    """Half-open token interval in the canonical source-position frame."""

    start: int
    end: int
    record_id: str = ""
    causal_group_id: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Live K/V intervals must be non-empty and non-negative.")

    @property
    def tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LiveKVSelectionPlan:
    """Validated resident-K/V selection for one request.

    ``source_position_base`` is the absolute position of the request suffix.
    It is intentionally independent of ``selected_tokens``: sparse selections
    can contain holes while their already-RoPE-encoded K/V retain original
    coordinates.
    """

    source_tokens: int
    source_position_base: int
    intervals: tuple[LiveKVInterval, ...]

    def __post_init__(self) -> None:
        if self.source_tokens < 0:
            raise ValueError("source_tokens cannot be negative.")
        if self.source_position_base < self.source_tokens:
            raise ValueError("source_position_base cannot precede the live source.")
        prior_end = 0
        identities: set[str] = set()
        for interval in self.intervals:
            if interval.end > self.source_tokens:
                raise ValueError("A live K/V interval extends past the source cache.")
            if interval.start < prior_end:
                raise ValueError("Live K/V intervals must be ordered and disjoint.")
            prior_end = interval.end
            if interval.record_id:
                if interval.record_id in identities:
                    raise ValueError("A stable record may occur only once in a plan.")
                identities.add(interval.record_id)

    @classmethod
    def full(cls, source_tokens: int) -> "LiveKVSelectionPlan":
        source_tokens = int(source_tokens)
        intervals = (
            (LiveKVInterval(0, source_tokens, "full-history", "full-history"),)
            if source_tokens
            else ()
        )
        return cls(source_tokens, source_tokens, intervals)

    @classmethod
    def create(
        cls,
        source_tokens: int,
        intervals: Iterable[LiveKVInterval | tuple[int, int]],
        *,
        source_position_base: int | None = None,
    ) -> "LiveKVSelectionPlan":
        normalized = tuple(
            value
            if isinstance(value, LiveKVInterval)
            else LiveKVInterval(int(value[0]), int(value[1]))
            for value in intervals
        )
        return cls(
            int(source_tokens),
            int(source_tokens if source_position_base is None else source_position_base),
            normalized,
        )

    @property
    def selected_tokens(self) -> int:
        return sum(interval.tokens for interval in self.intervals)

    @property
    def full_retention(self) -> bool:
        # Full retention is a coverage property, not an interval-identity
        # property. Record-aware plans commonly preserve adjacent logical
        # identities even when their union covers the complete source. Those
        # plans must take the dense semantic-no-op path instead of introducing
        # a segmented reduction solely because a record boundary is present.
        return self.source_tokens == 0 or (
            self.source_position_base == self.source_tokens
            and self.selected_tokens == self.source_tokens
        )

    @property
    def has_holes(self) -> bool:
        return self.selected_tokens != self.source_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "source_tokens": self.source_tokens,
            "source_position_base": self.source_position_base,
            "selected_tokens": self.selected_tokens,
            "full_retention": self.full_retention,
            "has_holes": self.has_holes,
            "intervals": [
                {
                    "start": row.start,
                    "end": row.end,
                    "record_id": row.record_id,
                    "causal_group_id": row.causal_group_id,
                }
                for row in self.intervals
            ],
        }


@dataclass(frozen=True)
class LiveKVSourceView:
    """Immutable lifecycle view of one canonical resident history source."""

    source_id: str
    tenant_id: str
    session_id: str
    generation: int
    tier: str
    active_request_ids: tuple[str, ...]


@dataclass
class _LiveKVSource(Generic[T]):
    tenant_id: str
    session_id: str
    generation: int
    hot_value: T | None
    offloaded_value: object | None = None
    active_request_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.active_request_ids is None:
            self.active_request_ids = set()


class LiveKVSourceRegistry(Generic[T]):
    """Own canonical history K/V independently from request membership.

    Engine-native adapters use this registry for the lifecycle invariant that
    a request may borrow an immutable source, but releasing or cancelling the
    request never deletes the source.  Eviction is forbidden while any request
    is borrowing the source.  Session termination is a tombstone: queued or
    stale work cannot recreate that session accidentally.

    The registry deliberately does not prescribe a tensor/page representation.
    ``dump`` and ``load`` callbacks provide engine-specific lossless offload.
    """

    def __init__(
        self,
        *,
        dump: Callable[[T], object] | None = None,
        load: Callable[[object], T] | None = None,
    ) -> None:
        if (dump is None) != (load is None):
            raise ValueError("Live K/V offload requires both dump and load callbacks.")
        self._dump = dump
        self._load = load
        self._sources: dict[str, _LiveKVSource[T]] = {}
        self._requests: dict[str, tuple[str, ...]] = {}
        self._terminated_sessions: set[tuple[str, str]] = set()
        self._lock = RLock()

    @staticmethod
    def _identity(value: str, label: str) -> str:
        normalized = str(value)
        if not normalized:
            raise ValueError(f"{label} cannot be empty.")
        return normalized

    def register(
        self,
        source_id: str,
        value: T,
        *,
        tenant_id: str,
        session_id: str,
        generation: int,
    ) -> None:
        """Register or atomically replace an unborrowed canonical source."""

        source = self._identity(source_id, "source_id")
        tenant = self._identity(tenant_id, "tenant_id")
        session = self._identity(session_id, "session_id")
        version = int(generation)
        if version < 0:
            raise ValueError("Live K/V generation cannot be negative.")
        with self._lock:
            if (tenant, session) in self._terminated_sessions:
                raise RuntimeError("Cannot register K/V for a terminated session.")
            prior = self._sources.get(source)
            if prior is not None:
                if prior.active_request_ids:
                    raise RuntimeError("Cannot replace live K/V while requests borrow it.")
                if (prior.tenant_id, prior.session_id) != (tenant, session):
                    raise RuntimeError("Live K/V source identity crossed tenant/session scope.")
                if version <= prior.generation:
                    raise RuntimeError("Live K/V replacement must advance its generation.")
            self._sources[source] = _LiveKVSource(
                tenant, session, version, value
            )

    def borrow(
        self,
        request_id: str,
        source_ids: Iterable[str],
        *,
        tenant_id: str,
        session_id: str,
        expected_generations: Iterable[int],
    ) -> tuple[T, ...]:
        """Pin selected sources for one request and return their hot values."""

        request = self._identity(request_id, "request_id")
        sources = tuple(self._identity(value, "source_id") for value in source_ids)
        generations = tuple(map(int, expected_generations))
        tenant = self._identity(tenant_id, "tenant_id")
        session = self._identity(session_id, "session_id")
        if not sources or len(sources) != len(set(sources)):
            raise ValueError("A live K/V borrow needs unique source identities.")
        if len(sources) != len(generations):
            raise ValueError("Expected generations must match selected sources.")
        with self._lock:
            if request in self._requests:
                raise RuntimeError(f"Live K/V request {request!r} is already active.")
            if (tenant, session) in self._terminated_sessions:
                raise RuntimeError("Cannot borrow K/V for a terminated session.")
            rows: list[_LiveKVSource[T]] = []
            for source, generation in zip(sources, generations):
                row = self._sources.get(source)
                if row is None:
                    raise KeyError(f"Unknown live K/V source {source!r}.")
                if (row.tenant_id, row.session_id) != (tenant, session):
                    raise RuntimeError("Live K/V borrow crossed tenant/session scope.")
                if row.generation != generation:
                    raise RuntimeError(
                        f"Stale live K/V generation for {source!r}: "
                        f"expected {generation}, current {row.generation}."
                    )
                if row.hot_value is None:
                    if row.offloaded_value is None or self._load is None:
                        raise RuntimeError("Live K/V source is evicted without a restore path.")
                    row.hot_value = self._load(row.offloaded_value)
                rows.append(row)
            self._requests[request] = sources
            for row in rows:
                assert row.active_request_ids is not None
                row.active_request_ids.add(request)
            return tuple(row.hot_value for row in rows)  # type: ignore[misc]

    def release(self, request_id: str) -> bool:
        """Detach request membership without deleting canonical source state."""

        request = str(request_id)
        with self._lock:
            sources = self._requests.pop(request, None)
            if sources is None:
                return False
            for source in sources:
                row = self._sources.get(source)
                if row is not None and row.active_request_ids is not None:
                    row.active_request_ids.discard(request)
            return True

    cancel = release

    def offload(self, source_id: str) -> object:
        """Losslessly offload an idle source and clear its hot representation."""

        source = str(source_id)
        with self._lock:
            row = self._sources[source]
            if row.active_request_ids:
                raise RuntimeError("Cannot offload live K/V while requests borrow it.")
            if row.hot_value is None:
                if row.offloaded_value is None:
                    raise RuntimeError("Live K/V source has neither hot nor offloaded state.")
                return row.offloaded_value
            if self._dump is None:
                raise RuntimeError("This live K/V registry has no offload codec.")
            row.offloaded_value = self._dump(row.hot_value)
            row.hot_value = None
            return row.offloaded_value

    def terminate_session(self, tenant_id: str, session_id: str) -> int:
        """Tombstone a session and remove its idle sources and request pins."""

        tenant, session = str(tenant_id), str(session_id)
        with self._lock:
            self._terminated_sessions.add((tenant, session))
            requests = tuple(
                request
                for request, sources in self._requests.items()
                if any(
                    (self._sources[source].tenant_id, self._sources[source].session_id)
                    == (tenant, session)
                    for source in sources
                )
            )
            for request in requests:
                self.release(request)
            removed = 0
            for source in tuple(self._sources):
                row = self._sources[source]
                if (row.tenant_id, row.session_id) == (tenant, session):
                    if row.active_request_ids:
                        raise RuntimeError("Session termination left a borrowed K/V source.")
                    del self._sources[source]
                    removed += 1
            return removed

    def view(self, source_id: str) -> LiveKVSourceView | None:
        with self._lock:
            row = self._sources.get(str(source_id))
            if row is None:
                return None
            return LiveKVSourceView(
                str(source_id),
                row.tenant_id,
                row.session_id,
                row.generation,
                "hot" if row.hot_value is not None else "offloaded",
                tuple(sorted(row.active_request_ids or ())),
            )

    def close(self) -> None:
        with self._lock:
            self._requests.clear()
            self._sources.clear()
            self._terminated_sessions.clear()
