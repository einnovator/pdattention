"""Shared request-isolation invariants for engine-native PRA adapters.

PRA-selected detail is request-scoped external memory.  It must never become
part of an engine's ordinary prefix or sequential cache namespace, where it
could leak into another request or be attached twice through cache reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Iterable


@dataclass(frozen=True)
class EnginePRARequestView:
    """Immutable diagnostic view of one active request's selected resources."""

    request_id: str
    logical_keys: tuple[str, ...]
    attached: bool


@dataclass
class _RequestState:
    logical_keys: tuple[str, ...]
    attached: bool = False


class EnginePRAIsolationGuard:
    """Enforce request isolation and exactly-once native-memory attachment.

    Engine adapters open a request after routing, call :meth:`attach_once`
    immediately before exposing selected K/V to attention, and close it during
    request cleanup.  Ordinary prefix/cache-pool paths call
    :meth:`assert_ordinary_pool_safe` before accepting cache identities.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _RequestState] = {}
        self._lock = RLock()

    @staticmethod
    def _normalize_keys(logical_keys: Iterable[str]) -> tuple[str, ...]:
        keys = tuple(map(str, logical_keys))
        if len(keys) != len(set(keys)):
            raise ValueError("PRA selected-memory keys must be unique per request.")
        return keys

    def open_request(self, request_id: str, logical_keys: Iterable[str]) -> None:
        """Register the complete selected-memory set for one active request."""

        identifier = str(request_id)
        keys = self._normalize_keys(logical_keys)
        with self._lock:
            if identifier in self._requests:
                raise RuntimeError(f"PRA request {identifier!r} is already active.")
            self._requests[identifier] = _RequestState(keys)

    def attach_once(
        self, request_id: str, logical_keys: Iterable[str] | None = None
    ) -> None:
        """Mark the request's selected K/V as attached to attention exactly once."""

        identifier = str(request_id)
        with self._lock:
            state = self._requests.get(identifier)
            if state is None:
                raise KeyError(f"PRA request {identifier!r} is not active.")
            if logical_keys is not None:
                keys = self._normalize_keys(logical_keys)
                if keys != state.logical_keys:
                    raise RuntimeError(
                        "PRA attachment keys differ from the request's routed keys."
                    )
            if state.attached:
                raise RuntimeError(
                    f"PRA selected memory is already attached to request {identifier!r}."
                )
            state.attached = True

    def close_request(self, request_id: str, *, require_attached: bool = True) -> None:
        """Remove all request visibility, optionally requiring an attachment."""

        identifier = str(request_id)
        with self._lock:
            state = self._requests.pop(identifier, None)
            if state is None:
                return
            if require_attached and state.logical_keys and not state.attached:
                raise RuntimeError(
                    f"PRA request {identifier!r} closed before selected memory was attached."
                )

    def visible_keys(self, request_id: str) -> tuple[str, ...]:
        """Return only keys explicitly selected for the active request."""

        with self._lock:
            state = self._requests.get(str(request_id))
            return () if state is None else state.logical_keys

    def assert_ordinary_pool_safe(self, logical_keys: Iterable[str]) -> None:
        """Reject PRA identities offered to an ordinary prefix/cache pool."""

        keys = self._normalize_keys(logical_keys)
        if keys:
            raise RuntimeError(
                "PRA-selected detail must not enter an ordinary sequential or prefix "
                f"cache pool: {keys!r}."
            )

    def view(self, request_id: str) -> EnginePRARequestView | None:
        """Return a stable diagnostic snapshot for tests and engine telemetry."""

        identifier = str(request_id)
        with self._lock:
            state = self._requests.get(identifier)
            if state is None:
                return None
            return EnginePRARequestView(identifier, state.logical_keys, state.attached)

    def close(self) -> None:
        """Clear request-scoped metadata during adapter shutdown."""

        with self._lock:
            self._requests.clear()
