"""V1 step metadata for non-prefix PRA pages.

The vLLM scheduler remains authoritative for sequential request tokens. PRA
block tables and their source-position frame travel beside that state so an
attention backend can consume them without turning selected memory into a
prefix-cache hit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class VLLMNativeBlockSet:
    """Immutable selected pages registered for one active request."""

    logical_keys: tuple[str, ...]
    block_ids_by_group: tuple[tuple[int, ...], ...]
    selected_token_count: int
    source_position_base: int
    consumer_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.logical_keys:
            raise ValueError("A vLLM PRA block set requires logical identities.")
        if not self.block_ids_by_group or any(
            not group for group in self.block_ids_by_group
        ):
            raise ValueError("A vLLM PRA block set requires physical pages per group.")
        if self.selected_token_count <= 0 or self.source_position_base < 0:
            raise ValueError("PRA token geometry must be positive and positioned.")
        if len(set(self.consumer_layers)) != len(self.consumer_layers) or any(
            layer < 0 for layer in self.consumer_layers
        ):
            raise ValueError("PRA consumer layers must be unique nonnegative indices.")


@dataclass(frozen=True)
class VLLMNativeStep:
    """One request's disjoint scheduler and selected-memory attention view."""

    request_id: str
    scheduler_cache_start: int
    query_token_count: int
    selected: VLLMNativeBlockSet | None

    @property
    def query_position_start(self) -> int:
        base = 0 if self.selected is None else self.selected.source_position_base
        return base + self.scheduler_cache_start

    @property
    def attention_key_tokens(self) -> int:
        selected_tokens = (
            0 if self.selected is None else self.selected.selected_token_count
        )
        return selected_tokens + self.scheduler_cache_start + self.query_token_count


class VLLMNativeStepRegistry:
    """Thread-safe metadata contract for a future V1 attention hook.

    Registration does not promote the adapter to native generation. It carries
    selected pages beside scheduler state and provides explicit cleanup.
    """

    def __init__(self) -> None:
        self._requests: dict[str, VLLMNativeBlockSet] = {}
        self._lock = threading.RLock()

    def register(self, request_id: str, selected: VLLMNativeBlockSet) -> None:
        request_id = str(request_id)
        if not request_id:
            raise ValueError("A vLLM PRA request ID cannot be empty.")
        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None and previous != selected:
                raise ValueError("A live vLLM request cannot change its PRA block set.")
            self._requests[request_id] = selected

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._requests.pop(str(request_id), None)

    def plan_step(
        self,
        request_ids: Sequence[str],
        *,
        scheduler_cache_starts: Mapping[str, int],
        query_token_counts: Mapping[str, int],
    ) -> tuple[VLLMNativeStep, ...]:
        """Build attention views without rewriting scheduler-owned lengths."""

        steps = []
        with self._lock:
            for request_id in request_ids:
                start = int(scheduler_cache_starts[request_id])
                query_tokens = int(query_token_counts[request_id])
                if start < 0 or query_tokens <= 0:
                    raise ValueError("Invalid vLLM scheduler step geometry.")
                steps.append(
                    VLLMNativeStep(
                        request_id=str(request_id),
                        scheduler_cache_start=start,
                        query_token_count=query_tokens,
                        selected=self._requests.get(str(request_id)),
                    )
                )
        return tuple(steps)

    def active_request_count(self) -> int:
        with self._lock:
            return len(self._requests)
