"""Request-owned lifecycle for sparse live vLLM-Metal K/V pages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Generic, TypeVar

from pra_hf.live_history import LiveKVSelectionPlan, LiveKVSourceRegistry

from .v1_native import VLLMMetalV1NativeBridge, capture_paged_memory


T = TypeVar("T")


@dataclass(frozen=True)
class VLLMMetalLiveSource:
    """Physical pages for one canonical, page-aligned agent history."""

    source_id: str
    block_ids: tuple[int, ...]
    source_tokens: int
    origin: str
    owning_handle: str | None = None
    restored_with_physical_copy: bool = False


@dataclass(frozen=True)
class VLLMMetalOffloadedSource:
    """Lossless non-resident representation used by the lifecycle registry."""

    source_id: str
    source_tokens: int
    payload: bytes


@dataclass(frozen=True)
class VLLMMetalLiveSelection:
    """Request-local non-owning page alias and accounting."""

    plan: LiveKVSelectionPlan
    page_indices: tuple[int, ...]
    block_ids: tuple[int, ...]
    selected_kv_tokens: int
    logical_key: str
    attachment_physical_kv_copy: bool = False
    selected_text_reencoded_tokens: int = 0
    source_restored_with_physical_copy: bool = False


@dataclass
class VLLMMetalLiveRequest(Generic[T]):
    """One engine request holding exactly one canonical source borrow."""

    runtime: "VLLMMetalLiveKVRuntime"
    request_id: str
    source_id: str
    tenant_id: str
    session_id: str
    generation: int
    selection: VLLMMetalLiveSelection
    _closed: bool = field(default=False, init=False, repr=False)
    _outcome: str | None = field(default=None, init=False, repr=False)

    @property
    def active(self) -> bool:
        return not self._closed

    @property
    def outcome(self) -> str | None:
        return self._outcome

    def _close(self, outcome: str) -> bool:
        return self.runtime._release_request(self, outcome)

    def finish(self) -> bool:
        return self._close("finished")

    def cancel(self) -> bool:
        return self._close("cancelled")

    def fail(self) -> bool:
        return self._close("error")

    def execute(self, operation: Callable[[], T]) -> T:
        """Run an engine operation under exact-once terminal cleanup."""

        if self._closed:
            raise RuntimeError("A closed vLLM-Metal live-K/V request cannot execute.")
        try:
            result = operation()
        except BaseException:
            self.fail()
            raise
        if not self.finish():
            raise RuntimeError("vLLM-Metal request closed before normal completion.")
        return result

    def __enter__(self) -> "VLLMMetalLiveRequest[T]":
        if self._closed:
            raise RuntimeError("A closed vLLM-Metal live-K/V request cannot be re-entered.")
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        self.finish() if exc_type is None else self.fail()
        return False


class VLLMMetalLiveKVRuntime:
    """Own scheduler-page pins independently from vLLM request membership.

    Initial sources alias the scheduler's prefix-cache pages with no K/V copy.
    Offload copies those values to a lossless byte payload and releases the
    scheduler pin. Restoration copies the canonical source once into bridge-
    owned reserve pages; each later request still aliases those restored pages
    without another copy or selected-token model evaluation.
    """

    def __init__(
        self,
        bridge: VLLMMetalV1NativeBridge,
        block_pool: object,
        *,
        dump_pages: Callable[[tuple[int, ...], int], bytes] | None = None,
        restore_pages: Callable[
            [str, bytes, int], tuple[tuple[int, ...], str]
        ]
        | None = None,
    ) -> None:
        self.bridge = bridge
        self.block_pool = block_pool
        self._dump_pages = dump_pages or self._default_dump_pages
        self._restore_pages = restore_pages or self._default_restore_pages
        self._restore_generation = 0
        self._hot_sources: dict[str, VLLMMetalLiveSource] = {}
        self._source_scopes: dict[str, tuple[str, str]] = {}
        self._requests: dict[str, VLLMMetalLiveRequest[object]] = {}
        self._terminal_counts = {"finished": 0, "cancelled": 0, "error": 0}
        self._lock = RLock()
        self.registry = LiveKVSourceRegistry[VLLMMetalLiveSource](
            dump=self._dump_source,
            load=self._restore_source,
        )

    def _blocks(self, block_ids: tuple[int, ...]) -> tuple[object, ...]:
        blocks = self.block_pool.blocks
        return tuple(blocks[index] for index in block_ids)

    def _default_dump_pages(
        self, block_ids: tuple[int, ...], source_tokens: int
    ) -> bytes:
        from pra_mlx.native import serialize_native_memory

        memory = capture_paged_memory(self.bridge, block_ids, source_tokens)
        return serialize_native_memory(memory, quantization="none")

    def _default_restore_pages(
        self, source_id: str, payload: bytes, source_tokens: int
    ) -> tuple[tuple[int, ...], str]:
        from pra_mlx.native import deserialize_native_memory

        memory = deserialize_native_memory(payload)
        if memory.source_tokens != source_tokens:
            raise RuntimeError("Restored vLLM-Metal source length changed.")
        self._restore_generation += 1
        handle = f"__pra_restored__:{source_id}:{self._restore_generation}"
        return self.bridge.materialize(handle, memory), handle

    def _dump_source(self, source: VLLMMetalLiveSource) -> object:
        return VLLMMetalOffloadedSource(
            source.source_id,
            source.source_tokens,
            self._dump_pages(source.block_ids, source.source_tokens),
        )

    def _restore_source(self, value: object) -> VLLMMetalLiveSource:
        if not isinstance(value, VLLMMetalOffloadedSource):
            raise TypeError("Invalid vLLM-Metal offloaded live-K/V payload.")
        blocks, handle = self._restore_pages(
            value.source_id, value.payload, value.source_tokens
        )
        source = VLLMMetalLiveSource(
            value.source_id,
            tuple(map(int, blocks)),
            value.source_tokens,
            "reserve",
            handle,
            True,
        )
        self._hot_sources[value.source_id] = source
        return source

    def register_source(
        self,
        source_id: str,
        block_ids,
        *,
        source_tokens: int,
        tenant_id: str,
        session_id: str,
        generation: int,
    ) -> None:
        """Pin complete scheduler pages as the canonical zero-copy source."""

        source_key = str(source_id)
        blocks = tuple(map(int, block_ids))
        tokens = int(source_tokens)
        if tokens <= 0 or tokens % self.bridge.block_size:
            raise ValueError("vLLM-Metal live sources must contain complete pages.")
        if len(blocks) * self.bridge.block_size != tokens:
            raise ValueError("vLLM-Metal source manifest does not match its pages.")
        if any(block < 0 or block >= self.bridge.scheduler_blocks for block in blocks):
            raise ValueError("Initial vLLM-Metal sources must use scheduler pages.")
        with self._lock:
            physical = self._blocks(blocks)
            self.block_pool.touch(physical)
            source = VLLMMetalLiveSource(
                source_key, blocks, tokens, "scheduler", None, False
            )
            try:
                self.registry.register(
                    source_key,
                    source,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    generation=generation,
                )
            except BaseException:
                self.block_pool.free_blocks(physical)
                raise
            self._hot_sources[source_key] = source
            self._source_scopes[source_key] = (str(tenant_id), str(session_id))

    def begin_request(
        self,
        request_id: str,
        source_id: str,
        plan: LiveKVSelectionPlan,
        *,
        tenant_id: str,
        session_id: str,
        expected_generation: int,
    ) -> VLLMMetalLiveRequest[object]:
        """Borrow a source, alias selected pages, then register engine metadata."""

        request_key = str(request_id)
        source_key = str(source_id)
        with self._lock:
            if request_key in self._requests:
                raise RuntimeError(
                    f"vLLM-Metal live-K/V request {request_key!r} is already active."
                )
            (source,) = self.registry.borrow(
                request_key,
                (source_key,),
                tenant_id=tenant_id,
                session_id=session_id,
                expected_generations=(expected_generation,),
            )
            alias_key = f"__pra_request__:{request_key}"
            try:
                if plan.source_tokens != source.source_tokens:
                    raise ValueError(
                        "vLLM-Metal source length does not match the selection plan."
                    )
                page_indices = self.selected_page_indices(plan, self.bridge.block_size)
                selected_blocks = tuple(source.block_ids[index] for index in page_indices)
                selected_tokens = len(selected_blocks) * self.bridge.block_size
                self.bridge.alias_resident_pages(
                    alias_key,
                    selected_blocks,
                    selected_token_count=selected_tokens,
                )
                self.bridge.register(
                    request_key,
                    (alias_key,),
                    selected_token_count=selected_tokens,
                    source_position_base=plan.source_position_base,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
            except BaseException:
                self.bridge.unregister(request_key)
                self.bridge.release(alias_key)
                self.registry.release(request_key)
                raise
            selection = VLLMMetalLiveSelection(
                plan,
                page_indices,
                selected_blocks,
                selected_tokens,
                alias_key,
                False,
                0,
                source.restored_with_physical_copy,
            )
            request = VLLMMetalLiveRequest(
                self,
                request_key,
                source_key,
                str(tenant_id),
                str(session_id),
                int(expected_generation),
                selection,
            )
            self._requests[request_key] = request
            return request

    @staticmethod
    def selected_page_indices(
        plan: LiveKVSelectionPlan, block_size: int
    ) -> tuple[int, ...]:
        """Round retained causal intervals outward to complete physical pages."""

        selected = []
        for index in range(math.ceil(plan.source_tokens / block_size)):
            start = index * block_size
            end = min((index + 1) * block_size, plan.source_tokens)
            if any(row.start < end and start < row.end for row in plan.intervals):
                selected.append(index)
        if not selected:
            raise ValueError("vLLM-Metal live selection contains no complete page.")
        return tuple(selected)

    def _release_request(
        self, request: VLLMMetalLiveRequest[object], outcome: str
    ) -> bool:
        with self._lock:
            if request._closed:
                return False
            active = self._requests.get(request.request_id)
            if active is not request:
                raise RuntimeError("vLLM-Metal request ownership is inconsistent.")
            self.bridge.unregister(request.request_id)
            self.bridge.release(request.selection.logical_key)
            if not self.registry.release(request.request_id):
                raise RuntimeError("vLLM-Metal registry lost an active source borrow.")
            del self._requests[request.request_id]
            request._closed = True
            request._outcome = outcome
            self._terminal_counts[outcome] += 1
            return True

    def cancel_request(
        self,
        request_id: str,
        *,
        abort_requests: Callable[[list[str]], None] | None = None,
    ) -> bool:
        with self._lock:
            request = self._requests.get(str(request_id))
            if request is None:
                return False
            if abort_requests is not None:
                try:
                    abort_requests([request.request_id])
                except BaseException:
                    request.fail()
                    raise
            return request.cancel()

    def _release_physical_source(self, source: VLLMMetalLiveSource) -> None:
        if source.origin == "scheduler":
            blocks = self._blocks(source.block_ids)
            self.block_pool.free_blocks(blocks)
            self.block_pool.evict_blocks(set(source.block_ids))
        elif source.origin == "reserve" and source.owning_handle is not None:
            self.bridge.release(source.owning_handle)
        else:
            raise RuntimeError("Unknown vLLM-Metal live source ownership mode.")

    def offload_source(self, source_id: str) -> VLLMMetalOffloadedSource:
        """Persist an idle source, then release its physical page ownership."""

        source_key = str(source_id)
        with self._lock:
            source = self._hot_sources[source_key]
            payload = self.registry.offload(source_key)
            if not isinstance(payload, VLLMMetalOffloadedSource):
                raise RuntimeError("vLLM-Metal registry returned an invalid payload.")
            self._release_physical_source(source)
            del self._hot_sources[source_key]
            return payload

    def terminate_session(
        self,
        tenant_id: str,
        session_id: str,
        *,
        abort_requests: Callable[[list[str]], None] | None = None,
    ) -> int:
        """Abort requests, release page ownership, then tombstone the session."""

        tenant, session = str(tenant_id), str(session_id)
        with self._lock:
            affected = tuple(
                request
                for request in self._requests.values()
                if (request.tenant_id, request.session_id) == (tenant, session)
            )
            if abort_requests is not None and affected:
                abort_requests([request.request_id for request in affected])
            for request in affected:
                request.cancel()
            source_ids = tuple(
                source_id
                for source_id, scope in self._source_scopes.items()
                if scope == (tenant, session)
            )
            for source_id in source_ids:
                source = self._hot_sources.pop(source_id, None)
                if source is not None:
                    self._release_physical_source(source)
            removed = self.registry.terminate_session(tenant, session)
            for source_id in source_ids:
                self._source_scopes.pop(source_id, None)
            return removed

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_request_ids": tuple(sorted(self._requests)),
                "hot_source_ids": tuple(sorted(self._hot_sources)),
                "terminal_counts": dict(self._terminal_counts),
            }

    def hot_source(self, source_id: str) -> VLLMMetalLiveSource | None:
        """Return immutable physical provenance for a currently hot source."""

        with self._lock:
            return self._hot_sources.get(str(source_id))
