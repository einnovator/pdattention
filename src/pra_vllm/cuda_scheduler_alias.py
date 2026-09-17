"""Scheduler-owned CUDA page aliases for vLLM V1.

This module deliberately works at the scheduler/KV-cache-manager boundary.
Worker-only KV connectors cannot implement zero-copy reuse: by the time their
load callback runs, the scheduler has already allocated a different block
table.  A qualifying PRA request instead adopts the source request's existing
``KVCacheBlock`` objects as local computed blocks.  vLLM's normal
``allocate_slots`` path then touches those objects and its normal request-free
path releases them.

The integration is intentionally narrow: vLLM 0.28, prefix caching enabled,
one homogeneous CUDA KV group, and complete pages.  Unsupported layouts fail
closed instead of silently falling back to copying or re-encoding.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class SchedulerPageSelection:
    """One compact selected prefix backed by pages from a live source."""

    logical_key: str
    source_logical_key: str
    source_generation: int
    selected_page_indices: tuple[int, ...]
    selected_token_count: int
    source_position_base: int

    def __post_init__(self) -> None:
        if not self.logical_key or not self.source_logical_key:
            raise ValueError("Scheduler aliases require stable logical keys.")
        if self.source_generation <= 0:
            raise ValueError("Source generation must be positive.")
        if not self.selected_page_indices:
            raise ValueError("A scheduler alias must select at least one page.")
        if len(set(self.selected_page_indices)) != len(self.selected_page_indices):
            raise ValueError("A scheduler alias cannot repeat a source page.")
        if tuple(sorted(self.selected_page_indices)) != self.selected_page_indices:
            raise ValueError("Selected source pages must retain causal order.")
        if self.selected_page_indices[0] < 0:
            raise ValueError("Selected source page indices cannot be negative.")
        if self.selected_token_count <= 0:
            raise ValueError("Selected token count must be positive.")
        if self.source_position_base < self.selected_token_count:
            raise ValueError(
                "Original position extent must cover the compact selected length."
            )


@dataclass
class _PinnedSource:
    generation: int
    source_tokens: int
    position_extent: int
    block_size: int
    blocks: tuple[Any, ...]
    block_pool: Any
    borrowers: set[str] = field(default_factory=set)
    pending: set[str] = field(default_factory=set)
    tombstoned: bool = False


@dataclass(frozen=True)
class SchedulerAliasTelemetry:
    source_pin_events: int
    alias_hit_events: int
    alias_prepare_events: int
    alias_commit_events: int
    alias_install_events: int
    alias_release_events: int
    physical_kv_copy_bytes: int
    host_to_device_bytes: int
    selected_history_reencoded_tokens: int
    materialized_history_encoded_tokens: int
    materialized_history_copy_bytes: int


class VLLMCudaSchedulerPageRegistry:
    """Own source pins while vLLM owns every request-level alias refcount."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, _PinnedSource] = {}
        self._pending: dict[str, SchedulerPageSelection] = {}
        self._active: dict[str, SchedulerPageSelection] = {}
        self._source_pin_events = 0
        self._alias_hit_events = 0
        self._alias_prepare_events = 0
        self._alias_commit_events = 0
        self._alias_install_events = 0
        self._alias_release_events = 0
        self._materialized_history_encoded_tokens = 0
        self._materialized_history_copy_bytes = 0

    def publish_source(
        self,
        logical_key: str,
        *,
        generation: int,
        source_tokens: int,
        blocks_by_group: Sequence[Sequence[Any]],
        block_sizes: Sequence[int],
        block_pool: Any,
        position_extent: int | None = None,
    ) -> None:
        """Pin a completed request's pages before scheduler request teardown."""

        if len(blocks_by_group) != 1 or len(block_sizes) != 1:
            raise NotImplementedError(
                "CUDA scheduler aliases currently require one homogeneous KV group."
            )
        block_size = int(block_sizes[0])
        if source_tokens <= 0:
            raise ValueError("A live CUDA source must contain at least one token.")
        # vLLM owns partially filled terminal pages as ordinary request state.
        # Receipt materialization needs to preserve that exact valid-token count
        # rather than padding model-visible history merely to fill a page.
        required = (int(source_tokens) + block_size - 1) // block_size
        blocks = tuple(blocks_by_group[0][:required])
        if len(blocks) != required:
            raise RuntimeError("Source request does not own all declared KV pages.")
        if any(getattr(block, "is_null", False) for block in blocks):
            raise RuntimeError("Null/sliding-window pages cannot back a PRA source.")
        logical_extent = (
            int(source_tokens) if position_extent is None else int(position_extent)
        )
        if logical_extent < int(source_tokens):
            raise ValueError(
                "A CUDA source position extent cannot be smaller than its "
                "valid token count."
            )

        key = str(logical_key)
        with self._lock:
            previous = self._sources.get(key)
            if previous is not None:
                same = (
                    previous.generation == int(generation)
                    and tuple(id(block) for block in previous.blocks)
                    == tuple(id(block) for block in blocks)
                    and previous.position_extent == logical_extent
                    and not previous.tombstoned
                )
                if same:
                    return
                if previous.borrowers or previous.pending:
                    raise RuntimeError(
                        "Cannot replace a source generation while it is borrowed."
                    )
                if not self._only_source_pin_remains(previous):
                    raise RuntimeError(
                        "Cannot replace a source with scheduler/in-flight aliases."
                    )
                self._drop_source_locked(key, previous)

            # This is the persistent source-owner reference. Request aliases get
            # separate references through allocate_slots/add_local_computed_blocks.
            block_pool.touch(blocks)
            self._sources[key] = _PinnedSource(
                generation=int(generation),
                source_tokens=int(source_tokens),
                position_extent=logical_extent,
                block_size=block_size,
                blocks=blocks,
                block_pool=block_pool,
            )
            self._source_pin_events += 1

    def publish_composite_source(
        self,
        logical_key: str,
        *,
        generation: int,
        position_extent: int,
        components: Sequence[tuple[str, int, Sequence[int]]],
        selected_token_count: int | None = None,
        materialized_history_encoded_tokens: int = 0,
        materialized_history_copy_bytes: int = 0,
    ) -> int:
        """Pin an ordered zero-copy page sequence drawn from resident sources.

        Receipt-aware agent memory is hybrid: unchanged records remain pages of
        the full-history source, while compact closure receipts are newly
        encoded into their own resident pages.  The final request must borrow
        both without packing or copying either.  ``position_extent`` is the
        original logical history extent and is intentionally independent of the
        compact number of physical pages in the composite.
        """

        rows = tuple(components)
        if not rows:
            raise ValueError("A CUDA composite source needs at least one component.")
        key = str(logical_key)
        if not key:
            raise ValueError("A CUDA composite source needs a stable logical key.")
        if key in {str(source_key) for source_key, _, _ in rows}:
            raise ValueError("A CUDA composite cannot replace one of its components.")
        encoded_tokens = int(materialized_history_encoded_tokens)
        copied_bytes = int(materialized_history_copy_bytes)
        if encoded_tokens < 0 or copied_bytes < 0:
            raise ValueError("CUDA materialization accounting cannot be negative.")
        with self._lock:
            selected: list[Any] = []
            pool = None
            block_size = None
            for source_key, source_generation, page_indices in rows:
                source = self._require_generation(
                    str(source_key), int(source_generation)
                )
                if source.tombstoned:
                    raise RuntimeError(
                        "A terminated CUDA source cannot enter a composite."
                    )
                if pool is None:
                    pool = source.block_pool
                    block_size = source.block_size
                elif source.block_pool is not pool or source.block_size != block_size:
                    raise NotImplementedError(
                        "CUDA composites require one homogeneous block pool."
                    )
                indices = tuple(map(int, page_indices))
                if not indices:
                    raise ValueError("A CUDA composite component cannot be empty.")
                if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
                    raise ValueError(
                        "Composite source pages must be unique and causally ordered."
                    )
                if indices[0] < 0 or indices[-1] >= len(source.blocks):
                    raise ValueError("Composite CUDA page is outside its source.")
                selected.extend(source.blocks[index] for index in indices)
            if len({id(block) for block in selected}) != len(selected):
                raise ValueError("A CUDA composite cannot repeat a physical page.")
            assert pool is not None and block_size is not None
            physical_capacity = len(selected) * block_size
            valid_tokens = (
                physical_capacity
                if selected_token_count is None
                else int(selected_token_count)
            )
            if not physical_capacity - block_size < valid_tokens <= physical_capacity:
                raise ValueError(
                    "A CUDA composite may have only one partially filled terminal page."
                )
            if int(position_extent) < valid_tokens:
                raise ValueError(
                    "Composite position extent cannot be smaller than its valid tokens."
                )
            previous = self._sources.get(key)
            if previous is not None:
                same = (
                    previous.generation == int(generation)
                    and previous.position_extent == int(position_extent)
                    and tuple(id(block) for block in previous.blocks)
                    == tuple(id(block) for block in selected)
                    and not previous.tombstoned
                )
                if same:
                    return valid_tokens
                if previous.borrowers or previous.pending:
                    raise RuntimeError(
                        "Cannot replace a composite source while it is borrowed."
                    )
                if not self._only_source_pin_remains(previous):
                    raise RuntimeError(
                        "Cannot replace a composite with scheduler/in-flight aliases."
                    )
                self._drop_source_locked(key, previous)
            pool.touch(selected)
            self._sources[key] = _PinnedSource(
                generation=int(generation),
                source_tokens=valid_tokens,
                position_extent=int(position_extent),
                block_size=block_size,
                blocks=tuple(selected),
                block_pool=pool,
            )
            self._source_pin_events += 1
            self._materialized_history_encoded_tokens += encoded_tokens
            self._materialized_history_copy_bytes += copied_bytes
            return valid_tokens

    def prepare_alias(
        self,
        request_id: str,
        selection: SchedulerPageSelection,
        *,
        prompt_token_count: int,
        create_kv_cache_blocks: Any,
    ) -> Any:
        """Return source block objects for vLLM's ordinary local-hit allocator."""

        request_key = str(request_id)
        with self._lock:
            active = self._active.get(request_key)
            if active is not None:
                if active != selection:
                    raise RuntimeError("A live request cannot change its PRA selection.")
                raise RuntimeError("A running request cannot reinstall its alias.")
            pending = self._pending.get(request_key)
            if pending is not None and pending != selection:
                raise RuntimeError("A pending request cannot change its PRA selection.")

            source = self._sources.get(selection.source_logical_key)
            if source is None or source.tombstoned:
                raise RuntimeError("Sparse CUDA source is unavailable or terminated.")
            if source.generation != selection.source_generation:
                raise RuntimeError(
                    "Stale sparse CUDA generation: requested "
                    f"{selection.source_generation}, live {source.generation}."
                )
            selected_capacity = (
                len(selection.selected_page_indices) * source.block_size
            )
            minimum_valid = selected_capacity - source.block_size + 1
            if not minimum_valid <= selection.selected_token_count <= selected_capacity:
                raise ValueError(
                    "Selected CUDA token count must cover all complete selected pages "
                    "and may end within only the terminal page."
                )
            if selection.selected_token_count != selected_capacity and (
                selection.selected_page_indices[-1] != len(source.blocks) - 1
                or selection.selected_token_count > source.source_tokens
            ):
                raise ValueError(
                    "A partial CUDA alias must end at the source's valid terminal page."
                )
            if selection.source_position_base > source.position_extent:
                raise ValueError("Original position extent exceeds the live source.")
            if selection.selected_page_indices[-1] >= len(source.blocks):
                raise ValueError("Selected CUDA page is outside the live source.")
            # vLLM still needs a compact visible token prefix so request length,
            # sampling, and worker input geometry remain authoritative. Those
            # tokens are marked computed and are never sent through the model.
            if prompt_token_count <= selection.selected_token_count:
                raise ValueError(
                    "Scheduler alias requests must contain the compact selected "
                    "token prefix followed by at least one query token."
                )

            selected = tuple(source.blocks[index] for index in selection.selected_page_indices)
            source.pending.add(request_key)
            self._pending[request_key] = selection
            self._alias_prepare_events += 1
            return create_kv_cache_blocks((selected,))

    def note_alias_hit(self, request_id: str) -> None:
        """Record that the scheduler lookup actually selected the PRA alias."""

        with self._lock:
            if str(request_id) not in self._pending:
                raise RuntimeError("Sparse CUDA alias hit was not prepared first.")
            self._alias_hit_events += 1

    def commit_alias(self, request_id: str, installed_blocks: Any) -> None:
        """Confirm allocate_slots installed the exact source block identities."""

        request_key = str(request_id)
        with self._lock:
            selection = self._pending.pop(request_key, None)
            if selection is None:
                return
            source = self._sources.get(selection.source_logical_key)
            if source is None or source.tombstoned:
                raise RuntimeError("Source disappeared before alias commit.")
            expected = tuple(
                source.blocks[index] for index in selection.selected_page_indices
            )
            groups = tuple(getattr(installed_blocks, "blocks", ()))
            if len(groups) != 1 or any(
                actual is not wanted
                for actual, wanted in zip(groups[0][: len(expected)], expected)
            ) or len(groups[0]) < len(expected):
                source.pending.discard(request_key)
                raise RuntimeError(
                    "vLLM did not install the authoritative source block table."
                )
            if any(
                int(getattr(actual, "block_id")) != int(getattr(wanted, "block_id"))
                for actual, wanted in zip(groups[0], expected)
            ):
                source.pending.discard(request_key)
                raise RuntimeError("Installed CUDA physical page IDs do not match source.")
            source.pending.discard(request_key)
            source.borrowers.add(request_key)
            self._active[request_key] = selection
            self._alias_commit_events += 1
            self._alias_install_events += 1

    def publish_extended_source(
        self,
        request_id: str,
        destination_logical_key: str,
        *,
        destination_generation: int,
        append_complete_pages: int,
        request_blocks: Any,
    ) -> int:
        """Publish old full pages plus newly evaluated request suffix pages.

        The request block table begins with its compact selected aliases.  Its
        following pages contain suffix K/V evaluated at ``source_position_base``.
        Those pages can extend the *full* canonical source without copying K/V.
        Partial suffix pages are deliberately excluded.
        """

        request_key = str(request_id)
        with self._lock:
            selection = self._active.get(request_key)
            if selection is None:
                raise RuntimeError("Cannot extend a source before alias commit.")
            source = self._sources.get(selection.source_logical_key)
            if source is None or source.tombstoned:
                raise RuntimeError("Canonical CUDA source is unavailable.")
            if selection.source_position_base != source.source_tokens:
                raise RuntimeError(
                    "Sparse request position base does not equal canonical source extent."
                )
            if source.source_tokens % source.block_size:
                raise NotImplementedError(
                    "Extending a shared partial CUDA source requires explicit "
                    "copy-on-write accounting."
                )
            count = int(append_complete_pages)
            if count < 0:
                raise ValueError("append_complete_pages cannot be negative.")
            groups = tuple(getattr(request_blocks, "blocks", ()))
            if len(groups) != 1:
                raise NotImplementedError(
                    "CUDA source extension requires one homogeneous KV group."
                )
            selected_pages = len(selection.selected_page_indices)
            end = selected_pages + count
            if len(groups[0]) < end:
                raise RuntimeError("Request does not own all committed suffix pages.")
            appended = tuple(groups[0][selected_pages:end])
            combined = (*source.blocks, *appended)
            tokens = source.source_tokens + count * source.block_size
            self.publish_source(
                destination_logical_key,
                generation=destination_generation,
                source_tokens=tokens,
                blocks_by_group=(combined,),
                block_sizes=(source.block_size,),
                block_pool=source.block_pool,
                position_extent=selection.source_position_base + count * source.block_size,
            )
            return tokens

    def finish_request(self, request_id: str) -> bool:
        """Release logical state once; vLLM frees its physical alias reference."""

        request_key = str(request_id)
        with self._lock:
            pending = self._pending.pop(request_key, None)
            selection = self._active.pop(request_key, None)
            target = selection or pending
            if target is None:
                return False
            source = self._sources.get(target.source_logical_key)
            if source is not None:
                source.pending.discard(request_key)
                source.borrowers.discard(request_key)
                if (
                    source.tombstoned
                    and not source.pending
                    and not source.borrowers
                    and self._only_source_pin_remains(source)
                ):
                    self._drop_source_locked(target.source_logical_key, source)
            self._alias_release_events += 1
            return True

    def evict_source(self, logical_key: str, *, generation: int) -> tuple[int, ...]:
        """Drop an unborrowed source pin once its pages remain safely owned."""

        key = str(logical_key)
        with self._lock:
            source = self._require_generation(key, generation)
            if source.borrowers or source.pending:
                raise RuntimeError("Cannot evict a borrowed CUDA source generation.")
            successor_covers_source = any(
                candidate is not source
                and not candidate.tombstoned
                and len(candidate.blocks) >= len(source.blocks)
                and all(
                    actual is expected
                    for actual, expected in zip(candidate.blocks, source.blocks)
                )
                for candidate in self._sources.values()
            )
            if not self._only_source_pin_remains(source) and not successor_covers_source:
                raise RuntimeError(
                    "Cannot evict CUDA source pages with scheduler/in-flight aliases."
                )
            # A committed successor pins the complete old prefix before the
            # scheduler releases the request's deferred physical aliases.  In
            # that rollover window it is safe to drop only the superseded
            # registry pin: both the successor and the request still own the
            # exact block objects, so no page can return to the free pool.
            ids = tuple(int(block.block_id) for block in source.blocks)
            self._drop_source_locked(key, source)
            return ids

    def terminate_source(self, logical_key: str, *, generation: int) -> bool:
        """Reject new borrows immediately and defer unpin until borrowers drain."""

        key = str(logical_key)
        with self._lock:
            source = self._require_generation(key, generation)
            if source.tombstoned:
                return False
            source.tombstoned = True
            if (
                not source.borrowers
                and not source.pending
                and self._only_source_pin_remains(source)
            ):
                self._drop_source_locked(key, source)
            return True

    def reap_tombstones(self) -> tuple[str, ...]:
        """Release tombstones after vLLM's deferred-free fence has drained."""

        with self._lock:
            ready = [
                key
                for key, source in self._sources.items()
                if source.tombstoned
                and not source.borrowers
                and not source.pending
                and self._only_source_pin_remains(source)
            ]
            for key in ready:
                self._drop_source_locked(key, self._sources[key])
            return tuple(sorted(ready))

    def _require_generation(self, key: str, generation: int) -> _PinnedSource:
        source = self._sources.get(key)
        if source is None or source.generation != int(generation):
            raise RuntimeError("Stale or unavailable CUDA source generation.")
        return source

    def _drop_source_locked(self, key: str, source: _PinnedSource) -> None:
        source.block_pool.free_blocks(reversed(source.blocks))
        self._sources.pop(key, None)

    def _only_source_pin_remains(self, source: _PinnedSource) -> bool:
        # A page can be owned by two canonical generations during atomic
        # rollover.  Discount all registry-owned source pins, but never an
        # authoritative request/deferred-free reference.
        def persistent_pins(block: Any) -> int:
            return sum(
                sum(candidate is block for candidate in row.blocks)
                for row in self._sources.values()
            )

        return all(
            int(block.ref_cnt) == persistent_pins(block)
            for block in source.blocks
        )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "sources": {
                    key: {
                        "generation": source.generation,
                        "source_tokens": source.source_tokens,
                        "position_extent": source.position_extent,
                        "block_ids": [int(block.block_id) for block in source.blocks],
                        "borrowers": sorted(source.borrowers),
                        "pending": sorted(source.pending),
                        "tombstoned": source.tombstoned,
                    }
                    for key, source in sorted(self._sources.items())
                },
                "active_requests": sorted(self._active),
                "pending_requests": sorted(self._pending),
                "telemetry": self.telemetry().__dict__,
            }

    def telemetry(self) -> SchedulerAliasTelemetry:
        return SchedulerAliasTelemetry(
            source_pin_events=self._source_pin_events,
            alias_hit_events=self._alias_hit_events,
            alias_prepare_events=self._alias_prepare_events,
            alias_commit_events=self._alias_commit_events,
            alias_install_events=self._alias_install_events,
            alias_release_events=self._alias_release_events,
            physical_kv_copy_bytes=0,
            host_to_device_bytes=0,
            selected_history_reencoded_tokens=0,
            materialized_history_encoded_tokens=(
                self._materialized_history_encoded_tokens
            ),
            materialized_history_copy_bytes=self._materialized_history_copy_bytes,
        )


_HOOK_LOCK = threading.Lock()
_HOOKED_SCHEDULERS: set[type[Any]] = set()


def install_vllm_scheduler_page_alias_hooks() -> None:
    """Install the minimal vLLM 0.28 scheduler admission hooks once."""

    from vllm.v1.core.sched.scheduler import Scheduler

    with _HOOK_LOCK:
        if Scheduler in _HOOKED_SCHEDULERS:
            return
        original_lookup = Scheduler._get_local_prefix_cache_hit
        original_finished = Scheduler._connector_finished

        @functools.wraps(original_lookup)
        def lookup(scheduler: Any, request: Any):
            ordinary = original_lookup(scheduler, request)
            connector = getattr(scheduler, "connector", None)
            hook = getattr(connector, "scheduler_owned_alias_hit", None)
            if hook is None:
                return ordinary
            return hook(scheduler.kv_cache_manager, request, ordinary)

        @functools.wraps(original_finished)
        def finished(scheduler: Any, request: Any):
            connector = getattr(scheduler, "connector", None)
            bind = getattr(connector, "bind_scheduler_kv_manager", None)
            if bind is not None:
                bind(scheduler.kv_cache_manager)
            return original_finished(scheduler, request)

        Scheduler._get_local_prefix_cache_hit = lookup
        Scheduler._connector_finished = finished
        _HOOKED_SCHEDULERS.add(Scheduler)
