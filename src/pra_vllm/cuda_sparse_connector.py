"""vLLM 0.28 CUDA connector with independent selected-count/position geometry.

The ordinary detached connector advances both attention length and RoPE query
positions by the selected-token count.  Sparse causal records need a distinct
position base: attention sees only selected pages, while new query tokens keep
the position they would have had after the complete live history.
"""

from __future__ import annotations

import contextvars
import functools
import json
from dataclasses import dataclass
from typing import Any

from pra_vllm.cuda_connector import (
    PRASemanticConnector,
    PRASemanticConnectorMetadata,
    _RequestTransfer,
)
from pra_vllm.cuda_scheduler_alias import (
    SchedulerPageSelection,
    VLLMCudaSchedulerPageRegistry,
    install_vllm_scheduler_page_alias_hooks,
)
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


_POSITION_BASES: contextvars.ContextVar[dict[str, tuple[int, int]]] = (
    contextvars.ContextVar("pra_vllm_sparse_position_bases", default={})
)
_HOOKED: set[type[Any]] = set()


def _completed_append_pages(manager: Any, request: Any, selected_tokens: int) -> int:
    """Count complete suffix pages without depending on a store-only local."""

    computed_suffix = max(0, int(request.num_computed_tokens) - int(selected_tokens))
    block_size = int(manager.coordinator.single_type_managers[0].block_size)
    return computed_suffix // block_size


def _position_delta(selected_tokens: int, source_position_base: int) -> int:
    """Return the RoPE offset not represented by compact selected pages."""

    selected = int(selected_tokens)
    position_base = int(source_position_base)
    if selected <= 0 or position_base < selected:
        raise ValueError("Invalid sparse CUDA position geometry.")
    return position_base - selected


@dataclass
class _SparseRequestTransfer(_RequestTransfer):
    source_position_base: int = 0
    scheduler_alias: bool = False

    @classmethod
    def create_sparse(
        cls,
        command: SparseCudaConnectorCommand,
        block_ids: list[int],
        block_size: int,
        request_id: str,
        detached: bool,
        scheduler_alias: bool = False,
    ) -> "_SparseRequestTransfer":
        base = _RequestTransfer.create(
            command,
            block_ids,
            block_size,
            request_id,
            # A scheduler alias already owns its authoritative slots; this
            # metadata exists only to carry original-position geometry.
            detached or scheduler_alias,
        )
        return cls(
            request_id=base.request_id,
            logical_key=base.logical_key,
            source_tokens=base.source_tokens,
            slot_mapping=base.slot_mapping,
            mode=base.mode,
            residency=base.residency,
            detached=bool(detached),
            source_position_base=command.source_position_base,
            scheduler_alias=bool(scheduler_alias),
        )


def _install_sparse_position_hook() -> None:
    """Adjust only query positions after the contiguous detached hook runs."""

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if GPUModelRunner in _HOOKED:
        return
    previous_prepare = GPUModelRunner._prepare_inputs
    previous_execute = GPUModelRunner.execute_model

    @functools.wraps(previous_prepare)
    def prepare(worker: Any, scheduler_output: Any, num_scheduled_tokens: Any):
        result = previous_prepare(worker, scheduler_output, num_scheduled_tokens)
        bases = _POSITION_BASES.get()
        cursor = 0
        for row, scheduled in enumerate(map(int, num_scheduled_tokens)):
            geometry = bases.get(str(worker.input_batch.req_ids[row]))
            if geometry is not None:
                selected_tokens, source_position_base = geometry
                delta = _position_delta(selected_tokens, source_position_base)
                if delta:
                    worker.positions[cursor : cursor + scheduled].add_(delta)
            cursor += scheduled
        return result

    @functools.wraps(previous_execute)
    def execute(worker: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        bases = {
            str(request.request_id): (
                int(request.source_tokens), int(request.source_position_base)
            )
            for request in getattr(metadata, "requests", ())
            if request.mode == "load"
            and (request.detached or getattr(request, "scheduler_alias", False))
            and hasattr(request, "source_position_base")
        }
        token = _POSITION_BASES.set(bases)
        try:
            return previous_execute(worker, scheduler_output, *args, **kwargs)
        finally:
            _POSITION_BASES.reset(token)

    GPUModelRunner._prepare_inputs = prepare
    GPUModelRunner.execute_model = execute
    _HOOKED.add(GPUModelRunner)


class PRASparseConnector(PRASemanticConnector):
    """Detached connector that accepts sparse original-position commands."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        raw_alias = self._kv_transfer_config.get_from_extra_config(
            "scheduler_page_aliases", False
        )
        self._scheduler_alias_enabled = (
            raw_alias.lower() in {"1", "true", "yes", "on"}
            if isinstance(raw_alias, str)
            else bool(raw_alias)
        )
        self._scheduler_alias_registry = VLLMCudaSchedulerPageRegistry()
        self._scheduler_kv_manager: Any | None = None
        self._scheduler_alias_requests: set[str] = set()
        self._scheduler_committed_sources: dict[str, tuple[str, int, int]] = {}
        if self._detached or self._scheduler_alias_enabled:
            _install_sparse_position_hook()
        if self._scheduler_alias_enabled:
            if self._detached:
                raise ValueError(
                    "scheduler_page_aliases and detached_pages are mutually exclusive."
                )
            install_vllm_scheduler_page_alias_hooks()

    def bind_scheduler_kv_manager(self, manager: Any) -> None:
        """Bind the authoritative scheduler allocator, never a worker mirror."""

        if not getattr(self, "_scheduler_alias_enabled", False):
            return
        if self._scheduler_kv_manager not in (None, manager):
            raise RuntimeError("PRA CUDA connector cannot cross scheduler allocators.")
        self._scheduler_kv_manager = manager

    def _scheduler_selection(
        self, command: SparseCudaConnectorCommand
    ) -> SchedulerPageSelection:
        manifest = self._directory(command.logical_key) / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        try:
            parent = str(payload["parent_logical_key"])
            page_indices = tuple(map(int, payload["selected_page_indices"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Scheduler-page alias manifest lacks its live source mapping."
            ) from error
        return SchedulerPageSelection(
            logical_key=command.logical_key,
            source_logical_key=parent,
            source_generation=command.source_generation,
            selected_page_indices=page_indices,
            selected_token_count=command.source_tokens,
            source_position_base=command.source_position_base,
        )

    def scheduler_owned_alias_hit(
        self, manager: Any, request: Any, ordinary_hit: Any
    ) -> Any:
        """Replace APC lookup with selected existing pages before allocation."""

        if not getattr(self, "_scheduler_alias_enabled", False):
            return ordinary_hit
        self.bind_scheduler_kv_manager(manager)
        command = self._commands.get(request.request_id)
        if not isinstance(command, SparseCudaConnectorCommand):
            return ordinary_hit
        if command.mode != "load":
            return ordinary_hit
        if not self._ready(command):
            raise RuntimeError(
                "Stale or unavailable sparse CUDA K/V generation: "
                f"{command.logical_key}@{command.source_generation}"
            )
        prompt_token_count = len(request.prompt_token_ids or ())
        blocks = self._scheduler_alias_registry.prepare_alias(
            request.request_id,
            self._scheduler_selection(command),
            prompt_token_count=prompt_token_count,
            create_kv_cache_blocks=manager.create_kv_cache_blocks,
        )
        self._scheduler_alias_registry.note_alias_hit(request.request_id)
        selected = int(command.source_tokens)
        # Keep compact attention length and full source position extent separate.
        # The worker position hook applies the latter; this scheduler result must
        # remain the selected length so no omitted token is re-encoded.
        return blocks, selected, selected, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        if getattr(self, "_scheduler_alias_enabled", False):
            before = set(
                self._scheduler_alias_registry.snapshot()["pending_requests"]
            )
            if str(request.request_id) in before:
                self._scheduler_alias_registry.commit_alias(request.request_id, blocks)
                self._scheduler_alias_requests.add(str(request.request_id))
                return
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def on_new_request(self, request: Any) -> None:
        command = SparseCudaConnectorCommand.parse(request.cache_salt)
        if command is not None:
            self._commands[request.request_id] = command

    def _ready(self, command: Any) -> bool:
        if not isinstance(command, SparseCudaConnectorCommand):
            return super()._ready(command)
        manifest = self._directory(command.logical_key) / "manifest.json"
        if not manifest.exists():
            return False
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return bool(
            payload.get("logical_key") == command.logical_key
            and int(payload.get("source_tokens", -1)) == command.source_tokens
            and int(payload.get("source_generation", -1))
            == command.source_generation
        )

    def _add_new_request(
        self,
        metadata: PRASemanticConnectorMetadata,
        req_id: str,
        block_ids: list[int],
    ) -> None:
        command = self._commands.get(req_id)
        if (
            getattr(self, "_scheduler_alias_enabled", False)
            and command is not None
            and command.mode == "store"
        ):
            # The scheduler pins the live pages; no worker-side D2H export is
            # part of the scheduler-alias path.
            return
        if not isinstance(command, SparseCudaConnectorCommand):
            return super()._add_new_request(metadata, req_id, block_ids)
        if command.mode == "load" and not self._ready(command):
            raise RuntimeError(
                "Stale or unavailable sparse CUDA K/V generation: "
                f"{command.logical_key}@{command.source_generation}"
            )
        scheduler_alias = req_id in getattr(self, "_scheduler_alias_requests", ())
        if (
            command.mode == "load"
            and not self._detached
            and not scheduler_alias
            and req_id not in self._loads
        ):
            return
        metadata.requests.append(
            _SparseRequestTransfer.create_sparse(
                command,
                block_ids,
                self._block_size,
                req_id,
                detached=self._detached and command.mode == "load",
                scheduler_alias=scheduler_alias,
            )
        )

    def request_finished(self, request: Any, block_ids: list[int]):
        """Pin sources and close aliases before vLLM frees request blocks."""

        command = self._commands.get(request.request_id)
        if (
            getattr(self, "_scheduler_alias_enabled", False)
            and getattr(self, "_scheduler_kv_manager", None) is not None
        ):
            manager = self._scheduler_kv_manager
            if command is not None and command.mode == "store":
                source_tokens = int(command.source_tokens)
                if int(request.num_computed_tokens) < source_tokens:
                    raise RuntimeError(
                        "Cannot publish an incompletely computed CUDA source."
                    )
                generation = int(getattr(command, "source_generation", 1))
                self._scheduler_alias_registry.publish_source(
                    command.logical_key,
                    generation=generation,
                    source_tokens=source_tokens,
                    blocks_by_group=manager.get_blocks(request.request_id).blocks,
                    block_sizes=tuple(
                        int(item.block_size)
                        for item in manager.coordinator.single_type_managers
                    ),
                    block_pool=manager.block_pool,
                )
            elif command is not None and command.mode == "load":
                manifest_path = self._directory(command.logical_key) / "manifest.json"
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                destination = payload.get("commit_source_logical_key")
                destination_generation = payload.get("commit_source_generation")
                if destination is not None or destination_generation is not None:
                    if not destination or destination_generation is None:
                        raise RuntimeError(
                            "Sparse CUDA source commit metadata is incomplete."
                        )
                    append_pages = _completed_append_pages(
                        manager, request, int(command.source_tokens)
                    )
                    committed_tokens = self._scheduler_alias_registry.publish_extended_source(
                        request.request_id,
                        str(destination),
                        destination_generation=int(destination_generation),
                        append_complete_pages=append_pages,
                        request_blocks=manager.get_blocks(request.request_id),
                    )
                    self._scheduler_committed_sources[str(request.request_id)] = (
                        str(destination),
                        int(destination_generation),
                        committed_tokens,
                    )
            self._scheduler_alias_registry.finish_request(request.request_id)
            self._scheduler_alias_requests.discard(str(request.request_id))
        return super().request_finished(request, block_ids)

    def pop_scheduler_committed_source(
        self, request_id: str
    ) -> tuple[str, int, int] | None:
        """Return the source-generation receipt emitted by request teardown."""

        return self._scheduler_committed_sources.pop(str(request_id), None)

    def evict_scheduler_source(
        self, logical_key: str, *, source_generation: int = 1
    ) -> tuple[int, ...]:
        if not getattr(self, "_scheduler_alias_enabled", False):
            raise RuntimeError("Scheduler-page aliases are not enabled.")
        return self._scheduler_alias_registry.evict_source(
            logical_key, generation=source_generation
        )

    def terminate_scheduler_source(
        self, logical_key: str, *, source_generation: int = 1
    ) -> bool:
        if not getattr(self, "_scheduler_alias_enabled", False):
            raise RuntimeError("Scheduler-page aliases are not enabled.")
        return self._scheduler_alias_registry.terminate_source(
            logical_key, generation=source_generation
        )

    def evict_detached_resource(
        self, logical_key: str, *, residency: str = "hot"
    ) -> tuple[int, ...]:
        """Evict an unborrowed detached resource and return its physical pages.

        vLLM calls ``request_finished`` for cancellation as well as normal
        completion.  That callback decrements the actual worker refcount.  An
        explicit eviction must fail closed until every borrower has left.
        """

        handle_key = (str(logical_key), str(residency))
        borrowers = int(self._detached_refcounts.get(handle_key, 0))
        if borrowers:
            raise RuntimeError(
                f"Cannot evict detached CUDA resource with {borrowers} borrower(s)."
            )
        owners = tuple(
            request_id
            for request_id, request_key in self._detached_active_requests.items()
            if request_key == handle_key
        )
        if owners:
            raise RuntimeError(
                "Detached CUDA ownership corruption for request(s) "
                + ", ".join(owners)
            )
        blocks = tuple(self._detached_handles.pop(handle_key, ()))
        self._detached_materialized.discard(handle_key)
        self._detached_tensor_bytes.pop(handle_key, None)
        if self._detached_free is not None:
            self._detached_free.extend(blocks)
            self._detached_free.sort()
        return blocks

    def reconcile_detached_requests(
        self, active_request_ids: set[str]
    ) -> tuple[str, ...]:
        """Release worker borrows absent from an authoritative active snapshot."""

        before = set(self._detached_active_requests)
        self._reap_inactive_detached_requests(set(map(str, active_request_ids)))
        return tuple(sorted(before - set(self._detached_active_requests)))

    def offload_detached_resource(
        self, logical_key: str, *, residency: str = "hot"
    ) -> tuple[int, ...]:
        """Release resident pages while retaining the persisted lossless K/V."""

        return self.evict_detached_resource(logical_key, residency=residency)
