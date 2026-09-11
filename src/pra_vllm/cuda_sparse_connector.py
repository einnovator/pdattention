"""vLLM 0.28 CUDA connector with independent selected-count/position geometry.

The ordinary detached connector advances both attention length and RoPE query
positions by the selected-token count.  Sparse causal records need a distinct
position base: attention sees only selected pages, while new query tokens keep
the position they would have had after the complete live history.
"""

from __future__ import annotations

import contextvars
import functools
from dataclasses import dataclass
from typing import Any

from pra_vllm.cuda_connector import (
    PRASemanticConnector,
    PRASemanticConnectorMetadata,
    _RequestTransfer,
)
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


_POSITION_BASES: contextvars.ContextVar[dict[str, tuple[int, int]]] = (
    contextvars.ContextVar("pra_vllm_sparse_position_bases", default={})
)
_HOOKED: set[type[Any]] = set()


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

    @classmethod
    def create_sparse(
        cls,
        command: SparseCudaConnectorCommand,
        block_ids: list[int],
        block_size: int,
        request_id: str,
        detached: bool,
    ) -> "_SparseRequestTransfer":
        base = _RequestTransfer.create(
            command, block_ids, block_size, request_id, detached
        )
        return cls(
            request_id=base.request_id,
            logical_key=base.logical_key,
            source_tokens=base.source_tokens,
            slot_mapping=base.slot_mapping,
            mode=base.mode,
            residency=base.residency,
            detached=base.detached,
            source_position_base=command.source_position_base,
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
            and request.detached
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
        if self._detached:
            _install_sparse_position_hook()

    def on_new_request(self, request: Any) -> None:
        command = SparseCudaConnectorCommand.parse(request.cache_salt)
        if command is not None:
            self._commands[request.request_id] = command

    def _add_new_request(
        self,
        metadata: PRASemanticConnectorMetadata,
        req_id: str,
        block_ids: list[int],
    ) -> None:
        command = self._commands.get(req_id)
        if not isinstance(command, SparseCudaConnectorCommand):
            return super()._add_new_request(metadata, req_id, block_ids)
        if command.mode == "load" and not self._detached and req_id not in self._loads:
            return
        metadata.requests.append(
            _SparseRequestTransfer.create_sparse(
                command,
                block_ids,
                self._block_size,
                req_id,
                detached=self._detached and command.mode == "load",
            )
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
