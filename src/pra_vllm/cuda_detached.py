"""Worker-side helpers for scheduler-invisible CUDA PRA pages.

The vLLM scheduler owns the ordinary block range ``[0, num_blocks)``.  The
worker allocates a small physical tail after that range, while the scheduler
continues to reason about the original capacity.  Query positions and
attention metadata are augmented only inside the worker, so selected memory
does not consume prompt tokens, scheduler blocks, or APC entries.
"""

from __future__ import annotations

import contextvars
import functools
import math
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class DetachedCommand:
    """Minimum request geometry needed before the connector is bound."""

    request_id: str
    source_tokens: int


@dataclass(frozen=True)
class DetachedLayout:
    """Physical worker cache range that is invisible to the scheduler."""

    scheduler_blocks: int
    reserve_blocks: int

    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.scheduler_blocks,
                self.scheduler_blocks + self.reserve_blocks,
            )
        )


_ACTIVE_COMMANDS: contextvars.ContextVar[Mapping[str, DetachedCommand]] = (
    contextvars.ContextVar("pra_vllm_detached_commands", default={})
)
_ACTIVE_ROWS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "pra_vllm_detached_rows", default=()
)
_LAYOUTS: dict[int, DetachedLayout] = {}
_HOOK_LOCK = threading.Lock()
_HOOKED_CLASSES: set[type[Any]] = set()
_RESERVE_BLOCKS = 0


def expand_worker_kv_cache_config(
    kv_cache_config: Any,
    reserve_blocks: int,
) -> tuple[Any, DetachedLayout]:
    """Clone and extend worker tensors while preserving scheduler capacity."""

    if reserve_blocks <= 0:
        raise ValueError("Detached CUDA reserve_blocks must be positive.")
    original_blocks = int(kv_cache_config.num_blocks)
    if original_blocks <= 0:
        raise ValueError("vLLM reported no scheduler K/V blocks.")
    expanded = deepcopy(kv_cache_config)
    for tensor in expanded.kv_cache_tensors:
        if int(tensor.size) % original_blocks:
            raise ValueError("vLLM K/V tensor size is not block divisible.")
        tensor.size += (int(tensor.size) // original_blocks) * reserve_blocks
    expanded.num_blocks = original_blocks + reserve_blocks
    return expanded, DetachedLayout(original_blocks, reserve_blocks)


def apply_detached_query_geometry(
    runner: Any,
    num_scheduled_tokens: Sequence[int],
    commands: Mapping[str, DetachedCommand],
) -> None:
    """Offset query RoPE positions after local slot mapping is computed."""

    cursor = 0
    request_ids = tuple(runner.input_batch.req_ids)
    for row, scheduled in enumerate(map(int, num_scheduled_tokens)):
        command = commands.get(str(request_ids[row]))
        if command is not None:
            end = cursor + scheduled
            runner.positions[cursor:end].add_(command.source_tokens)
            runner.seq_lens[row].add_(command.source_tokens)
            runner.optimistic_seq_lens_cpu[row].add_(command.source_tokens)
        cursor += scheduled


def _metadata_objects(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = (
            item
            for group in value
            for item in (group.values() if isinstance(group, Mapping) else (group,))
        )
    else:
        values = (value,)
    unique: list[Any] = []
    seen: set[int] = set()
    for item in values:
        identity = id(item)
        if identity not in seen:
            unique.append(item)
            seen.add(identity)
    return unique


def prepend_detached_blocks(
    attention_metadata: Any,
    *,
    row: int,
    block_ids: Sequence[int],
    source_tokens: int,
    block_size: int,
) -> None:
    """Expose reserved pages to one attention row without changing slot maps."""

    if source_tokens % block_size:
        raise ValueError("Detached CUDA source K/V must be block aligned.")
    source_blocks = len(block_ids)
    if source_blocks != source_tokens // block_size:
        raise ValueError("Detached CUDA block count does not match source tokens.")
    for metadata in _metadata_objects(attention_metadata):
        table = getattr(metadata, "block_table", None)
        seq_lens = getattr(metadata, "seq_lens", None)
        if table is None or seq_lens is None:
            raise NotImplementedError(
                "Detached CUDA currently requires paged attention metadata with "
                "block_table and seq_lens."
            )
        total_tokens = int(seq_lens[row].item())
        local_tokens = total_tokens - source_tokens
        if local_tokens <= 0:
            raise RuntimeError("Detached CUDA request has no local query suffix.")
        local_blocks = math.ceil(local_tokens / block_size)
        if source_blocks + local_blocks > table.shape[1]:
            raise RuntimeError("Detached CUDA attention block table is too narrow.")
        local = table[row, :local_blocks].clone()
        table[row, source_blocks : source_blocks + local_blocks].copy_(local)
        table[row, :source_blocks].copy_(
            torch.as_tensor(block_ids, dtype=table.dtype, device=table.device)
        )


def active_detached_commands() -> Mapping[str, DetachedCommand]:
    """Return commands bound around the current worker execute call."""

    return _ACTIVE_COMMANDS.get()


def active_detached_rows() -> tuple[str, ...]:
    """Return model-runner row identities for the current execute call."""

    return _ACTIVE_ROWS.get()


def detached_layout(runner: Any | None = None) -> DetachedLayout | None:
    """Return the reserved physical tail associated with a model runner."""

    if runner is not None:
        return _LAYOUTS.get(id(runner))
    return next(iter(_LAYOUTS.values()), None) if len(_LAYOUTS) == 1 else None


def install_detached_runner_hooks(reserve_blocks: int) -> None:
    """Install version-bounded vLLM worker hooks once per process."""

    global _RESERVE_BLOCKS
    if reserve_blocks <= 0:
        raise ValueError("Detached CUDA reserve_blocks must be positive.")
    with _HOOK_LOCK:
        _RESERVE_BLOCKS = max(_RESERVE_BLOCKS, int(reserve_blocks))
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if GPUModelRunner in _HOOKED_CLASSES:
            return
        original_initialize = GPUModelRunner.initialize_kv_cache
        original_prepare = GPUModelRunner._prepare_inputs
        original_execute = GPUModelRunner.execute_model

        @functools.wraps(original_initialize)
        def initialize(worker: Any, kv_cache_config: Any, is_profiling: bool = False):
            if is_profiling:
                return original_initialize(worker, kv_cache_config, is_profiling)
            expanded, layout = expand_worker_kv_cache_config(
                kv_cache_config, _RESERVE_BLOCKS
            )
            result = original_initialize(worker, expanded, is_profiling)
            _LAYOUTS[id(worker)] = layout
            return result

        @functools.wraps(original_prepare)
        def prepare(worker: Any, scheduler_output: Any, num_scheduled_tokens: Any):
            result = original_prepare(worker, scheduler_output, num_scheduled_tokens)
            _ACTIVE_ROWS.set(tuple(map(str, worker.input_batch.req_ids)))
            apply_detached_query_geometry(
                worker, num_scheduled_tokens, active_detached_commands()
            )
            return result

        @functools.wraps(original_execute)
        def execute(worker: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
            metadata = getattr(scheduler_output, "kv_connector_metadata", None)
            commands = {
                str(request.request_id): DetachedCommand(
                    str(request.request_id), int(request.source_tokens)
                )
                for request in getattr(metadata, "requests", ())
                if request.mode == "load" and getattr(request, "detached", False)
            }
            token = _ACTIVE_COMMANDS.set(commands)
            row_token = _ACTIVE_ROWS.set(())
            try:
                return original_execute(worker, scheduler_output, *args, **kwargs)
            finally:
                _ACTIVE_ROWS.reset(row_token)
                _ACTIVE_COMMANDS.reset(token)

        GPUModelRunner.initialize_kv_cache = initialize
        GPUModelRunner._prepare_inputs = prepare
        GPUModelRunner.execute_model = execute
        _HOOKED_CLASSES.add(GPUModelRunner)
