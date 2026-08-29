"""Live vLLM-Metal V1 generation bridge for native PRA K/V pages.

The bridge expands the runner-owned physical cache with a scheduler-invisible
tail. Selected post-RoPE K/V is written into that tail, then prepended only to
the attention block table for a registered request. Scheduler slot mappings,
prefix-cache identities, and sequential token counts remain unchanged.
"""

from __future__ import annotations

import contextvars
import math
import threading
import types
from dataclasses import dataclass
from dataclasses import replace
from typing import Mapping, Sequence

from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_mlx.native import MLXNativeMemory
from pra_vllm.v1_metadata import VLLMNativeBlockSet, VLLMNativeStepRegistry


@dataclass(frozen=True)
class VLLMSchedulerObservation:
    """One scheduler-owned prefill row observed before PRA augmentation.

    ``scheduler_cache_start`` is vLLM-Metal's reconciled APC boundary. PRA
    pages are added later, at the attention metadata boundary, so this record
    distinguishes ordinary prefix reuse from selected non-prefix memory.
    """

    request_id: str
    scheduler_cache_start: int
    scheduled_query_tokens: int
    prompt_tokens: int | None
    selected_registered: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "scheduler_cache_start": self.scheduler_cache_start,
            "scheduled_query_tokens": self.scheduled_query_tokens,
            "prompt_tokens": self.prompt_tokens,
            "selected_registered": self.selected_registered,
        }


def observe_prefill_rows(
    prefill_requests: Sequence[object],
    registered_request_ids: set[str],
) -> tuple[VLLMSchedulerObservation, ...]:
    """Capture scheduler geometry before selected pages enter attention."""

    return tuple(
        VLLMSchedulerObservation(
            request_id=str(request.req_id),
            scheduler_cache_start=int(request.start_pos),
            scheduled_query_tokens=len(request.token_ids),
            prompt_tokens=(
                None if request.prompt_len is None else int(request.prompt_len)
            ),
            selected_registered=str(request.req_id) in registered_request_ids,
        )
        for request in prefill_requests
    )


def augment_paged_context(
    context: object,
    request_ids: Sequence[str],
    selected_by_request: Mapping[str, VLLMNativeBlockSet],
) -> None:
    """Prepend selected pages to V1 attention metadata in request-row order."""

    groups = getattr(context, "kv_groups", None)
    if groups is None:
        groups = (context,)
    if len(request_ids) != len(context.context_lens):
        raise RuntimeError("vLLM PRA request rows do not match paged context rows.")
    for row, request_id in enumerate(request_ids):
        selected = selected_by_request.get(str(request_id))
        if selected is None:
            continue
        if len(selected.block_ids_by_group) != len(groups):
            raise RuntimeError("vLLM PRA pages do not match scheduler cache groups.")
        for group, selected_blocks in zip(groups, selected.block_ids_by_group):
            if selected.selected_token_count % int(group.block_size):
                raise ValueError(
                    "Live vLLM PRA currently requires page-aligned selected tokens."
                )
            group.block_tables[row][:0] = list(selected_blocks)
        context.context_lens[row] += selected.selected_token_count
        context.offsets[row] += selected.source_position_base
    if getattr(context, "kv_groups", None) is not None:
        context.block_tables = context.kv_groups[0].block_tables
        context.slot_mapping = context.kv_groups[0].slot_mapping
    context.kernel_metadata_cache.clear()


class VLLMMetalV1NativeBridge:
    """Attach scheduler-invisible selected pages to live V1 generation.

    This first measured bridge supports the standard, unquantized MHA Metal
    cache with one scheduler K/V group and a uniform page size. Selected spans
    must fill complete pages and must be consumed by every attention layer.
    """

    integration_level = "E2"

    def __init__(self, runner: object, *, reserve_blocks: int = 64) -> None:
        if reserve_blocks <= 0:
            raise ValueError("vLLM PRA reserve_blocks must be positive.")
        self.runner = runner
        self.runtime = runner.paged_attention_runtime
        if self.runtime is None:
            raise RuntimeError("vLLM PRA requires the V1 paged-attention runtime.")
        cache = self.runtime.kv_cache
        if getattr(cache, "turboquant", False):
            raise NotImplementedError("Live vLLM PRA does not yet support TurboQuant.")
        group_sizes = tuple(self.runtime.kv_group_block_sizes())
        if len(group_sizes) != 1:
            raise NotImplementedError(
                "Live vLLM PRA currently supports one scheduler K/V group."
            )
        self.block_size = int(group_sizes[0])
        self.scheduler_blocks = int(cache.num_blocks)
        self.reserve_blocks = int(reserve_blocks)
        self.registry = VLLMNativeStepRegistry()
        self.isolation = EnginePRAIsolationGuard()
        self._observation_lock = threading.RLock()
        self._scheduler_observations: list[VLLMSchedulerObservation] = []
        self._handles: dict[str, tuple[int, ...]] = {}
        self._free = list(
            range(self.scheduler_blocks, self.scheduler_blocks + self.reserve_blocks)
        )
        self._active_rows: contextvars.ContextVar[
            tuple[tuple[str, ...], tuple[str, ...]] | None
        ] = contextvars.ContextVar("vllm_pra_rows", default=None)
        self._original_start = runner._start_paged_forward
        self._original_reconcile = runner._reconcile_request_lifecycle
        self._model_runner_module = __import__(
            runner.__class__.__module__, fromlist=["prepare_grouped"]
        )
        self._original_prepare = self._model_runner_module.prepare_grouped
        self._expand_runtime_cache()
        self._install_hooks()

    def _expand_runtime_cache(self) -> None:
        """Append physical pages and rebind every patched attention layer."""

        import mlx.core as mx
        from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache

        old = self.runtime.kv_cache
        total = self.scheduler_blocks + self.reserve_blocks
        layout = getattr(old, "_layout", None)
        if layout is None:
            expanded = MetalPagedKVCache(
                num_layers=old.num_layers,
                num_kv_heads=old.num_kv_heads,
                head_dim=old.head_dim,
                num_blocks=total,
                block_size=old.block_size,
                dtype=old.dtype,
                kv_heads_per_layer=old.kv_heads_per_layer,
                head_dim_per_layer=old.head_dim_per_layer,
                sliding_window_per_layer=old.sliding_window_per_layer,
            )
        else:
            expanded_layout = replace(layout, num_blocks=total)
            expanded = MetalPagedKVCache.from_layout(expanded_layout, old.dtype)
            self.runtime._layout = expanded_layout
        for index in range(old.num_layers):
            expanded.key_caches[index][: self.scheduler_blocks] = old.key_caches[index]
            expanded.value_caches[index][: self.scheduler_blocks] = old.value_caches[index]
        mx.eval(*expanded.key_caches, *expanded.value_caches)
        self.runtime._cache = expanded
        self.runtime.patch_model(self.runner.model)

    def materialize(self, logical_key: str, memory: MLXNativeMemory) -> tuple[int, ...]:
        """Write one immutable, page-aligned native memory into reserved pages."""

        import mlx.core as mx

        key = str(logical_key)
        existing = self._handles.get(key)
        if existing is not None:
            return existing
        if len(memory.layers) != self.runtime.kv_cache.num_layers:
            raise ValueError("Selected memory does not match vLLM model layers.")
        if memory.source_tokens % self.block_size:
            raise ValueError(
                "Live vLLM PRA selected memory must contain complete physical pages."
            )
        block_count = math.ceil(memory.source_tokens / self.block_size)
        if block_count > len(self._free):
            raise MemoryError("vLLM PRA reserved tail has insufficient free pages.")
        block_ids = tuple(self._free[:block_count])
        del self._free[:block_count]
        cache = self.runtime.kv_cache
        for index, layer in enumerate(memory.layers):
            keys = layer.keys[0].transpose(1, 0, 2).reshape(
                block_count,
                self.block_size,
                cache.kv_heads_per_layer[index],
                cache.head_dim_per_layer[index],
            )
            values = layer.values[0].transpose(1, 0, 2).reshape(
                block_count,
                self.block_size,
                cache.kv_heads_per_layer[index],
                cache.head_dim_per_layer[index],
            )
            cache.key_caches[index][list(block_ids)] = keys
            cache.value_caches[index][list(block_ids)] = values
        mx.eval(*cache.key_caches, *cache.value_caches)
        self._handles[key] = block_ids
        return block_ids

    def release(self, logical_key: str) -> None:
        """Release reserved pages after no active request can reference them."""

        blocks = self._handles.pop(str(logical_key), None)
        if blocks is not None:
            self._free.extend(blocks)
            self._free.sort()

    def register(
        self,
        request_id: str,
        logical_keys: Sequence[str],
        *,
        selected_token_count: int,
        source_position_base: int,
        consumer_layers: Sequence[int] | None = None,
    ) -> None:
        """Bind materialized pages to one forthcoming vLLM request ID."""

        keys = tuple(map(str, logical_keys))
        missing = [key for key in keys if key not in self._handles]
        if missing:
            raise KeyError(f"vLLM PRA pages are not materialized: {missing}")
        expected_layers = tuple(range(self.runtime.kv_cache.num_layers))
        consumers = expected_layers if consumer_layers is None else tuple(consumer_layers)
        if consumers != expected_layers:
            raise NotImplementedError(
                "Live vLLM PRA currently requires every attention layer as a consumer."
            )
        selected = VLLMNativeBlockSet(
            logical_keys=keys,
            block_ids_by_group=(
                tuple(block for key in keys for block in self._handles[key]),
            ),
            selected_token_count=int(selected_token_count),
            source_position_base=int(source_position_base),
            consumer_layers=consumers,
        )
        if selected.selected_token_count % self.block_size:
            raise ValueError("vLLM PRA request selection must be page-aligned.")
        physical_tokens = (
            len(selected.block_ids_by_group[0]) * self.block_size
        )
        if selected.selected_token_count != physical_tokens:
            raise ValueError(
                "vLLM PRA selected_token_count must cover every registered page."
            )
        self.registry.register(request_id, selected)
        self.isolation.open_request(request_id, keys)

    def unregister(self, request_id: str) -> None:
        self.registry.unregister(request_id)
        self.isolation.close_request(request_id, require_attached=False)

    def _selected(self, request_id: str) -> VLLMNativeBlockSet | None:
        rows = self.registry.plan_step(
            (request_id,),
            scheduler_cache_starts={request_id: 0},
            query_token_counts={request_id: 1},
        )
        return rows[0].selected

    def _install_hooks(self) -> None:
        bridge = self

        def start(_runner, batch, prefill_reqs, decode_reqs, scheduler_output):
            registered = set(bridge.registry.active_request_ids())
            observations = observe_prefill_rows(prefill_reqs, registered)
            if observations:
                with bridge._observation_lock:
                    bridge._scheduler_observations.extend(observations)
            token = bridge._active_rows.set(
                (
                    tuple(str(req_id) for req_id, _ in decode_reqs),
                    tuple(str(req.req_id) for req in prefill_reqs),
                )
            )
            try:
                return bridge._original_start(
                    batch, prefill_reqs, decode_reqs, scheduler_output
                )
            finally:
                bridge._active_rows.reset(token)

        def prepare(decode_requests, prefill_requests, block_sizes, **kwargs):
            bridge._original_prepare(
                decode_requests, prefill_requests, block_sizes, **kwargs
            )
            active = bridge._active_rows.get()
            if active is None:
                return
            decode_ids, prefill_ids = active
            expanded_ids: list[str] = []
            merge = bool(kwargs.get("merge_verify_windows", False))
            for request_id, request in zip(decode_ids, decode_requests):
                query_tokens = 1 if len(request) == 2 else int(request[2])
                expanded_ids.extend(
                    [request_id] if merge or query_tokens == 1 else [request_id] * query_tokens
                )
            expanded_ids.extend(prefill_ids)
            from vllm_metal.attention.context import get_context

            context = get_context()
            selected = {
                request_id: value
                for request_id in set(expanded_ids)
                if (value := bridge._selected(request_id)) is not None
            }
            for request_id, value in selected.items():
                view = bridge.isolation.view(request_id)
                if view is not None and not view.attached:
                    bridge.isolation.attach_once(request_id, value.logical_keys)
            if context is not None:
                augment_paged_context(context, expanded_ids, selected)

        def reconcile(_runner, evicted_req_ids, **kwargs):
            result = bridge._original_reconcile(evicted_req_ids, **kwargs)
            for request_id in evicted_req_ids:
                bridge.unregister(request_id)
            return result

        self.runner._start_paged_forward = types.MethodType(start, self.runner)
        self.runner._reconcile_request_lifecycle = types.MethodType(
            reconcile, self.runner
        )
        self._model_runner_module.prepare_grouped = prepare

    def scheduler_observations(
        self, request_ids: Sequence[str] | None = None
    ) -> tuple[Mapping[str, object], ...]:
        """Return immutable pre-augmentation scheduler observations."""

        selected_ids = None if request_ids is None else set(map(str, request_ids))
        with self._observation_lock:
            rows = tuple(self._scheduler_observations)
        return tuple(
            row.as_dict()
            for row in rows
            if selected_ids is None or row.request_id in selected_ids
        )

    def close(self) -> None:
        """Restore runner hooks and clear request-scoped metadata."""

        self.runner._start_paged_forward = self._original_start
        self.runner._reconcile_request_lifecycle = self._original_reconcile
        self._model_runner_module.prepare_grouped = self._original_prepare
        self.isolation.close()

    def capabilities(self) -> Mapping[str, object]:
        return {
            "integration_level": self.integration_level,
            "native_kv_generation": True,
            "scheduler_invisible_tail_pages": self.reserve_blocks,
            "ordinary_prefix_namespace_used": False,
            "page_aligned_selection_required": True,
            "consumer_layers": "all",
            "scheduler_prefix_observability": True,
        }
