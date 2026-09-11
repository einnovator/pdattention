"""Native selected-K/V integration for SGLang's MLX model runner.

SGLang's scheduler counts only sequential request tokens.  PRA memory is not a
prefix and must therefore stay out of that count even though attention sees it.
This module keeps the scheduler-facing cache offset local while an attention
view exposes the source-position offset and ``selected + local`` K/V tensors.
"""

from __future__ import annotations

import contextvars
import types
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_hf.live_history import LiveKVSelectionPlan, LiveKVSourceRegistry
from pra_hf.storage_lifecycle import PRAStorageManager
from pra_mlx.native import (
    MLXDisjointLayerKV,
    MLXDisjointNativeMemory,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXResidentKVSelection,
    capture_live_native_memory,
    combine_native_memories,
    deserialize_native_memory,
    disjoint_segmented_selected_attention,
    select_live_native_memory_disjoint,
    serialize_native_memory,
)
from pra_sglang.hicache import SGLangPRAHiCache


class _AttentionCacheView:
    """mlx-lm cache protocol presented only inside one attention module."""

    def __init__(self, owner: "SGLangSelectedKVCache") -> None:
        self.owner = owner

    @property
    def offset(self) -> int:
        return self.owner.rope_offset

    @property
    def state(self):
        return self.owner.state

    def update_and_fetch(self, keys, values):
        return self.owner.update_and_fetch(keys, values)

    def make_mask(self, n: int, return_array: bool = False, **kwargs: object):
        return self.owner.make_mask(n, return_array=return_array, **kwargs)


class SGLangSelectedKVCache:
    """SGLang cache carrying immutable PRA K/V beside local sequential K/V.

    ``offset`` is deliberately the local sequence length used by scheduler and
    Radix bookkeeping. ``rope_offset`` is the absolute source-position extent
    used by model attention. K/V shapes are ``[B,Hkv,T,D]``.
    """

    def __init__(
        self,
        local_cache: object,
        memory: MLXNativeLayerKV | MLXDisjointLayerKV,
        *,
        position_base: int,
    ) -> None:
        if position_base < 0:
            raise ValueError("SGLang PRA position base cannot be negative.")
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)
        self.attention_view = _AttentionCacheView(self)

    @property
    def offset(self) -> int:
        return int(self.local_cache.offset)

    @offset.setter
    def offset(self, value: int) -> None:
        self.local_cache.offset = int(value)

    @property
    def rope_offset(self) -> int:
        return self.position_base + self.offset

    @property
    def memory_tokens(self) -> int:
        if isinstance(self.memory, MLXDisjointLayerKV):
            return self.memory.tokens
        return int(self.memory.keys.shape[2])

    @property
    def disjoint(self) -> bool:
        return isinstance(self.memory, MLXDisjointLayerKV)

    @property
    def keys(self):
        """Local K only; shared Radix pools must not ingest PRA as a prefix."""

        return self.local_cache.keys

    @property
    def values(self):
        return self.local_cache.values

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        if self.disjoint:
            selected = tuple(
                value
                for segment in self.memory.segments
                for value in (segment.keys, segment.values)
            )
            return (*selected, *local)
        return (self.memory.keys, self.memory.values, *local)

    def reset(self) -> None:
        self.local_cache.reset()

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        if self.disjoint:
            raise RuntimeError(
                "Disjoint SGLang PRA cache requires interval-addressed attention."
            )

        prior_offset = self.offset
        local_k, local_v = self.local_cache.update_and_fetch(keys, values)
        expected_local_tokens = prior_offset + int(keys.shape[2])
        if int(local_k.shape[2]) != expected_local_tokens:
            raise RuntimeError(
                "SGLang local cache returned contaminated K/V: "
                f"expected {expected_local_tokens} tokens, got {local_k.shape[2]} "
                f"from {type(self.local_cache).__name__}."
            )
        return (
            mx.concatenate((self.memory.keys, local_k), axis=2),
            mx.concatenate((self.memory.values, local_v), axis=2),
        )

    def update_and_fetch_segments(self, keys, values):
        """Update request-local K/V while retaining source intervals separately."""

        if not self.disjoint:
            raise RuntimeError("Dense SGLang PRA cache has no disjoint segments.")
        local_k, local_v = self.local_cache.update_and_fetch(keys, values)
        return (
            tuple(segment.keys for segment in self.memory.segments),
            tuple(segment.values for segment in self.memory.segments),
            local_k,
            local_v,
        )

    def write_token(self, keys, values) -> None:
        self.local_cache.write_token(keys, values)

    def get_kv(self, window: int | None = None):
        import mlx.core as mx

        if self.disjoint:
            raise RuntimeError(
                "Disjoint SGLang PRA cache cannot materialize get_kv(); "
                "use interval-addressed attention."
            )
        local_k, local_v = self.local_cache.get_kv(window)
        return (
            mx.concatenate((self.memory.keys, local_k), axis=2),
            mx.concatenate((self.memory.values, local_v), axis=2),
        )

    def make_mask(
        self,
        n: int,
        return_array: bool = False,
        window_size: int | None = None,
        **_: object,
    ):
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        if window_size is None:
            local = create_causal_mask(n, self.offset)
        else:
            local = create_causal_mask(
                n, self.offset, window_size=min(window_size, self.offset + n)
            )
        selected = mx.ones((n, self.memory_tokens), dtype=mx.bool_)
        return mx.concatenate((selected, local), axis=1)


def _base_attention(attention: object) -> object:
    """Return the mlx-lm attention wrapped by SGLang's decode adapter."""

    return getattr(attention, "_inner", attention)


def _qwen_projections(attention: object, x: object):
    """Project and normalize Qwen3 Q/K/V without changing model weights."""

    base = _base_attention(attention)
    batch, length, _ = x.shape
    queries = base.q_proj(x)
    keys = base.k_proj(x)
    values = base.v_proj(x)
    queries = base.q_norm(
        queries.reshape(batch, length, base.n_heads, -1)
    ).transpose(0, 2, 1, 3)
    keys = base.k_norm(
        keys.reshape(batch, length, base.n_kv_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(batch, length, base.n_kv_heads, -1).transpose(
        0, 2, 1, 3
    )
    return base, queries, keys, values


def _disjoint_qwen_attention(
    attention: object, x: object, cache: SGLangSelectedKVCache
):
    """Consume an unbatched SGLang request without packing source intervals."""

    base, queries, keys, values = _qwen_projections(attention, x)
    length = int(x.shape[1])
    queries = base.rope(queries, offset=cache.rope_offset)
    keys = base.rope(keys, offset=cache.rope_offset)
    layer_mask = cache.make_mask(length, return_array=True)
    memory_k, memory_v, local_k, local_v = cache.update_and_fetch_segments(
        keys, values
    )
    output = disjoint_segmented_selected_attention(
        queries,
        memory_k,
        memory_v,
        local_k,
        local_v,
        scale=float(base.scale),
        mask=layer_mask,
        source_keys=cache.memory.source_keys,
        source_values=cache.memory.source_values,
        source_intervals=cache.memory.intervals,
    )
    output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], length, -1)
    return base.o_proj(output)


def _disjoint_qwen_batched_attention(
    attention: object,
    x: object,
    context: object,
    layer_caches: Sequence[SGLangSelectedKVCache],
):
    """Consume an all-disjoint SGLang decode batch one request at a time."""

    import mlx.core as mx

    base, queries, keys, values = _qwen_projections(attention, x)
    offsets = context.offsets
    queries = base.rope(queries, offset=offsets)
    keys = base.rope(keys, offset=offsets)
    rows = []
    for index, cache in enumerate(layer_caches):
        # SGLang normally uses write_token/get_kv during batched decode. The
        # local cache protocol's update_and_fetch is equivalent here and lets
        # the selected source intervals remain separate.
        mask = cache.make_mask(1, return_array=True)
        memory_k, memory_v, local_k, local_v = cache.update_and_fetch_segments(
            keys[index : index + 1], values[index : index + 1]
        )
        rows.append(
            disjoint_segmented_selected_attention(
                queries[index : index + 1],
                memory_k,
                memory_v,
                local_k,
                local_v,
                scale=float(base.scale),
                mask=mask,
                source_keys=cache.memory.source_keys,
                source_values=cache.memory.source_values,
                source_intervals=cache.memory.intervals,
            )
        )
    output = mx.concatenate(rows, axis=0)
    output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], 1, -1)
    return base.o_proj(output)


def install_selected_kv_attention(model: object) -> int:
    """Patch supported mlx-lm attention modules to consume the PRA cache view.

    The patch wraps, rather than forks, the model-specific attention. It also
    composes with SGLang's existing ``MLXAttentionWrapper`` during prefill.
    """

    import mlx.nn as nn

    class _PRAAttentionWrapper(nn.Module):
        def __init__(self, inner: object) -> None:
            super().__init__()
            object.__setattr__(self, "_inner", inner)

        def __call__(self, x, mask=None, cache=None):
            if isinstance(cache, SGLangSelectedKVCache):
                if cache.disjoint:
                    return _disjoint_qwen_attention(self._inner, x, cache)
                cache = cache.attention_view
            # SGLang's batched decode passes shim caches through the model and
            # keeps the real per-request caches in a thread-local context.
            # Intercept an all-disjoint batch before MLXAttentionWrapper calls
            # get_kv(), which would otherwise pack selected history.
            try:
                from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
                    get_context,
                )

                context = get_context()
            except (ImportError, AttributeError):
                context = None
            layer_index = getattr(self._inner, "_layer_idx", None)
            if context is not None and layer_index is not None:
                cache_index = context.attention_pool_index_by_layer[layer_index]
                layer_caches = context.attention_layer_caches[cache_index]
                disjoint = [
                    isinstance(row, SGLangSelectedKVCache) and row.disjoint
                    for row in layer_caches
                ]
                if any(disjoint):
                    if not all(disjoint):
                        raise RuntimeError(
                            "A SGLang PRA decode batch cannot mix disjoint and "
                            "ordinary cache protocols."
                        )
                    return _disjoint_qwen_batched_attention(
                        self._inner, x, context, layer_caches
                    )
            return self._inner(x, mask=mask, cache=cache)

    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise TypeError("SGLang PRA currently requires a model.layers attention stack.")
    patched = 0
    for layer in layers:
        attribute = "self_attn" if hasattr(layer, "self_attn") else "attention"
        attention = getattr(layer, attribute, None)
        if attention is None:
            continue
        if getattr(attention, "_pra_selected_kv_wrapper", False):
            continue
        wrapper = _PRAAttentionWrapper(attention)
        object.__setattr__(wrapper, "_pra_selected_kv_wrapper", True)
        setattr(layer, attribute, wrapper)
        patched += 1
    if patched == 0:
        raise TypeError("No supported SGLang MLX attention layers were found.")
    return patched


@dataclass(frozen=True)
class SGLangNativeRequest:
    """Selected immutable memory registered for one SGLang request ID."""

    memory: MLXNativeMemory | MLXDisjointNativeMemory
    source_position_base: int
    logical_keys: tuple[str, ...] = ()
    storage_pinned: bool = False


@dataclass(frozen=True)
class SGLangMLXLiveKVSource:
    """Canonical live-history K/V and the SGLang request that owns it."""

    memory: MLXNativeMemory
    owner_request_id: str


@dataclass
class SGLangMLXLiveKVRequest:
    """One request-scoped borrow of a canonical SGLang-MLX K/V source."""

    runtime: "SGLangMLXLiveKVRuntime"
    request_id: str
    source_id: str
    tenant_id: str
    session_id: str
    generation: int
    selection: MLXResidentKVSelection
    owner_request_id: str
    _closed: bool = field(default=False, init=False, repr=False)
    _outcome: str | None = field(default=None, init=False, repr=False)

    @property
    def active(self) -> bool:
        return not self._closed

    @property
    def outcome(self) -> str | None:
        return self._outcome

    def _close(self, outcome: str) -> bool:
        return self.runtime._terminate_request(self, outcome)

    def finish(self) -> bool:
        return self._close("finished")

    def cancel(self) -> bool:
        return self._close("cancelled")

    def fail(self) -> bool:
        return self._close("error")


class SGLangMLXNativeBridge:
    """Install selected K/V into SGLang MLX runner-owned request caches.

    Initial prefill and batched decode are supported. Radix pooling continues
    to synchronize local sequential K/V only. HiCache placement of the separate
    PRA namespace remains an explicit later integration level.
    """

    integration_level = "E2"

    def __init__(
        self,
        runner: object,
        *,
        hicache: SGLangPRAHiCache | None = None,
        storage_manager: PRAStorageManager | None = None,
    ) -> None:
        self.runner = runner
        self.hicache = hicache
        self.storage = storage_manager
        if getattr(runner._cache_layout, "has_auxiliary_state", False):
            raise NotImplementedError(
                "SGLang PRA does not yet wrap hybrid auxiliary-state caches."
            )
        if getattr(runner._cache_layout, "has_sliding_window_layers", False):
            raise NotImplementedError(
                "SGLang PRA batched decode does not yet support sliding-window layers."
            )
        self._requests: dict[str, SGLangNativeRequest] = {}
        self._release_callbacks: dict[str, Callable[[str, str], None]] = {}
        self._terminal_outcomes: dict[str, str] = {}
        self._source_owner_pins: dict[str, set[str]] = {}
        self._lifecycle_lock = RLock()
        self.isolation = EnginePRAIsolationGuard()
        self._active_req: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "sglang_pra_request", default=None
        )
        self._original_acquire = runner._acquire_cache
        self._original_release = runner._release_cache
        self._original_prefill_start = runner.prefill_start
        self._original_build_context = runner._build_batched_decode_context
        self._original_cache_from_prefix = runner._cache_with_pool_backed_attention
        self._original_materialize_prefix = runner._materialize_pool_backed_attention
        self._original_remove_request = runner.remove_request
        self.patched_layers = install_selected_kv_attention(runner.model)
        self._install_hooks()

    def register(
        self,
        req_id: str,
        memory: (
            MLXNativeMemory | MLXDisjointNativeMemory | MLXResidentKVSelection | None
        ) = None,
        *,
        logical_keys: tuple[str, ...] = (),
        tenant_id: str | None = None,
        session_id: str | None = None,
        source_position_base: int | None = None,
        on_release: Callable[[str, str], None] | None = None,
    ) -> None:
        identifier = str(req_id)
        storage_pinned = False
        live_selection = (
            memory if isinstance(memory, MLXResidentKVSelection) else None
        )
        if live_selection is not None:
            if source_position_base is not None:
                raise ValueError(
                    "A live SGLang selection already defines source_position_base."
                )
            source_position_base = live_selection.plan.source_position_base
            memory = live_selection.memory
        if memory is None:
            if not logical_keys:
                raise ValueError(
                    "SGLang PRA registration requires memory or lifecycle logical keys."
                )
            if self.storage is not None:
                promoted = []
                try:
                    for key in logical_keys:
                        promoted.append(
                            self.storage.promote(key, request_id=identifier)
                        )
                except Exception:
                    for key in logical_keys[: len(promoted)]:
                        self.storage.unpin(key, identifier)
                    raise
                memory = combine_native_memories(tuple(promoted))
                storage_pinned = True
            elif self.hicache is not None:
                memory = combine_native_memories(
                    tuple(self.hicache.get(key) for key in logical_keys)
                )
            else:
                raise ValueError(
                    "SGLang PRA logical-key registration requires shared storage or HiCache."
                )
        try:
            if len(memory.layers) != self.runner._cache_layout.num_layers:
                raise ValueError("Selected memory does not match SGLang model layers.")
            expected_tokens = (
                live_selection.plan.selected_tokens
                if live_selection is not None
                else memory.source_tokens
            )
            if any(
                (
                    layer.tokens
                    if isinstance(layer, MLXDisjointLayerKV)
                    else int(layer.keys.shape[2])
                )
                != expected_tokens
                for layer in memory.layers
            ):
                raise ValueError(
                    "Selected memory token geometry disagrees with its position base."
                )
            position_base = int(
                memory.source_tokens
                if source_position_base is None
                else source_position_base
            )
            if position_base < expected_tokens:
                raise ValueError(
                    "SGLang source_position_base cannot be smaller than selected K/V."
                )
            with self._lifecycle_lock:
                self.isolation.open_request(
                    identifier,
                    logical_keys,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
                self._requests[identifier] = SGLangNativeRequest(
                    memory, position_base, logical_keys, storage_pinned
                )
                if on_release is not None:
                    self._release_callbacks[identifier] = on_release
        except BaseException:
            if storage_pinned and self.storage is not None:
                for key in logical_keys:
                    self.storage.unpin(key, identifier)
            raise

    def unregister(self, req_id: str, *, outcome: str = "finished") -> bool:
        """Release bridge and source ownership once for one terminal request.

        ``MlxModelRunner.remove_request`` is the native completion/cancellation
        teardown point.  The installed wrapper calls this method in ``finally``
        so an engine cleanup error cannot strand a live-source borrow.
        """

        identifier = str(req_id)
        if outcome not in {"finished", "cancelled", "error"}:
            raise ValueError(f"Unknown SGLang request outcome {outcome!r}.")
        with self._lifecycle_lock:
            request = self._requests.pop(identifier, None)
            callback = self._release_callbacks.pop(identifier, None)
            self._terminal_outcomes.pop(identifier, None)
            if request is None:
                return False
            if request.storage_pinned and self.storage is not None:
                for key in request.logical_keys:
                    self.storage.unpin(key, identifier)
            self.isolation.close_request(identifier, require_attached=False)
        # Do not call an owner callback under the bridge lock. A concurrent
        # begin path takes the runtime lock before the bridge lock; reversing
        # that order here would permit teardown/registration deadlock.
        if callback is not None:
            callback(identifier, outcome)
        return True

    def set_terminal_outcome(self, req_id: str, outcome: str) -> None:
        """Annotate the next native removal as finish, cancellation, or error."""

        if outcome not in {"finished", "cancelled", "error"}:
            raise ValueError(f"Unknown SGLang request outcome {outcome!r}.")
        identifier = str(req_id)
        with self._lifecycle_lock:
            if identifier not in self._requests:
                raise KeyError(f"SGLang PRA request {identifier!r} is not active.")
            self._terminal_outcomes[identifier] = outcome

    def pin_source_owner(self, owner_req_id: str, borrower_req_id: str) -> None:
        """Prevent Radix/request-pool teardown of a borrowed source owner."""

        owner = str(owner_req_id)
        borrower = str(borrower_req_id)
        if not owner or not borrower:
            raise ValueError("SGLang source owner and borrower IDs cannot be empty.")
        with self._lifecycle_lock:
            self._source_owner_pins.setdefault(owner, set()).add(borrower)

    def unpin_source_owner(self, owner_req_id: str, borrower_req_id: str) -> bool:
        owner = str(owner_req_id)
        borrower = str(borrower_req_id)
        with self._lifecycle_lock:
            borrowers = self._source_owner_pins.get(owner)
            if borrowers is None or borrower not in borrowers:
                return False
            borrowers.remove(borrower)
            if not borrowers:
                del self._source_owner_pins[owner]
            return True

    def _wrap_cache(self, req_id: str, caches: list[object]) -> list[object]:
        request = self._requests.get(req_id)
        if request is None:
            return caches
        wrapped = [isinstance(cache, SGLangSelectedKVCache) for cache in caches]
        if any(wrapped):
            if not all(wrapped[index] for index in self.runner._cache_layout.attention_layer_indices):
                raise RuntimeError("SGLang PRA cache list is only partially wrapped.")
            return caches
        view = self.isolation.view(req_id)
        if view is not None and not view.attached:
            self.isolation.attach_once(req_id, request.logical_keys)
        memory = request.memory
        return [
            (
                SGLangSelectedKVCache(
                    cache,
                    memory.layers[index],
                    position_base=request.source_position_base,
                )
                if index in self.runner._cache_layout.attention_layer_indices
                else cache
            )
            for index, cache in enumerate(caches)
        ]

    @staticmethod
    def _unwrap_cache(caches: list[object]) -> list[object]:
        """Return only request-local caches to SGLang's reusable pool."""

        return [
            cache.local_cache if isinstance(cache, SGLangSelectedKVCache) else cache
            for cache in caches
        ]

    def _install_hooks(self) -> None:
        bridge = self

        def acquire(_runner):
            caches = bridge._original_acquire()
            req_id = bridge._active_req.get()
            return caches if req_id is None else bridge._wrap_cache(req_id, caches)

        def release(_runner, caches):
            return bridge._original_release(bridge._unwrap_cache(caches))

        def prefill(_runner, req_id, *args, **kwargs):
            token = bridge._active_req.set(str(req_id))
            try:
                return bridge._original_prefill_start(req_id, *args, **kwargs)
            finally:
                bridge._active_req.reset(token)

        def cache_from_prefix(_runner, prefix_slot_ids, prefix_len):
            caches = bridge._original_cache_from_prefix(prefix_slot_ids, prefix_len)
            req_id = bridge._active_req.get()
            return caches if req_id is None else bridge._wrap_cache(req_id, caches)

        def materialize_prefix(_runner, caches):
            # Pool materialization consumes only scheduler-owned prefix K/V.
            # The patched acquire path wraps the resulting contiguous cache
            # again for selected-memory attention without a second attach.
            return bridge._original_materialize_prefix(bridge._unwrap_cache(caches))

        def build_context(_runner, caches, req_ids):
            ctx = bridge._original_build_context(caches, req_ids)
            selected = [
                cache_list[_runner._cache_layout.first_attention_layer_index]
                for cache_list in caches
            ]
            if not any(isinstance(cache, SGLangSelectedKVCache) for cache in selected):
                return ctx
            if not all(isinstance(cache, SGLangSelectedKVCache) for cache in selected):
                raise RuntimeError("A SGLang PRA decode batch cannot mix cache protocols.")
            import mlx.core as mx

            ctx.seq_lens = [cache.rope_offset for cache in selected]
            ctx.offsets = mx.array(ctx.seq_lens, dtype=mx.int32)
            visible = [cache.memory_tokens + cache.offset + 1 for cache in selected]
            ctx.max_len = max(visible)
            ctx.valid_lens = mx.array(visible, dtype=mx.int32)
            ctx.needs_padding = min(visible) < ctx.max_len
            ctx.pad_sizes = [ctx.max_len - value for value in visible]
            ctx.positions = mx.arange(ctx.max_len) if ctx.needs_padding else None
            ctx._padding_by_window = {}
            return ctx

        def remove_request(_runner, req_id, *args, **kwargs):
            identifier = str(req_id)
            with bridge._lifecycle_lock:
                borrowers = tuple(
                    sorted(bridge._source_owner_pins.get(identifier, ()))
                )
                outcome = bridge._terminal_outcomes.get(identifier, "finished")
            if borrowers:
                raise RuntimeError(
                    "Cannot remove a SGLang source owner while live-K/V requests "
                    f"borrow it: {borrowers!r}."
                )
            try:
                result = bridge._original_remove_request(req_id, *args, **kwargs)
            except BaseException:
                bridge.unregister(identifier, outcome="error")
                raise
            bridge.unregister(identifier, outcome=outcome)
            return result

        self.runner._acquire_cache = types.MethodType(acquire, self.runner)
        self.runner._release_cache = types.MethodType(release, self.runner)
        self.runner.prefill_start = types.MethodType(prefill, self.runner)
        self.runner._cache_with_pool_backed_attention = types.MethodType(
            cache_from_prefix, self.runner
        )
        self.runner._materialize_pool_backed_attention = types.MethodType(
            materialize_prefix, self.runner
        )
        self.runner._build_batched_decode_context = types.MethodType(
            build_context, self.runner
        )
        self.runner.remove_request = types.MethodType(remove_request, self.runner)

    def close(self) -> None:
        self.runner._acquire_cache = self._original_acquire
        self.runner._release_cache = self._original_release
        self.runner.prefill_start = self._original_prefill_start
        self.runner._cache_with_pool_backed_attention = self._original_cache_from_prefix
        self.runner._materialize_pool_backed_attention = self._original_materialize_prefix
        self.runner._build_batched_decode_context = self._original_build_context
        self.runner.remove_request = self._original_remove_request
        for request_id in tuple(self._requests):
            self.unregister(request_id, outcome="cancelled")
        self._source_owner_pins.clear()
        self.isolation.close()

    def capabilities(self) -> Mapping[str, object]:
        return {
            "integration_level": self.integration_level,
            "native_kv": True,
            "radix_prefix_identity_separate": True,
            "radix_prefix_and_native_same_request": True,
            "hicache_external_namespace": (
                self.hicache is not None or self.storage is not None
            ),
            "shared_storage_lifecycle": self.storage is not None,
            "live_prefix_kv_capture": True,
            "live_prefix_kv_subset": True,
            "source_positions_preserved": True,
            "zero_selected_text_reencoding": True,
            "disjoint_selection_requires_pack_copy": False,
            "disjoint_attention_runtime_qualification_required": True,
            "physical_kv_copy_reported": True,
            "native_remove_request_terminal_hook": True,
            "source_owner_pin_guard": True,
            "request_activation_callback": "MlxModelRunner.prefill_start",
            "selected_cache_callbacks": (
                "MlxModelRunner._acquire_cache",
                "MlxModelRunner._cache_with_pool_backed_attention",
            ),
            "request_teardown_callback": "MlxModelRunner.remove_request",
            "radix_pool_release_strips_selected_memory": True,
            "hicache_metrics": (
                None if self.hicache is None else self.hicache.metrics().to_dict()
            ),
            "patched_layers": self.patched_layers,
        }


def _select_live_source_memory(
    source: MLXNativeMemory, plan: LiveKVSelectionPlan
) -> MLXResidentKVSelection:
    """Select existing MLX K/V cells without evaluating their source tokens."""

    if source.source_tokens != plan.source_tokens:
        raise ValueError(
            "SGLang live-K/V plan does not describe the complete source cache: "
            f"plan={plan.source_tokens}, source={source.source_tokens}."
        )
    packed = len(plan.intervals) > 1
    layers: list[MLXNativeLayerKV] = []
    for layer in source.layers:
        if int(layer.keys.shape[2]) < plan.source_tokens:
            raise ValueError("SGLang live K/V source is shorter than its selection plan.")
        key_parts = tuple(
            layer.keys[:, :, interval.start : interval.end, :]
            for interval in plan.intervals
        )
        value_parts = tuple(
            layer.values[:, :, interval.start : interval.end, :]
            for interval in plan.intervals
        )
        if not key_parts:
            keys = layer.keys[:, :, :0, :]
            values = layer.values[:, :, :0, :]
        elif len(key_parts) == 1:
            keys = key_parts[0]
            values = value_parts[0]
        else:
            import mlx.core as mx

            keys = mx.concatenate(key_parts, axis=2)
            values = mx.concatenate(value_parts, axis=2)
        layers.append(MLXNativeLayerKV(keys, values))
    memory = MLXNativeMemory(tuple(layers), plan.selected_tokens)
    if packed:
        import mlx.core as mx

        mx.eval(*(value for layer in memory.layers for value in (layer.keys, layer.values)))
    return MLXResidentKVSelection(
        memory,
        plan,
        physical_kv_copy=packed,
        selected_text_reencoded_tokens=0,
    )


class SGLangMLXLiveKVRuntime:
    """Bind canonical SGLang-MLX history K/V to native request teardown.

    The live registry owns the canonical source independently from RadixCache.
    A request borrow and source-owner pin are established before a selected
    cache can be constructed.  The bridge's ``remove_request`` hook drives the
    one terminal callback for normal completion, cancellation, and failures.
    """

    def __init__(
        self,
        bridge: SGLangMLXNativeBridge,
        *,
        dump: Callable[[SGLangMLXLiveKVSource], object] | None = None,
        load: Callable[[object], SGLangMLXLiveKVSource] | None = None,
    ) -> None:
        if dump is None and load is None:
            dump = self._dump_source
            load = self._load_source
        self.bridge = bridge
        self.registry = LiveKVSourceRegistry[SGLangMLXLiveKVSource](
            dump=dump, load=load
        )
        self._requests: dict[str, SGLangMLXLiveKVRequest] = {}
        self._terminal_counts = {"finished": 0, "cancelled": 0, "error": 0}
        self._lock = RLock()

    @staticmethod
    def _dump_source(source: SGLangMLXLiveKVSource) -> object:
        return (source.owner_request_id, serialize_native_memory(source.memory))

    @staticmethod
    def _load_source(payload: object) -> SGLangMLXLiveKVSource:
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise TypeError("Malformed offloaded SGLang live-K/V source.")
        owner, encoded = payload
        if not isinstance(encoded, bytes):
            raise TypeError("SGLang live-K/V offload payload must contain bytes.")
        return SGLangMLXLiveKVSource(
            deserialize_native_memory(encoded), str(owner)
        )

    def register_source(
        self,
        source_id: str,
        caches: Sequence[object] | MLXNativeMemory,
        *,
        owner_request_id: str,
        tenant_id: str,
        session_id: str,
        generation: int,
        source_tokens: int | None = None,
    ) -> None:
        """Capture a full live cache once under a persistent source identity."""

        owner = str(owner_request_id)
        if not owner:
            raise ValueError("SGLang live-K/V source owner ID cannot be empty.")
        runner_requests = getattr(self.bridge.runner, "_req_caches", None)
        if runner_requests is not None and owner not in runner_requests:
            raise KeyError(
                f"SGLang live-K/V source owner {owner!r} is not resident."
            )
        if isinstance(caches, MLXNativeMemory):
            memory = caches
            if source_tokens is not None and int(source_tokens) != memory.source_tokens:
                raise ValueError("Explicit source_tokens disagree with native memory.")
        else:
            cache_rows = tuple(caches)
            if not cache_rows:
                raise ValueError("SGLang live-K/V source caches cannot be empty.")
            if source_tokens is None:
                first = self.bridge.runner._cache_layout.first_attention_layer_index
                source_tokens = int(cache_rows[first].offset)
            full_plan = LiveKVSelectionPlan.full(int(source_tokens))
            memory = capture_live_native_memory(cache_rows, full_plan).memory
        self.registry.register(
            source_id,
            SGLangMLXLiveKVSource(memory, owner),
            tenant_id=tenant_id,
            session_id=session_id,
            generation=generation,
        )

    def begin_request(
        self,
        request_id: str,
        source_id: str,
        plan: LiveKVSelectionPlan,
        *,
        tenant_id: str,
        session_id: str,
        expected_generation: int,
        disjoint: bool = False,
    ) -> SGLangMLXLiveKVRequest:
        """Borrow and pin the source before constructing selected request K/V."""

        request_key = str(request_id)
        with self._lock:
            if request_key in self._requests:
                raise RuntimeError(
                    f"SGLang live-K/V request {request_key!r} is already active."
                )
            (source,) = self.registry.borrow(
                request_key,
                (source_id,),
                tenant_id=tenant_id,
                session_id=session_id,
                expected_generations=(expected_generation,),
            )
            self.bridge.pin_source_owner(source.owner_request_id, request_key)
            try:
                selection = (
                    select_live_native_memory_disjoint(source.memory, plan)
                    if disjoint
                    else _select_live_source_memory(source.memory, plan)
                )
                request = SGLangMLXLiveKVRequest(
                    self,
                    request_key,
                    str(source_id),
                    str(tenant_id),
                    str(session_id),
                    int(expected_generation),
                    selection,
                    source.owner_request_id,
                )
                self._requests[request_key] = request
                self.bridge.register(
                    request_key,
                    selection,
                    logical_keys=(str(source_id),),
                    tenant_id=tenant_id,
                    session_id=session_id,
                    on_release=self._native_released,
                )
            except BaseException:
                self._requests.pop(request_key, None)
                self.bridge.unpin_source_owner(source.owner_request_id, request_key)
                self.registry.release(request_key)
                raise
            return request

    def _native_released(self, request_id: str, outcome: str) -> None:
        """Complete the registry transition behind the native teardown hook."""

        with self._lock:
            request = self._requests.pop(request_id, None)
            if request is None:
                return
            if not self.registry.release(request_id):
                raise RuntimeError("SGLang live-K/V registry lost an active borrow.")
            if not self.bridge.unpin_source_owner(
                request.owner_request_id, request_id
            ):
                raise RuntimeError("SGLang live-K/V source-owner pin was lost.")
            request._closed = True
            request._outcome = outcome
            self._terminal_counts[outcome] += 1

    def _terminate_request(
        self, request: SGLangMLXLiveKVRequest, outcome: str
    ) -> bool:
        with self._lock:
            if request._closed:
                return False
            if self._requests.get(request.request_id) is not request:
                raise RuntimeError("SGLang live-K/V request ownership is inconsistent.")
            self.bridge.set_terminal_outcome(request.request_id, outcome)
            view = self.bridge.isolation.view(request.request_id)
            if view is not None and view.attached:
                self.bridge.runner.remove_request(request.request_id)
            else:
                # A queued request may be cancelled before SGLang constructs
                # its cache.  There is then no native cache to remove, but the
                # same bridge callback still owns exact-once release.
                self.bridge.unregister(request.request_id, outcome=outcome)
            return True

    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(str(request_id))
            return False if request is None else request.cancel()

    def offload_source(self, source_id: str) -> object:
        return self.registry.offload(source_id)

    evict_source = offload_source

    def terminate_session(self, tenant_id: str, session_id: str) -> int:
        """Cancel native borrowers before installing a session tombstone."""

        with self._lock:
            affected = tuple(
                request
                for request in self._requests.values()
                if (request.tenant_id, request.session_id)
                == (str(tenant_id), str(session_id))
            )
            for request in affected:
                request.cancel()
            return self.registry.terminate_session(tenant_id, session_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_request_ids": tuple(sorted(self._requests)),
                "terminal_counts": dict(self._terminal_counts),
                "physical_kv_copy_policy": (
                    "dense_interval_pack_or_unqualified_disjoint_views"
                ),
                "selected_text_reencoded_tokens": 0,
            }

    def close(self) -> None:
        with self._lock:
            for request in tuple(self._requests.values()):
                request.cancel()
            self.registry.close()
