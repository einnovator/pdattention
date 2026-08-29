"""Native selected-K/V integration for SGLang's MLX model runner.

SGLang's scheduler counts only sequential request tokens.  PRA memory is not a
prefix and must therefore stay out of that count even though attention sees it.
This module keeps the scheduler-facing cache offset local while an attention
view exposes the source-position offset and ``selected + local`` K/V tensors.
"""

from __future__ import annotations

import contextvars
import types
from dataclasses import dataclass
from typing import Any, Mapping

from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory


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
        memory: MLXNativeLayerKV,
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
        return int(self.memory.keys.shape[2])

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
        return (self.memory.keys, self.memory.values, *local)

    def reset(self) -> None:
        self.local_cache.reset()

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

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

    def write_token(self, keys, values) -> None:
        self.local_cache.write_token(keys, values)

    def get_kv(self, window: int | None = None):
        import mlx.core as mx

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
                cache = cache.attention_view
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

    memory: MLXNativeMemory
    logical_keys: tuple[str, ...] = ()


class SGLangMLXNativeBridge:
    """Install selected K/V into SGLang MLX runner-owned request caches.

    Initial prefill and batched decode are supported. Radix pooling continues
    to synchronize local sequential K/V only. HiCache placement of the separate
    PRA namespace remains an explicit later integration level.
    """

    integration_level = "E1"

    def __init__(self, runner: object) -> None:
        self.runner = runner
        if getattr(runner._cache_layout, "has_auxiliary_state", False):
            raise NotImplementedError(
                "SGLang PRA does not yet wrap hybrid auxiliary-state caches."
            )
        if getattr(runner._cache_layout, "has_sliding_window_layers", False):
            raise NotImplementedError(
                "SGLang PRA batched decode does not yet support sliding-window layers."
            )
        self._requests: dict[str, SGLangNativeRequest] = {}
        self.isolation = EnginePRAIsolationGuard()
        self._active_req: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "sglang_pra_request", default=None
        )
        self._original_acquire = runner._acquire_cache
        self._original_release = runner._release_cache
        self._original_prefill_start = runner.prefill_start
        self._original_build_context = runner._build_batched_decode_context
        self.patched_layers = install_selected_kv_attention(runner.model)
        self._install_hooks()

    def register(
        self,
        req_id: str,
        memory: MLXNativeMemory,
        *,
        logical_keys: tuple[str, ...] = (),
    ) -> None:
        if len(memory.layers) != self.runner._cache_layout.num_layers:
            raise ValueError("Selected memory does not match SGLang model layers.")
        if any(int(layer.keys.shape[2]) != memory.source_tokens for layer in memory.layers):
            raise ValueError("Selected memory token geometry disagrees with its position base.")
        identifier = str(req_id)
        self.isolation.open_request(identifier, logical_keys)
        self._requests[identifier] = SGLangNativeRequest(memory, logical_keys)

    def unregister(self, req_id: str) -> None:
        identifier = str(req_id)
        self._requests.pop(identifier, None)
        self.isolation.close_request(identifier, require_attached=False)

    def _wrap_cache(self, req_id: str, caches: list[object]) -> list[object]:
        request = self._requests.get(req_id)
        if request is None:
            return caches
        self.isolation.attach_once(req_id, request.logical_keys)
        memory = request.memory
        return [
            SGLangSelectedKVCache(cache, layer, position_base=memory.source_tokens)
            for cache, layer in zip(caches, memory.layers)
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

        self.runner._acquire_cache = types.MethodType(acquire, self.runner)
        self.runner._release_cache = types.MethodType(release, self.runner)
        self.runner.prefill_start = types.MethodType(prefill, self.runner)
        self.runner._build_batched_decode_context = types.MethodType(
            build_context, self.runner
        )

    def close(self) -> None:
        self.runner._acquire_cache = self._original_acquire
        self.runner._release_cache = self._original_release
        self.runner.prefill_start = self._original_prefill_start
        self.runner._build_batched_decode_context = self._original_build_context
        self._requests.clear()
        self.isolation.close()

    def capabilities(self) -> Mapping[str, object]:
        return {
            "integration_level": self.integration_level,
            "native_kv": True,
            "radix_prefix_identity_separate": True,
            "hicache_external_namespace": False,
            "patched_layers": self.patched_layers,
        }
