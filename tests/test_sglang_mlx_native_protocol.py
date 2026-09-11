from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_hf.live_history import LiveKVSelectionPlan
from pra_mlx.native import (
    MLXDisjointLayerKV,
    MLXNativeLayerKV,
    MLXNativeMemory,
)
import pra_sglang.mlx_native as sglang_native
from pra_sglang.mlx_native import (
    SGLangMLXLiveKVRuntime,
    SGLangMLXNativeBridge,
    SGLangNativeRequest,
    SGLangSelectedKVCache,
)


@dataclass(frozen=True)
class _Array:
    shape: tuple[int, ...]


class _LocalCache:
    def __init__(self) -> None:
        self.offset = 7
        self.keys = object()
        self.values = object()
        self.state = ()

    def reset(self) -> None:
        self.offset = 0


@dataclass(frozen=True)
class _Layout:
    attention_layer_indices: tuple[int, ...] = (0,)
    first_attention_layer_index: int = 0
    num_layers: int = 1
    has_auxiliary_state: bool = False
    has_sliding_window_layers: bool = False


@dataclass(frozen=True)
class _Runner:
    _cache_layout: _Layout = _Layout()


class _LifecycleRunner:
    def __init__(self) -> None:
        self._cache_layout = _Layout()
        self.model = object()
        self._req_caches: dict[str, list[object]] = {"source-owner": [_LocalCache()]}
        self.pool_releases: list[list[object]] = []
        self.fail_remove = False

    def _acquire_cache(self):
        cache = _LocalCache()
        cache.offset = 0
        return [cache]

    def _release_cache(self, caches):
        self.pool_releases.append(caches)

    def prefill_start(self, req_id, *_args, **_kwargs):
        self._req_caches[str(req_id)] = self._acquire_cache()
        return object()

    def _cache_with_pool_backed_attention(self, *_args):
        return self._acquire_cache()

    @staticmethod
    def _materialize_pool_backed_attention(caches):
        return caches

    @staticmethod
    def _build_batched_decode_context(_caches, _req_ids):
        return types.SimpleNamespace()

    def remove_request(self, req_id):
        caches = self._req_caches.pop(str(req_id))
        self._release_cache(caches)
        if self.fail_remove:
            raise RuntimeError("native cleanup failed")


def _bridge(monkeypatch) -> tuple[_LifecycleRunner, SGLangMLXNativeBridge]:
    monkeypatch.setattr(sglang_native, "install_selected_kv_attention", lambda _model: 1)
    runner = _LifecycleRunner()
    return runner, SGLangMLXNativeBridge(runner)


def _memory(tokens: int = 6) -> MLXNativeMemory:
    keys = np.arange(tokens, dtype=np.float32).reshape(1, 1, tokens, 1)
    return MLXNativeMemory((MLXNativeLayerKV(keys, keys + 100),), tokens)


def _fake_mlx(monkeypatch) -> None:
    core = types.ModuleType("mlx.core")
    core.concatenate = np.concatenate
    core.eval = lambda *_values: None
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_sglang_cache_keeps_scheduler_and_rope_offsets_separate() -> None:
    local = _LocalCache()
    memory = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    cache = SGLangSelectedKVCache(local, memory, position_base=31)

    assert cache.offset == 7
    assert cache.rope_offset == 38
    assert cache.memory_tokens == 11
    assert cache.attention_view.offset == 38
    assert cache.keys is local.keys


def test_sglang_cache_reset_preserves_immutable_memory() -> None:
    local = _LocalCache()
    memory = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    cache = SGLangSelectedKVCache(local, memory, position_base=31)

    cache.reset()

    assert cache.offset == 0
    assert cache.rope_offset == 31
    assert cache.memory is memory


def test_sglang_pool_release_strips_selected_memory_wrapper() -> None:
    local = _LocalCache()
    memory = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    wrapped = SGLangSelectedKVCache(local, memory, position_base=31)

    assert SGLangMLXNativeBridge._unwrap_cache([wrapped]) == [local]


def test_sglang_radix_transition_wraps_twice_but_attaches_once() -> None:
    local = _LocalCache()
    layer = MLXNativeLayerKV(_Array((1, 2, 11, 8)), _Array((1, 2, 11, 8)))
    bridge = object.__new__(SGLangMLXNativeBridge)
    bridge.runner = _Runner()
    bridge._requests = {
        "request": SGLangNativeRequest(
            MLXNativeMemory((layer,), source_tokens=11),
            source_position_base=31,
            logical_keys=("resource-R",),
        )
    }
    bridge.isolation = EnginePRAIsolationGuard()
    bridge.isolation.open_request("request", ("resource-R",))

    pool_backed = bridge._wrap_cache("request", [local])
    same_stage = bridge._wrap_cache("request", pool_backed)
    contiguous = bridge._wrap_cache("request", [_LocalCache()])

    assert same_stage is pool_backed
    assert isinstance(contiguous[0], SGLangSelectedKVCache)
    assert contiguous[0].position_base == 31
    assert contiguous[0].rope_offset == 31 + contiguous[0].offset
    assert bridge.isolation.view("request").attached


def test_sglang_live_request_pins_owner_and_releases_on_native_finish(
    monkeypatch,
) -> None:
    _fake_mlx(monkeypatch)
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: source,
        load=lambda source: source,
    )
    runtime.register_source(
        "history",
        _memory(),
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=3,
    )
    plan = LiveKVSelectionPlan.create(
        6, ((0, 2), (4, 6)), source_position_base=11
    )

    request = runtime.begin_request(
        "candidate",
        "history",
        plan,
        tenant_id="tenant",
        session_id="session",
        expected_generation=3,
    )

    # The source is borrowed and its SGLang owner is pinned before prefill can
    # construct the selected request cache.
    assert bridge.isolation.view("candidate").attached is False
    assert runtime.registry.view("history").active_request_ids == ("candidate",)
    with pytest.raises(RuntimeError, match="source owner"):
        runner.remove_request("source-owner")

    runner.prefill_start("candidate", [1], [1], [], [], 0)
    selected = runner._req_caches["candidate"][0]
    assert isinstance(selected, SGLangSelectedKVCache)
    assert selected.position_base == 11
    assert selected.memory_tokens == 4
    assert request.selection.physical_kv_copy is True
    assert request.selection.selected_text_reencoded_tokens == 0

    assert request.finish()
    assert request.outcome == "finished"
    assert request.finish() is False
    assert runtime.registry.view("history").active_request_ids == ()
    assert bridge.isolation.view("candidate") is None
    assert runner.pool_releases[-1][0].__class__ is _LocalCache
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 1,
        "cancelled": 0,
        "error": 0,
    }
    runner.remove_request("source-owner")


def test_sglang_disjoint_live_request_keeps_source_intervals_unpacked(
    monkeypatch,
) -> None:
    _fake_mlx(monkeypatch)
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: source,
        load=lambda source: source,
    )
    runtime.register_source(
        "history",
        _memory(),
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=3,
    )
    plan = LiveKVSelectionPlan.create(
        6, ((0, 2), (4, 6)), source_position_base=11
    )

    request = runtime.begin_request(
        "candidate",
        "history",
        plan,
        tenant_id="tenant",
        session_id="session",
        expected_generation=3,
        disjoint=True,
    )
    assert request.selection.physical_kv_copy is None
    assert request.selection.selected_text_reencoded_tokens == 0
    assert isinstance(request.selection.memory.layers[0], MLXDisjointLayerKV)
    assert [segment.keys.shape[2] for segment in request.selection.memory.layers[0].segments] == [2, 2]

    runner.prefill_start("candidate", [1], [1], [], [], 0)
    selected = runner._req_caches["candidate"][0]
    assert isinstance(selected, SGLangSelectedKVCache)
    assert selected.disjoint
    assert selected.memory_tokens == 4
    assert selected.position_base == 11
    with pytest.raises(RuntimeError, match="interval-addressed attention"):
        selected.get_kv()
    assert request.cancel()


def test_sglang_borrows_and_pins_before_selected_cache_construction(
    monkeypatch,
) -> None:
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: source,
        load=lambda source: source,
    )
    runtime.register_source(
        "history",
        _memory(),
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=1,
    )
    select = sglang_native._select_live_source_memory
    observed = {}

    def inspect_order(source, plan):
        observed["borrowers"] = runtime.registry.view(
            "history"
        ).active_request_ids
        observed["owner_pins"] = tuple(
            sorted(bridge._source_owner_pins["source-owner"])
        )
        return select(source, plan)

    monkeypatch.setattr(sglang_native, "_select_live_source_memory", inspect_order)
    request = runtime.begin_request(
        "candidate",
        "history",
        LiveKVSelectionPlan.create(6, ((0, 6),)),
        tenant_id="tenant",
        session_id="session",
        expected_generation=1,
    )

    assert observed == {
        "borrowers": ("candidate",),
        "owner_pins": ("candidate",),
    }
    assert request.cancel()


def test_sglang_captures_full_live_source_once_and_never_reencodes_selected_text(
    monkeypatch,
) -> None:
    _fake_mlx(monkeypatch)
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: source,
        load=lambda source: source,
    )

    class SourceCache:
        offset = 6
        state_reads = 0

        @property
        def state(self):
            self.state_reads += 1
            memory = _memory()
            return (memory.layers[0].keys, memory.layers[0].values)

    source_cache = SourceCache()
    runner._req_caches["source-owner"] = [source_cache]
    runtime.register_source(
        "history",
        [source_cache],
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=1,
    )
    assert source_cache.state_reads == 1

    for request_id in ("first", "second"):
        request = runtime.begin_request(
            request_id,
            "history",
            LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
            tenant_id="tenant",
            session_id="session",
            expected_generation=1,
        )
        assert request.selection.selected_text_reencoded_tokens == 0
        assert request.cancel()
    assert source_cache.state_reads == 1


def test_sglang_live_request_cancel_and_error_release_exactly_once(monkeypatch) -> None:
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: source,
        load=lambda source: source,
    )
    runtime.register_source(
        "history",
        _memory(),
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=1,
    )
    plan = LiveKVSelectionPlan.create(6, ((1, 6),))

    queued = runtime.begin_request(
        "queued",
        "history",
        plan,
        tenant_id="tenant",
        session_id="session",
        expected_generation=1,
    )
    assert queued.cancel()
    assert queued.outcome == "cancelled"
    assert queued.cancel() is False

    failed = runtime.begin_request(
        "failed",
        "history",
        plan,
        tenant_id="tenant",
        session_id="session",
        expected_generation=1,
    )
    runner.prefill_start("failed", [1], [1], [], [], 0)
    runner.fail_remove = True
    with pytest.raises(RuntimeError, match="native cleanup failed"):
        runner.remove_request("failed")
    assert failed.outcome == "error"
    assert failed.fail() is False
    assert runtime.registry.view("history").active_request_ids == ()
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 0,
        "cancelled": 1,
        "error": 1,
    }


def test_sglang_live_registry_rejects_stale_and_borrowed_eviction_and_tombstones(
    monkeypatch,
) -> None:
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: ("offloaded", source),
        load=lambda payload: payload[1],
    )
    source = _memory()
    runtime.register_source(
        "history",
        source,
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=4,
    )
    plan = LiveKVSelectionPlan.create(6, ((0, 6),))

    with pytest.raises(RuntimeError, match="Stale"):
        runtime.begin_request(
            "stale",
            "history",
            plan,
            tenant_id="tenant",
            session_id="session",
            expected_generation=3,
        )

    active = runtime.begin_request(
        "active",
        "history",
        plan,
        tenant_id="tenant",
        session_id="session",
        expected_generation=4,
    )
    with pytest.raises(RuntimeError, match="while requests borrow"):
        runtime.offload_source("history")
    with pytest.raises(RuntimeError, match="while requests borrow"):
        runtime.evict_source("history")

    assert runtime.terminate_session("tenant", "session") == 1
    assert active.outcome == "cancelled"
    assert runtime.registry.view("history") is None
    assert runtime.snapshot()["active_request_ids"] == ()
    with pytest.raises(RuntimeError, match="terminated"):
        runtime.register_source(
            "replacement",
            source,
            owner_request_id="source-owner",
            tenant_id="tenant",
            session_id="session",
            generation=5,
        )


def test_sglang_radix_pool_never_receives_selected_memory_for_two_borrowers(
    monkeypatch,
) -> None:
    runner, bridge = _bridge(monkeypatch)
    runtime = SGLangMLXLiveKVRuntime(
        bridge,
        dump=lambda source: source,
        load=lambda source: source,
    )
    runtime.register_source(
        "history",
        _memory(),
        owner_request_id="source-owner",
        tenant_id="tenant",
        session_id="session",
        generation=1,
    )
    plan = LiveKVSelectionPlan.create(6, ((1, 6),))
    requests = [
        runtime.begin_request(
            request_id,
            "history",
            plan,
            tenant_id="tenant",
            session_id="session",
            expected_generation=1,
        )
        for request_id in ("request-a", "request-b")
    ]
    for request in requests:
        runner.prefill_start(request.request_id, [1], [1], [], [], 0)

    assert runtime.registry.view("history").active_request_ids == (
        "request-a",
        "request-b",
    )
    assert requests[0].cancel()
    assert runtime.registry.view("history").active_request_ids == ("request-b",)
    with pytest.raises(RuntimeError, match="source owner"):
        runner.remove_request("source-owner")
    assert requests[1].finish()
    assert all(
        not isinstance(cache, SGLangSelectedKVCache)
        for released in runner.pool_releases
        for cache in released
    )
