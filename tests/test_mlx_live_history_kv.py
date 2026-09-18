from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from pra_hf.live_history import LiveKVSelectionPlan
from pra_mlx.mlx_live_kv import MLXLiveKVRequestCancelled, MLXLiveKVRuntime
from pra_mlx.native import (
    MLXDisjointLayerKV,
    MLXDisjointSelectedKVCache,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXSelectedKVCache,
)


class _FakeMX(ModuleType):
    int32 = np.int32
    float32 = np.float32
    bool_ = np.bool_

    @staticmethod
    def array(value, dtype=None):
        return np.array(value, dtype=dtype)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def arange(*args):
        return np.arange(*args)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis)

    @staticmethod
    def ones(shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def eval(*_values):
        return None

    @staticmethod
    def argmax(value):
        return np.argmax(value)


class _FakeLocalCache:
    def __init__(self) -> None:
        self.offset = 0

    @property
    def state(self):
        empty = np.zeros((1, 1, 0, 2), dtype=np.float32)
        return empty, empty

    @property
    def nbytes(self) -> int:
        return 0

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        removed = min(self.offset, int(n))
        self.offset -= removed
        return removed


@pytest.fixture()
def fake_mlx(monkeypatch):
    mlx = ModuleType("mlx")
    core = _FakeMX("mlx.core")
    mlx.core = core
    mlx_lm = ModuleType("mlx_lm")
    models = ModuleType("mlx_lm.models")
    cache = ModuleType("mlx_lm.models.cache")
    base = ModuleType("mlx_lm.models.base")
    cache.make_prompt_cache = lambda model, max_kv_size=None: [
        _FakeLocalCache() for _ in model.layers
    ]
    base.create_causal_mask = lambda n, offset, **_kwargs: np.arange(
        offset + n
    )[None, :] <= np.arange(offset, offset + n)[:, None]
    models.cache = cache
    models.base = base
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", base)
    return core


class _DeterministicMLXModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.layers = (object(),)
        self.fail = fail
        self.position_offsets: list[int] = []

    def __call__(self, input_ids, *, cache):
        if self.fail:
            raise RuntimeError("model failure")
        self.position_offsets.append(int(cache[0].offset))
        width = int(input_ids.shape[1])
        for layer in cache:
            layer.local_cache.offset += width
        logits = np.zeros((1, width, 8), dtype=np.float32)
        logits[..., 3] = 1.0
        return logits


def _memory() -> MLXNativeMemory:
    keys = np.arange(12, dtype=np.float32).reshape(1, 1, 6, 2)
    return MLXNativeMemory((MLXNativeLayerKV(keys, keys + 100),), 6)


def _runtime() -> MLXLiveKVRuntime:
    return MLXLiveKVRuntime(dump=lambda value: value, load=lambda value: value)


def _begin(
    runtime: MLXLiveKVRuntime,
    request_id: str,
    *,
    generation: int = 1,
    segmented: bool = False,
):
    return runtime.begin_request(
        request_id,
        "source",
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
        tenant_id="tenant",
        session_id="session",
        expected_generation=generation,
        segmented=segmented,
    )


def test_mlx_request_owns_borrow_through_decode_and_releases_once(fake_mlx) -> None:
    runtime = _runtime()
    runtime.register_source(
        "source", _memory(), tenant_id="tenant", session_id="session", generation=1
    )
    candidate = _begin(runtime, "candidate")
    reference = _begin(runtime, "reference")
    assert runtime.registry.view("source").active_request_ids == (
        "candidate", "reference"
    )
    with pytest.raises(RuntimeError, match="while requests borrow"):
        runtime.offload_source("source")

    model = _DeterministicMLXModel()
    result = candidate.generate(model, [5, 6], max_new_tokens=2)
    assert result.token_ids == (3, 3)
    assert result.source_position_base == 6
    assert result.selected_kv_tokens == 4
    assert result.selected_text_reencoded_tokens == 0
    assert result.physical_kv_copy is True
    assert result.selected_kv_segments == 2
    assert result.selection_pack_bytes == candidate.selection.memory.nbytes
    assert result.materialization_policy == "dense_pack"
    assert model.position_offsets == [6, 8]
    assert candidate.outcome == "finished"
    assert candidate.finish() is False
    assert runtime.registry.view("source").active_request_ids == ("reference",)

    reference.generate(_DeterministicMLXModel(), [5], max_new_tokens=1)
    assert runtime.registry.view("source").active_request_ids == ()
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 2,
        "cancelled": 0,
        "error": 0,
    }


def test_mlx_cancel_error_stale_restore_and_termination_lifecycle(fake_mlx) -> None:
    runtime = _runtime()
    memory = _memory()
    runtime.register_source(
        "source", memory, tenant_id="tenant", session_id="session", generation=1
    )

    cancelled = _begin(runtime, "cancelled")
    checks = iter((False, True))
    with pytest.raises(MLXLiveKVRequestCancelled):
        cancelled.generate(
            _DeterministicMLXModel(),
            [5],
            max_new_tokens=2,
            cancelled=lambda: next(checks),
        )
    assert cancelled.outcome == "cancelled"
    assert cancelled.cancel() is False

    failed = _begin(runtime, "failed")
    with pytest.raises(RuntimeError, match="model failure"):
        failed.generate(_DeterministicMLXModel(fail=True), [5], max_new_tokens=1)
    assert failed.outcome == "error"
    assert failed.fail() is False

    with pytest.raises(RuntimeError, match="Stale"):
        _begin(runtime, "stale", generation=0)
    assert runtime.registry.view("source").active_request_ids == ()

    assert runtime.offload_source("source") is memory
    assert runtime.registry.view("source").tier == "offloaded"
    restored = _begin(runtime, "restored")
    assert runtime.registry.view("source").tier == "hot"
    restored.generate(_DeterministicMLXModel(), [5], max_new_tokens=1)

    active = _begin(runtime, "active-at-termination")
    assert runtime.terminate_session("tenant", "session") == 1
    assert active.outcome == "cancelled"
    assert active.cancel() is False
    assert runtime.registry.view("source") is None
    with pytest.raises(RuntimeError, match="terminated"):
        runtime.register_source(
            "replacement",
            memory,
            tenant_id="tenant",
            session_id="session",
            generation=2,
        )
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 1,
        "cancelled": 2,
        "error": 1,
    }


def test_mlx_selection_failure_releases_source_borrow(fake_mlx) -> None:
    runtime = _runtime()
    runtime.register_source(
        "source", _memory(), tenant_id="tenant", session_id="session", generation=1
    )
    with pytest.raises(ValueError, match="does not match"):
        runtime.begin_request(
            "invalid",
            "source",
            LiveKVSelectionPlan.create(5, ((0, 5),)),
            tenant_id="tenant",
            session_id="session",
            expected_generation=1,
        )
    assert runtime.registry.view("source").active_request_ids == ()
    assert runtime.snapshot()["active_request_ids"] == ()


def test_mlx_disjoint_request_keeps_source_views_without_pack_copy(fake_mlx) -> None:
    runtime = _runtime()
    source = _memory()
    runtime.register_source(
        "source", source, tenant_id="tenant", session_id="session", generation=1
    )

    request = _begin(runtime, "segmented", segmented=True)
    assert request.segmented is True
    assert request.selection.physical_kv_copy is None
    assert request.selection.plan.selected_tokens == 4
    selected_layer = request.selection.memory.layers[0]
    assert len(selected_layer.segments) == 2
    assert np.shares_memory(selected_layer.segments[0].keys, source.layers[0].keys)
    assert np.shares_memory(selected_layer.segments[1].values, source.layers[0].values)

    result = request.generate(_DeterministicMLXModel(), [5], max_new_tokens=1)
    assert result.token_ids == (3,)
    assert result.selected_text_reencoded_tokens == 0
    assert result.physical_kv_copy is None
    assert result.selected_kv_segments == 2
    assert result.selection_pack_bytes == 0
    assert result.materialization_policy == "disjoint_segmented"
    assert request.outcome == "finished"


def test_mlx_disjoint_request_materializes_original_position_history(fake_mlx) -> None:
    runtime = _runtime()
    runtime.register_source(
        "source", _memory(), tenant_id="tenant", session_id="session", generation=1
    )
    request = _begin(runtime, "materialized", segmented=True)
    model = _DeterministicMLXModel()

    result = request.generate(
        model,
        [5],
        max_new_tokens=1,
        materialized_history=(([9, 10], 2),),
    )

    assert model.position_offsets == [2, 3, 6]
    assert result.selected_kv_tokens == 4
    assert result.selected_text_reencoded_tokens == 2
    assert result.materialization_policy == "disjoint_segmented"


def test_mlx_disjoint_mask_hides_selected_future_from_old_receipt(fake_mlx) -> None:
    keys = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    memory = MLXDisjointLayerKV(
        (
            MLXNativeLayerKV(keys[:, :, :2], keys[:, :, :2] + 10),
            MLXNativeLayerKV(keys[:, :, 2:], keys[:, :, 2:] + 10),
        ),
        intervals=((0, 2), (4, 6)),
    )
    cache = MLXDisjointSelectedKVCache(
        _FakeLocalCache(), memory, position_base=2
    )

    old_receipt_mask = cache.make_mask(1)
    assert old_receipt_mask.tolist() == [[True, True, False, False, True]]

    cache.position_base = 6
    active_tail_mask = cache.make_mask(1)
    assert active_tail_mask.tolist() == [[True, True, True, True, True]]


def test_mlx_dense_selected_cache_uses_native_mask_dispatch(fake_mlx) -> None:
    keys = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    cache = MLXSelectedKVCache(
        _FakeLocalCache(),
        MLXNativeLayerKV(keys, keys + 10),
        position_base=4,
    )

    assert cache.make_mask(1) is None
    assert cache.make_mask(3) == "causal"
    explicit = cache.make_mask(3, return_array=True)
    assert explicit.shape == (3, 7)


def test_mlx_disjoint_oracle_separates_physical_and_logical_intervals(
    fake_mlx,
) -> None:
    keys = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    memory = MLXDisjointLayerKV(
        (
            MLXNativeLayerKV(keys[:, :, :2], keys[:, :, :2] + 10),
            MLXNativeLayerKV(keys[:, :, 2:], keys[:, :, 2:] + 10),
        ),
        source_keys=keys,
        source_values=keys + 10,
        intervals=((0, 2), (2, 4)),
        logical_intervals=((0, 2), (8, 10)),
    )
    cache = MLXDisjointSelectedKVCache(
        _FakeLocalCache(), memory, position_base=4
    )

    mask = cache.make_mask(1)
    assert mask.tolist() == [[True, True, False, False, True]]
