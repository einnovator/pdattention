from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from pra_hf.live_history import LiveKVSelectionPlan
from pra_mlx.mlx_live_kv import MLXLiveKVRequestCancelled, MLXLiveKVRuntime
from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory


class _FakeMX(ModuleType):
    int32 = np.int32
    float32 = np.float32

    @staticmethod
    def array(value, dtype=None):
        return np.array(value, dtype=dtype)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

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
    cache.make_prompt_cache = lambda model, max_kv_size=None: [
        _FakeLocalCache() for _ in model.layers
    ]
    models.cache = cache
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
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


def _begin(runtime: MLXLiveKVRuntime, request_id: str, *, generation: int = 1):
    return runtime.begin_request(
        request_id,
        "source",
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
        tenant_id="tenant",
        session_id="session",
        expected_generation=generation,
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
