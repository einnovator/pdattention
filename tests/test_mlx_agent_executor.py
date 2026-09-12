from __future__ import annotations

import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest

from pra_hf.deployment import PRAWireRequest
from pra_mlx.agent_executor import MLXAgentHistoryExecutor


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

    @staticmethod
    def max(value):
        return np.max(value)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def get_active_memory():
        return 0

    @staticmethod
    def get_peak_memory():
        return 0

    @staticmethod
    def reset_peak_memory():
        return None


class _FakeCache:
    def __init__(self) -> None:
        self.keys = np.zeros((1, 1, 0, 2), dtype=np.float32)
        self.values = np.zeros((1, 1, 0, 2), dtype=np.float32)
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values

    @property
    def nbytes(self):
        return self.keys.nbytes + self.values.nbytes

    def append(self, token_ids) -> None:
        width = len(token_ids)
        start = self.offset
        values = np.arange(start, start + width, dtype=np.float32).reshape(1, 1, width, 1)
        values = np.repeat(values, 2, axis=3)
        self.keys = np.concatenate((self.keys, values), axis=2)
        self.values = np.concatenate((self.values, values + 100), axis=2)
        self.offset += width


class _Tokenizer:
    eos_token_id = 0
    chat_template = "fake-append-stable-template"

    @staticmethod
    def _content(value: str):
        if value == "A":
            return [3]
        return [30 + (ord(char) % 50) for char in value]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_):
        result = []
        for row in messages:
            role = row["role"]
            if role == "assistant":
                result.extend([90, *self._content(row.get("content", "")), 0])
            else:
                result.extend([
                    10 if role == "system" else 20,
                    *self._content(row.get("content", "")),
                ])
        if add_generation_prompt:
            result.append(90)
        return result

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join("A" if int(token) == 3 else "?" for token in token_ids)


class _Model:
    def __init__(self, *, sparse_delta: float = 0.0) -> None:
        self.layers = (object(),)
        self.sparse_delta = float(sparse_delta)
        self.calls = 0

    def __call__(self, input_ids, *, cache):
        self.calls += 1
        values = list(map(int, input_ids[0]))
        for wrapped in cache:
            getattr(wrapped, "local_cache", wrapped).append(values)
        logits = np.zeros((1, len(values), 8), dtype=np.float32)
        logits[..., 3 if values[-1] != 3 else 0] = 1.0
        if type(cache[0]).__name__ == "MLXDisjointSelectedKVCache":
            logits[..., 7] += self.sparse_delta
        return logits


@pytest.fixture()
def fake_mlx(monkeypatch):
    mlx = ModuleType("mlx")
    core = _FakeMX("mlx.core")
    mlx.core = core
    mlx_lm = ModuleType("mlx_lm")
    models = ModuleType("mlx_lm.models")
    cache = ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda model, max_kv_size=None: [
        _FakeCache() for _ in model.layers
    ]
    models.cache = cache
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
    monkeypatch.setattr(
        "pra_mlx.agent_executor.install_qwen3_segmented_attention",
        lambda model, compiled=False: 1,
    )
    return core


def _manifest(messages):
    return [{
        "message_index": index,
        "role": row["role"],
        "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
    } for index, row in enumerate(messages)]


def _executor(model=None):
    return MLXAgentHistoryExecutor(
        model or _Model(),
        _Tokenizer(),
        model_id="fake",
        model_revision="pinned",
        wire_tail_tokens=1,
        max_abs_logit_delta=0.005,
    )


def _request(messages, *, request_messages=None, retention=1.0, request_id="r1"):
    rows = tuple(messages)
    sent = tuple(request_messages or rows)
    mandatory = [rows.index(row) for row in sent]
    return PRAWireRequest(
        model="fake",
        messages=sent,
        request_id=request_id,
        tenant_id="tenant",
        session_id="session",
        max_new_tokens=2,
        metadata={
            "history_projection": "live-agent-kv-v1",
            "logical_message_manifest": _manifest(rows),
            "mandatory_message_indices": mandatory,
            "target_retention_fraction": retention,
        },
    )


def test_plain_request_does_not_enter_resident_pra(fake_mlx) -> None:
    executor = _executor()
    result = executor.generate(PRAWireRequest(
        model="fake",
        messages=({"role": "user", "content": "plain"},),
        max_new_tokens=1,
    ))
    assert result.text == "A"
    assert result.trace[0]["native_kv_used"] is False
    assert result.raw["usage"]["prompt_tokens"] == result.trace[0]["prompt_tokens"]
    assert result.raw["usage"]["completion_tokens"] == 1
    assert result.raw["usage"]["total_tokens"] == (
        result.raw["usage"]["prompt_tokens"] + 1
    )
    assert not executor._sessions


def test_pra100_then_pra90_reuses_history_and_separates_copy_metrics(fake_mlx) -> None:
    executor = _executor()
    initial = (
        {"role": "system", "content": "S" * 40},
        {"role": "user", "content": "U" * 40},
    )
    first = executor.generate(_request(initial))
    assert first.text == "A"
    first_trace = first.trace[0]
    assert first_trace["pra_100_semantic_noop"] is True
    assert first_trace["exact_trajectory_eligible"] is True
    assert first_trace["selected_history_reencoded_tokens"] == 0
    assert first_trace["selected_history_kv_copy_bytes"] is None
    assert first_trace["canonical_suffix_graft_d2d_bytes"] > 0

    logical = (*initial, {"role": "assistant", "content": "A"}, {
        "role": "user", "content": "O" * 40,
    })
    second = executor.generate(_request(
        logical,
        request_messages=(logical[0], logical[1], logical[3]),
        retention=0.9,
        request_id="r2",
    ))
    trace = second.trace[0]
    assert second.text == "A"
    assert trace["pra_100_semantic_noop"] is False
    assert 0.9 <= trace["realized_historical_kv_retention_fraction"] < 1
    assert trace["selected_history_reencoded_tokens"] == 0
    assert trace["selected_interval_pack_bytes"] == 0
    assert trace["physical_kv_copy"] is None
    assert trace["same_subset_gate_passed"] is True
    assert trace["same_subset_reference_pack_bytes"] > 0
    assert trace["canonical_suffix_graft_d2d_bytes"] > 0
    assert trace["known_total_kv_copy_bytes"] > 0
    assert trace["total_kv_copy_bytes"] is None
    assert executor.runtime.snapshot()["active_request_ids"] == ()

    source_id = executor._sessions["session"].source_id
    assert executor.runtime.registry.view(source_id) is not None
    executor.close_session("session")
    assert executor.runtime.registry.view(source_id) is None


def test_sparse_same_subset_mismatch_fails_closed_and_releases_borrows(fake_mlx) -> None:
    executor = _executor(_Model(sparse_delta=0.01))
    initial = (
        {"role": "system", "content": "S" * 40},
        {"role": "user", "content": "U" * 40},
    )
    executor.generate(_request(initial))
    logical = (*initial, {"role": "assistant", "content": "A"}, {
        "role": "user", "content": "O" * 40,
    })
    with pytest.raises(RuntimeError, match="same-subset correctness gate failed"):
        executor.generate(_request(
            logical,
            request_messages=(logical[0], logical[1], logical[3]),
            retention=0.9,
            request_id="mismatch",
        ))
    assert executor.runtime.snapshot()["active_request_ids"] == ()


def test_append_rewrite_and_tenant_crossing_fail_closed(fake_mlx) -> None:
    executor = _executor()
    initial = (
        {"role": "system", "content": "S" * 40},
        {"role": "user", "content": "U" * 40},
    )
    executor.generate(_request(initial))
    with pytest.raises(RuntimeError, match="append-stable template"):
        executor.generate(_request((
            {"role": "system", "content": "changed"},
            initial[1],
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "next"},
        ), request_id="rewrite"))
    with pytest.raises(RuntimeError, match="tenant scope"):
        executor.generate(PRAWireRequest(
            model="fake",
            messages=initial,
            tenant_id="other",
            session_id="session",
            metadata={"history_projection": "live-agent-kv-v1"},
        ))
