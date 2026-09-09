"""Lifecycle and isolation contracts for direct-MLX subagent state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from experiments.paper9_subagents.run_mlx_lifecycle import resolve_model_snapshot
from pra_hf.subagent_mlx_native import (
    MLXHostSubagentMemory,
    MLXLayerKV,
    MLXStatePoolError,
    MLXSubagentMemory,
    MLXSubagentStatePool,
    _mlx_memory_to_device,
    _mlx_memory_to_host,
)


def _memory(value: float) -> MLXSubagentMemory:
    keys = np.full((1, 1, 4, 4), value, dtype=np.float32)
    values = np.full((1, 1, 4, 4), value + 1, dtype=np.float32)
    return MLXSubagentMemory((MLXLayerKV(keys, values),), token_count=4)


def _to_host(memory: MLXSubagentMemory) -> MLXHostSubagentMemory:
    return MLXHostSubagentMemory(
        tuple(MLXLayerKV(layer.keys.copy(), layer.values.copy()) for layer in memory.layers),
        memory.token_count,
    )


def _to_device(memory: MLXHostSubagentMemory) -> MLXSubagentMemory:
    return MLXSubagentMemory(
        tuple(MLXLayerKV(layer.keys.copy(), layer.values.copy()) for layer in memory.layers),
        memory.token_count,
    )


def _pool(*, capacity: int = 1024, max_records: int = 4) -> MLXSubagentStatePool:
    return MLXSubagentStatePool(
        device_capacity_bytes=capacity,
        max_records=max_records,
        export_to_host=_to_host,
        import_to_device=_to_device,
    )


def test_model_snapshot_resolves_the_requested_revision(tmp_path) -> None:
    snapshot = tmp_path / "snapshots" / "0123456789abcdef"
    snapshot.mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return snapshot

    path, resolved = resolve_model_snapshot(
        "mlx-community/Qwen3-0.6B-4bit",
        "01234567",
        snapshot_download=download,
    )

    assert calls == [{
        "repo_id": "mlx-community/Qwen3-0.6B-4bit",
        "revision": "01234567",
    }]
    assert path == snapshot.resolve()
    assert resolved == "0123456789abcdef"


def test_mlx_bfloat16_host_round_trip_preserves_exact_bits() -> None:
    mx = pytest.importorskip("mlx.core")
    keys = mx.array([[[[1.25, -2.5]]]], dtype=mx.bfloat16)
    values = mx.array([[[[3.75, -4.5]]]], dtype=mx.bfloat16)
    memory = MLXSubagentMemory((MLXLayerKV(keys, values),), token_count=1)

    host = _mlx_memory_to_host(memory)
    restored = _mlx_memory_to_device(host)

    assert host.layers[0].keys.dtype == np.dtype("uint16")
    assert host.layers[0].key_dtype == "bfloat16"
    assert restored.layers[0].keys.dtype == mx.bfloat16
    assert bool(mx.array_equal(restored.layers[0].keys, keys).item())
    assert bool(mx.array_equal(restored.layers[0].values, values).item())


def test_concurrent_sessions_are_isolated_even_for_equal_record_ids() -> None:
    pool = _pool()
    pool.put("session-a", "shared", _memory(1))
    pool.put("session-b", "shared", _memory(9))

    def acquire(session: str, agent: str) -> tuple[str, float]:
        lease, memory = pool.acquire(session, "shared", agent)
        value = float(memory.layers[0].keys[0, 0, 0, 0])
        pool.release(lease.lease_uuid)
        return session, value

    with ThreadPoolExecutor(max_workers=2) as executor:
        values = dict(executor.map(lambda row: acquire(*row), (("session-a", "a"), ("session-b", "b"))))

    assert values == {"session-a": 1.0, "session-b": 9.0}
    with pytest.raises(MLXStatePoolError, match="requested session"):
        pool.acquire("session-c", "shared", "intruder")


def test_cancellation_releases_lease_and_session_termination_is_scoped() -> None:
    pool = _pool()
    pool.put("session-a", "record", _memory(1))
    pool.put("session-b", "record", _memory(2))
    pool.acquire("session-a", "record", "child")

    assert pool.cancel_target("session-a", "child") is True
    assert pool.cancel_target("session-a", "child") is False
    assert pool.terminate_session("session-a") == 1
    snapshot = pool.snapshot()
    assert {(row["session_uuid"], row["record_uuid"]) for row in snapshot["records"]} == {
        ("session-b", "record")
    }
    assert snapshot["active_leases"] == 0


def test_active_state_is_protected_while_inactive_state_demotes_and_evicts() -> None:
    one_record_bytes = _memory(1).nbytes
    pool = _pool(capacity=one_record_bytes, max_records=2)
    pool.put("session", "active", _memory(1))
    lease, _ = pool.acquire("session", "active", "child")

    with pytest.raises(MLXStatePoolError, match="active native records"):
        pool.put("session", "blocked", _memory(2))

    pool.release(lease.lease_uuid)
    pool.put("session", "second", _memory(2))
    snapshot = pool.snapshot()
    assert {row["tier"] for row in snapshot["records"]} == {"device", "host"}

    lease, restored = pool.acquire("session", "active", "child-2")
    assert float(restored.layers[0].keys[0, 0, 0, 0]) == 1.0
    pool.release(lease.lease_uuid)
    pool.put("session", "third", _memory(3))
    snapshot = pool.snapshot()
    assert len(snapshot["records"]) == 2
    assert any(event["action"] == "demote" for event in snapshot["events"])
    assert any(event["action"] == "promote" for event in snapshot["events"])
    assert any(event.get("reason") == "record_capacity" for event in snapshot["events"])
