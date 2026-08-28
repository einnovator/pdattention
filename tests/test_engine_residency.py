from __future__ import annotations

import time

import pytest

from pra_hf.engine_memory import (
    LogicalPRABlock,
    LogicalPRABlockId,
    LogicalPRABlockStore,
    PRAResidencyState,
)
from pra_hf.engine_residency import EnginePRAResidencyManager, PRAEvictionPolicy


def _register(store: LogicalPRABlockStore, resource: str, size: int = 32) -> str:
    identity = LogicalPRABlockId(
        tenant_id="tenant-a",
        session_id="session-a",
        resource_id=resource,
        resource_version="v1",
        record_type="document",
        token_start=0,
        token_end=8,
        layer=0,
        model_revision="model-v1",
        dtype="float16",
        layout="bhld",
        materialization_profile="reference",
        position_policy="source",
    )
    return store.register(LogicalPRABlock(identity, address_bytes=8, detail_bytes=size))


def test_prefetch_is_shared_and_pinned_until_request_cleanup() -> None:
    store = LogicalPRABlockStore()
    key = _register(store, "record-1")
    manager = EnginePRAResidencyManager(store, max_resident_bytes=64)
    calls = 0

    def materialize():
        nonlocal calls
        calls += 1
        return {"native": True}, 32

    first = manager.prefetch(key, materialize)
    assert first.result() == {"native": True}
    second = manager.prefetch(key, materialize)
    assert second.result() == {"native": True}
    assert manager.resolve(key, materialize, request_id="request-1") == {"native": True}
    assert calls == 1
    with manager.pin_request("request-1", (key,)):
        assert store.get(key).state == PRAResidencyState.PINNED
    assert store.get(key).state == PRAResidencyState.RESIDENT
    metrics = manager.metrics()
    assert metrics.loads == 1
    assert metrics.prefetches == 1
    assert metrics.prefetch_hits == 1
    assert metrics.duplicate_transfer_bytes_avoided == 32
    manager.close()


@pytest.mark.parametrize("policy", list(PRAEvictionPolicy))
def test_bounded_residency_evicts_and_reloads(policy: PRAEvictionPolicy) -> None:
    store = LogicalPRABlockStore()
    first = _register(store, "record-1")
    second = _register(store, "record-2")
    manager = EnginePRAResidencyManager(
        store, max_resident_bytes=32, policy=policy
    )
    manager.resolve(first, lambda: ("first", 32))
    manager.resolve(second, lambda: ("second", 32))
    assert manager.metrics().resident_bytes == 32
    assert manager.metrics().evictions == 1
    manager.resolve(first, lambda: ("first-reloaded", 32))
    metrics = manager.metrics()
    assert metrics.reloads == 1
    assert metrics.evictions == 2
    assert metrics.bytes_loaded == 96
    manager.close()


def test_pinned_payload_blocks_eviction() -> None:
    store = LogicalPRABlockStore()
    first = _register(store, "record-1")
    second = _register(store, "record-2")
    manager = EnginePRAResidencyManager(store, max_resident_bytes=32)
    manager.resolve(first, lambda: ("first", 32))
    with manager.pin_request("request-1", (first,)):
        with pytest.raises(MemoryError, match="Pinned"):
            manager.resolve(second, lambda: ("second", 32))
    manager.close()


def test_unused_prefetch_is_reported_as_waste() -> None:
    store = LogicalPRABlockStore()
    key = _register(store, "record-1")
    manager = EnginePRAResidencyManager(store, max_resident_bytes=64)
    manager.prefetch(key, lambda: ("payload", 32)).result()
    manager.close()
    assert manager.metrics().wasted_prefetches == 1
    assert any(row["action"] == "prefetch_wasted" for row in manager.events())


def test_waiting_for_prefetch_records_late_stall() -> None:
    store = LogicalPRABlockStore()
    key = _register(store, "record-1")
    manager = EnginePRAResidencyManager(store, max_resident_bytes=64)

    def slow():
        time.sleep(0.01)
        return "payload", 32

    manager.prefetch(key, slow)
    assert manager.resolve(key, slow) == "payload"
    assert manager.metrics().late_block_stall_ns > 0
    manager.close()


def test_payloads_are_disposed_on_eviction_invalidation_and_close() -> None:
    store = LogicalPRABlockStore()
    keys = tuple(_register(store, f"record-{index}") for index in range(3))
    disposed = []
    manager = EnginePRAResidencyManager(
        store,
        max_resident_bytes=64,
        payload_disposer=disposed.append,
    )
    payloads = tuple(object() for _ in keys)
    by_key = dict(zip(keys, payloads))

    for key, payload in zip(keys, payloads):
        manager.resolve(key, lambda payload=payload: (payload, 32))
    assert len(disposed) == 1
    assert disposed[0] in payloads

    resident_key = next(
        key for key in keys if store.get(key).state == PRAResidencyState.RESIDENT
    )
    manager.invalidate((resident_key,))
    assert by_key[resident_key] in disposed
    manager.close()
    assert set(disposed) == set(payloads)
