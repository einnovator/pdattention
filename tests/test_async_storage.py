from __future__ import annotations

import asyncio
import time

from pra_hf.async_storage import (
    PRAAsyncPromotionScheduler,
    PRAHotAdmissionCandidate,
)
from pra_hf.storage_lifecycle import (
    PRAStorageEntry,
    PRAStorageManager,
    PRAStoragePolicy,
    PRAStorageTierConfig,
    PRARetentionClass,
)


def _manager(tmp_path) -> PRAStorageManager:
    policy = PRAStoragePolicy(
        hot=PRAStorageTierConfig(max_bytes=16),
        warm=PRAStorageTierConfig(path=str(tmp_path / "warm"), max_bytes=128),
        cold=PRAStorageTierConfig(enabled=False),
    )
    return PRAStorageManager(policy)


def _entry(key: str, size: int = 8) -> PRAStorageEntry:
    return PRAStorageEntry(
        logical_key=key,
        record_type="document",
        retention_class=PRARetentionClass.RECONSTRUCTABLE,
        tenant_id="tenant-a",
        session_id="session-a",
        task_id=None,
        task_status=None,
        resource_version="v1",
        detail_bytes=size,
    )


def test_event_loop_prefetch_coalesces_and_reports_ready_demand(tmp_path) -> None:
    manager = _manager(tmp_path)
    manager.register(_entry("resource"), b"native-kv")
    manager.demote_hot("resource")

    async def scenario() -> None:
        scheduler = PRAAsyncPromotionScheduler(manager)
        first = scheduler.prefetch("resource", tenant_id="tenant-a")
        second = scheduler.prefetch("resource", tenant_id="tenant-a")
        assert first is second
        await first
        assert await scheduler.resolve("resource", tenant_id="tenant-a") == b"native-kv"
        await scheduler.close()
        metrics = scheduler.metrics()
        assert metrics.scheduled == 1
        assert metrics.coalesced == 1
        assert metrics.ready_at_demand == 1
        assert metrics.late_demands == 0
        assert metrics.bytes_promoted == len(b"native-kv")
        assert metrics.wasted_prefetches == 0

    asyncio.run(scenario())


def test_late_demand_records_only_remaining_promotion_stall(tmp_path) -> None:
    manager = _manager(tmp_path)
    manager.register(_entry("slow"), b"native-kv")
    manager.demote_hot("slow")
    original = manager.warm.get

    def delayed(*args, **kwargs):
        time.sleep(0.03)
        return original(*args, **kwargs)

    manager.warm.get = delayed

    async def scenario() -> None:
        scheduler = PRAAsyncPromotionScheduler(manager)
        scheduler.prefetch("slow")
        await asyncio.sleep(0.005)
        assert await scheduler.resolve("slow") == b"native-kv"
        await scheduler.close()
        metrics = scheduler.metrics()
        assert metrics.late_demands == 1
        assert metrics.demand_stall_ns >= 10_000_000

    asyncio.run(scenario())


def test_hot_set_admission_prefers_reuse_within_byte_budget(tmp_path) -> None:
    manager = _manager(tmp_path)
    for key in ("high", "medium", "low"):
        manager.register(_entry(key), key.encode().ljust(8, b"-"))
        manager.demote_hot(key)

    async def scenario() -> None:
        scheduler = PRAAsyncPromotionScheduler(manager)
        decisions = scheduler.admit_hot_set(
            (
                PRAHotAdmissionCandidate("low", expected_reuse=0.5),
                PRAHotAdmissionCandidate("medium", expected_reuse=2.0),
                PRAHotAdmissionCandidate("high", expected_reuse=4.0),
            ),
            max_prefetch_bytes=16,
        )
        assert [row.logical_key for row in decisions if row.admitted] == [
            "high",
            "medium",
        ]
        assert decisions[-1].reason == "reuse_below_threshold"
        await scheduler.close()
        assert scheduler.metrics().admission_rejected == 1
        assert scheduler.metrics().wasted_prefetches == 2

    asyncio.run(scenario())
