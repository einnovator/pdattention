from __future__ import annotations

from dataclasses import dataclass

import pytest

from pra_hf.live_history import LiveKVSelectionPlan
from pra_vllm.metal_live_kv import VLLMMetalLiveKVRuntime


@dataclass
class _Block:
    block_id: int
    ref_cnt: int = 0


class _BlockPool:
    def __init__(self, count: int = 16) -> None:
        self.blocks = [_Block(index) for index in range(count)]
        self.evicted: list[set[int]] = []

    def touch(self, blocks) -> None:
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks) -> None:
        for block in blocks:
            block.ref_cnt -= 1
            assert block.ref_cnt >= 0

    def evict_blocks(self, block_ids: set[int]) -> None:
        self.evicted.append(set(block_ids))


class _Bridge:
    block_size = 4
    scheduler_blocks = 8
    reserve_blocks = 8

    def __init__(self) -> None:
        self.handles: dict[str, tuple[int, ...]] = {}
        self.registrations: dict[str, tuple[str, ...]] = {}
        self.released: list[str] = []

    def alias_resident_pages(self, key, blocks, *, selected_token_count):
        values = tuple(blocks)
        assert selected_token_count == len(values) * self.block_size
        self.handles[str(key)] = values
        return values

    def register(
        self,
        request_id,
        keys,
        *,
        selected_token_count,
        source_position_base,
        tenant_id,
        session_id,
    ) -> None:
        assert selected_token_count > 0
        assert source_position_base == 24
        assert tenant_id == "tenant"
        assert session_id == "session"
        self.registrations[str(request_id)] = tuple(keys)

    def unregister(self, request_id) -> None:
        self.registrations.pop(str(request_id), None)

    def release(self, key) -> None:
        value = str(key)
        self.handles.pop(value, None)
        self.released.append(value)


def _plan() -> LiveKVSelectionPlan:
    return LiveKVSelectionPlan.create(24, ((0, 4), (8, 24)))


def _runtime():
    bridge = _Bridge()
    pool = _BlockPool()
    restored = []

    def restore(source_id: str, payload: bytes, source_tokens: int):
        assert source_id == "source"
        assert payload == b"source:0,1,2,3,4,5"
        assert source_tokens == 24
        restored.append(source_id)
        bridge.handles["restored-source"] = (8, 9, 10, 11, 12, 13)
        return bridge.handles["restored-source"], "restored-source"

    runtime = VLLMMetalLiveKVRuntime(
        bridge,
        pool,
        dump_pages=lambda blocks, _tokens: (
            f"source:{','.join(map(str, blocks))}".encode("ascii")
        ),
        restore_pages=restore,
    )
    runtime.register_source(
        "source",
        (0, 1, 2, 3, 4, 5),
        source_tokens=24,
        tenant_id="tenant",
        session_id="session",
        generation=1,
    )
    return runtime, bridge, pool, restored


def _begin(runtime, request_id: str, *, generation: int = 1):
    return runtime.begin_request(
        request_id,
        "source",
        _plan(),
        tenant_id="tenant",
        session_id="session",
        expected_generation=generation,
    )


def test_vllm_metal_source_pin_and_concurrent_request_cleanup_are_exact_once() -> None:
    runtime, bridge, pool, _restored = _runtime()
    assert [block.ref_cnt for block in pool.blocks[:6]] == [1] * 6

    first = _begin(runtime, "first")
    second = _begin(runtime, "second")
    assert runtime.registry.view("source").active_request_ids == ("first", "second")
    assert first.selection.page_indices == (0, 2, 3, 4, 5)
    assert first.selection.selected_kv_tokens == 20
    assert first.selection.plan.source_position_base == 24
    assert not first.selection.attachment_physical_kv_copy
    assert first.selection.selected_text_reencoded_tokens == 0
    with pytest.raises(RuntimeError, match="while requests borrow"):
        runtime.offload_source("source")

    assert first.execute(lambda: "complete") == "complete"
    assert first.outcome == "finished"
    assert first.finish() is False
    assert runtime.registry.view("source").active_request_ids == ("second",)
    assert second.finish()
    assert bridge.registrations == {}
    assert [block.ref_cnt for block in pool.blocks[:6]] == [1] * 6
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 2,
        "cancelled": 0,
        "error": 0,
    }


def test_vllm_metal_cancel_error_stale_offload_restore_and_tombstone() -> None:
    runtime, bridge, pool, restored = _runtime()
    aborted: list[list[str]] = []

    cancelled = _begin(runtime, "cancelled")
    assert runtime.cancel_request(
        "cancelled", abort_requests=lambda ids: aborted.append(ids)
    )
    assert aborted == [["cancelled"]]
    assert cancelled.outcome == "cancelled"
    assert cancelled.cancel() is False

    errored = _begin(runtime, "errored")
    with pytest.raises(RuntimeError, match="engine failure"):
        errored.execute(lambda: (_ for _ in ()).throw(RuntimeError("engine failure")))
    assert errored.outcome == "error"
    assert errored.fail() is False

    with pytest.raises(RuntimeError, match="Stale"):
        _begin(runtime, "stale", generation=0)
    payload = runtime.offload_source("source")
    assert payload.payload == b"source:0,1,2,3,4,5"
    assert runtime.registry.view("source").tier == "offloaded"
    assert [block.ref_cnt for block in pool.blocks[:6]] == [0] * 6
    assert pool.evicted == [{0, 1, 2, 3, 4, 5}]

    request = _begin(runtime, "restored")
    assert restored == ["source"]
    assert runtime.registry.view("source").tier == "hot"
    assert request.selection.source_restored_with_physical_copy
    assert not request.selection.attachment_physical_kv_copy
    assert request.selection.block_ids == (8, 10, 11, 12, 13)
    request.finish()

    active = _begin(runtime, "active-at-termination")
    terminated_aborts: list[list[str]] = []
    assert runtime.terminate_session(
        "tenant",
        "session",
        abort_requests=lambda ids: terminated_aborts.append(ids),
    ) == 1
    assert terminated_aborts == [["active-at-termination"]]
    assert active.outcome == "cancelled"
    assert active.cancel() is False
    assert "restored-source" in bridge.released
    assert runtime.registry.view("source") is None
    with pytest.raises(RuntimeError, match="terminated"):
        runtime.register_source(
            "replacement",
            (0, 1, 2, 3, 4, 5),
            source_tokens=24,
            tenant_id="tenant",
            session_id="session",
            generation=2,
        )
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 1,
        "cancelled": 2,
        "error": 1,
    }


def test_vllm_metal_failed_selection_releases_source_borrow() -> None:
    runtime, _bridge, _pool, _restored = _runtime()
    with pytest.raises(ValueError, match="does not match"):
        runtime.begin_request(
            "bad-plan",
            "source",
            LiveKVSelectionPlan.create(20, ((0, 20),)),
            tenant_id="tenant",
            session_id="session",
            expected_generation=1,
        )
    assert runtime.registry.view("source").active_request_ids == ()
