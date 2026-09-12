from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from pra_vllm.cuda_scheduler_alias import (
    SchedulerPageSelection,
    VLLMCudaSchedulerPageRegistry,
)
from pra_vllm.cuda_sparse_connector import _completed_append_pages


@dataclass(eq=False)
class _Block:
    block_id: int
    ref_cnt: int = 1
    is_null: bool = False


class _Pool:
    def __init__(self) -> None:
        self.touch_calls: list[tuple[int, ...]] = []
        self.free_calls: list[tuple[int, ...]] = []

    def touch(self, blocks) -> None:
        rows = tuple(blocks)
        self.touch_calls.append(tuple(block.block_id for block in rows))
        for block in rows:
            block.ref_cnt += 1

    def free_blocks(self, blocks) -> None:
        rows = tuple(blocks)
        self.free_calls.append(tuple(block.block_id for block in rows))
        for block in rows:
            block.ref_cnt -= 1


def _selection(request_key: str = "selected") -> SchedulerPageSelection:
    return SchedulerPageSelection(
        logical_key=request_key,
        source_logical_key="source",
        source_generation=7,
        selected_page_indices=(0, 2),
        selected_token_count=32,
        source_position_base=48,
    )


def _blocks(value):
    return SimpleNamespace(blocks=tuple(tuple(group) for group in value))


def _published():
    pool = _Pool()
    pages = tuple(_Block(index) for index in range(3))
    registry = VLLMCudaSchedulerPageRegistry()
    registry.publish_source(
        "source",
        generation=7,
        source_tokens=48,
        blocks_by_group=(pages,),
        block_sizes=(16,),
        block_pool=pool,
    )
    return registry, pool, pages


def test_load_completion_counts_pages_from_explicit_scheduler_manager() -> None:
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[SimpleNamespace(block_size=16)]
        )
    )
    assert _completed_append_pages(
        manager, SimpleNamespace(num_computed_tokens=101), 64
    ) == 2


def test_authoritative_alias_uses_exact_source_objects_and_native_refcounts() -> None:
    registry, pool, pages = _published()

    # The producer request then exits. The registry's source-owner pin remains.
    pool.free_blocks(reversed(pages))
    assert [page.ref_cnt for page in pages] == [1, 1, 1]

    installed = registry.prepare_alias(
        "consumer",
        _selection(),
        prompt_token_count=34,
        create_kv_cache_blocks=_blocks,
    )
    assert installed.blocks[0] == (pages[0], pages[2])
    assert installed.blocks[0][0] is pages[0]
    assert installed.blocks[0][1] is pages[2]

    # This is what vLLM add_local_computed_blocks does: touch, then append the
    # exact objects to the request table. No tensor copy or H2D path is called.
    pool.touch(installed.blocks[0])
    registry.commit_alias("consumer", installed)
    assert [page.ref_cnt for page in pages] == [2, 1, 2]

    assert registry.finish_request("consumer")
    assert not registry.finish_request("consumer")
    pool.free_blocks(reversed(installed.blocks[0]))
    assert [page.ref_cnt for page in pages] == [1, 1, 1]
    assert registry.evict_source("source", generation=7) == (0, 1, 2)
    assert [page.ref_cnt for page in pages] == [0, 0, 0]

    telemetry = registry.telemetry()
    assert telemetry.source_pin_events == 1
    assert telemetry.alias_install_events == 1
    assert telemetry.alias_release_events == 1
    assert telemetry.physical_kv_copy_bytes == 0
    assert telemetry.host_to_device_bytes == 0
    assert telemetry.selected_history_reencoded_tokens == 0


def test_concurrent_finish_cancel_and_error_release_exactly_once() -> None:
    registry, pool, pages = _published()
    pool.free_blocks(reversed(pages))
    aliases = {}
    for request_id in ("finish", "cancel", "error"):
        aliases[request_id] = registry.prepare_alias(
            request_id,
            _selection(request_id),
            prompt_token_count=33,
            create_kv_cache_blocks=_blocks,
        )
        pool.touch(aliases[request_id].blocks[0])
        registry.commit_alias(request_id, aliases[request_id])

    with pytest.raises(RuntimeError, match="borrowed"):
        registry.evict_source("source", generation=7)

    for request_id in ("cancel", "error", "finish"):
        assert registry.finish_request(request_id)
        assert not registry.finish_request(request_id)
        pool.free_blocks(reversed(aliases[request_id].blocks[0]))

    assert registry.telemetry().alias_release_events == 3
    assert [page.ref_cnt for page in pages] == [1, 1, 1]
    registry.evict_source("source", generation=7)
    assert [page.ref_cnt for page in pages] == [0, 0, 0]


def test_stale_generation_and_out_of_range_selection_fail_before_touch() -> None:
    registry, pool, _ = _published()
    stale = SchedulerPageSelection(
        logical_key="stale",
        source_logical_key="source",
        source_generation=8,
        selected_page_indices=(0,),
        selected_token_count=16,
        source_position_base=48,
    )
    with pytest.raises(RuntimeError, match="Stale"):
        registry.prepare_alias(
            "stale", stale, prompt_token_count=17, create_kv_cache_blocks=_blocks
        )

    outside = SchedulerPageSelection(
        logical_key="outside",
        source_logical_key="source",
        source_generation=7,
        selected_page_indices=(3,),
        selected_token_count=16,
        source_position_base=48,
    )
    with pytest.raises(ValueError, match="outside"):
        registry.prepare_alias(
            "outside", outside, prompt_token_count=17, create_kv_cache_blocks=_blocks
        )
    assert pool.touch_calls == [(0, 1, 2)]


def test_termination_is_a_tombstone_and_defers_unpin_until_borrower_exit() -> None:
    registry, pool, pages = _published()
    pool.free_blocks(reversed(pages))
    installed = registry.prepare_alias(
        "consumer",
        _selection(),
        prompt_token_count=33,
        create_kv_cache_blocks=_blocks,
    )
    pool.touch(installed.blocks[0])
    registry.commit_alias("consumer", installed)

    assert registry.terminate_source("source", generation=7)
    assert not registry.terminate_source("source", generation=7)
    with pytest.raises(RuntimeError, match="unavailable or terminated"):
        registry.prepare_alias(
            "late",
            _selection("late"),
            prompt_token_count=33,
            create_kv_cache_blocks=_blocks,
        )
    assert registry.snapshot()["sources"]["source"]["tombstoned"]

    registry.finish_request("consumer")
    # The logical callback fires before vLLM's ordinary/deferred physical free.
    assert "source" in registry.snapshot()["sources"]
    assert [page.ref_cnt for page in pages] == [2, 1, 2]
    pool.free_blocks(reversed(installed.blocks[0]))
    assert registry.reap_tombstones() == ("source",)
    assert [page.ref_cnt for page in pages] == [0, 0, 0]


def test_compact_selected_length_and_original_position_extent_are_independent() -> None:
    selection = _selection()
    assert selection.selected_token_count == 32
    assert selection.source_position_base == 48
    assert selection.source_position_base - selection.selected_token_count == 16

    registry, _, _ = _published()
    with pytest.raises(ValueError, match="compact selected token prefix"):
        registry.prepare_alias(
            "no-query",
            selection,
            prompt_token_count=selection.selected_token_count,
            create_kv_cache_blocks=_blocks,
        )


def test_commit_rejects_a_worker_or_detached_copy() -> None:
    registry, pool, _ = _published()
    registry.prepare_alias(
        "consumer",
        _selection(),
        prompt_token_count=33,
        create_kv_cache_blocks=_blocks,
    )
    copied_pages = (_Block(0), _Block(2))
    pool.touch(copied_pages)

    with pytest.raises(RuntimeError, match="authoritative source block table"):
        registry.commit_alias("consumer", _blocks((copied_pages,)))


def test_eviction_waits_for_scheduler_deferred_physical_release() -> None:
    registry, pool, pages = _published()
    pool.free_blocks(reversed(pages))
    installed = registry.prepare_alias(
        "consumer",
        _selection(),
        prompt_token_count=33,
        create_kv_cache_blocks=_blocks,
    )
    pool.touch(installed.blocks[0])
    registry.commit_alias("consumer", installed)
    registry.finish_request("consumer")

    with pytest.raises(RuntimeError, match="in-flight aliases"):
        registry.evict_source("source", generation=7)

    pool.free_blocks(reversed(installed.blocks[0]))
    assert registry.evict_source("source", generation=7) == (0, 1, 2)
