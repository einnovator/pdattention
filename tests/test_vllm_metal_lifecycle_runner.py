from __future__ import annotations

from experiments.paper4_5_agent.run_vllm_metal_live_kv_lifecycle import (
    _complete_source_blocks,
    _page_prefix_plan,
)
from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan


def test_frozen_plan_projects_to_page_prefix_and_preserves_identity() -> None:
    plan = LiveKVSelectionPlan.create(
        21,
        (
            LiveKVInterval(0, 4, "system", "preamble"),
            LiveKVInterval(8, 14, "turn-a", "turn-a"),
            LiveKVInterval(18, 21, "wire-record", "current"),
        ),
    )

    projected = _page_prefix_plan(plan, 16, full_retention=False)

    assert projected.source_tokens == 16
    assert projected.source_position_base == 16
    assert [(row.start, row.end, row.record_id) for row in projected.intervals] == [
        (0, 4, "system"),
        (8, 14, "turn-a"),
    ]


def test_full_frozen_plan_covers_page_prefix() -> None:
    plan = LiveKVSelectionPlan.full(21)
    projected = _page_prefix_plan(plan, 16, full_retention=True)

    assert projected.full_retention
    assert projected.selected_tokens == 16


def test_chunked_prefill_uses_longest_cumulative_page_table() -> None:
    observations = [
        {"block_ids_by_group": [[10, 11]], "scheduler_cache_start": 0},
        {"block_ids_by_group": [[10, 11, 12, 13]], "scheduler_cache_start": 32},
        {"block_ids_by_group": [[10, 11, 12, 13, 14]], "scheduler_cache_start": 64},
    ]

    assert _complete_source_blocks(observations, 5) == (10, 11, 12, 13, 14)
