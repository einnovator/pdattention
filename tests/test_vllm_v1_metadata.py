from __future__ import annotations

import pytest

from pra_vllm.v1_metadata import VLLMNativeBlockSet, VLLMNativeStepRegistry


def _selected() -> VLLMNativeBlockSet:
    return VLLMNativeBlockSet(
        logical_keys=("resource-v1",),
        block_ids_by_group=((7, 8),),
        selected_token_count=23,
        source_position_base=31,
        consumer_layers=(20, 21, 22, 23),
    )


def test_native_step_keeps_scheduler_and_attention_geometry_separate() -> None:
    registry = VLLMNativeStepRegistry()
    registry.register("native", _selected())

    native, ordinary = registry.plan_step(
        ("native", "ordinary"),
        scheduler_cache_starts={"native": 5, "ordinary": 5},
        query_token_counts={"native": 2, "ordinary": 2},
    )

    assert native.scheduler_cache_start == ordinary.scheduler_cache_start == 5
    assert native.query_position_start == 36
    assert ordinary.query_position_start == 5
    assert native.attention_key_tokens == 30
    assert ordinary.attention_key_tokens == 7


def test_live_request_cannot_change_selected_pages() -> None:
    registry = VLLMNativeStepRegistry()
    registry.register("request", _selected())
    replacement = VLLMNativeBlockSet(
        logical_keys=("resource-v2",),
        block_ids_by_group=((9,),),
        selected_token_count=8,
        source_position_base=8,
        consumer_layers=(23,),
    )

    with pytest.raises(ValueError, match="cannot change"):
        registry.register("request", replacement)

    registry.unregister("request")
    assert registry.active_request_count() == 0


def test_registry_exposes_active_request_snapshot() -> None:
    registry = VLLMNativeStepRegistry()
    registry.register("req-a", _selected())
    registry.register("req-b", _selected())

    assert registry.active_request_ids() == ("req-a", "req-b")

    registry.unregister("req-a")
    assert registry.active_request_ids() == ("req-b",)
