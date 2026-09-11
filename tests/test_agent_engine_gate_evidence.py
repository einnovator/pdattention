from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/"
    "engine_gates"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_sglang_same_state_gate_uses_a_pinned_source_owner() -> None:
    result = _load("sglang_same_state_task02_v3.json")
    assert result["same_resident_kv_fork"] is True
    assert result["source_owner_pinned_through_both_forks"] is True
    assert result["completed_turns"] == result["exact_turns"] == 10
    assert result["zero_selected_text_reencoding"] is True
    assert result["zero_physical_kv_copy"] is True
    assert result["same_state_gate_valid"] is True


def test_sglang_rematerialization_control_cannot_qualify_the_engine() -> None:
    result = _load("sglang_same_state_task02_v1.json")
    assert result["same_resident_kv_fork"] is False
    assert result["qualification_valid"] is False
    assert result["exact_turns"] == 7
    assert result["first_divergent_turn"] == 1


def test_vllm_overlay_result_is_exact_but_not_a_packaged_runtime_gate() -> None:
    result = _load("vllm_same_state_task02_v2.json")
    assert result["completed_turns"] == result["exact_turns"] == 10
    assert result["zero_selected_text_reencoding"] is True
    assert result["zero_physical_kv_copy"] is True
    assert result["exact_packaged_runtime"] is False
    assert result["same_state_gate_valid"] is False


def test_hf_no_cast_sparse_consumer_clears_strict_gate() -> None:
    result = _load("hf_nonpacking_sparse_position_task02_qwen15_strict_v2.json")
    assert result["max_abs_logit_delta"] == 0.0
    assert result["transient_kv_copy_bytes"] == 0
    assert result["max_transient_kv_tile_bytes"] == 0
    assert result["zero_physical_kv_copy"] is True
    assert result["zero_copy_engine_gate_valid"] is True


def test_mlx_interval_metal_consumer_is_exact_and_not_kv_sized() -> None:
    result = _load("mlx_interval_metal_sparse_lifecycle_task02_090_v2.json")
    allocation = result["disjoint_attention_allocation"]
    assert result["consumer_implementation"] == "fused_interval_addressed_metal"
    assert result["max_abs_logit_delta_same_subset"] == 0.0
    assert result["physical_kv_copy"] is False
    assert allocation["peak_delta_bytes"] < allocation["selected_layer_kv_bytes"] * 0.01
    assert result["engine_lifecycle_qualified"] is True


def test_sglang_interval_metal_consumer_is_exact_through_radix_lifecycle() -> None:
    result = _load("sglang_interval_metal_sparse_lifecycle_task02_090_v2.json")
    allocation = result["disjoint_attention_allocation"]
    assert result["consumer_implementation"] == "fused_interval_addressed_metal"
    assert result["max_abs_logit_delta_same_subset"] == 0.0
    assert result["physical_kv_copy"] is False
    assert allocation["peak_delta_bytes"] < allocation["selected_layer_kv_bytes"] * 0.01
    assert result["engine_lifecycle_qualified"] is True
