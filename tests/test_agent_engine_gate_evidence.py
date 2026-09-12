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


def test_hf_task02_full_history_cache_control_is_sequence_exact() -> None:
    result = _load(
        "hf_qwen25coder15b_task02_incremental_full_history_20260912.json"
    )
    assert result["model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert result["completed_turns"] == result["exact_turns"] == 7
    assert result["all_exact"] is True
    assert result["first_divergent_turn"] is None
    assert max(row["max_abs_logit_delta"] for row in result["rows"]) == 0.34375


def test_hf_task02_pra100_is_logit_exact_from_identical_resident_state() -> None:
    result = _load(
        "hf_qwen25coder15b_task02_pra100_same_state_cuda_20260912.json"
    )
    assert result["schema_version"] == "paper4.5.agent-history-kv-gate.v2"
    assert result["model_revision"] == (
        "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    )
    assert result["retention_fraction"] == 1.0
    assert result["completed_turns"] == result["exact_turns"] == 7
    assert result["max_abs_logit_delta"] == 0.0
    assert result["pra100_same_state_gate_valid"] is True
    assert result["zero_copy_engine_gate_valid"] is True
    assert all(row["dense_semantic_noop_identity"] for row in result["rows"])
    accounting = result["copy_accounting"]
    assert accounting["selected_history_reencoded_tokens"] == 0
    assert accounting["selected_history_kv_copy_bytes"] == 0
    assert accounting["physical_kv_copy_bytes"] == 0
    assert accounting["host_to_device_bytes"] == 0
    assert accounting["interval_pack_bytes"] == 0
    assert accounting["measurement_state_fork_copy_bytes"] > 0
    assert accounting["measurement_state_fork_in_production_path"] is False


def test_vllm_task02_pra100_matches_dense_and_closes_every_alias() -> None:
    result = _load(
        "vllm_qwen25coder15b_task02_pra100_dense_reference_20260912.json"
    )
    assert result["engine_version"] == "0.28.0"
    assert result["model_revision"] == (
        "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    )
    assert result["retention_fractions"] == [1.0]
    assert len(result["arms"]) == 1
    rows = result["arms"][0]["rows"]
    assert len(rows) == 7
    assert all(row["dense_reference_exact"] for row in rows)
    assert all(row["trace"]["realized_retention_fraction"] == 1.0 for row in rows)
    accounting = result["copy_accounting"]
    assert accounting["selected_history_reencoded_tokens"] == 0
    assert accounting["selected_history_kv_copy_bytes"] == 0
    assert accounting["physical_kv_copy_bytes"] == 0
    assert accounting["host_to_device_bytes"] == 0
    assert accounting["total_kv_copy_bytes"] == 0
    assert accounting["consumer_temporary_bytes_peak"] == 206446080
    callbacks = result["consumer"]["scheduler_alias_calls"]
    assert set(callbacks.values()) == {6}
    lifecycle = result["arms"][0]["lifecycle"]
    assert lifecycle["source_count_after_close"] == 0
    assert lifecycle["active_request_count_after_close"] == 0
    assert lifecycle["pending_commit_count_after_close"] == 0
    assert result["passed"] is True


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
