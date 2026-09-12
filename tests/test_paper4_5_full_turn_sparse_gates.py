"""Publication guards for the frozen seven-turn sparse-engine evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATES = (
    ROOT
    / "docs"
    / "papers"
    / "shared"
    / "results"
    / "paper4_5_runtime_productization"
    / "coding_agents"
    / "engine_gates"
)


def _read(name: str) -> dict[str, object]:
    return json.loads((GATES / name).read_text(encoding="utf-8"))


def test_vllm_cuda_alias_covers_all_seven_eligible_turns() -> None:
    result = _read("vllm_cuda_scheduler_alias_task02_full7_090_qwen15_v2.json")
    telemetry = result["final_registry_snapshot"]["telemetry"]

    assert result["qualified"] is True
    assert result["expected_eligible_turns"] == 7
    assert result["completed_turns"] == result["exact_turns"] == 7
    assert [row["turn"] for row in result["rows"]] == list(range(4, 11))
    assert telemetry["source_pin_events"] == 7
    assert telemetry["alias_install_events"] == 14
    assert telemetry["alias_release_events"] == 14
    assert telemetry["physical_kv_copy_bytes"] == 0
    assert telemetry["host_to_device_bytes"] == 0
    assert telemetry["selected_history_reencoded_tokens"] == 0
    assert result["final_registry_snapshot"]["sources"] == {}
    assert result["final_registry_snapshot"]["active_requests"] == []
    assert result["final_registry_snapshot"]["pending_requests"] == []


def test_hf_triton_consumer_covers_all_seven_sparse_turns() -> None:
    result = _read("hf_fused_sparse_position_task02_full7_090_qwen15_v1.json")

    assert result["consumer_implementation"] == "fused_interval_addressed_triton"
    assert result["completed_turns"] == result["exact_turns"] == 7
    assert result["sparse_turns"] == 7
    assert result["zero_copy_engine_gate_valid"] is True
    assert result["zero_selected_text_reencoding"] is True
    assert result["zero_physical_kv_copy"] is True
    assert result["zero_interval_pack_bytes"] is True
    assert result["request_tail_copy_bytes"] == 0
    assert result["fused_attention_calls"] == 196
    assert result["max_abs_logit_delta"] <= result["logit_tolerance"]


def test_metal_consumers_cover_all_seven_sparse_turns() -> None:
    for name, engine in (
        ("mlx_fused_sparse_position_task02_full7_090_qwen06_v1.json", "mlx-lm"),
        (
            "sglang_fused_sparse_position_task02_full7_090_qwen06_v1.json",
            "sglang-mlx",
        ),
    ):
        result = _read(name)
        assert result["engine"] == engine
        assert result["consumer_implementation"] == "fused_interval_addressed_metal"
        assert result["completed_turns"] == result["qualified_turns"] == 7
        assert result["full_coverage_qualified"] is True
        assert result["selection_pack_bytes"] == 0
        assert result["selection_active_memory_delta_bytes"] == 0
        assert result["selected_history_reencoded_tokens"] == 0
        assert result["physical_kv_copy_turns"] == 0
        assert result["full_selected_kv_sized_allocation_turns"] == 0
        assert result["max_abs_logit_delta_same_subset"] == 0.0
        assert result["max_consumer_peak_delta_to_selected_kv_ratio"] < 0.01
        assert [row["turn"] for row in result["rows"]] == list(range(4, 11))
