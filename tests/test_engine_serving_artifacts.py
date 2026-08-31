from __future__ import annotations

import json
from pathlib import Path

from experiments.engine_serving.summarize import build_registry


ROOT = Path(__file__).resolve().parents[1]


def test_engine_registry_separates_smoke_from_native_evidence() -> None:
    registry = build_registry()
    assert registry["registry_version"] == "2026-08-paper6-engine-native-v10"
    assert len(registry["rows"]) == 15
    assert {row["engine"] for row in registry["rows"]} == {"vllm", "sglang", "mlx"}
    assert all(row["evidence_tier"] == "SMOKE" for row in registry["rows"])
    assert all(row["native_pra_status"] == "NOT_MEASURED" for row in registry["rows"])
    rotating = registry["mlx_rotating_archive"]
    assert rotating["evidence_tier"] == "CONTROLLED"
    assert rotating["seeds"] == [11, 23, 37, 53, 71]
    assert rotating["native_pra_status"] == "NOT_MEASURED"
    native = registry["native_results"]
    assert native["mlx"]["status"] == "MEASURED"
    assert native["mlx"]["exact_recovery"] == 1.0
    assert len(native["mlx"]["models"]) == 3
    assert all(row["max_logit_error"] == 0.0 for row in native["mlx"]["models"])
    pressure = native["mlx"]["residency_pressure_curve"]
    assert len(pressure) == 12
    assert all(row["seed_count"] == 5 for row in pressure)
    assert all(row["request_count"] == 85 for row in pressure)
    assert all(
        row["reload_fraction"] == 1.0
        for row in pressure
        if row["resident_resource_budget"] < 8
    )
    assert all(
        row["reload_fraction"] == 0.0
        for row in pressure
        if row["resident_resource_budget"] == 8
    )
    assert native["sglang"]["status"] == "MEASURED_SGLANG_CACHE_PATH"
    assert native["sglang"]["radix_identity_separation_rate"] == 1.0
    prefetch = native["sglang"]["builtin_hicache_prefetch"]
    assert prefetch["seed_count"] == 5
    assert prefetch["native_async_overlap_status"] == (
        "OPEN_PYTHON_THREAD_BLOCKS_CALLER"
    )
    assert all(
        row["exact_tensor_recovery"] == 1.0
        for row in prefetch["rows_by_requested_lead_ms"]
    )
    assert min(
        row["actual_lead_ms_mean"]
        for row in prefetch["rows_by_requested_lead_ms"]
        if row["requested_lead_ms"]
    ) > 190.0
    assert native["vllm"]["status"] == "MEASURED_KERNEL_PATH"
    assert native["mac_engine_extension"] is not None
    expanded_matched = native["expanded_matched_e0_e2"]
    assert expanded_matched is not None
    assert len(expanded_matched["parity"]) >= 24
    assert all(
        row["exact_output_parity"] == 1.0
        for row in expanded_matched["parity"]
    )
    assert len(registry["product_matrix"]) == 13
    assert any(
        row["model"].endswith("gemma-3-1b-it-4bit")
        and row["engine"] == "SGLang MLX"
        and "blocked before PRA" in row["status"]
        for row in registry["product_matrix"]
    )
    bounded_e0 = [
        row
        for row in registry["product_matrix"]
        if row["evidence_tier"] == "SERVING_BOUNDED_LOAD_CONTEXT_PRESSURE"
    ]
    assert {row["engine"] for row in bounded_e0} == {
        "vLLM CUDA V1",
        "TensorRT-LLM 1.2.1",
        "OpenVINO GenAI 2026.3.1",
    }
    assert all("E0" in row["profile"] for row in bounded_e0)


def test_selected_text_recovers_codeword_with_fewer_tokens_than_full_context() -> None:
    registry = build_registry()
    for engine in ("vllm", "sglang", "mlx"):
        rows = {row["condition"]: row for row in registry["rows"] if row["engine"] == engine}
        assert rows["prefix_plus_pra"]["quality_absolute"] == 1.0
        assert rows["no_prefix_no_pra"]["quality_absolute"] == 0.0
        assert rows["prefix_plus_pra"]["visible_initial_tokens"] < rows["full_context"]["visible_initial_tokens"]


def test_generated_registry_matches_builder_after_summary_run() -> None:
    generated = json.loads(
        (ROOT / "docs" / "papers" / "shared" / "results" / "pra_engine_benchmarks.json").read_text()
    )
    assert generated == build_registry()
