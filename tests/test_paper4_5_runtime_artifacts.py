"""Consistency checks for the generated Paper 4.5 runtime artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime"
LAYER_RESULTS = RESULTS / "layer_profiles"
PAPER = ROOT / "docs" / "papers" / "paper4_5_runtime_productization"
DEMO = ROOT / "pra-hf-demo" / "pra_runtime_productization.ipynb"


def test_runtime_artifacts_are_complete_and_internally_consistent() -> None:
    findings = json.loads((RESULTS / "findings.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    macros = (RESULTS / "generated_runtime_results.tex").read_text(encoding="utf-8")

    assert findings["status"] == "portable_eager_measured_compile_and_engine_gates_negative"
    assert findings["optional_engines_measured"] == []
    assert findings["pinned_over_hbm_latency_ratio"] > 1.0
    assert findings["ordinary_over_hbm_latency_ratio"] > findings["pinned_over_hbm_latency_ratio"]
    assert findings["cuda_layout_slowest_over_fastest_batch1"] < 1.1
    assert findings["cuda_layout_slowest_over_fastest_batch4"] < 1.1

    assert manifest["engine_speed_claims"] is False
    assert manifest["quality_metrics_recomputed"] is False
    assert manifest["paper8_native_geometry_integrated"] is True
    assert manifest["hf_reference_gate"]["decode_lifetime_pass"] is True
    assert manifest["hf_reference_gate"]["gateway_streaming_pass"] is True
    assert manifest["hf_reference_gate"]["prefix_equivalence_max_abs_error_fp32"] < 1e-6
    assert manifest["agent_plugin_profile"].startswith("two public event vocabularies")
    assert manifest["protocol"]["quality_selection_frozen"] is True
    assert manifest["protocol"]["scope"] == "portable selected-KV mechanism microbenchmark"
    assert "\\newcommand{\\RuntimeCudaIndexBatchOneMs}" in macros
    assert "\\newcommand{\\RuntimeLayoutSpreadBatchFour}" in macros
    assert manifest["execution_policy_profile"].startswith("five-seed")
    assert findings["token_per_layer_routing_operations"] == pytest.approx(
        3 * findings["token_shared_routing_operations"]
    )
    assert findings["agent_plugin_cases"] == 10
    assert findings["deepseek_agent_contract_pass_rate"] == 1
    assert findings["pi_agent_contract_pass_rate"] == 1
    assert "\\newcommand{\\ExecutionRoutingReduction}{3.0}" in macros
    assert (RESULTS / "execution_policy_tradeoff.pdf").is_file()
    assert (PAPER / "paper.pdf").stat().st_size > 100_000

    required = (
        "pra_profile_benchmarks.json",
        "generated_profile_matrix.tex",
        "generated_profile_technical_matrix.tex",
        "generated_runtime_matrix.tex",
        "hf_native_prefix_equivalence.csv",
        "hf_native_full_scope_parity.csv",
        "hf_interval_dedup_results.csv",
        "hf_decode_lifetime_results.csv",
        "hf_materialization_profile_results.csv",
        "hf_modern_gpu_compile_results.csv",
        "hf_async_transfer_results.csv",
        "hf_multitenant_cache_results.csv",
        "hf_gateway_streaming_results.csv",
        "engine_feature_applicability.md",
    )
    assert all((RESULTS / name).is_file() for name in required)


def test_compatibility_matrix_keeps_claim_levels_explicit() -> None:
    with (RESULTS / "compatibility_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["runtime"]: row["status"] for row in csv.DictReader(handle)}

    assert rows["HF/PyTorch eager"] == "measured"
    assert rows["torch.compile"] == "blocked on host"
    assert rows["vLLM thin"] == "contract only"
    assert rows["Standalone gateway"] == "contract tested"
    assert rows["OpenAI-compatible HTTP"] == "E0 implemented"
    assert rows["SGLang/FreeToken"] == "E0 feasible, not run"
    assert rows["TensorRT-LLM/MLX"] == "architectural only"


def test_agent_plugin_contract_artifacts_preserve_explicit_fallback() -> None:
    summary = json.loads(
        (RESULTS / "agent_plugin_contract_summary.json").read_text(encoding="utf-8")
    )
    with (RESULTS / "agent_plugin_contract_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 10
    assert set(summary) == {"deepseek_harness", "pi_coding_agent"}
    assert all(value["contract_pass_rate"] == 1 for value in summary.values())
    assert all(row["native_kv_claimed"] == "0" for row in rows)
    assert all(row["fallback_contains_evidence"] == "1" for row in rows)


def test_layer_profile_calibration_has_cross_model_roles_costs_and_provenance() -> None:
    manifest = json.loads(
        (LAYER_RESULTS / "layer_calibration_manifest.json").read_text(encoding="utf-8")
    )
    selected = json.loads(
        (LAYER_RESULTS / "layer_calibration_selected_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["models"] == ["gemma", "llama", "qwen"]
    assert manifest["transport"].startswith("corrected_native_kv")
    assert selected["qwen"]["reference_correctness"] == "all_layers"
    assert selected["llama"]["balanced"] == "all_layers"
    assert selected["gemma"]["balanced"] == "all_layers"

    required = (
        "pra_layer_profile_schema.json",
        "pra_layer_profile_registry.json",
        "layer_calibration_candidates.csv",
        "layer_calibration_pareto.csv",
        "layer_calibration_selected_profiles.json",
        "layer_profile_portability.csv",
        "layer_profiles_qwen.csv",
        "layer_profiles_llama.csv",
        "layer_profiles_gemma.csv",
        "detail_kv_encoding_policy_schema.json",
        "address_encoding_policy_schema.json",
        "layer_profile_storage_costs.csv",
        "layer_profile_switching_costs.csv",
        "layer_profile_detail_union.json",
        "partial_native_index_lifecycle.csv",
        "layer_profile_quality_cost.pdf",
        "layer_profile_quality_cost.png",
    )
    assert all((LAYER_RESULTS / name).is_file() for name in required)


def test_runtime_demo_is_executed_and_covers_the_unified_sdk() -> None:
    notebook = json.loads(DEMO.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", ())
        if output.get("output_type") == "error"
    ]

    assert len(code_cells) >= 18
    assert [cell["execution_count"] for cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    assert not errors
    for phrase in (
        "How this differs from the Paper 2 model-family demo",
        "Authenticated external memory",
        "Four physical layouts",
        "Byte-bounded hot-cache behavior",
        "Capability-graph disclosure",
        "Thin vLLM handoff",
        "Multi-axis execution policy",
        "Standalone gateway and explicit downgrade",
        "Size-adaptive native indexing and lazy region promotion",
        "Product profiles and evidence status",
        "Independent cheap indexes for oversized typed records",
    ):
        assert phrase in markdown

    outputs = json.dumps(
        [output for cell in code_cells for output in cell.get("outputs", ())]
    )
    assert "'in_budget_state': 'BUILT'" in outputs
    assert "'oversized_state': 'SKIPPED_SIZE_LIMIT'" in outputs
    assert "'references_after_close': 0" in outputs
