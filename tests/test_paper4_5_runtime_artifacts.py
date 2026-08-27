"""Consistency checks for the generated Paper 4.5 runtime artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime"
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
    assert manifest["protocol"]["quality_selection_frozen"] is True
    assert manifest["protocol"]["scope"] == "portable selected-KV mechanism microbenchmark"
    assert "\\newcommand{\\RuntimeCudaIndexBatchOneMs}" in macros
    assert "\\newcommand{\\RuntimeLayoutSpreadBatchFour}" in macros
    assert manifest["execution_policy_profile"].startswith("five-seed")
    assert findings["token_per_layer_routing_operations"] == pytest.approx(
        3 * findings["token_shared_routing_operations"]
    )
    assert "\\newcommand{\\ExecutionRoutingReduction}{3.0}" in macros
    assert (RESULTS / "execution_policy_tradeoff.pdf").is_file()
    assert (PAPER / "paper.pdf").stat().st_size > 100_000


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
    ):
        assert phrase in markdown

    outputs = json.dumps(
        [output for cell in code_cells for output in cell.get("outputs", ())]
    )
    assert "'in_budget_state': 'BUILT'" in outputs
    assert "'oversized_state': 'SKIPPED_SIZE_LIMIT'" in outputs
    assert "'references_after_close': 0" in outputs
