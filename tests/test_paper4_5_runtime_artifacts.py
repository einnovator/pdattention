"""Consistency checks for the generated Paper 4.5 runtime artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime"
PAPER = ROOT / "docs" / "papers" / "paper4_5_runtime_productization"


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
    assert (PAPER / "paper.pdf").stat().st_size > 100_000


def test_compatibility_matrix_keeps_claim_levels_explicit() -> None:
    with (RESULTS / "compatibility_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["runtime"]: row["status"] for row in csv.DictReader(handle)}

    assert rows["HF/PyTorch eager"] == "measured"
    assert rows["torch.compile"] == "blocked on host"
    assert rows["vLLM thin"] == "contract only"
    assert rows["SGLang/TensorRT-LLM/MLX"] == "architectural only"
