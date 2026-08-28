"""Regression gates for the checked-in Paper 4.5 cross-model artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.paper4_5_runtime.run_cross_model_validation import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_cross_model_manifest_pins_exact_checkpoint_provenance() -> None:
    rows = {row["key"]: row for row in _rows("hf_cross_model_manifest.csv")}

    assert set(rows) == set(MODEL_SPECS)
    assert rows["qwen"]["result_status"] == "passed"
    assert rows["llama"]["result_status"] == "passed"
    assert rows["llama"]["official_checkpoint_tested"] == "0"
    assert "access_blocked" in rows["llama"]["checkpoint_status"]
    assert rows["gemma"]["result_status"] == "partial_topology"


def test_each_model_records_all_semantic_conditions_and_runtime_axes() -> None:
    expected = {
        "VISIBLE_PREFIX",
        "NATIVE_FULL_REFERENCE",
        "NATIVE_SELECTED_FULL_RECORD",
        "NATIVE_CANONICAL_SPARSE_PROFILE",
    }
    for model in MODEL_SPECS:
        rows = _rows(f"hf_{model}_semantic_gate.csv")
        assert {row["condition"] for row in rows} == expected
        assert len(rows) == 3 * len(expected)
        assert all(int(row["active_layers"]) >= 0 for row in rows)
        assert all(int(row["physical_kv_bytes"]) >= 0 for row in rows)


def test_full_native_gate_and_partial_gemma_boundary_are_explicit() -> None:
    rows = {
        row["model"]: row for row in _rows("hf_cross_model_native_parity.csv")
    }

    assert rows["qwen"]["full_native_top_token_equal"] == "True"
    assert rows["llama"]["full_native_top_token_equal"] == "True"
    assert rows["gemma"]["status"] == "partial_topology"
    assert float(rows["gemma"]["full_native_max_logit_error"]) > 1.0
    assert "local_sliding_layers" in rows["gemma"]["topology_coverage"]


def test_task_smoke_uses_structural_scope_and_exact_selected_ids() -> None:
    rows = _rows("hf_cross_model_task_smoke.csv")

    assert len(rows) == 18
    assert all(row["scope"] == "paper8_task_structural" for row in rows)
    assert all(row["selected_id_exact"] == "1" for row in rows)


def test_reuse_and_multitenant_runtime_controls_pass() -> None:
    reuse = _rows("hf_payload_reuse_results.csv")
    concurrency = _rows("hf_multitenant_concurrency_results.csv")

    assert all(row["passed"] == "1" for row in reuse)
    assert sum(int(row["reuse_allowed"]) for row in reuse) == 1
    assert concurrency == [{
        "cancellation_cleanup": "1",
        "case": "same_uri_cross_tenant_concurrent_publish",
        "evictions": "1",
        "passed": "1",
        "tenant_isolation": "1",
    }]
    summary = json.loads((RESULTS / "hf_cross_model_summary.json").read_text())
    assert summary["status"] == "complete"
