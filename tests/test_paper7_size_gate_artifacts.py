from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "docs/papers/shared/results/paper7_records/native_index_size_gate"
)
PAPER = (
    ROOT
    / "docs/papers/paper7_records/paper7_typed_adaptive_context_inception.tex"
)


def test_size_gate_artifact_family_is_complete():
    required = {
        "native_index_size_gate_results.csv",
        "native_index_size_gate_manifest.json",
        "ingestion_latency_by_size.csv",
        "time_to_usable_context.csv",
        "lazy_native_region_results.csv",
        "oversized_record_policy_results.csv",
        "native_index_ingestion_ttuc.pdf",
        "native_index_ingestion_ttuc.png",
        "generated_native_index_size_gate_results.tex",
    }

    assert required <= {path.name for path in RESULTS.iterdir()}


def test_profile_has_five_seeds_exact_boundary_and_oversized_lazy_rows():
    manifest = json.loads(
        (RESULTS / "native_index_size_gate_manifest.json").read_text(encoding="utf-8")
    )
    with (RESULTS / "native_index_size_gate_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert manifest["seeds"] == [11, 23, 37, 53, 71]
    assert manifest["sizes"] == [1024, 4096, 16384, 65536, 262144]
    boundary = [
        row
        for row in rows
        if row["condition"] == "SIZE_GATED_CHEAP"
        and int(row["payload_tokens"]) == 4096
    ]
    oversized = [
        row
        for row in rows
        if row["condition"] == "SIZE_GATED_LAZY_NATIVE"
        and int(row["payload_tokens"]) > 4096
    ]

    assert len(boundary) == 5
    assert all(row["native_index_state"] == "BUILT" for row in boundary)
    assert len(oversized) == 15
    assert all(row["native_index_state"] == "SKIPPED_SIZE_LIMIT" for row in oversized)
    assert all(int(row["selected_region_tokens"]) == 32 for row in oversized)
    assert all(float(row["evidence_recall"]) == 1.0 for row in oversized)


def test_generated_macros_match_largest_payload_summary_and_paper_claims():
    with (RESULTS / "ingestion_latency_by_size.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))
    by_condition = {
        row["condition"]: row
        for row in summary
        if int(row["payload_tokens"]) == 262144
    }
    macros = (
        RESULTS / "generated_native_index_size_gate_results.tex"
    ).read_text(encoding="utf-8")
    paper = PAPER.read_text(encoding="utf-8")

    eager_index = float(
        by_condition["FULL_BODY_NATIVE"]["native_index_latency_ms_median"]
    )
    gated_ttuc = float(
        by_condition["SIZE_GATED_CHEAP"]["time_to_usable_context_ms_median"]
    )
    assert f"{{{eager_index:.1f}}}" in macros
    assert f"{{{gated_ttuc:.1f}}}" in macros
    assert "\\PaperSevenSizeGateLimit" in paper
    assert "\\PaperSevenGatedRecall" in paper
    assert "not a pretrained LM" in paper
    assert eager_index == pytest.approx(693.6, abs=0.1)
