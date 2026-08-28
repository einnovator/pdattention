"""Contract checks for the frozen Paper 7 Headroom cross-evaluation package."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/papers/shared/results/paper7_records/headroom_cross_eval"
REQUIRED = {
    "headroom_official_manifest.json",
    "headroom_on_paper7_results.csv",
    "pra_on_headroom_results.csv",
    "headroom_trigger_analysis.csv",
    "headroom_oracle_results.csv",
    "headroom_ccr_style_faithfulness.csv",
    "headroom_cost_accounting.csv",
    "headroom_cross_eval_summary.csv",
    "headroom_cross_eval_stats.json",
}


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_required_cross_eval_artifacts_are_present() -> None:
    assert REQUIRED <= {path.name for path in RESULTS.iterdir()}


def test_manifest_pins_official_headroom_and_marks_boundaries() -> None:
    manifest = json.loads((RESULTS / "headroom_official_manifest.json").read_text())
    assert manifest["headroom"]["package"] == "headroom-ai==0.37.0"
    assert manifest["headroom"]["python"] == "3.10.11"
    assert manifest["feature_audit"]["state_reset"].startswith("reset_compression_store")
    assert "No matched official full-CCR run" in manifest["claim_boundary"]


def test_official_and_ccr_style_conditions_are_not_conflated() -> None:
    rows = _rows("headroom_on_paper7_results.csv")
    sources = {(row["condition"], row["source"]) for row in rows}
    assert ("HEADROOM_OFFICIAL_DEFAULT", "HEADROOM_OFFICIAL") in sources
    assert ("HEADROOM_OFFICIAL_TUNED", "HEADROOM_OFFICIAL") in sources
    assert ("CCR_STYLE", "CCR_STYLE") in sources


def test_reverse_direction_records_balanced_sources_and_eligibility() -> None:
    cases = json.loads((RESULTS / "headroom_eval_cases.json").read_text())["rows"]
    assert {dataset: sum(row["dataset"] == dataset for row in cases) for dataset in {
        "tool_outputs", "ccr_needle", "hotpotqa", "msmarco"
    }} == {"tool_outputs": 8, "ccr_needle": 8, "hotpotqa": 8, "msmarco": 8}

    rows = _rows("pra_on_headroom_results.csv")
    eligible = {
        dataset: sum(
            row["condition"] == "PRA_FROZEN"
            and row["dataset"] == dataset
            and row["evidence_eligible"] == "1"
            for row in rows
        )
        for dataset in {"tool_outputs", "ccr_needle", "hotpotqa", "msmarco"}
    }
    assert eligible == {"tool_outputs": 8, "ccr_needle": 8, "hotpotqa": 3, "msmarco": 5}
