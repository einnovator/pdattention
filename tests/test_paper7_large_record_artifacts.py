from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/large_record_hybrid"


def _rows(name: str):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_required_large_record_artifacts_are_nonempty():
    names = (
        "large_record_index_policy_matrix.csv",
        "large_record_hybrid_results.csv",
        "large_record_channel_overlap.csv",
        "large_record_index_costs.csv",
        "type_aware_compression_results.csv",
        "type_aware_compression_retention.csv",
        "compression_recovery_decomposition.csv",
        "headroom_pra_cost_frontier.csv",
        "headroom_reverse_eval_hybrid.csv",
    )
    assert all(_rows(name) for name in names)
    assert (OUTPUT / "generated_large_record_hybrid_results.tex").is_file()
    assert (OUTPUT / "figures/large_record_hybrid_recall.pdf").is_file()
    assert (OUTPUT / "figures/headroom_pra_multi_axis_cost.pdf").is_file()


def test_policy_matrix_keeps_cheap_indexes_when_native_is_gated():
    by_state = {row["state"]: row for row in _rows("large_record_index_policy_matrix.csv")}
    skipped = by_state["SKIPPED_SIZE_LIMIT"]
    assert skipped["typed_index"] == "BUILT"
    assert skipped["bm25_index"] == "BUILT"
    assert skipped["embedding_index"] == "BUILT"
    assert skipped["native_qk_index"] == "SKIPPED_SIZE_LIMIT"


def test_reverse_eval_reports_all_four_datasets_and_channels():
    rows = _rows("headroom_reverse_eval_hybrid.csv")
    assert {row["dataset"] for row in rows} == {
        "tool_outputs", "ccr_needle", "hotpotqa", "msmarco"
    }
    assert {row["channel"] for row in rows} == {
        "typed", "bm25", "embedding", "hybrid"
    }


def test_type_aware_budget_and_recovery_are_separate_axes():
    frontier = {row["condition"]: row for row in _rows("headroom_pra_cost_frontier.csv")}
    aware = frontier["PRA_TYPE_AWARE_BM25_EMBED"]
    assert float(aware["initial_visible_tokens"]) <= 96
    assert float(aware["selected_region_tokens"]) > 0
    assert float(aware["cheap_index_bytes"]) > 0
    assert "separate axes" in aware["cost_axes_note"]
