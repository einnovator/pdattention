"""Regression contracts for the controlled Paper 9 SCRB evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/papers/shared/results/paper9_subagents/scrb_v1"


def test_scrb_scaling_grid_and_conditions_are_complete() -> None:
    with (RESULTS / "scaling_rows.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 * 4 * 4 * 4 * 5
    assert {int(row["shared_tokens"]) for row in rows} == {2048, 8192, 32768, 65536}
    assert {int(row["fanout"]) for row in rows} == {2, 4, 8, 16}
    assert {row["condition"] for row in rows} == {
        "isolated",
        "harness_memoization",
        "pra_no_visibility",
        "pra_payload",
        "pra_native_kv",
    }


def test_native_reuse_reduces_prefill_beyond_memoization() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    target = summary["target_8k_fanout8"]
    assert target["harness_memoization"]["tool_calls"]["mean"] == 1
    assert target["pra_payload"]["prefill_tokens"]["mean"] == 73728
    assert target["pra_native_kv"]["prefill_tokens"]["mean"] == 8192
    assert target["pra_native_kv"]["speedup_vs_isolated"] > 6
    assert "not live model latency" in summary["evidence_scope"]


def test_selective_invalidation_preserves_valid_reuse_without_stale_reads() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    selective = summary["invalidation_fanout16"]["selective_resource"]
    coarse = summary["invalidation_fanout16"]["coarse_domain"]
    unsafe = summary["invalidation_fanout16"]["unsafe_no_invalidation"]
    assert selective["reuse_rate"] == 0.5
    assert selective["stale_reads"] == 0
    assert coarse["false_invalidations"] > 0
    assert unsafe["stale_reads"] > 0
    assert summary["runtime_contract_check"]["stale_reads"] == 0


def test_descendant_routing_recovers_detail_with_bounded_context_growth() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    descendant = summary["descendant_evidence"]
    assert descendant["pra_descendant_route"]["accuracy"] == 1
    assert descendant["summary_only"]["accuracy"] < 0.6
    assert descendant["pra_descendant_route"]["parent_context_growth_tokens"] < (
        descendant["full_transcript_copy"]["parent_context_growth_tokens"] / 40
    )


def test_natural_trace_audit_is_labeled_as_opportunity_not_cross_agent_evidence() -> None:
    summary = json.loads(
        (RESULTS.parent / "natural_trace_opportunity.json").read_text(encoding="utf-8")
    )
    assert summary["trace_count"] == 12
    assert summary["read_search_calls"] == 40
    assert summary["exact_repeated_read_search_calls"] == 8
    assert "not observed cross-agent reuse" in summary["scope"]


def test_natural_repository_cohort_separates_validity_from_bounded_routing() -> None:
    path = RESULTS.parent / "natural_repository_v1" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["seeds"] == [11, 23, 37, 71, 101]
    assert summary["routing"]["all_valid"]["evidence_recall"] == 1
    assert summary["routing"]["oracle"]["evidence_recall"] == 1
    assert summary["routing"]["learned"]["evidence_recall"] == 0.6
    assert summary["routing"]["all_valid"]["mean_selected_tokens"] > (
        summary["routing"]["oracle"]["mean_selected_tokens"] * 10
    )


def test_natural_repository_cohort_executes_both_schedulers_without_duplicate_reads() -> None:
    path = RESULTS.parent / "natural_repository_v1" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    for scheduler in ("sequential", "parallel"):
        row = summary["scheduling"][scheduler]
        assert row["runs"] == 5
        assert row["mean_logical_file_reads"] == 35
        assert row["mean_physical_file_reads"] == 13
        assert row["orientation_reuses"] == 110
        assert row["dag_join_parents_visible"] == 2
        assert row["completed_peer_visible_rate"] == 1
