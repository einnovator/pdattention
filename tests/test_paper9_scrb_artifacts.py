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


def test_fielded_router_closes_the_small_tracked_source_gap_without_hiding_baselines() -> None:
    baseline = json.loads(
        (RESULTS.parent / "natural_repository_v1" / "summary.json").read_text(encoding="utf-8")
    )
    improved = json.loads(
        (RESULTS.parent / "natural_repository_v2" / "summary.json").read_text(encoding="utf-8")
    )
    assert baseline["routing"]["lexical"]["evidence_recall"] == 0.6
    assert improved["routing"]["lexical"]["evidence_recall"] == 0.6
    assert improved["routing"]["learned"]["evidence_recall"] < improved["routing"][
        "fielded_bm25"
    ]["evidence_recall"]
    assert improved["routing"]["fielded_bm25"]["evidence_recall"] == 1
    assert improved["routing"]["fielded_bm25"]["mean_selected_tokens"] == improved[
        "routing"
    ]["oracle"]["mean_selected_tokens"]
    assert improved["fielded_bm25_protocol"]["model_calls"] == 0
    assert (
        improved["fielded_bm25_protocol"]["development_scope"]
        == "post_hoc_in_cohort_diagnostic"
    )


def test_router_transfer_uses_frozen_external_repositories_and_retains_an_oracle_gap() -> None:
    summary = json.loads(
        (RESULTS.parent / "router_transfer_v1" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["development_examples"] == 11
    assert set(summary["repositories"]) == {"dynaspike", "cognitive_coprocessors"}
    pooled = summary["routing"]["pooled"]
    assert pooled["fielded_bm25"]["queries"] == 16
    assert pooled["lexical"]["top1_recall"] == 0.8125
    assert pooled["fielded_bm25"]["top1_recall"] == 0.875
    assert pooled["oracle"]["top1_recall"] == 1


def test_autonomous_campaign_is_complete_and_keeps_consumption_separate() -> None:
    summary = json.loads(
        (RESULTS.parent / "autonomous_repository_v1" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["seeds"] == [11, 23, 37]
    assert summary["agent_runs"] == 96
    assert summary["execution_failures"] == 0
    assert len(summary["source_hashes"]) == 5
    assert set(summary["conditions"]) == {
        "sequential_isolated",
        "parallel_isolated",
        "sequential_completed",
        "parallel_completed",
    }
    assert all(row["seeds"] == 3 for row in summary["conditions"].values())
    assert summary["paired_summary"]["completed_parallel_speedup"]["mean"] > 1.2
    assert (
        summary["paired_summary"]["sequential_context_physical_call_delta"]["mean"]
        < -12
    )
    assert summary["paired_summary"]["sequential_context_accuracy_delta"]["mean"] < 0


def test_live_mlx_primary_sweep_is_five_seed_and_exact() -> None:
    path = RESULTS.parent / "mlx_live_m4_final" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["seeds"] == [11, 23, 37, 71, 101]
    assert summary["shared_token_targets"] == [512, 2048, 8192]
    assert summary["fanouts"] == [1, 2, 4, 8, 16]
    for condition in (
        "host_split_text",
        "harness_memo_text",
        "pra_record_reprefill",
        "pra_native_kv",
    ):
        result = summary["conditions"][condition]
        assert result["argmax_parity_vs_host_split"] == 1
        assert result["max_abs_logit_delta_vs_host_split"] == 0
    for point in summary["session_economics"]["max_fanout_by_shared_tokens"].values():
        assert point["fanout"] == 16
        assert point["physical_token_reduction_mean"] > 0.90


def test_live_mlx_32k_sweep_exposes_capacity_dependent_economics() -> None:
    m4 = json.loads(
        (RESULTS.parent / "mlx_32_m4_final" / "summary.json").read_text(encoding="utf-8")
    )
    m5 = json.loads(
        (RESULTS.parent / "mlx_32_m5_replication" / "summary.json").read_text(encoding="utf-8")
    )
    assert m4["seeds"] == [11, 23, 37, 71, 101]
    assert m5["seeds"] == [37, 101]
    assert m4["conditions"]["pra_native_kv"]["argmax_parity_vs_host_split"] == 1
    assert m5["conditions"]["pra_native_kv"]["argmax_parity_vs_host_split"] == 1
    m4_point = m4["session_economics"]["max_fanout_by_shared_tokens"]["32768"]
    m5_point = m5["session_economics"]["max_fanout_by_shared_tokens"]["32768"]
    assert m4_point["amortized_speedup_mean"] > 3
    assert m5_point["amortized_speedup_mean"] < 1
    assert m4_point["physical_token_reduction_mean"] > 0.74
    assert m5_point["physical_token_reduction_mean"] > 0.74


def test_live_mlx_generation_parity_is_exact_for_all_five_seeds() -> None:
    path = RESULTS.parent / "mlx_generation_parity_v1" / "generation_parity.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["seeds"] == [11, 23, 37, 71, 101]
    assert summary["generation_tokens"] == 16
    assert summary["exact_sequence_matches"] == 5
    assert summary["max_abs_logit_delta"] == 0
