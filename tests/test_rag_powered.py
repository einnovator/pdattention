from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiments.rag_vs_pra.analyze_powered_decomposition import (
    _normalize_failure_classes,
    _paper_quality_table,
    _paper_runtime_table,
    _paper_table,
)
from pra_hf.rag_evaluation import ContextCondition
from pra_hf.rag_powered import (
    bootstrap_mean_ci,
    official_multihop_rag_score,
    paired_delta,
    percentile,
    qualification_gates,
    summarize_rows,
    validate_selector_frozen_rows,
    validate_strong_reranker_parity,
    write_results,
)


def _row(condition: ContextCondition, receipt: str, *, regime: str = "COLD"):
    return {
        "example_id": "q1",
        "condition": condition.value,
        "selector_profile": "pra_generic",
        "candidate_count": 20,
        "token_budget": 2048,
        "regime": regime,
        "status": "MEASURED",
        "selection_receipt_id": receipt,
        "exact_match": 1.0,
        "token_f1": 0.8,
        "official_multihop_rag_score": 1.0,
        "failure_class": "SUCCESS",
        "retrieval_context_metrics": {
            "supporting_document_coverage": 1.0,
            "materialization_avoidance": 0.5,
        },
        "serving_metrics": {
            "ttft_ms": 10.0,
            "itl_ms": 2.0,
            "total_latency_ms": 20.0,
            "native_reuse": float(regime == "WARM"),
        },
    }


def test_official_score_matches_upstream_token_intersection() -> None:
    assert official_multihop_rag_score("The answer is Lisbon", ("Lisbon",)) == 1.0
    assert official_multihop_rag_score("Oslo", ("Lisbon",)) == 0.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    low, high = bootstrap_mean_ci([0.0, 1.0], seed=7, samples=200)
    assert low == 0.0
    assert high == 1.0


def test_selector_freeze_validation_and_paired_delta() -> None:
    rows = [
        _row(ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR, "same"),
        _row(ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR, "same"),
    ]
    validate_selector_frozen_rows(rows)
    summary = summarize_rows(rows)
    assert len(summary) == 2
    delta = paired_delta(
        rows,
        left_condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value,
        right_condition=ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value,
        metric="token_f1",
        selector_profile="pra_generic",
        regime="COLD",
    )
    assert delta["paired_examples"] == 1
    assert delta["mean_delta"] == 0.0

    rows[1]["selection_receipt_id"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        validate_selector_frozen_rows(rows)


def test_cold_and_warm_are_never_aggregated_together() -> None:
    rows = [
        _row(ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR, "cold", regime="COLD"),
        _row(ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR, "cold", regime="COLD"),
        _row(ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR, "warm", regime="WARM"),
        _row(ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR, "warm", regime="WARM"),
    ]
    summaries = summarize_rows(rows)
    assert {row["regime"] for row in summaries} == {"COLD", "WARM"}
    assert all(row["examples"] == 1 for row in summaries)


def test_bundle_and_card_gate_stays_closed_without_qualified_adapter() -> None:
    summaries = summarize_rows(
        [
            _row(ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR, "same"),
            _row(ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR, "same"),
        ]
    )
    gates = qualification_gates(summaries, minimum_examples=1)
    assert gates["bundle_gate"] == "NO_QUALIFIED_ADAPTER"
    assert gates["card_gate"] == "FAILED_OR_CANDIDATE_ONLY"


def test_condition_results_jsonl_is_compressed_and_roundtrips(tmp_path: Path) -> None:
    rows = [_row(ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR, "same")]
    path = tmp_path / "condition_results.jsonl.gz"
    write_results(path, rows)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["example_id"] == "q1"


def test_failure_normalization_and_paper_table_generation() -> None:
    selected = _row(ContextCondition.NO_PRA_STANDARD_RAG, "baseline")
    selected.update(
        {
            "selector_profile": "standard_bm25",
            "gold_document_ids": ["gold-a", "gold-b"],
            "candidate_document_ids": ["gold-a", "gold-b", "noise"],
            "selected_document_ids": ["gold-a", "noise"],
            "failure_class": "GENERATION_FAILURE",
        }
    )
    _normalize_failure_classes([selected])
    assert selected["failure_class"] == "STANDARD_RAG_PACKING_MISS"
    table = _paper_table(summarize_rows([selected]))
    assert "standard\\_bm25" in table
    assert "Token F1" not in table
    assert "Answer avail." in _paper_quality_table(summarize_rows([selected]))
    assert "TTFT p95" in _paper_runtime_table([])


def test_strong_reranker_visible_path_requires_exact_output_parity() -> None:
    standard = _row(ContextCondition.NO_PRA_STANDARD_RAG, "shared")
    standard.update(
        {
            "selector_profile": "strong_conventional_reranker",
            "prediction": "Lisbon",
        }
    )
    selected = _row(ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR, "shared")
    selected.update({"selector_profile": "pra_strong_reranker", "prediction": "Lisbon"})
    validate_strong_reranker_parity([standard, selected])

    selected["prediction"] = "Oslo"
    with pytest.raises(ValueError, match="output mismatch"):
        validate_strong_reranker_parity([standard, selected])
