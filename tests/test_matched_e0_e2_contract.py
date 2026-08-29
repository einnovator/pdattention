from __future__ import annotations

import pytest

from experiments.engine_serving.matched_e0_e2_contract import (
    CONDITIONS,
    METRIC_FAMILIES,
    REGIMES,
    FrozenSelectionIdentity,
    FrozenSourceInterval,
    benchmark_row,
    regime_schedule,
    validate_payload,
)


def _selection() -> FrozenSelectionIdentity:
    return FrozenSelectionIdentity(
        dataset="qasper",
        example_id="paper-question",
        candidate_document_ids=("a", "b", "c"),
        selected_document_ids=("b",),
        selected_intervals=(FrozenSourceInterval("b", 0, 12),),
        selected_source_sha256="a" * 64,
    )


def _metrics() -> dict[str, dict[str, object]]:
    return {
        "quality": {
            "exact_match": 1.0,
            "token_f1": 1.0,
            "task_score": 1.0,
            "gold_answer_logprob": None,
            "evidence_recall": 1.0,
        },
        "input": {
            "candidate_tokens": 100,
            "selected_source_tokens": 20,
            "visible_prompt_tokens": 8,
        },
        "pra": {
            "selected_native_kv_tokens": 20,
            "active_detail_bytes": 1024,
            "retained_detail_bytes": 1024,
        },
        "ingestion": {
            "text_preparation_ms": 0.1,
            "kv_encode_ms": 2.0,
            "index_construction_ms": 0.2,
            "time_to_usable_context_ms": 2.3,
        },
        "serving": {
            "ttft_ms": 3.0,
            "itl_ms": 1.0,
            "tpot_ms": 1.0,
            "total_latency_ms": 5.0,
            "generated_tokens": 3,
            "tokens_per_second": 600.0,
            "requests_per_second": None,
        },
        "reuse": {
            "ordinary_prefix_cache_hit_tokens": 0,
            "pra_hot_hit": False,
            "pra_warm_hit": False,
            "bytes_read": 1024,
            "bytes_promoted": 1024,
            "bytes_avoided": 0,
            "duplicate_physical_kv_avoided_bytes": 0,
        },
    }


def test_schedule_covers_four_regimes_and_reuses_query_variants() -> None:
    schedule = regime_schedule("What is the answer?", concurrency=4)
    assert {request.regime for request in schedule} == set(REGIMES)
    assert sum(request.regime == "cold_one_shot" for request in schedule) == 1
    assert sum(request.regime == "warm_repeated" for request in schedule) == 2
    assert sum(request.regime == "multi_query_same_resource" for request in schedule) == 3
    assert sum(request.regime == "concurrent_shared_resource" for request in schedule) == 4


def test_selection_identity_covers_candidates_ordered_intervals_and_source() -> None:
    selection = _selection()
    serialized = selection.to_dict()
    assert len(serialized["candidate_set_sha256"]) == 64
    assert len(serialized["selection_id"]) == 64
    assert serialized["selected_intervals"][0]["coordinate_space"] == (
        "unicode_codepoint"
    )


def test_payload_requires_every_condition_for_every_scheduled_request() -> None:
    selection = _selection()
    rows = []
    for request in regime_schedule("What is the answer?", concurrency=2):
        for condition in CONDITIONS:
            rows.append(
                benchmark_row(
                    condition=condition,
                    selection=selection,
                    request=request,
                    output="answer",
                    metrics=_metrics(),
                )
            )
    payload = {"schema_version": "2.0", "rows": rows}
    validate_payload(payload)
    assert set(rows[0]["metrics"]) == set(METRIC_FAMILIES)

    payload["rows"] = rows[:-1]
    with pytest.raises(ValueError, match="unmatched"):
        validate_payload(payload)


def test_metric_families_require_explicit_unavailable_values() -> None:
    metrics = _metrics()
    del metrics["reuse"]["bytes_read"]
    with pytest.raises(ValueError, match="bytes_read"):
        benchmark_row(
            condition="e2_native_kv",
            selection=_selection(),
            request=regime_schedule("Question?", concurrency=1)[0],
            output="answer",
            metrics=metrics,
        )
