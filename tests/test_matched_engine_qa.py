from experiments.engine_serving.matched_qa import (
    manifest_entries_from_rows,
    selection_identity,
    selected_source,
    source_digest,
)
from experiments.paper6_2_mlx.run_answer_quality_pressure import QADocument, QAExample
from experiments.engine_serving.summarize_matched_e0_e2 import _percentile
from experiments.engine_serving.summarize_matched_e0_e2 import (
    _aggregate,
    _normalize,
    _parity,
)
from experiments.engine_serving.matched_e0_e2_contract import (
    SCHEMA_VERSION,
    benchmark_metrics,
    benchmark_row,
    regime_schedule,
)


def _example() -> QAExample:
    return QAExample(
        dataset="fixture",
        example_id="example-1",
        question="Where is the answer?",
        answer="Second",
        source="",
        source_scope="fixture",
        documents=(
            QADocument("first", "First", "First body."),
            QADocument("second", "Second", "Second body."),
        ),
        evidence_document_ids=frozenset(("second",)),
    )


def test_selected_source_preserves_frozen_rank_order():
    source = selected_source(_example(), ("second", "first"))
    assert source == "Document: Second\nSecond body.\n\nDocument: First\nFirst body."
    assert source_digest(source) == source_digest(source)


def test_manifest_entries_keep_one_native_row_per_example():
    example = _example()
    rows = [
        {
            "condition": "routed_native",
            "seed": 11,
            "example_id": example.example_id,
            "selected_document_ids": ["second", "first"],
            "evidence_recall_at_4": 1.0,
        },
        {
            "condition": "routed_native",
            "seed": 11,
            "example_id": example.example_id,
            "selected_document_ids": ["second", "first"],
            "evidence_recall_at_4": 1.0,
        },
        {
            "condition": "no_memory",
            "seed": 11,
            "example_id": example.example_id,
            "selected_document_ids": ["first"],
            "evidence_recall_at_4": 0.0,
        },
    ]
    [entry] = manifest_entries_from_rows((example,), rows)
    assert entry["selected_document_ids"] == ["second", "first"]
    assert entry["candidate_document_ids"] == ["first", "second"]
    assert [interval["document_id"] for interval in entry["selected_intervals"]] == [
        "second",
        "first",
    ]
    assert len(entry["candidate_set_sha256"]) == 64
    assert len(entry["selection_id"]) == 64
    assert entry["seed"] == 11
    assert entry["selected_source_characters"] > 0


def test_selected_source_rejects_unknown_document():
    try:
        selected_source(_example(), ("missing",))
    except ValueError as error:
        assert "Unknown selected document" in str(error)
    else:
        raise AssertionError("unknown selected document should fail")


def test_selection_identity_freezes_full_document_intervals():
    identity = selection_identity(_example(), ("second", "first"))
    assert identity.candidate_document_ids == ("first", "second")
    assert identity.selected_document_ids == ("second", "first")
    assert [(item.start, item.end) for item in identity.selected_intervals] == [
        (0, len("Second body.")),
        (0, len("First body.")),
    ]


def test_matched_summary_percentile_interpolates_small_cohort():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.8499999999999996
    assert _percentile([], 0.99) is None


def test_v2_summary_preserves_regimes_and_paired_identity(tmp_path):
    identity = selection_identity(_example(), ("second",))
    requests = regime_schedule(
        _example().question, warm_repeats=1, multi_query_count=1, concurrency=1
    )
    rows = []
    for request in requests:
        for condition in ("e0_selected_text", "e2_native_kv"):
            native = condition == "e2_native_kv"
            rows.append(
                benchmark_row(
                    condition=condition,
                    selection=identity,
                    request=request,
                    output="Second",
                    metrics=benchmark_metrics(
                        exact_match=1.0,
                        token_f1=1.0,
                        gold_answer_logprob=-0.5,
                        evidence_recall=1.0,
                        candidate_tokens=20,
                        selected_source_tokens=10,
                        visible_prompt_tokens=5 if native else 15,
                        selected_native_kv_tokens=10 if native else 0,
                        active_detail_bytes=100 if native else 0,
                        retained_detail_bytes=100 if native else 0,
                        text_preparation_ms=1.0,
                        kv_encode_ms=2.0,
                        index_construction_ms=0.0,
                        time_to_usable_context_ms=3.0,
                        ttft_ms=4.0,
                        itl_ms=1.0,
                        total_latency_ms=10.0,
                        generated_tokens=2,
                        ordinary_prefix_cache_hit_tokens=0,
                        pra_hot_hit=native and request.regime != "cold_one_shot",
                        pra_warm_hit=False,
                        bytes_read=100 if native else 0,
                        bytes_promoted=0,
                        bytes_avoided=100 if native else 0,
                        duplicate_physical_kv_avoided_bytes=100 if native else 0,
                    ),
                    extra={"dataset": "fixture"},
                )
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine": "fixture-engine",
        "model_id": "fixture-model",
        "rows": rows,
    }
    normalized = _normalize(payload, tmp_path / "fixture.json")
    aggregates = _aggregate(normalized)
    parity = _parity(normalized)
    assert {row["regime"] for row in aggregates} == {
        "cold_one_shot",
        "warm_repeated",
        "multi_query_same_resource",
        "concurrent_shared_resource",
    }
    assert len(parity) == 4
    assert all(row["exact_output_parity"] == 1.0 for row in parity)
