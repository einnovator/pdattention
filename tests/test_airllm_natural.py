from __future__ import annotations

import pytest

from experiments.paper6_6_airllm.run_cuda_natural import (
    TokenTimingStreamer,
    _summary,
    quality,
    select_entries,
)
from experiments.paper6_6_airllm.summarize_cuda_natural import render_table, summarize
from experiments.paper6_6_airllm.summarize_results import cuda_product_row


def test_airllm_natural_quality_is_bounded_and_token_aware() -> None:
    exact, f1, containment = quality("The answer is Cyan-Orbit-47.", "CYAN ORBIT 47")

    assert exact == 0.0
    assert 0.0 < f1 < 1.0
    assert containment == 1.0


def test_airllm_natural_cohort_is_bounded_per_dataset() -> None:
    manifest = {
        "entries": [
            {"dataset": "qasper", "example_id": "q1"},
            {"dataset": "qasper", "example_id": "q2"},
            {"dataset": "hotpotqa", "example_id": "h1"},
        ]
    }

    selected = select_entries(manifest, ("qasper", "hotpotqa"), 1)

    assert [entry["example_id"] for entry in selected] == ["q1", "h1"]


def test_token_timing_streamer_excludes_prompt_and_measures_decode() -> None:
    times = iter((1.125, 1.145, 1.185))
    streamer = TokenTimingStreamer(started_at=1.0, clock=lambda: next(times))

    streamer.put([11, 12, 13])
    streamer.put(14)
    streamer.put(15)
    streamer.put(16)

    assert streamer.metrics() == {
        "ttft_ms": pytest.approx(125.0),
        "itl_ms": pytest.approx(30.0),
        "timed_output_tokens": 3,
    }


def test_airllm_summary_keeps_optional_token_timing_explicit() -> None:
    base = {
        "dataset": "qasper",
        "condition": "selected_text_e0",
        "regime": "cold_one_shot",
        "exact_match": 0.0,
        "token_f1": 0.25,
        "answer_containment": 0.0,
        "completion_seconds": 1.0,
        "visible_prompt_tokens": 20,
        "selected_native_kv_tokens": 0,
        "peak_cuda_bytes": 100,
    }

    [measured] = _summary(
        [{**base, "ttft_ms": 100.0, "itl_ms": 20.0}, {**base, "ttft_ms": 120.0, "itl_ms": 30.0}]
    )
    [legacy] = _summary([base])

    assert measured["mean_ttft_ms"] == 110.0
    assert measured["mean_itl_ms"] == 25.0
    assert measured["ttft_ms"] == {
        "p50": 110.0,
        "p95": pytest.approx(119.0),
        "p99": pytest.approx(119.8),
    }
    assert measured["completion_seconds"]["p95"] == 1.0
    assert legacy["mean_ttft_ms"] is None
    assert legacy["mean_itl_ms"] is None
    assert legacy["ttft_ms"] is None


def test_airllm_cuda_smoke_cannot_promote_product_row_to_e2() -> None:
    row = cuda_product_row(
        {
            "device": "CUDA",
            "model_id": "tiny",
            "evidence_tier": "LIVE_ENGINE_MECHANISM_SMOKE",
            "status": "COMPLETE",
            "disabled_exact_sequence": True,
        },
        "fallback",
    )

    assert row["pra_level"] == "E1"
    assert row["status"] == "RESEARCH_ONLY"
    assert row["run_status"] == "COMPLETE"
    assert row["native_e2_candidate"] is True
    assert row["semantic_parity_gate"] == "OPEN"


def test_airllm_natural_summary_compares_matched_e0_e2() -> None:
    aggregate = {
        "dataset": "qasper",
        "regime": "cold_one_shot",
        "samples": 2,
        "mean_token_f1": 0.25,
        "mean_answer_containment": 0.5,
        "mean_completion_seconds": 10.0,
        "mean_visible_prompt_tokens": 400.0,
        "mean_native_kv_tokens": 0.0,
        "peak_cuda_bytes": 100,
    }
    payload = {
        "status": "COMPLETE",
        "schema_version": "raw-v1",
        "selector_frozen": True,
        "aggregates": [
            {**aggregate, "condition": "selected_text_e0"},
            {
                **aggregate,
                "condition": "native_pra_e2",
                "mean_token_f1": 0.2,
                "mean_completion_seconds": 11.0,
                "mean_visible_prompt_tokens": 40.0,
                "mean_native_kv_tokens": 384.0,
                "peak_cuda_bytes": 120,
            },
        ],
        "rows": [
            {
                "dataset": "qasper",
                "example_id": "a",
                "repeat": 0,
                "condition": "selected_text_e0",
                "output_text": "alpha",
            },
            {
                "dataset": "qasper",
                "example_id": "a",
                "repeat": 0,
                "condition": "native_pra_e2",
                "regime": "cold_one_shot",
                "reference_encode_seconds": 4.0,
                "output_text": "beta",
            }
        ],
    }

    result = summarize(payload)

    [comparison] = result["comparisons"]
    assert comparison["e2_over_e0_completion"] == 1.1
    assert comparison["visible_token_reduction"] == 0.9
    assert comparison["token_f1_delta"] == pytest.approx(-0.05)
    assert comparison["exact_output_pair_parity"] == 0.0
    assert comparison["e2_over_e0_ttft"] is None
    assert "QASPER" in render_table(result)


def test_airllm_natural_summary_preserves_token_latency_axes() -> None:
    aggregate = {
        "dataset": "qasper",
        "regime": "cold_one_shot",
        "samples": 1,
        "mean_token_f1": 0.25,
        "mean_answer_containment": 0.5,
        "mean_completion_seconds": 1.0,
        "mean_visible_prompt_tokens": 20.0,
        "mean_native_kv_tokens": 0.0,
        "peak_cuda_bytes": 100,
        "mean_ttft_ms": 100.0,
        "mean_itl_ms": 20.0,
    }
    payload = {
        "status": "COMPLETE",
        "aggregates": [
            {**aggregate, "condition": "selected_text_e0"},
            {
                **aggregate,
                "condition": "native_pra_e2",
                "mean_ttft_ms": 120.0,
                "mean_itl_ms": 25.0,
            },
        ],
        "rows": [],
    }

    [comparison] = summarize(payload)["comparisons"]

    assert comparison["e2_over_e0_ttft"] == 1.2
    assert comparison["e0_itl_ms"] == 20.0
    assert comparison["e2_itl_ms"] == 25.0
