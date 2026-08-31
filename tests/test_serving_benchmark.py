from __future__ import annotations

import pytest

from pra_hf.serving_benchmark import benchmark_messages, percentile, run_serving_benchmark


def test_benchmark_conditions_separate_prefix_and_pra_memory() -> None:
    conditions = benchmark_messages()
    assert list(conditions) == [
        "no_prefix_no_pra",
        "prefix_only",
        "pra_only",
        "prefix_plus_pra",
        "full_context",
    ]
    assert "PRA_EVIDENCE_4821" not in str(conditions["prefix_only"])
    assert "PRA_EVIDENCE_4821" in str(conditions["pra_only"])
    assert len(str(conditions["full_context"])) > len(str(conditions["prefix_plus_pra"]))


def test_percentile_interpolates_and_handles_empty_input() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_benchmark_distractor_dimensions_are_explicit_and_bounded() -> None:
    small = benchmark_messages(distractor_count=2, distractor_repeat=3)
    default = benchmark_messages()
    assert len(str(small["full_context"])) < len(str(default["full_context"]))
    with pytest.raises(ValueError, match="dimensions"):
        benchmark_messages(distractor_count=0)


def test_serving_benchmark_rejects_single_repeat() -> None:
    with pytest.raises(ValueError, match="At least two"):
        run_serving_benchmark(
            "http://engine", model="model", engine="mlx", repeats=1
        )


def test_serving_benchmark_rejects_nonpositive_decode_budget() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        run_serving_benchmark(
            "http://engine", model="model", engine="mlx", repeats=2, max_tokens=0
        )


def test_serving_benchmark_pairs_tail_and_quality_adjusted_rates(monkeypatch) -> None:
    def fake_request(*args, **kwargs):
        return {
            "ttft_ms": 10.0,
            "completion_latency_ms": 100.0,
            "mean_itl_ms": 5.0,
            "output_events": 2,
            "output_text": "PRA_EVIDENCE_4821",
            "prompt_tokens": 32,
            "completion_tokens": 4,
            "cached_tokens": None,
        }

    monkeypatch.setattr("pra_hf.serving_benchmark.stream_chat_completion", fake_request)
    result = run_serving_benchmark(
        "http://engine",
        model="model",
        engine="mlx",
        repeats=2,
        hardware_metadata={"accelerator": "test"},
    )
    row = result["aggregates"][0]

    assert row["ttft_ms_p99"] == 10.0
    assert row["itl_ms_p95"] == 5.0
    assert row["requests_per_second"] == pytest.approx(10.0)
    assert row["successful_requests_per_second"] == pytest.approx(10.0)
    assert row["successful_tasks_per_accelerator_hour"] == pytest.approx(36_000.0)
    assert row["cache_metric_status"] == "NOT_MEASURED"
    assert result["hardware"]["accelerator"] == "test"
