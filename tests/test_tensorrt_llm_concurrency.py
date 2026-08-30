from __future__ import annotations

from experiments.paper6_4_tensorrt_llm.run_concurrency import _aggregate
from pra_hf import serving_benchmark


def test_concurrency_aggregate_reports_tail_and_throughput() -> None:
    rows = [
        {
            "quality_ok": True,
            "ttft_ms": value,
            "completion_latency_ms": value * 2,
            "completion_tokens": 10,
            "cached_tokens": 20,
        }
        for value in (10.0, 20.0, 30.0, 40.0, 50.0)
    ]
    result = _aggregate(serving_benchmark, rows, 2.0)
    assert result["request_throughput_s"] == 2.5
    assert result["output_throughput_tokens_s"] == 25.0
    assert result["quality_success_rate"] == 1.0
    assert result["ttft_ms_p99"] > result["ttft_ms_p95"]
