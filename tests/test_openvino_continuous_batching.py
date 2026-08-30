from __future__ import annotations

from experiments.paper6_3_openvino.run_continuous_batching import _aggregate
from pra_hf import serving_benchmark


def test_continuous_batching_aggregate_reports_tails_and_throughput() -> None:
    rows = [
        {
            "quality_ok": True,
            "ttft_ms": value,
            "generation_ms": value * 2,
            "output_tokens": 10,
        }
        for value in (10.0, 20.0, 30.0, 40.0, 50.0)
    ]
    result = _aggregate(serving_benchmark, rows, 2.0)
    assert result["request_throughput_s"] == 2.5
    assert result["output_throughput_tokens_s"] == 25.0
    assert result["quality_success_rate"] == 1.0
    assert result["ttft_ms_p99"] > result["ttft_ms_p95"]
