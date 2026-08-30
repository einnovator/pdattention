from __future__ import annotations

from experiments.paper6_3_openvino.run_genai_e0 import _aggregate, _metric
from pra_hf.serving_benchmark import percentile


class _Stats:
    mean = 12.5


class _Perf:
    def get_ttft(self):
        return _Stats()


def test_openvino_metric_reads_perf_statistics() -> None:
    assert _metric(_Perf(), "get_ttft") == 12.5
    assert _metric(_Perf(), "missing") is None


def test_openvino_aggregate_keeps_cold_warm_and_tails_distinct() -> None:
    rows = [
        {
            "ttft_ms": value,
            "wall_latency_ms": value * 2,
            "expected_answer_present": True,
            "input_tokens": 10,
            "rss_bytes": 100,
        }
        for value in (10.0, 5.0, 6.0, 7.0, 8.0)
    ]
    result = _aggregate("pra_only", rows, percentile=percentile)
    assert result["cold_ttft_ms"] == 10.0
    assert result["warm_ttft_ms_mean"] == 6.5
    assert result["quality_success_rate"] == 1.0
    assert result["ttft_ms_p99"] > result["ttft_ms_p95"]
