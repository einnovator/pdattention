from __future__ import annotations

from experiments.engine_serving.summarize_cross_engine_context_pressure import (
    _paired_rows,
    _table,
)


def test_pairing_computes_within_engine_ratios() -> None:
    payload = {
        "rows": [
            {
                "size": "Large",
                "workload": "shared_resource",
                "representation": "pra_only",
                "mean_prompt_tokens": 64,
                "quality_success_rate": 1.0,
                "request_throughput_s": 80.0,
                "ttft_ms_p99": 50.0,
            },
            {
                "size": "Large",
                "workload": "shared_resource",
                "representation": "full_context",
                "mean_prompt_tokens": 1024,
                "quality_success_rate": 1.0,
                "request_throughput_s": 40.0,
                "ttft_ms_p99": 150.0,
            },
        ]
    }
    row = _paired_rows(payload)[0]
    assert row["throughput_ratio"] == 2.0
    assert row["ttft_ratio"] == 3.0
    assert "2.00$\\times$" in _table([{"engine": "Test", **row}])
