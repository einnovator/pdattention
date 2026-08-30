from experiments.engine_serving.summarize_mac_engine_extension import (
    _async_rows,
    _nearest,
    _write_concurrency_table,
)


def test_async_summary_separates_ready_rate_from_demand_stall() -> None:
    rows = _async_rows(
        [
            {
                "engine": "mlx",
                "rows": [
                    {
                        "lead_ms": 100,
                        "ready_at_demand": False,
                        "output_exact": True,
                        "demand_stall_ms": 20.0,
                        "demand_to_hot_ratio": 1.2,
                        "prefetch_to_completion_ms": 120.0,
                    },
                    {
                        "lead_ms": 100,
                        "ready_at_demand": True,
                        "output_exact": True,
                        "demand_stall_ms": 0.0,
                        "demand_to_hot_ratio": 1.0,
                        "prefetch_to_completion_ms": 130.0,
                    },
                ],
            }
        ]
    )

    assert rows[0]["ready_rate"] == 0.5
    assert rows[0]["exact_rate"] == 1.0
    assert rows[0]["demand_stall_p95_ms"] == 20.0
    assert _nearest([3.0, 1.0, 2.0], 0.50) == 2.0


def test_concurrency_table_reports_model_queue(tmp_path) -> None:
    output = tmp_path / "concurrency.tex"
    _write_concurrency_table(
        output,
        [
            {
                "workload": "shared_resource",
                "tier": "warm",
                "concurrency": 4,
                "requests_per_second": 2.0,
                "request_p50_ms": 10.0,
                "request_p95_ms": 20.0,
                "request_p99_ms": 30.0,
                "model_queue_p95_ms": 42.0,
                "exact_vs_hot_rate": 1.0,
            }
        ],
    )

    rendered = output.read_text(encoding="utf-8")
    assert "queue p95" in rendered
    assert "42" in rendered
