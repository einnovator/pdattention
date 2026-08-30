from experiments.engine_serving.summarize_mac_engine_extension import (
    _async_rows,
    _nearest,
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
