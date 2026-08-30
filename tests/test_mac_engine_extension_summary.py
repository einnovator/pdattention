from experiments.engine_serving.summarize_mac_engine_extension import (
    _async_rows,
    _nearest,
    _pressure_rows,
    _tier_window_rows,
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


def test_pressure_summary_keeps_final_revisit_separate() -> None:
    rows = _pressure_rows(
        [
            {
                "dataset": "qasper",
                "resources_per_seed": 8,
                "session_rounds": 3,
                "resident_resource_budgets": [2],
                "rows": [
                    {
                        "resident_resource_budget": 2,
                        "final_revisit": False,
                        "reload_on_request": False,
                        "token_f1": 0.5,
                        "gold_answer_logprob": -2.0,
                        "resolve_ms": 1.0,
                        "completion_latency_ms": 10.0,
                    },
                    {
                        "resident_resource_budget": 2,
                        "final_revisit": True,
                        "reload_on_request": True,
                        "token_f1": 1.0,
                        "gold_answer_logprob": -1.0,
                        "resolve_ms": 2.0,
                        "completion_latency_ms": 20.0,
                    },
                ],
                "seed_summaries": [
                    {
                        "resident_resource_budget": 2,
                        "reloads": 4,
                        "evictions": 3,
                    }
                ],
            }
        ]
    )

    assert rows[0]["final_revisit_reload_rate"] == 1.0
    assert rows[0]["token_f1"] == 0.75
    assert rows[0]["completion_p95_ms"] == 20.0


def test_tier_window_summary_keeps_physical_start_tiers_disjoint() -> None:
    rows = _tier_window_rows(
        {
            "hot_resource_budgets": [2],
            "warm_resource_budgets": [4],
            "local_kv_sizes": [64],
            "rows": [
                {
                    "hot_resource_budget": 2,
                    "warm_resource_budget": 4,
                    "local_kv_size": 64,
                    "tier_before": tier,
                    "token_f1": 1.0,
                    "resolve_ms": latency,
                    "completion_latency_ms": 10.0,
                }
                for tier, latency in (("hot", 1.0), ("warm", 2.0), ("source", 3.0))
            ],
        }
    )

    assert rows[0]["hot_start_rate"] == 1 / 3
    assert rows[0]["warm_start_rate"] == 1 / 3
    assert rows[0]["source_start_rate"] == 1 / 3
    assert rows[0]["resolve_p95_ms"] == 3.0
