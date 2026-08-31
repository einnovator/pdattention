from experiments.paper6_2_mlx.summarize_m4_pressure import summarize


def test_pressure_summary_preserves_quality_and_reports_reload_threshold() -> None:
    payload = {
        "dataset": "qasper",
        "model_id": "mlx-community/Qwen3-4B-4bit",
        "resources_per_seed": 8,
        "rows": [
            {
                "resident_resource_budget": budget,
                "seed": seed,
                "token_f1": 0.25,
                "gold_answer_logprob": -4.0,
                "reload_on_request": budget < 8,
                "resolve_ms": 10.0 if budget < 8 else 2.0,
                "completion_latency_ms": 5.0,
                "resident_bytes_after_request": budget * 1048576,
            }
            for budget in (1, 8)
            for seed in (11, 23)
        ],
        "seed_summaries": [
            {
                "resident_resource_budget": budget,
                "loads": 2 if budget < 8 else 1,
                "evictions": 1 if budget < 8 else 0,
                "reloads": 1 if budget < 8 else 0,
            }
            for budget in (1, 8)
            for _ in (11, 23)
        ],
    }

    rows = summarize([payload])

    assert [row["resident_resource_budget"] for row in rows] == [1, 8]
    assert rows[0]["reload_fraction"] == 1.0
    assert rows[1]["reload_fraction"] == 0.0
    assert rows[0]["token_f1"] == rows[1]["token_f1"] == 0.25
