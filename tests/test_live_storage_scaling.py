from experiments.engine_serving.summarize_live_storage_scaling import (
    _artifact_row,
    _percentile,
)


def test_percentile_interpolates_finite_cohort() -> None:
    assert _percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert _percentile([10.0, 20.0], 0.95) == 19.5


def test_artifact_row_keeps_quality_and_latency_disjoint() -> None:
    rows = []
    for index, warm_exact in enumerate((True, False)):
        rows.append(
            {
                "dataset": "qasper",
                "native_bytes": 100 + index,
                "hot_warm_exact": warm_exact,
                "hot_cold_int8_exact": False,
                "hot_cold_first_token_equal": index == 0,
                "hot_cold_common_prefix_tokens": index + 1,
                "cold_int8_f1_delta": 0.1 * index,
                "lifecycle_request_latency_ms": {
                    "hot": 10 + index,
                    "warm": 20 + index,
                    "cold": 30 + index,
                },
                "background_transition_latency_ms": {
                    "hot": 0,
                    "warm": 5 + index,
                    "cold": 15 + index,
                },
            }
        )
    result = _artifact_row(
        {
            "engine": "mlx-lm",
            "model_id": "test/model",
            "rows": rows,
            "summary": {"restart_recovered": True},
        }
    )

    assert result["examples"] == 2
    assert result["warm_exact_rate"] == 0.5
    assert result["int8_exact_rate"] == 0.0
    assert result["int8_first_token_rate"] == 0.5
    assert result["request_warm_p50_ms"] == 20.5
    assert result["request_warm_p95_ms"] == 20.95
    assert result["restart_recovered"] is True
