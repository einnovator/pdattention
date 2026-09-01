from experiments.paper6_3_openvino.run_genai_e0 import _aggregate


def _percentile(values, probability):
    return sorted(values)[round((len(values) - 1) * probability)]


def test_cache_admission_failures_remain_visible_without_fake_zero_metrics():
    rows = [
        {
            "measurement_status": "NOT_RUN_CACHE_ADMISSION",
            "error": "request did not fit in the available cache budget",
        },
        {
            "measurement_status": "NOT_RUN_CACHE_ADMISSION",
            "error": "request did not fit in the available cache budget",
        },
    ]
    result = _aggregate("selected_text_e0", rows, percentile=_percentile)
    assert result["sample_count"] == 0
    assert result["requested_sample_count"] == 2
    assert result["cache_admission_failures"] == 2
    assert result["ttft_ms_p50"] is None
    assert result["quality_success_rate"] is None


def test_cache_aggregate_excludes_failed_requests_from_latency():
    rows = [
        {
            "measurement_status": "MEASURED",
            "expected_answer_present": True,
            "ttft_ms": 10.0,
            "wall_latency_ms": 20.0,
            "input_tokens": 30,
            "rss_bytes": 40,
        },
        {
            "measurement_status": "NOT_RUN_CACHE_ADMISSION",
            "error": "request did not fit in the available cache budget",
        },
    ]
    result = _aggregate("selected_text_e0", rows, percentile=_percentile)
    assert result["sample_count"] == 1
    assert result["cache_admission_failures"] == 1
    assert result["ttft_ms_p50"] == 10.0
