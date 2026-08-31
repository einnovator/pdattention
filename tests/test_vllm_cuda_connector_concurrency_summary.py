from experiments.paper6_vllm.summarize_cuda_connector_concurrency import (
    render_table,
    summarize,
)


def test_summary_keeps_recovery_and_cross_request_leakage_disjoint() -> None:
    payload = {
        "schema_version": "raw-v1",
        "evidence_tier": "CONTROLLED_CUDA_BATCHED_NATIVE_TRANSFER",
        "integration_status": "E2_CANDIDATE_PREFIX_SHAPED",
        "engine_version": "0.28.0",
        "model_id": "tiny",
        "device": "CUDA",
        "source_tokens": 16,
        "source_content_visible": False,
        "source_slots_scheduler_visible": True,
        "rows": [
            {
                "condition": "mixed_native_ordinary",
                "concurrency": 2,
                "requests": 2,
                "expected_recoveries": 1,
                "expected_requests": 1,
                "forbidden_leaks": 0,
                "requests_per_second": 4.0,
                "output_tokens_per_second": 20.0,
                "completion_ms": 500.0,
                "peak_allocated_bytes": 2**20,
                "peak_reserved_bytes": 2 * 2**20,
            }
        ],
    }

    result = summarize(payload)

    assert result["all_expected_recovered"] is True
    assert result["total_forbidden_leaks"] == 0
    assert result["rows"][0]["recovery_rate"] == 1.0
    assert result["rows"][0]["leakage_rate"] == 0.0
    assert result["rows"][0]["peak_allocated_mib"] == 1.0
    assert "mixed native/ordinary" in render_table(result)
