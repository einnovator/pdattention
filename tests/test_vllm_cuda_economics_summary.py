from __future__ import annotations

from experiments.paper6_vllm.run_cuda_economics_matrix import percentile
from experiments.paper6_vllm.summarize_cuda_economics import summarize


def test_percentile_interpolates_finite_sample() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.99) is None


def test_summary_preserves_candidate_boundary_and_disjoint_metrics() -> None:
    row = {
        "condition": "e2_hot",
        "requests": 100,
        "success_rate": 1.0,
        "successful_requests_per_second": 48.9,
        "ttft_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
        "mean_itl_ms": {"p50": 2.0, "p95": 3.0, "p99": 4.0},
        "peak_allocated_bytes": 100 * 2**20,
        "apc_blocks_mean": 0.0,
        "pra_logical_blocks": 2,
        "pra_hot_source_bytes": 4 * 2**20,
        "pra_warm_persisted_bytes": 0,
        "h2d_bytes_per_request": 0,
        "d2d_bytes_per_request": 4 * 2**20,
        "tail_status": "MEASURED",
    }
    payload = {
        "schema_version": "paper6-vllm-cuda-economics-v1",
        "evidence_tier": "CUDA_MATCHED_CONNECTOR_CANDIDATE",
        "integration_status": "E2_CANDIDATE_PREFIX_SHAPED",
        "engine_version": "0.28.0",
        "model_id": "tiny",
        "device": "cuda",
        "selector_frozen": True,
        "source_slots_scheduler_visible_in_e2": True,
        "concurrency": 8,
        "requests_per_condition": 100,
        "hbm_decomposition": {},
        "rows": [row],
        "limitations": ["prefix-shaped"],
    }

    result = summarize(payload)

    assert result["integration_status"] == "E2_CANDIDATE_PREFIX_SHAPED"
    assert result["rows"][0]["h2d_mib_per_request"] == 0.0
    assert result["rows"][0]["d2d_mib_per_request"] == 4.0
    assert result["rows"][0]["pra_hot_source_mib"] == 4.0
