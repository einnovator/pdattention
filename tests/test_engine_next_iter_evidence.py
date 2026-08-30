from pathlib import Path

from experiments.engine_serving.build_next_iter_evidence import _mlx, _sglang, _vllm
from experiments.engine_serving.run_platform_gate_audit import audit


RESULTS = Path("docs/papers/shared/results")


def test_next_iter_synthesis_keeps_unmeasured_gates_explicit() -> None:
    platform = audit()
    mlx = _mlx(RESULTS)
    sglang = _sglang(RESULTS, platform)
    vllm = _vllm(RESULTS, platform)

    assert mlx["staged_pareto_status"]["status"] == "PARTIAL_STAGED_FRONTIER"
    assert 32768 in mlx["staged_pareto_status"]["required_followup_windows"]
    assert mlx["segmented_attention"]["model_runner_fused"] is False
    assert mlx["segmented_attention"]["one_attention_normalization"] is True
    row_32k = next(
        row
        for row in mlx["segmented_attention"]["rows"]
        if row["local_tokens"] == 32768
    )
    assert row_32k["kv_concat_temporary_bytes_avoided"] > 128 * 1024 * 1024
    assert sglang["off_node_claim_allowed"] is False
    assert vllm["production_cuda_claim_allowed"] is False
    assert all(
        row["argmax_parity_rate"] == 1.0
        for row in mlx["consumer_profiles"]
        if row["profile"] == "all"
    )
