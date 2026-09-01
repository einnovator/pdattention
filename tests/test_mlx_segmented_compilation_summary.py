import json
from pathlib import Path

from experiments.paper6_2_mlx.summarize_segmented_compilation import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_segmented_compilation_summary_separates_kernel_and_model_evidence() -> None:
    kernel = json.loads(
        (ROOT / "docs/papers/shared/results/paper6_2_mlx/segmented_attention_compiled_m4.json").read_text()
    )
    compiled = json.loads(
        (ROOT / "docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling/qwen3_32b_compiled.json").read_text()
    )
    eager = json.loads(
        (ROOT / "docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling/qwen3_32b.json").read_text()
    )

    report = summarize(kernel, compiled, eager)

    assert [row["occupied_context_tokens"] for row in report["kernel_rows"]] == [2048, 8192, 32768]
    assert report["kernel_rows"][-1]["concat_temporary_bytes_avoided"] > 128 * 2**20
    assert report["compiled_model_conditions"]["E2_CONCAT_WARM"]["sequence_agreement_vs_e0"] == 1.0
    assert report["matched_model_examples"] == 15
    assert report["claim_boundary"].startswith("MLX graph compilation")
