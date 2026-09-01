from __future__ import annotations

from experiments.mac_scaling.summarize_mlx_scaling import summarize


def _payload(model: str, latency: float) -> dict[str, object]:
    aggregate = []
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
        for index, condition in enumerate(
            (
                "E0_SELECTED",
                "E2_CONCAT_ALL",
                "E2_SEGMENTED_ALL",
                "E2_SEGMENTED_LAST_3_4",
                "E2_SEGMENTED_LAST_2_3",
                "E2_SEGMENTED_LAST_1_2",
                "E2_SEGMENTED_LAST_1_3",
                "E2_SEGMENTED_LAST_1_4",
            )
        ):
            aggregate.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "samples": 5,
                    "token_f1": 0.5,
                    "gold_answer_logprob": -2.0 - index,
                    "completion_latency_ms": latency + index,
                    "active_detail_bytes": index * 1024,
                    "peak_unified_memory_bytes": 2**30,
                    "sequence_agreement_vs_e0": 1.0 if index < 3 else 0.5,
                }
            )
    return {
        "model_id": model,
        "model_revision": "a" * 40,
        "layer_count": 36,
        "seeds": [11, 23, 37, 53, 71],
        "aggregate": aggregate,
        "rows": [{"model_resident_bytes": 2**30, "peak_unified_memory_bytes": 2**30}],
    }


def test_summary_keeps_concat_segmented_and_profile_metrics_disjoint() -> None:
    rows = summarize([_payload("mlx-community/Qwen3-8B-4bit", 10.0)])
    row = rows[0]

    assert row["samples"] == 15
    assert row["concat_sequence_agreement"] == 1.0
    assert row["segmented_sequence_agreement"] == 1.0
    assert row["profiles"]["E2_SEGMENTED_LAST_3_4"]["sequence_agreement"] == 0.5
