from experiments.paper6_2_mlx.summarize_m4_scaling import (
    comparison_rows,
    summarize,
)


def test_summarizes_cross_model_transport_and_controls() -> None:
    rows = []
    values = {
        "ordinary_split": (0.4, -2.0, 20.0, 0),
        "native_fp": (0.4, -2.0, 10.0, 8 * 1048576),
        "native_int8_resident": (0.4, -2.1, 12.0, 4 * 1048576),
        "native_shuffled": (0.1, -8.0, 10.0, 8 * 1048576),
        "no_memory": (0.0, -9.0, 8.0, 0),
    }
    for condition, (f1, logprob, latency, resident) in values.items():
        for seed in (11, 23):
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "token_f1": f1,
                    "gold_answer_logprob": logprob,
                    "completion_latency_ms": latency,
                    "resident_selected_kv_bytes": resident,
                }
            )
    payload = {"model_id": "mlx-community/Qwen3-4B-4bit", "dataset": "qasper", "rows": rows}

    result = comparison_rows(summarize([payload]))

    assert len(result) == 1
    assert result[0]["native_over_ordinary"] == 0.5
    assert result[0]["native_resident_mib"] == 8.0
    assert result[0]["int8_resident_mib"] == 4.0
    assert result[0]["gold_logprob_delta_shuffled"] == -6.0
