from pathlib import Path

from experiments.mac_scaling.summarize_mlx_long_context import (
    _model_sort_key,
    summarize,
)


def _row(condition: str, logprob: float, rmse: float, agreement: float) -> dict:
    return {
        "status": "MEASURED",
        "model_id": "model",
        "context_target_tokens": 8192,
        "condition": condition,
        "dataset": "qasper",
        "example_id": "example-1",
        "token_f1": 0.5,
        "gold_answer_logprob": logprob,
        "completion_latency_ms": 10.0,
        "first_token_agreement_vs_full": agreement,
        "sequence_agreement_vs_full": agreement,
        "first_logit_rmse_vs_full": rmse,
        "active_detail_bytes": 64,
        "peak_unified_memory_bytes": 128,
        "output_token_ids": [1] if agreement else [2],
    }


def test_summary_separates_dilution_from_position_error(tmp_path: Path) -> None:
    import json

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "model_id": "model",
                "runtime": {},
                "rows": [
                    _row("FULL_VISIBLE", -4.0, 0.0, 1.0),
                    _row("E0_SELECTED", -3.0, 0.2, 1.0),
                    _row("E2_SELECTED", -3.0, 0.2, 1.0),
                    _row("E2_SOURCE_RELATIVE", -4.1, 0.1, 1.0),
                    _row("E2_QUERY_RESTART", -8.0, 2.0, 0.0),
                ],
            }
        ),
        encoding="utf-8",
    )

    row = summarize([source])["comparisons"][0]

    assert row["full_minus_selected_gold_logprob"] == -1.0
    assert row["source_relative_logit_rmse"] == 0.1
    assert row["query_restart_sequence_agreement"] == 0.0
    assert row["selected_native_sequence_agreement"] == 1.0
    assert row["selected_native_minus_text_gold_logprob"] == 0.0


def test_model_size_sort_is_numeric() -> None:
    models = ["Qwen3-32B-4bit", "Qwen3-8B-4bit", "Qwen3-14B-4bit"]

    assert sorted(models, key=_model_sort_key) == [
        "Qwen3-8B-4bit",
        "Qwen3-14B-4bit",
        "Qwen3-32B-4bit",
    ]
