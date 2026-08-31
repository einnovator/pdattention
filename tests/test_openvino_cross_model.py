from __future__ import annotations

import pytest

from experiments.paper6_3_openvino.summarize_cross_model import (
    render_table,
    summarize,
)


def _payload(model: str, selected_f1: float, full_f1: float) -> dict:
    rows = []
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
        for condition, f1, ttft, tokens in (
            ("selected_context", selected_f1, 100.0, 200.0),
            ("full_context", full_f1, 250.0, 800.0),
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "sample_count": 20,
                    "token_f1": f1,
                    "answer_containment": f1,
                    "ttft_ms": {"p50": ttft, "p95": ttft * 2},
                    "mean_prompt_tokens": tokens,
                }
            )
    return {
        "model_id": model,
        "engine_version": "2026.3.1.0",
        "device": "GPU",
        "evidence_tier": "CONTROLLED_NATURAL_QA",
        "aggregates": rows,
    }


def test_cross_model_summary_pairs_quality_and_ttft() -> None:
    result = summarize(
        [
            _payload("Qwen2-0.5B-Instruct-int4-ov", 0.2, 0.3),
            _payload("Qwen2.5-1.5B-Instruct-int4-ov", 0.4, 0.45),
            _payload("TinyLlama-1.1B-Chat-v1.0-int4-ov", 0.35, 0.4),
        ]
    )

    assert len(result["rows"]) == 9
    first = result["rows"][0]
    assert first["selected_minus_full_f1"] == pytest.approx(-0.1)
    assert first["full_over_selected_ttft"] == 2.5
    assert first["selected_ttft_p95_ms"] == 200.0
    assert result["engine_version"] == "2026.3.1.0"
    table = render_table(result)
    assert "Qwen2.5 1.5B" in table
    assert "TinyLlama 1.1B" in table
