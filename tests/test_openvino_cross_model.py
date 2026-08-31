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
                    "ttft_ms": {"p50": ttft},
                    "mean_prompt_tokens": tokens,
                }
            )
    return {"model_id": model, "aggregates": rows}


def test_cross_model_summary_pairs_quality_and_ttft() -> None:
    result = summarize(
        [_payload("small", 0.2, 0.3), _payload("large", 0.4, 0.45)]
    )

    assert len(result["rows"]) == 6
    first = result["rows"][0]
    assert first["selected_minus_full_f1"] == pytest.approx(-0.1)
    assert first["full_over_selected_ttft"] == 2.5
    assert "1.5B" in render_table(result)
