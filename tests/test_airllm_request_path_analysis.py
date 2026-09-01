from __future__ import annotations

from experiments.paper6_6_airllm.analyze_request_path import analyze


def _row(condition: str, repeat: int, ttft: float, completion: float) -> dict:
    return {
        "condition": condition,
        "repeat": repeat,
        "visible_prompt_tokens": 20 if condition == "native_pra_e2" else 400,
        "selected_native_kv_tokens": 384 if condition == "native_pra_e2" else 0,
        "ttft_ms": ttft,
        "itl_ms": ttft / 2,
        "completion_seconds": completion,
        "peak_cuda_bytes": 100 * 2**20,
        "reference_encode_seconds": 10.0 if condition == "native_pra_e2" and repeat == 0 else 0.0,
    }


def test_analysis_does_not_invent_break_even_when_e2_request_is_slower() -> None:
    payload = {
        "schema_version": "source",
        "model_id": "tiny",
        "device": "cuda",
        "rows": [
            _row("full_context_e0", 0, 5_000.0, 20.0),
            _row("selected_text_e0", 1, 5_000.0, 20.0),
            _row("native_pra_e2", 0, 11_000.0, 23.0),
            _row("native_pra_e2", 1, 11_000.0, 23.0),
        ],
    }

    report = analyze(payload)

    assert report["pooled"]["reuse_break_even_queries"] is None
    assert report["pooled"]["e2_over_e0_ttft"] == 2.2
    assert report["favorable_workload_hypothesis"]["current_trace_has_finite_break_even"] is False
