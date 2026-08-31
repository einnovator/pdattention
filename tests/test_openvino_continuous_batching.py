from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serving = _load(ROOT / "src" / "pra_hf" / "serving_benchmark.py", "ov_batch_serving")
batching = _load(
    ROOT / "experiments" / "paper6_3_openvino" / "run_continuous_batching.py",
    "ov_batching",
)


def test_request_spec_uses_parameterized_context() -> None:
    small, expected = batching._request_spec(
        serving,
        "full_context",
        2,
        distractor_count=1,
        distractor_repeat=1,
    )
    large, _ = batching._request_spec(
        serving,
        "full_context",
        2,
        distractor_count=4,
        distractor_repeat=8,
    )

    assert expected == "PRA_EVIDENCE_4823"
    assert expected in small[1]["content"]
    assert len(large[1]["content"]) > len(small[1]["content"])


def test_aggregate_reports_token_and_tpot_metrics() -> None:
    rows = [
        {
            "quality_ok": True,
            "ttft_ms": 10,
            "tpot_ms": 2,
            "generation_ms": 20,
            "input_tokens": 100,
            "output_tokens": 4,
        },
        {
            "quality_ok": True,
            "ttft_ms": 30,
            "tpot_ms": 6,
            "generation_ms": 40,
            "input_tokens": 200,
            "output_tokens": 6,
        },
    ]

    result = batching._aggregate(serving, rows, wall_seconds=2.0)

    assert result["tpot_ms_p50"] == 4.0
    assert result["mean_input_tokens"] == 150.0
    assert result["mean_output_tokens"] == 5.0
    assert result["output_throughput_tokens_s"] == 5.0


def test_openvino_counter_contract_is_documented_as_asymmetric() -> None:
    source = (
        ROOT
        / "experiments"
        / "paper6_3_openvino"
        / "run_continuous_batching.py"
    ).read_text(encoding="utf-8")
    assert "input_tokens /= batch_size" not in source
    assert "output_tokens /= batch_size" in source
