from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serving = _load(ROOT / "src" / "pra_hf" / "serving_benchmark.py", "load_serving")
load = _load(
    ROOT / "experiments" / "engine_serving" / "run_openai_load_e0.py",
    "load_benchmark",
)


def test_parameterized_messages_preserve_target_and_scale_context() -> None:
    small = serving.benchmark_messages(distractor_count=1, distractor_repeat=1)
    large = serving.benchmark_messages(distractor_count=4, distractor_repeat=8)

    assert "PRA_EVIDENCE_4821" in small["pra_only"][0]["content"]
    assert len(large["full_context"][1]["content"]) > len(
        small["full_context"][1]["content"]
    )
    with pytest.raises(ValueError):
        serving.benchmark_messages(distractor_count=-1)


def test_request_spec_renames_target_for_independent_resources() -> None:
    messages, expected = load._request_spec(
        serving,
        "pra_only",
        9,
        distractor_count=2,
        distractor_repeat=3,
    )

    assert expected == "PRA_EVIDENCE_4830"
    assert expected in messages[0]["content"]
    assert "PRA_EVIDENCE_4821" not in messages[0]["content"]


def test_load_aggregate_reports_tail_and_throughput() -> None:
    rows = [
        {
            "quality_ok": True,
            "ttft_ms": 10.0,
            "mean_itl_ms": 2.0,
            "completion_latency_ms": 20.0,
            "completion_tokens": 4,
            "cached_tokens": 8,
        },
        {
            "quality_ok": False,
            "ttft_ms": 30.0,
            "mean_itl_ms": 4.0,
            "completion_latency_ms": 40.0,
            "completion_tokens": 6,
            "cached_tokens": 12,
        },
    ]

    result = load._aggregate(serving, rows, elapsed=2.0)

    assert result["quality_success_rate"] == 0.5
    assert result["request_throughput_s"] == 1.0
    assert result["output_throughput_tokens_s"] == 5.0
    assert result["ttft_ms"]["p50"] == 20.0
    assert result["mean_cached_tokens"] == 10.0
