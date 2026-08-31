from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "experiments" / "engine_serving" / "summarize_context_pressure.py"
spec = importlib.util.spec_from_file_location("context_pressure", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_summary_uses_peak_concurrency_and_prompt_tokens() -> None:
    payload = {
        "engine": "test",
        "model_id": "model",
        "concurrency": [1, 16],
        "samples": [
            {
                "representation": "pra_only",
                "workload": "shared_resource",
                "concurrency": 16,
                "prompt_tokens": 64,
            }
        ],
        "aggregates": [
            {
                "representation": "pra_only",
                "workload": "shared_resource",
                "concurrency": 1,
                "quality_success_rate": 1,
                "request_throughput_s": 1,
                "output_throughput_tokens_s": 4,
                "ttft_ms": {"p99": 10},
            },
            {
                "representation": "pra_only",
                "workload": "shared_resource",
                "concurrency": 16,
                "quality_success_rate": 1,
                "request_throughput_s": 8,
                "output_throughput_tokens_s": 32,
                "ttft_ms": {"p99": 20},
            },
        ],
    }
    result = module.summarize([("Small", payload)])
    assert len(result["rows"]) == 1
    assert result["rows"][0]["mean_prompt_tokens"] == 64
    assert result["rows"][0]["request_throughput_s"] == 8


def test_table_includes_success_and_tail() -> None:
    row = {
        "size": "Large",
        "workload": "independent_resources",
        "representation": "full_context",
        "mean_prompt_tokens": 1500,
        "quality_success_rate": 0.5,
        "request_throughput_s": 12,
        "output_throughput_tokens_s": 120,
        "ttft_ms_p99": 400,
    }
    assert "Large & Independent & Full & 1500 & 0.50" in module._table([row])
