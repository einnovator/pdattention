from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "experiments" / "engine_serving" / "summarize_portable_e0.py"
spec = importlib.util.spec_from_file_location("portable_summary", path)
assert spec is not None and spec.loader is not None
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary
spec.loader.exec_module(summary)


def test_natural_table_contains_comparable_metrics() -> None:
    payload = {
        "aggregates": [
            {
                "dataset": "hotpotqa",
                "condition": "selected_context",
                "token_f1": 0.25,
                "answer_containment": 0.5,
                "mean_prompt_tokens": 400,
                "ttft_ms": {"p50": 20, "p95": 30},
            }
        ]
    }
    table = summary._natural_table(payload)
    assert "hotpotqa & Selected & 0.250 & 0.500 & 400 & 20.0 & 30.0" in table


def test_load_table_selects_policy_endpoints() -> None:
    template = {
        "representation": "pra_only",
        "workload": "shared_resource",
        "request_throughput_s": 10,
        "ttft_ms": {"p50": 20, "p99": 40},
        "itl_ms": {"p99": 5},
    }
    payload = {
        "aggregates": [
            {**template, "concurrency": 1},
            {**template, "concurrency": 2},
            {**template, "concurrency": 16},
        ]
    }
    table = summary._load_table(payload)
    assert "Shared & Selected & 1" in table
    assert "Shared & Selected & 16" in table
    assert "Shared & Selected & 2" not in table
