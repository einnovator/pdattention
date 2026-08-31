from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.paper6_vllm.run_cuda_connector_concurrency import _condition_row
from experiments.paper6_vllm.summarize_cuda_connector import summarize


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/papers/shared/results/paper6_vllm"


def test_cuda_connector_summary_preserves_candidate_boundary() -> None:
    controlled = json.loads(
        (RESULTS / "cuda_connector_candidate.json").read_text(encoding="utf-8")
    )
    natural = json.loads(
        (RESULTS / "cuda_connector_natural_apc.json").read_text(encoding="utf-8")
    )

    summary = summarize(controlled, natural)

    assert summary["natural_samples"] == 60
    assert summary["natural_exact_pairs"] == 60
    assert summary["controlled"]["wrong_memory_follows_wrong_code"] == 5
    assert summary["qualification"]["native_kv_consumed"] is True
    assert summary["qualification"]["full_detached_e2_validated"] is False


def test_cuda_concurrency_row_separates_recovery_from_leakage() -> None:
    outputs = [
        SimpleNamespace(outputs=[SimpleNamespace(text="4821", token_ids=[1])]),
        SimpleNamespace(outputs=[SimpleNamespace(text="ordinary", token_ids=[2])]),
    ]

    row = _condition_row(
        concurrency=2,
        condition="mixed_native_ordinary",
        outputs=outputs,
        metrics={"requests": 2, "completion_ms": 10.0},
        expected_by_request=["4821", None],
        forbidden_by_request=["7394", "4821"],
    )

    assert row["expected_recoveries"] == 1
    assert row["expected_requests"] == 1
    assert row["forbidden_leaks"] == 0
