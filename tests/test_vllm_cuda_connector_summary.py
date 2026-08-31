from __future__ import annotations

import json
from pathlib import Path

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

