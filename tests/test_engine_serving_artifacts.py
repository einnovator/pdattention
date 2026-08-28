from __future__ import annotations

import json
from pathlib import Path

from experiments.engine_serving.summarize import build_registry


ROOT = Path(__file__).resolve().parents[1]


def test_engine_smoke_registry_is_complete_and_does_not_claim_native_pra() -> None:
    registry = build_registry()
    assert registry["registry_version"] == "2026-08-paper6-engine-smoke-v1"
    assert len(registry["rows"]) == 15
    assert {row["engine"] for row in registry["rows"]} == {"vllm", "sglang", "mlx"}
    assert all(row["evidence_tier"] == "SMOKE" for row in registry["rows"])
    assert all(row["native_pra_status"] == "NOT_MEASURED" for row in registry["rows"])


def test_selected_text_recovers_codeword_with_fewer_tokens_than_full_context() -> None:
    registry = build_registry()
    for engine in ("vllm", "sglang", "mlx"):
        rows = {row["condition"]: row for row in registry["rows"] if row["engine"] == engine}
        assert rows["prefix_plus_pra"]["quality_absolute"] == 1.0
        assert rows["no_prefix_no_pra"]["quality_absolute"] == 0.0
        assert rows["prefix_plus_pra"]["visible_initial_tokens"] < rows["full_context"]["visible_initial_tokens"]


def test_generated_registry_matches_builder_after_summary_run() -> None:
    generated = json.loads(
        (ROOT / "docs" / "papers" / "shared" / "results" / "pra_engine_benchmarks.json").read_text()
    )
    assert generated == build_registry()

