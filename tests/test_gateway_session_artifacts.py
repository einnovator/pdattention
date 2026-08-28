from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/papers/shared/results/paper4_5_runtime"


def _rows(name: str):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_required_gateway_session_artifacts_exist_and_preserve_claim_scope():
    required = (
        "engine_cache_capability_matrix.csv",
        "gateway_session_policy_matrix.csv",
        "prefix_preservation_results.csv",
        "session_delta_results.csv",
        "gateway_cache_trace_examples.json",
        "engine_profile_registry.json",
        "generated_gateway_session_table.tex",
        "gateway_prefix_reuse.pdf",
        "gateway_prefix_reuse.png",
        "gateway_two_cache_architecture.pdf",
        "gateway_two_cache_architecture.png",
    )
    assert all((RESULTS / name).is_file() for name in required)
    traces = json.loads(
        (RESULTS / "gateway_cache_trace_examples.json").read_text(encoding="utf-8")
    )
    assert traces["physical_prefix_cache_hits_measured"] is False
    assert all(row["engine_prefix_cache_hit"] == "UNKNOWN" for row in traces["examples"])


def test_prefix_preservation_and_delta_results_are_generated_from_real_gateway_path():
    summary = {row["condition"]: row for row in _rows("session_delta_results.csv")}
    legacy = summary["CURRENT_G10_PREPEND"]
    preserving = summary["PREFIX_PRESERVING_G10"]
    delta = summary["SESSION_DELTA_E0"]
    pra = summary["PRA_ENABLED_SESSION"]

    assert legacy["stable_turns_after_first"] == "0"
    assert legacy["prefix_invalidations"] == "4"
    assert preserving["stable_turns_after_first"] == "4"
    assert preserving["prefix_invalidations"] == "0"
    assert int(delta["message_bytes_sent"]) < int(preserving["message_bytes_sent"])
    assert delta["engine_session_reuse_turns"] == "4"
    assert pra["engine_session_reuse_turns"] == "4"
    assert pra["resource_bytes_sent"] != "0"
    assert all(row["engine_prefix_cache_hits"] == "NOT_MEASURED" for row in summary.values())


def test_engine_registry_copy_matches_packaged_registry():
    packaged = ROOT / "src/pra_hf/model_profiles/engine_profile_registry.json"
    assert json.loads((RESULTS / "engine_profile_registry.json").read_text()) == json.loads(
        packaged.read_text()
    )
    capabilities = {row["engine_type"]: row for row in _rows("engine_cache_capability_matrix.csv")}
    assert capabilities["openai_generic"]["prefix_cache_mode"] == "unknown"
    assert capabilities["vllm"]["prefix_cache_mode"] == "automatic_prefix_cache"
    assert capabilities["huggingface"]["default_pra_level"] == "E1"
