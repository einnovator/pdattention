from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/"
    "engine_gates"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_sglang_same_state_gate_uses_a_pinned_source_owner() -> None:
    result = _load("sglang_same_state_task02_v3.json")
    assert result["same_resident_kv_fork"] is True
    assert result["source_owner_pinned_through_both_forks"] is True
    assert result["completed_turns"] == result["exact_turns"] == 10
    assert result["zero_selected_text_reencoding"] is True
    assert result["zero_physical_kv_copy"] is True
    assert result["same_state_gate_valid"] is True


def test_sglang_rematerialization_control_cannot_qualify_the_engine() -> None:
    result = _load("sglang_same_state_task02_v1.json")
    assert result["same_resident_kv_fork"] is False
    assert result["qualification_valid"] is False
    assert result["exact_turns"] == 7
    assert result["first_divergent_turn"] == 1


def test_vllm_overlay_result_is_exact_but_not_a_packaged_runtime_gate() -> None:
    result = _load("vllm_same_state_task02_v2.json")
    assert result["completed_turns"] == result["exact_turns"] == 10
    assert result["zero_selected_text_reencoding"] is True
    assert result["zero_physical_kv_copy"] is True
    assert result["exact_packaged_runtime"] is False
    assert result["same_state_gate_valid"] is False

