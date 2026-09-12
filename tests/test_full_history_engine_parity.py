import json
from pathlib import Path

import pytest

from experiments.paper4_5_agent.run_full_history_engine_parity import _turn_gates


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/"
    "engine_gates/m5_full_history_parity"
)


def _row(turn: int) -> dict:
    source = 32 + turn * 8
    new = source if turn == 0 else 8
    trace = {
        "full_retention": True,
        "selected_kv_tokens": source,
        "source_tokens": source,
        "source_position_base": source,
        "selected_history_reencoded_tokens": 0,
        "physical_kv_copy": False,
        "physical_kv_copy_bytes": 0,
        "selected_history_kv_copy_bytes": 0,
        "new_history_encoded_tokens": new,
        "canonical_suffix_graft_d2d_bytes": 1024,
        "total_kv_copy_bytes": 2048,
    }
    return {
        "exact": True,
        "trace": trace,
        "response_pra": {"native_attach_bytes": 0},
    }


def test_full_history_gates_separate_selection_from_canonical_copy() -> None:
    gates = _turn_gates([_row(0), _row(1), _row(2)])
    assert all(gates.values())


def test_full_history_gate_rejects_missing_physical_copy_counter() -> None:
    rows = [_row(0), _row(1)]
    del rows[1]["trace"]["physical_kv_copy_bytes"]
    gates = _turn_gates(rows)
    assert gates["byte_exact_outputs"] is True
    assert gates["zero_selected_history_kv_copy"] is False


@pytest.mark.parametrize(
    ("name", "engine"),
    (
        ("mlx_qwen3_4b_pra100.json", "mlx-lm"),
        ("sglang_mlx_qwen3_4b_pra100.json", "sglang-mlx"),
    ),
)
def test_m5_full_history_evidence_closes_every_engine_gate(
    name: str, engine: str
) -> None:
    payload = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "paper4.5.full-history-engine-parity.v1"
    assert payload["engine"] == engine
    assert payload["model_revision"] == (
        "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
    )
    assert payload["pra_commit"] == "0da2e47b031f45d056d5781f50a704f96468b3e2"
    assert payload["turns_completed"] == 3
    assert payload["passed"] is True
    assert all(payload["gates"].values())
    assert [row["plain_text"] for row in payload["rows"]] == [
        "ALPHA", "BETA", "GAMMA"
    ]
    assert all(row["exact"] for row in payload["rows"])
