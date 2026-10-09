import json

import pytest

from experiments.paper4_5_agent.summarize_cross_engine_frozen_sequences import (
    summarize,
)


def _summary(tmp_path, engine, *, selected=40, ledger="ledger", copy=0):
    path = tmp_path / f"{engine}.json"
    path.write_text(
        json.dumps(
            {
                "engine": engine,
                "model": "model",
                "claim_boundary": "mechanism-only",
                "sequence_qualified": True,
                "frozen_ledger_bound": True,
                "frozen_ledger_files": {
                    "selection_fixture_sha256": ledger,
                    "request_replay_sha256": "replay",
                },
                "expected_requests": [1],
                "cumulative_source_kv_tokens": 100,
                "cumulative_selected_kv_tokens": selected,
                "visible_context_omission_fraction": 0.5,
                "selected_history_reencoded_tokens": 0,
                "selection_pack_bytes": 0,
                "attachment_physical_kv_copy_requests": copy,
                "rows": [
                    {
                        "request_input_sha256": "request",
                        "source_plan_digest": "plan",
                        "selected_resource_digest": "selected",
                        "source_wire_plan_digest": "wire",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cross_engine_gate_matches_ledger_and_reports_mapping_delta(tmp_path):
    result = summarize(
        {"hf": _summary(tmp_path, "hf"), "mlx": _summary(tmp_path, "mlx")},
        required_engines=("hf", "mlx"),
        paper8_logical_omission_fraction=0.55,
    )
    assert result["cross_engine_sequence_qualified"] is True
    assert result["engines"]["hf"]["resident_kv_omission_fraction"] == pytest.approx(0.6)
    assert result["engines"]["hf"]["resident_minus_paper8_percentage_points"] == pytest.approx(5.0)


def test_cross_engine_gate_rejects_different_frozen_ledger(tmp_path):
    result = summarize(
        {
            "hf": _summary(tmp_path, "hf"),
            "mlx": _summary(tmp_path, "mlx", ledger="different"),
        }
    )
    assert result["cross_engine_sequence_qualified"] is False
    assert "mlx:ledger_hashes_identical" in result["qualification_blockers"]


def test_cross_engine_gate_rejects_attachment_copy(tmp_path):
    result = summarize({"hf": _summary(tmp_path, "hf", copy=1)})
    assert result["cross_engine_sequence_qualified"] is False
    assert "hf:selected_kv_attachment_not_copied" in result["qualification_blockers"]


def test_cross_engine_gate_requires_declared_engines(tmp_path):
    result = summarize(
        {"hf": _summary(tmp_path, "hf")},
        required_engines=("hf", "mlx"),
    )
    assert result["cross_engine_sequence_qualified"] is False
    assert result["engines"]["mlx"]["status"] == "missing_required_engine_summary"
    assert "mlx:summary_missing" in result["qualification_blockers"]
