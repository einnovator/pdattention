from __future__ import annotations

import json

from click.testing import CliRunner

from pra_hf.cli import cli


def test_adaptive_profiles_command_emits_control_vectors() -> None:
    result = CliRunner().invoke(cli, ["adaptive", "profiles"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["name"] for row in payload["profiles"]] == ["E0_low", "E1_medium", "E2_high"]
    assert payload["profiles"][1]["control_vector"]["R"] == 2


def test_adaptive_plan_respects_hard_kv_cap(tmp_path) -> None:
    features = tmp_path / "features.json"
    features.write_text(
        json.dumps({"routing_entropy": 0.95, "root_score_gap": 0.01}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli,
        [
            "adaptive",
            "plan",
            "--mode",
            "auto",
            "--features",
            str(features),
            "--max-active-kv",
            "300",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selected_profile"]["name"] == "E0_low"
    assert payload["selected_profile"]["native_kv_budget"] <= 300


def test_adaptive_plan_rejects_oracle_features(tmp_path) -> None:
    features = tmp_path / "leaked.json"
    features.write_text(json.dumps({"oracle_evidence_recall": 1.0}), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        ["adaptive", "plan", "--features", str(features)],
    )
    assert result.exit_code != 0
    assert "Evaluator-only fields" in str(result.exception)
