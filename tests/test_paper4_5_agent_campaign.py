"""Reproduction gates for the long-running Paper 4.5 agent campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.paper4_5_agent.reproduction import OfficialResult, review_result
from experiments.paper4_5_agent.harness_matrix import (
    HarnessMatrixConfig,
    harbor_command,
    matrix_cells,
    run_matrix,
)
from experiments.paper4_5_agent.runners.r2egym import (
    locate_official_report,
    normalize_official_report,
    trajectories_to_predictions,
    write_task_results,
)
from experiments.paper4_5_agent.summarize_baseline import summarize
from experiments.paper4_5_agent.run_campaign import record_result, run_campaign
from experiments.paper4_5_agent.schema import CampaignConfig, ReproductionStatus
from experiments.agents.schema import BenchmarkManifest


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/fim14b_r2egym.yaml"
MATRIX_CONFIG = ROOT / "experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pilot.yaml"


def test_fim14b_campaign_pins_published_identity_and_orders_treatments() -> None:
    campaign = CampaignConfig.load(CONFIG)
    baseline = campaign.baselines[0]
    assert baseline.model == "TIGER-Lab/FIM-14B"
    assert baseline.published_score == 0.292
    assert baseline.published_total == 500
    assert baseline.max_steps_absolute == 100
    assert baseline.function_calling is False
    assert baseline.prefix_caching is True
    assert campaign.cells[0].baseline_cell is None
    assert all(cell.baseline_cell == "fim14b-no-pra" for cell in campaign.cells[1:])


def test_changed_engine_is_attempted_not_reproduced() -> None:
    campaign = CampaignConfig.load(CONFIG)
    result = OfficialResult(
        official_grader=True, score=0.30, resolved=150, total=500,
        configuration_differences=("engine: MLX instead of vLLM",),
    )
    review = review_result(
        campaign.baselines[0], result, absolute_tolerance=0.05, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_ATTEMPTED
    assert not review.compatible


def test_compatible_official_result_admits_baseline() -> None:
    campaign = CampaignConfig.load(CONFIG)
    result = OfficialResult(official_grader=True, score=0.29, resolved=145, total=500)
    review = review_result(
        campaign.baselines[0], result, absolute_tolerance=0.05, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_REPRODUCED


def test_treatment_without_baseline_dependency_is_rejected() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["cells"][1]["baseline_cell"] = None
    with pytest.raises(ValueError, match="require baseline_cell"):
        CampaignConfig.model_validate(payload)


def test_dry_run_persists_reports_and_blocks_treatments(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "campaign")
    for cell in payload["cells"]:
        cell["enabled"] = True
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # The runner resolves output relative to the repository inferred from config;
    # an absolute path remains absolute on every supported host.
    state = run_campaign(config, max_hours=1, resume=False, dry_run=True)
    assert state["cells"]["fim14b-no-pra"]["state"] == "PENDING"
    assert state["cells"]["fim14b-gateway-passthrough"]["state"] == "BLOCKED"
    output = tmp_path / "campaign"
    assert (output / "campaign_state.json").is_file()
    assert (output / "reproduction_report.md").is_file()
    assert json.loads((output / "summary.json").read_text())["pra_interpretation_allowed"] is False


def test_r2egym_converter_uses_only_visible_trajectory_patch(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.jsonl"
    source.write_text(json.dumps({
        "ds": {"instance_id": "repo__project-1", "patch": "hidden-gold"},
        "output_patch": "diff --git a/a.py b/a.py\n",
    }) + "\n", encoding="utf-8")
    destination = tmp_path / "predictions.jsonl"
    rows = trajectories_to_predictions(source, destination, "TIGER-Lab/FIM-7B")
    assert rows[0]["model_patch"].startswith("diff --git")
    assert "hidden-gold" not in destination.read_text(encoding="utf-8")


def test_official_swebench_report_is_normalized(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "submitted_ids": ["a", "b"], "resolved_ids": ["a"],
    }), encoding="utf-8")
    receipt = normalize_official_report(
        report, tmp_path / "official_result.json",
        configuration_differences=("engine: llama.cpp instead of vLLM",),
    )
    assert receipt["score"] == 0.5
    assert receipt["official_grader"] is True
    assert receipt["configuration_differences"]


def test_official_report_written_in_harness_root_is_retained(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    report = tmp_path / "model.smoke.json"
    report.write_text('{"submitted_ids": ["a"]}', encoding="utf-8")

    located = locate_official_report(output, tmp_path, "smoke")

    assert located == output / report.name
    assert located.read_bytes() == report.read_bytes()


def test_r2egym_task_telemetry_keeps_no_pra_costs_separate(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectory.jsonl"
    trajectories.write_text(json.dumps({
        "ds": {"instance_id": "repo__project-1"},
        "max_token_limit": 32768,
        "exit_reason": "agent",
        "output_patch": "diff\n+line\n",
        "trajectory_steps": [
            {
                "token_usage_prompt": 100, "token_usage_completion": 20,
                "llm_exec_time": 2.0, "env_exec_time": 0.5,
                "total_time_traj": 2.5, "action": "read file",
            },
            {
                "token_usage_prompt": 150, "token_usage_completion": 10,
                "llm_exec_time": 3.0, "env_exec_time": 0.25,
                "total_time_traj": 5.75, "tool_calls": [{"name": "patch"}],
            },
        ],
    }) + "\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "submitted_ids": ["repo__project-1"],
        "resolved_ids": ["repo__project-1"], "error_ids": [],
    }), encoding="utf-8")
    rows = write_task_results(
        trajectories, report, tmp_path / "results.jsonl",
        model="model", model_revision="revision", engine="llama.cpp",
        engine_version="version", quantization="Q4_K_M", harness_version="commit",
    )
    assert rows[0]["resolved"] is True
    assert rows[0]["physical_input_tokens"] == rows[0]["logical_input_tokens"] == 250
    assert rows[0]["unique_context_tokens_estimate"] == 150
    assert rows[0]["repeated_context_tokens_estimate"] == 100
    assert rows[0]["repeated_context_fraction_estimate"] == pytest.approx(0.4)
    assert rows[0]["model_call_count"] == rows[0]["trajectory_length"] == 2
    assert rows[0]["tool_call_count"] == 2
    assert rows[0]["p95_model_call_s"] == pytest.approx(2.95)
    assert rows[0]["patch_bytes"] == 11
    assert rows[0]["patch_lines"] == 2
    assert rows[0]["grader_outcome"] == "resolved"
    assert rows[0]["pra_route_time_s"] == rows[0]["pra_memory_bytes"] == 0
    assert rows[0]["prefill_time_s"] is None


def test_distributed_result_import_preserves_attempted_status(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "campaign")
    payload["cells"] = [payload["cells"][0]]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    result = tmp_path / "official_result.json"
    result.write_text(json.dumps({
        "official_grader": True, "score": 0.5, "resolved": 1, "total": 2,
        "configuration_differences": ["cohort=2-not-500"],
    }), encoding="utf-8")
    state = record_result(config, cell_id="fim14b-no-pra", result_path=result)
    cell = state["cells"]["fim14b-no-pra"]
    assert cell["reproduction_status"] == "BASELINE_ATTEMPTED"
    assert (tmp_path / "campaign/reproduction_report.md").is_file()


def test_agent_baseline_summary_keeps_small_cohort_locked() -> None:
    rows = [
        {
            "resolved": resolved,
            "cumulative_prompt_tokens": 100,
            "unique_context_tokens_estimate": 25,
            "repeated_context_tokens_estimate": 75,
            "output_tokens": 10,
            "model_call_count": 2,
            "tool_call_count": 2,
            "trajectory_length": 2,
            "patch_bytes": 20,
            "wall_time_s": 5.0,
            "decode_time_s": 3.0,
            "tool_time_s": 1.0,
            "repeated_context_fraction_estimate": 0.75,
            "ttft_ms": None,
            "prefill_time_s": None,
            "peak_memory_bytes": None,
            "kv_bytes": None,
        }
        for resolved in (True, False)
    ]
    result = summarize(rows, minimum_tasks=20)
    assert result["cohort_status"] == "INSUFFICIENT_COHORT"
    assert result["pra_treatment_unlocked"] is False
    assert result["token_totals"]["repeated_context_fraction_estimate"] == 0.75
    assert result["success_wilson_95_ci"] == pytest.approx(
        [0.09453120573423074, 0.9054687942657693]
    )


def test_stronger_model_matrix_crosses_10_tasks_and_three_harnesses() -> None:
    config = HarnessMatrixConfig.load(MATRIX_CONFIG)
    manifest = BenchmarkManifest.load(ROOT / config.manifest)
    cells = matrix_cells(config, manifest)
    assert len(cells) == 30
    assert {cell[2].harness_id for cell in cells[:3]} == {
        "opencode-1.18.26", "pi-0.73.1", "openhands-0.57.0",
    }
    assert len({cell[3] for cell in cells[:15]}) == 5


def test_harbor_matrix_command_is_one_task_and_redactable() -> None:
    config = HarnessMatrixConfig.load(MATRIX_CONFIG)
    manifest = BenchmarkManifest.load(ROOT / config.manifest)
    _, model, harness, task_id, _ = matrix_cells(config, manifest)[0]
    command = harbor_command(
        harbor="harbor", manifest=manifest, model=model, harness=harness,
        task_id=task_id, job_directory=Path("jobs"),
        base_url="http://model-host:11435/v1", api_key="secret",
    )
    assert command.count("-i") == 1
    assert f"terminal-bench/{task_id}" in command
    assert "OPENAI_BASE_URL=http://model-host:11435/v1" in command
    assert "version=1.18.26" in command


def test_harbor_matrix_retry_uses_isolated_attempt_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(MATRIX_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "matrix")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.delenv("PRA_AGENT_QWEN3_CODER_30B_URL", raising=False)
    state = run_matrix(config_path, resume=False, dry_run=True, max_cells=1)
    record = next(iter(state["cells"].values()))
    assert record["active_attempt"] == "a001"
    assert "attempts" in record["command"][record["command"].index("--jobs-dir") + 1]


def test_harbor_matrix_dry_run_is_pra_locked_and_writes_all_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(MATRIX_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "matrix")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.delenv("PRA_AGENT_QWEN3_CODER_30B_URL", raising=False)
    state = run_matrix(config_path, resume=False, dry_run=True)
    assert state["pra_enabled"] is False
    assert len(state["cells"]) == 30
    assert {record["state"] for record in state["cells"].values()} == {"PENDING"}
    summary = json.loads((tmp_path / "matrix/summary.json").read_text(encoding="utf-8"))
    assert summary["expected_runs"] == 30
    assert summary["completed_runs"] == 0
    assert summary["admission_gate"]["eligible"] is False
    report = (tmp_path / "matrix/report.md").read_text(encoding="utf-8")
    assert "official-Harbor **No-PRA baseline**" in report
    assert "Completed: `0/30`" in report
