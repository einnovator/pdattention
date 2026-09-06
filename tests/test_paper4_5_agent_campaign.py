"""Reproduction gates for the long-running Paper 4.5 agent campaign."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.paper4_5_agent.reproduction import OfficialResult, review_result
from experiments.paper4_5_agent.reports import _next_tier, _success_band
from experiments.paper4_5_agent.benchmark import (
    load_benchmark_card,
    precision_diagnostic_ids,
)
from experiments.paper4_5_agent.build_easy_cohorts import (
    DIFFICULTY,
    build_card,
    ids_digest,
    select_easy_rows,
)
from experiments.paper4_5_agent.build_swebench_lite_cohort import (
    build_card as build_lite_card,
    select_rows as select_lite_rows,
)
from experiments.paper4_5_agent.promote_nested_baseline import promote
from experiments.paper4_5_agent.analyze_easy_frontier import (
    _bucket_effects,
    summarize as summarize_frontier,
)
from experiments.paper4_5_agent.context_treatment import (
    ContextTreatment,
    TreatmentProxy,
    transform_chat_payload,
)
from experiments.paper4_5_agent.harness_matrix import (
    HarnessMatrixConfig,
    _completed_attempt,
    _invalid_trial_reason,
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
from experiments.paper4_5_agent.run_campaign import (
    _campaign_environment,
    record_result,
    run_campaign,
)
from experiments.paper4_5_agent.schema import (
    CampaignConfig,
    PublishedBaseline,
    ReproductionStatus,
)
from experiments.paper4_5_agent.runners.swebench_verified import (
    _aggregate_traces,
    _cleanup_owned_containers,
    _execute_chunks,
    _grader_error_type,
    _is_h100_80gb,
    _normalize_report,
    _trajectory_metrics,
    _write_empty_predictions,
    package_versions,
)
from experiments.agents.schema import BenchmarkManifest


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/fim14b_r2egym.yaml"
MATRIX_CONFIG = ROOT / "experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pilot.yaml"
SWEBENCH_CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/swebench_pra_frontier.yaml"
EASY20_CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/swebench_easy20_calibration.yaml"
EASY50_CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/swebench_easy50_pra_frontier.yaml"
FIXED50 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_fixed50.json"
EASY20 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_easy20.json"
EASY50 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_easy50.json"
LITE50 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_lite50.json"


def test_fixed50_card_is_exact_unique_and_digest_protected() -> None:
    card = load_benchmark_card(FIXED50)
    assert len(card["instance_ids"]) == len(set(card["instance_ids"])) == 50
    assert card["source_revision"] == "8f894c2284b9f73a515024d7c1f32e4d0fb14a04"
    assert card["canonical_ids_sha256"] == (
        "20acb5f7e30fb3c854091e47c4214afb7304a5d47f353408a71ffaa418318131"
    )
    assert precision_diagnostic_ids(card) == tuple(card["instance_ids"][:10])


def test_easy_cards_are_nested_frozen_and_digest_protected() -> None:
    easy20 = load_benchmark_card(EASY20)
    easy50 = load_benchmark_card(EASY50)
    assert easy20["instance_ids"] == easy50["instance_ids"][:20]
    assert len(easy20["instance_ids"]) == 20
    assert len(easy50["instance_ids"]) == 50
    assert {row["difficulty"] for row in easy50["task_metadata"]} == {DIFFICULTY}


def test_lite50_card_is_frozen_and_outcome_blind() -> None:
    card = load_benchmark_card(LITE50)
    assert len(card["instance_ids"]) == 50
    assert card["source_revision"] == "b0dde1093fe417d83b7184254edf8199c1f0dff5"
    assert card["canonical_ids_sha256"] == (
        "3dab12f2b8e1fe3cbddeb20cc7522991ad76f25eba9dbd147223297e555ab93d"
    )
    rows = [
        {
            "instance_id": f"repo__project-{index}", "repo": "repo/project",
            "base_commit": str(index), "version": "1", "model_outcome": index % 2,
        }
        for index in range(60)
    ]
    assert select_lite_rows(rows, limit=10) == select_lite_rows(reversed(rows), limit=10)
    assert "model_outcome" not in json.dumps(build_lite_card(rows, count=10))


def test_easy_selection_is_deterministic_and_outcome_blind() -> None:
    rows = [
        {
            "instance_id": f"repo__project-{index}",
            "repo": "repo/project",
            "base_commit": str(index),
            "version": "1",
            "difficulty": DIFFICULTY if index != 2 else "1-4 hours",
            "model_outcome": index % 2,
        }
        for index in range(8)
    ]
    first = select_easy_rows(rows, limit=4)
    second = select_easy_rows(reversed(rows), limit=4)
    assert [row["instance_id"] for row in first] == [row["instance_id"] for row in second]
    card = build_card(rows, count=4, eligible_count=7)
    assert "model_outcome" not in json.dumps(card)


def test_campaign_children_use_scheduler_interpreter() -> None:
    environment = _campaign_environment({"PRA_TEST_VALUE": "present"})
    selected = Path(environment["PATH"].split(os.pathsep)[0]).resolve()
    assert selected == Path(sys.executable).absolute().parent.resolve()
    assert environment["PRA_TEST_VALUE"] == "present"


def test_nested_baseline_promotion_reuses_only_exact_completed_prefix(tmp_path: Path) -> None:
    source_ids = ["repo__project-1", "repo__project-2"]
    destination_ids = [*source_ids, "repo__project-3"]
    source_card = tmp_path / "source.json"
    destination_card = tmp_path / "destination.json"
    for path, ids in ((source_card, source_ids), (destination_card, destination_ids)):
        path.write_text(json.dumps({
            "expected_count": len(ids),
            "canonical_ids_sha256": ids_digest(ids),
            "instance_ids": ids,
        }), encoding="utf-8")
    source_output = tmp_path / "source"
    for index, instance_id in enumerate(source_ids):
        chunk = source_output / f"chunk_{index:02d}"
        chunk.mkdir(parents=True)
        (chunk / "official_chunk_result.json").write_text(json.dumps({
            "submitted_ids": [instance_id], "resolved_ids": [], "error_ids": [],
        }), encoding="utf-8")
    destination_output = tmp_path / "destination"

    manifest = promote(
        source_card, destination_card, source_output, destination_output,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["copied_task_ids"] == source_ids
    assert (destination_output / "chunk_01" / "official_chunk_result.json").is_file()
    assert not (destination_output / "chunk_02").exists()


def test_easy_frontier_summary_preserves_paired_outcomes_and_missing_metrics(tmp_path: Path) -> None:
    baseline = tmp_path / "no_pra"
    treatment = tmp_path / "truncation_50"
    baseline.mkdir()
    treatment.mkdir()
    rows = [
        {"instance_id": "a", "resolved": True, "mode": "no-pra",
         "context_budget_fraction": 1.0, "physical_input_tokens": 100,
         "logical_input_tokens": 100, "cumulative_prompt_tokens": 100},
        {"instance_id": "b", "resolved": False, "mode": "no-pra",
         "context_budget_fraction": 1.0, "physical_input_tokens": 200,
         "logical_input_tokens": 200, "cumulative_prompt_tokens": 200},
    ]
    (baseline / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    treated = [
        {**rows[0], "resolved": False, "mode": "truncation",
         "context_budget_fraction": 0.5, "physical_input_tokens": 50},
        {**rows[1], "resolved": True, "mode": "truncation",
         "context_budget_fraction": 0.5, "physical_input_tokens": 100},
    ]
    (treatment / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in treated) + "\n", encoding="utf-8"
    )

    summary = summarize_frontier(tmp_path)

    conditions = {row["condition"]: row for row in summary["conditions"]}
    assert conditions["no_pra"]["physical_input_tokens"] == 300
    assert conditions["no_pra"]["wall_time_s"] is None
    assert conditions["truncation_50"]["token_saving_fraction"] == 0.5
    assert {row["outcome"] for row in summary["paired"]} == {"regressed", "recovered"}


def test_context_demand_buckets_are_contiguous_tertiles() -> None:
    rows = [
        {
            "baseline_trajectory_length": index,
            "baseline_resolved": index < 3,
            "treatment_resolved": index >= 3,
        }
        for index in range(6)
    ]
    assert _bucket_effects(rows, "baseline_trajectory_length") == [-1.0, 0.0, 1.0]


def test_frontier_reports_matched_pra_vs_truncation_delta(tmp_path: Path) -> None:
    for condition, mode, resolved, tokens in (
        ("truncation_50", "truncation", False, 50),
        ("pra_selected_50", "gateway-pra", True, 45),
    ):
        directory = tmp_path / condition
        directory.mkdir()
        (directory / "results.jsonl").write_text(json.dumps({
            "instance_id": "task", "resolved": resolved, "mode": mode,
            "context_budget_fraction": 0.5, "physical_input_tokens": tokens,
        }) + "\n", encoding="utf-8")

    summary = summarize_frontier(tmp_path)

    assert summary["matched_budget"][0]["success_delta"] == 1.0
    assert summary["matched_budget"][0]["physical_input_token_delta"] == -5


def test_easy20_campaign_is_local_no_pra_calibration() -> None:
    campaign = CampaignConfig.load(EASY20_CONFIG)
    baseline = campaign.baselines[0]
    assert baseline.admission_kind == "local_calibration"
    assert baseline.published_score is None
    assert baseline.minimum_admission_score == 0.2
    assert len(baseline.task_ids) == 20
    assert len(campaign.cells) == 1
    assert campaign.cells[0].mode.value == "native"
    assert baseline.context_limit == 32768
    assert baseline.max_steps == 50
    assert campaign.cells[0].command[
        campaign.cells[0].command.index("--max-steps") + 1
    ] == "50"
    assert "--local-calibration" in campaign.cells[0].command


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.09, "FLOOR"), (0.15, "MARGINAL"), (0.25, "USEFUL"),
     (0.50, "PREFERRED"), (0.75, "USEFUL"), (0.81, "SATURATED")],
)
def test_calibration_success_bands(score: float, band: str) -> None:
    assert _success_band(score) == band


def test_calibration_next_tier_follows_admission_policy() -> None:
    assert "correction" in _next_tier(0.15)
    assert "Easy-50" in _next_tier(0.50)
    assert "harder" in _next_tier(0.90)


def test_easy50_frontier_is_nested_gated_and_budget_matched() -> None:
    campaign = CampaignConfig.load(EASY50_CONFIG)
    assert len(campaign.baselines[0].task_ids) == 50
    assert len(campaign.cells) == 8
    baseline, *treatments = campaign.cells
    assert baseline.cell_id == "easy50-no-pra"
    assert all(cell.baseline_cell == baseline.cell_id for cell in treatments)
    assert all(cell.minimum_baseline_score == 0.2 for cell in treatments)
    truncation = {
        cell.cell_id.rsplit("-", 1)[-1]: cell
        for cell in treatments if cell.mode.value == "truncation"
    }
    selected = {
        cell.cell_id.rsplit("-", 1)[-1]: cell
        for cell in treatments if cell.mode.value == "gateway_pra"
    }
    assert truncation.keys() == selected.keys() == {"50", "25", "5"}
    for key in truncation:
        truncation_budget = truncation[key].command[
            truncation[key].command.index("--budget-fraction") + 1
        ]
        selected_budget = selected[key].command[
            selected[key].command.index("--budget-fraction") + 1
        ]
        assert truncation_budget == selected_budget


def test_fixed50_campaign_hydrates_ids_and_keeps_treatments_locked() -> None:
    campaign = CampaignConfig.load(SWEBENCH_CONFIG)
    assert [baseline.published_resolved for baseline in campaign.baselines] == [7, 19]
    assert all(len(baseline.task_ids) == 50 for baseline in campaign.baselines)
    assert all(not cell.enabled for cell in campaign.cells)
    treatments = [cell for cell in campaign.cells if cell.baseline_cell]
    assert treatments
    assert all(cell.baseline_cell == "gemma4-31b-no-pra" for cell in treatments)
    assert all(cell.minimum_baseline_score == 0.20 for cell in treatments)
    assert all("raise SystemExit" not in " ".join(cell.command) for cell in treatments)


def test_fixed50_campaign_dry_run_cannot_admit_pra(tmp_path: Path) -> None:
    payload = yaml.safe_load(SWEBENCH_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "swebench")
    for cell in payload["cells"]:
        cell["enabled"] = True
    config = ROOT / "experiments/paper4_5_agent/configs/campaigns/.tmp_swebench_test.yaml"
    try:
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        state = run_campaign(config, max_hours=1, resume=False, dry_run=True)
    finally:
        config.unlink(missing_ok=True)
    assert state["cells"]["qwen3-coder-30b-no-pra"]["state"] == "PENDING"
    assert state["cells"]["gemma4-31b-no-pra"]["state"] == "PENDING"
    assert state["cells"]["gemma4-pra-50"]["state"] == "BLOCKED"


def _fixed50_execution_identity(baseline: object) -> dict[str, object]:
    return {
        "cohort_sha256": baseline.task_ids_sha256,
        "benchmark_revision": baseline.benchmark_revision,
        "harness": baseline.harness,
        "harness_version": baseline.harness_version,
        "model": baseline.model,
        "engine": baseline.engine,
        "engine_version": baseline.engine_version,
        "dtype": baseline.dtype,
        "quantization": baseline.quantization,
        "kv_cache_dtype": baseline.kv_cache_dtype,
        "scaffold": baseline.scaffold,
        "context_limit": baseline.context_limit,
        "max_steps": baseline.max_steps,
        "temperature": baseline.temperature,
        "function_calling": baseline.function_calling,
        "prefix_caching": baseline.prefix_caching,
        "grading": baseline.grading,
    }


def test_baseline_score_floor_blocks_weak_but_reproduced_cell(tmp_path: Path) -> None:
    payload = yaml.safe_load(SWEBENCH_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "swebench")
    treatment = payload["cells"][2]
    treatment["baseline_id"] = "qwen3-coder-30b-fixed50"
    treatment["baseline_cell"] = "qwen3-coder-30b-no-pra"
    payload["cells"] = [payload["cells"][0], treatment]
    config = ROOT / "experiments/paper4_5_agent/configs/campaigns/.tmp_swebench_gate.yaml"
    try:
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        loaded = CampaignConfig.load(config)
        baseline_ids = list(loaded.baselines[0].task_ids)
        result = tmp_path / "weak.json"
        result.write_text(json.dumps({
            "official_grader": True, "score": 0.14, "resolved": 7, "total": 50,
            "task_ids": baseline_ids, "configuration_differences": [],
            "execution_identity": _fixed50_execution_identity(loaded.baselines[0]),
        }), encoding="utf-8")
        record_result(config, cell_id="qwen3-coder-30b-no-pra", result_path=result)
        with pytest.raises(ValueError, match="baseline score >= 0.200"):
            record_result(config, cell_id="gemma4-gateway-passthrough", result_path=result)
    finally:
        config.unlink(missing_ok=True)


def test_fixed50_score_without_execution_identity_does_not_unlock() -> None:
    campaign = CampaignConfig.load(SWEBENCH_CONFIG)
    baseline = campaign.baselines[1]
    result = OfficialResult(
        official_grader=True, score=0.38, resolved=19, total=50,
        task_ids=baseline.task_ids,
    )
    review = review_result(
        baseline, result, absolute_tolerance=0.10, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_ATTEMPTED
    assert any("structured execution identity" in reason for reason in review.reasons)


def test_local_calibration_admits_only_official_identity_matched_useful_score() -> None:
    payload = yaml.safe_load(SWEBENCH_CONFIG.read_text(encoding="utf-8"))["baselines"][0]
    payload.update(
        admission_kind="local_calibration",
        published_score=None,
        minimum_admission_score=0.2,
        maximum_admission_score=0.8,
    )
    baseline = PublishedBaseline.model_validate(payload)
    identity = _fixed50_execution_identity(baseline)
    result = OfficialResult(
        official_grader=True,
        score=0.4,
        resolved=20,
        total=50,
        task_ids=baseline.task_ids,
        execution_identity=identity,
    )
    review = review_result(
        baseline, result, absolute_tolerance=0.0, require_exact_cohort=True
    )
    assert review.status == ReproductionStatus.BASELINE_REPRODUCED
    assert review.published_score is None


def test_swebench_chunk_report_requires_exact_ids(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "submitted_ids": ["a", "b"], "resolved_ids": ["b"], "error_ids": [],
    }), encoding="utf-8")
    assert _normalize_report(report, ["a", "b"])["resolved_ids"] == ["b"]
    with pytest.raises(RuntimeError, match="frozen chunk"):
        _normalize_report(report, ["a", "c"])


def test_swebench_completed_chunks_reach_final_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index, instance_id in enumerate(("repo__project-1", "repo__project-2")):
        chunk = tmp_path / f"chunk_{index:02d}"
        chunk.mkdir()
        (chunk / "official_chunk_result.json").write_text(json.dumps({
            "submitted_ids": [instance_id],
            "resolved_ids": [instance_id] if index == 0 else [],
            "error_ids": [],
        }), encoding="utf-8")
    monkeypatch.setattr(
        "experiments.paper4_5_agent.runners.swebench_verified._write_task_rows",
        lambda *args, **kwargs: None,
    )
    args = SimpleNamespace(
        chunk_size=1, model="model", model_revision="model-revision",
        tokenizer_revision="tokenizer-revision", engine="ollama",
        engine_version="1", dtype="mixed", quantization="Q4_K_M",
        kv_cache_dtype="f16", scaffold="swebench_backticks.yaml",
        context_limit=32768, max_steps=50, grading="official",
        benchmark_revision="revision", harness_version="2.4.6",
    )
    card = {
        "instance_ids": ["repo__project-1", "repo__project-2"],
        "canonical_ids_sha256": "digest", "source_revision": "revision",
    }

    result_path = _execute_chunks(
        args, card, tmp_path, "http://unused", {"configuration_differences": []},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["score"] == 0.5
    assert result["resolved"] == 1
    assert result["timeouts"] == 0
    assert result["configuration_differences"] == []


def test_timed_out_agent_becomes_an_empty_official_prediction(tmp_path: Path) -> None:
    destination = tmp_path / "preds.json"
    _write_empty_predictions(destination, ["repo__project-1"], "model")
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "repo__project-1": {
            "model_name_or_path": "openai/model",
            "instance_id": "repo__project-1",
            "model_patch": "",
        }
    }


def test_timeout_cleanup_targets_only_emitted_owned_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = (
        "Started container minisweagent-a1b2 with ID x\n"
        "container sweb.eval.repo__task.run-1 completed\n"
        "unrelated production-container must remain"
    )

    cleaned = _cleanup_owned_containers(output)

    assert cleaned == ["minisweagent-a1b2", "sweb.eval.repo__task.run-1"]
    assert commands == [
        ["docker", "rm", "-f", "minisweagent-a1b2"],
        ["docker", "rm", "-f", "sweb.eval.repo__task.run-1"],
    ]


def test_swebench_patch_apply_failure_is_not_a_generic_grader_error(tmp_path: Path) -> None:
    log = tmp_path / "grader.log"
    log.write_text(
        "sympy__sympy-21847: >>>>> Patch Apply Failed: only garbage found",
        encoding="utf-8",
    )
    assert _grader_error_type(log, "sympy__sympy-21847") == "patch_apply_failed"
    assert _grader_error_type(log, "another__task-1") == "official_grader_error"


def test_swebench_package_probe_uses_null_for_missing_distributions() -> None:
    versions = package_versions()
    assert set(versions) == {"mini-swe-agent", "swebench", "vllm"}
    assert all(value is None or isinstance(value, str) for value in versions.values())


def test_official_result_rejects_inconsistent_score_and_ids() -> None:
    with pytest.raises(ValueError, match="score must equal"):
        OfficialResult(official_grader=True, score=0.5, resolved=1, total=3)
    with pytest.raises(ValueError, match="task_ids length"):
        OfficialResult(
            official_grader=True, score=0.5, resolved=1, total=2,
            task_ids=("only-one",),
        )
    with pytest.raises(ValueError, match="timeout count"):
        OfficialResult(
            official_grader=True, score=0.0, resolved=0, total=2, timeouts=3,
        )


def test_h100_preflight_accepts_nvidia_smi_mib_format() -> None:
    assert _is_h100_80gb("NVIDIA H100 80GB HBM3, 81559 MiB")
    assert not _is_h100_80gb("NVIDIA H100 PCIe, 61440 MiB")
    assert not _is_h100_80gb("NVIDIA RTX 4090, 24564 MiB")


def test_context_treatments_share_budget_and_keep_mandatory_messages() -> None:
    payload = {
        "model": "model",
        "messages": [
            {"role": "system", "content": "work carefully"},
            {"role": "user", "content": "repository task statement"},
            {"role": "assistant", "content": "I inspected alpha module ordinary details"},
            {"role": "user", "content": "tool output needle_value appears in alpha.py"},
            {"role": "assistant", "content": "more unrelated trajectory words here"},
            {"role": "user", "content": "where is needle_value defined"},
        ],
    }
    truncated, truncation = transform_chat_payload(
        payload, mode=ContextTreatment.TRUNCATION, budget_fraction=0.5,
    )
    selected, pra = transform_chat_payload(
        payload, mode=ContextTreatment.PRA_SELECTED_CONTEXT, budget_fraction=0.5,
    )
    assert truncated["messages"][0] == payload["messages"][0]
    assert truncated["messages"][-1] == payload["messages"][-1]
    assert selected["messages"] == [payload["messages"][0], payload["messages"][-1]]
    assert any("needle_value" in row["text"] for row in selected["pra"]["resources"])
    assert truncation.logical_input_tokens_estimate == pra.logical_input_tokens_estimate
    assert truncation.session_id == pra.session_id == selected["pra"]["session_id"]
    assert truncation.mandatory_tokens_estimate == pra.mandatory_tokens_estimate
    assert truncation.physical_input_tokens_estimate <= truncation.logical_input_tokens_estimate
    assert pra.physical_input_tokens_estimate <= pra.logical_input_tokens_estimate


def test_passthrough_does_not_rewrite_openai_payload() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hello world"}]}
    transformed, trace = transform_chat_payload(
        payload, mode=ContextTreatment.PASSTHROUGH, budget_fraction=1.0,
    )
    assert transformed == payload
    assert trace.tokens_avoided_estimate == 0
    assert trace.selected_tokens_estimate == 0


def test_minisweagent_trajectory_metrics_preserve_exact_usage(tmp_path: Path) -> None:
    path = tmp_path / "repo__project-1.traj.json"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task statement"},
        {
            "role": "assistant", "content": "inspect",
            "extra": {
                "timestamp": 10.0, "actions": [{"command": "cat file"}],
                "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            },
        },
        {"role": "user", "content": "output", "extra": {"timestamp": 11.5}},
        {
            "role": "assistant", "content": "finish",
            "extra": {
                "timestamp": 15.0, "actions": [{"command": "patch"}],
                "response": {"usage": {"prompt_tokens": 180, "completion_tokens": 30}},
            },
        },
        {"role": "exit", "content": "done", "extra": {"timestamp": 16.0}},
    ]
    path.write_text(json.dumps({
        "messages": messages,
        "info": {"exit_status": "Submitted", "submission": "diff\n+line\n"},
    }), encoding="utf-8")

    metrics = _trajectory_metrics(path)

    assert metrics["cumulative_prompt_tokens"] == 280
    assert metrics["max_prompt_tokens"] == 180
    assert metrics["repeated_context_tokens_estimate"] == 100
    assert metrics["repeated_context_fraction_estimate"] == pytest.approx(100 / 280)
    assert metrics["output_tokens"] == 50
    assert metrics["model_call_count"] == metrics["tool_call_count"] == 2
    assert metrics["wall_time_s"] == 6.0
    assert metrics["tool_time_s"] == 1.5
    assert metrics["termination_reason"] == "Submitted"
    assert metrics["patch"] == "diff\n+line\n"


def test_treatment_trace_aggregation_keeps_estimates_disjoint() -> None:
    rows = [
        {
            "logical_input_tokens_estimate": 100,
            "physical_input_tokens_estimate": 60,
            "selected_tokens_estimate": 30,
            "route_time_s": 0.1,
            "token_estimator": "whitespace_v1",
        },
        {
            "logical_input_tokens_estimate": 200,
            "physical_input_tokens_estimate": 100,
            "selected_tokens_estimate": 50,
            "route_time_s": 0.2,
            "token_estimator": "whitespace_v1",
        },
    ]

    aggregate = _aggregate_traces(rows)

    assert aggregate["logical_input_tokens_estimate"] == 300
    assert aggregate["physical_input_tokens_estimate"] == 160
    assert aggregate["selected_tokens_estimate"] == 80
    assert aggregate["tokens_avoided_estimate"] == 140
    assert aggregate["token_saving_fraction_estimate"] == pytest.approx(140 / 300)
    assert aggregate["route_time_s"] == pytest.approx(0.3)


def test_treatment_proxy_forwards_selected_context_and_writes_trace(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            observed.update(json.loads(self.rfile.read(length)))
            body = b'{"choices":[{"message":{"content":"ok"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    trace_path = tmp_path / "request_telemetry.jsonl"
    proxy = TreatmentProxy(
        f"http://127.0.0.1:{target.server_port}/v1",
        mode=ContextTreatment.PRA_SELECTED_CONTEXT,
        budget_fraction=0.5,
        trace_path=trace_path,
    )
    proxy_url = proxy.start()
    try:
        payload = json.dumps({
            "model": "model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task alpha"},
                {"role": "assistant", "content": "alpha evidence details"},
                {"role": "user", "content": "find alpha"},
            ],
        }).encode()
        request = urllib.request.Request(
            f"{proxy_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["choices"][0]["message"]["content"] == "ok"
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=5)
    assert observed["pra"]["metadata"]["benchmark_fairness"] == "agent-visible-messages-only"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["mode"] == "gateway-pra"
    assert trace["physical_input_tokens_estimate"] <= trace["logical_input_tokens_estimate"]


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
    assert (output / "precision_report.md").is_file()
    assert (output / "engine_report.md").is_file()
    assert (output / "pra_frontier_report.md").is_file()
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
        "mini-swe-agent-2.4.6", "qwen-code-0.23.0", "aider-0.86.2",
    }
    assert len({cell[3] for cell in cells[:15]}) == 5
    aider = next(row for row in config.harnesses if row.agent == "aider")
    assert aider.kwargs == {
        "stream": False, "auto_lint": False, "auto_test": False,
    }


def test_harness_matrix_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        HarnessMatrixConfig.load(path)


def test_completed_attempt_finds_terminal_interrupted_job(tmp_path: Path) -> None:
    attempt = tmp_path / "attempts/a002/job"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1, "n_cancelled_trials": 0,
            "n_running_trials": 0,
        },
    }), encoding="utf-8")
    assert _completed_attempt(tmp_path) == tmp_path / "attempts/a002"


def test_completed_attempt_ignores_running_job(tmp_path: Path) -> None:
    attempt = tmp_path / "attempts/a001/job"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "n_total_trials": 1,
        "stats": {"n_completed_trials": 0, "n_running_trials": 1},
    }), encoding="utf-8")
    assert _completed_attempt(tmp_path) is None


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
    assert "version=2.4.6" in command
    assert "max_tokens=8192" in command

    aider = next(row for row in config.harnesses if row.agent == "aider")
    aider_command = harbor_command(
        harbor="harbor", manifest=manifest, model=model, harness=aider,
        task_id=task_id, job_directory=Path("jobs"),
        base_url="http://model-host:11435/v1", api_key="secret",
    )
    assert aider_command[aider_command.index("-m") + 1] == (
        "openai/openai/qwen3-coder:30b"
    )


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
    assert summary["summary"]["tasks_solved_any"] == 0
    assert summary["summary"]["token_reported_runs"] == 0
    assert summary["admission_gate"]["eligible"] is False
    report = (tmp_path / "matrix/report.md").read_text(encoding="utf-8")
    assert "official-Harbor **No-PRA baseline**" in report
    assert "Completed: `0/30`" in report


def test_pre_inference_harness_failure_is_not_quality_evidence() -> None:
    row = SimpleNamespace(
        outcome=SimpleNamespace(failure_kind="NonZeroAgentExitCodeError"),
        behavior=SimpleNamespace(model_calls=0),
        tokens=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    assert "excluded" in (_invalid_trial_reason(row) or "")


def test_scored_model_failure_remains_admissible() -> None:
    row = SimpleNamespace(
        outcome=SimpleNamespace(failure_kind="official_score_below_success"),
        behavior=SimpleNamespace(model_calls=4),
        tokens=SimpleNamespace(input_tokens=1200, output_tokens=80),
    )
    assert _invalid_trial_reason(row) is None


def test_agent_timeout_without_adapter_telemetry_remains_admissible() -> None:
    row = SimpleNamespace(
        outcome=SimpleNamespace(failure_kind="AgentTimeoutError"),
        behavior=SimpleNamespace(model_calls=0),
        tokens=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    assert _invalid_trial_reason(row) is None
