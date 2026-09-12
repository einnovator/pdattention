from __future__ import annotations

import json
from pathlib import Path

from experiments.paper4_5_agent.summarize_cross_engine_agent_gate import (
    REQUIRED_ENGINES,
    summarize,
)


def _gate(path: Path, *, consumer_bytes: int | None = 10) -> None:
    payload = {
        "plain": {"task_success": True, "calls_to_solution": 12},
        "pra_100": {
            "task_success": True,
            "calls_to_solution": 12,
            "selected_history_reencoded_tokens": 0,
            "physical_kv_copy_bytes": 0,
            "total_kv_copy_bytes": 0,
            "consumer_temporary_bytes": consumer_bytes,
            "consumer_temporary_peak_bytes": consumer_bytes,
        },
        "pra_90": {
            "realized_retention_fraction": 0.91,
            "selected_history_reencoded_tokens": 0,
            "physical_kv_copy_bytes": 0,
            "total_kv_copy_bytes": 0,
            "consumer_temporary_bytes": consumer_bytes,
            "consumer_temporary_peak_bytes": consumer_bytes,
            "calls_to_solution": 12,
        },
        "gates": {
            "pra_100_exact_action_trajectory": True,
            "engine_task_gate_complete": True,
        },
        "classification": "retention_quality_result_at_90_percent",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_engine_blocks_expansion(tmp_path: Path) -> None:
    path = tmp_path / "llama.json"
    _gate(path)
    result = summarize({"llama.cpp": path})
    assert result["behavioral_complete_engines"] == 1
    assert result["measurement_complete_engines"] == 1
    assert result["easy14_expansion_allowed"] is False


def test_missing_consumer_telemetry_is_not_zero(tmp_path: Path) -> None:
    path = tmp_path / "llama.json"
    _gate(path, consumer_bytes=None)
    result = summarize({"llama.cpp": path})
    assert result["behavioral_complete_engines"] == 1
    assert result["measurement_complete_engines"] == 0
    assert result["engines"]["llama.cpp"]["status"] == (
        "behavioral_complete_measurement_pending"
    )


def test_plain_admission_failure_is_not_reported_as_pra_failure(tmp_path: Path) -> None:
    path = tmp_path / "mlx-admission.json"
    path.write_text(json.dumps({
        "model": "mlx-community/Qwen3-14B-4bit",
        "task": "django__django-15277",
        "chat_template": {
            "two_turn_byte_prefix_stable": True,
            "two_turn_token_prefix_stable": True,
        },
        "plain_admission": {
            "resolved": False,
            "model_calls": 50,
            "termination_reason": "LimitsExceeded",
        },
        "pra_100": {"status": "not_run"},
        "pra_090": {"status": "not_run"},
    }), encoding="utf-8")
    result = summarize({}, {"mlx": path})
    row = result["engines"]["mlx"]
    assert row["status"] == "admission_failed_model_task_pair"
    assert row["classification"] == (
        "plain_model_task_admission_failure_not_pra_failure"
    )
    assert row["behavioral_gate_complete"] is False
    assert result["easy14_expansion_allowed"] is False


def test_vllm_conditions_admission_schema_is_reduced(tmp_path: Path) -> None:
    path = tmp_path / "vllm-admission.json"
    path.write_text(json.dumps({
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "task": {"instance_id": "django__django-15368"},
        "conditions": {
            "plain": {
                "status": "failed_model_task_admission_no_progress_cutoff",
                "task_success": None,
                "termination": "bounded cutoff",
                "interface_controls": [
                    {"completed_api_calls": 19, "status": "accepted_actions"}
                ],
            },
            "pra_100": {"status": "not_run_plain_admission_gate_failed"},
            "pra_90": {"status": "not_run_pra_100_gate_not_reached"},
        },
    }), encoding="utf-8")
    result = summarize({}, {"vllm": path})
    admission = result["engines"]["vllm"]["admission"]
    assert result["engines"]["vllm"]["status"] == (
        "admission_failed_model_task_pair"
    )
    assert admission["task"] == "django__django-15368"
    assert admission["model_calls"] == 19
    assert admission["plain_official_solve"] is False
    assert admission["pra_100_run"] is False
    assert admission["pra_90_run"] is False


def test_sglang_bounded_plain_screen_is_reduced(tmp_path: Path) -> None:
    path = tmp_path / "sglang-admission.json"
    path.write_text(json.dumps({
        "schema_version": "paper4.5.sglang-agent-plain-admission.v1",
        "model": "mlx-community/Qwen3-14B-4bit",
        "instance_id": "django__django-15368",
        "status": "stopped_by_bounded_no_progress_gate",
        "task_success": None,
        "accepted_tool_actions": 4,
        "pra_100_executed": False,
        "pra_90_executed": False,
    }), encoding="utf-8")
    result = summarize({}, {"sglang-mlx": path})
    admission = result["engines"]["sglang-mlx"]["admission"]
    assert result["engines"]["sglang-mlx"]["status"] == (
        "admission_failed_model_task_pair"
    )
    assert admission["task"] == "django__django-15368"
    assert admission["model_calls"] == 4
    assert admission["plain_official_solve"] is False


def test_sglang_completed_diagnostic_schema_is_reduced(tmp_path: Path) -> None:
    path = tmp_path / "sglang-diagnostic.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "classification": "completed_plain_admission_failure",
        "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "task": "django__django-15368",
        "outcome": {
            "official_resolved": False,
            "exit_status": "RepeatedFormatError",
            "api_calls": 4,
            "accepted_actions": 1,
            "files_modified": 0,
        },
        "pra_arms_run": False,
    }), encoding="utf-8")
    result = summarize({}, {"sglang-mlx": path})
    row = result["engines"]["sglang-mlx"]
    admission = row["admission"]
    assert row["status"] == "admission_failed_model_task_pair"
    assert admission["task"] == "django__django-15368"
    assert admission["model_calls"] == 4
    assert admission["plain_official_solve"] is False
    assert admission["termination_reason"] == "RepeatedFormatError"
    assert admission["pra_100_run"] is False
    assert admission["pra_90_run"] is False


def test_vllm_hardware_throughput_failure_stays_distinct(tmp_path: Path) -> None:
    path = tmp_path / "vllm-14b-admission.json"
    path.write_text(json.dumps({
        "schema_version": "paper4.5.frozen-agent-engine-admission.v1",
        "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        "task": {"instance_id": "django__django-15368"},
        "conditions": {
            "plain": {
                "status": "hardware_inadmissible_for_autonomous_task_throughput",
                "task_success": None,
                "completed_tool_calls": 2,
            },
            "pra_100": {"status": "not_run_plain_admission_gate_failed"},
            "pra_90": {"status": "not_run_pra_100_gate_not_reached"},
        },
    }), encoding="utf-8")
    result = summarize({}, {"vllm": path})
    row = result["engines"]["vllm"]
    assert row["status"] == "admission_failed_hardware_throughput"
    assert row["classification"] == (
        "plain_hardware_throughput_admission_failure_not_pra_failure"
    )
    assert row["admission"]["model_calls"] == 2
    assert row["admission"]["plain_official_solve"] is False


def test_hf_frozen_task_hardware_admission_schema_is_reduced(tmp_path: Path) -> None:
    path = tmp_path / "hf-bnb-admission.json"
    path.write_text(json.dumps({
        "schema_version": "paper4.5-hf-bnb-agent-admission-v1",
        "model": {"model_id": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        "frozen_task01_plain": {
            "instance_id": "django__django-15277",
            "resolved": False,
            "agent_actions_completed": 0,
            "failure_type": "CUDA_out_of_memory",
            "classification": (
                "model_server_admitted_but_task_prompt_hardware_inadmissible"
            ),
        },
        "pra_100": None,
        "pra_90": None,
        "outcome": {"plain_model_task_pair_admitted": False},
    }), encoding="utf-8")
    result = summarize({}, {"huggingface": path})
    row = result["engines"]["huggingface"]
    assert row["status"] == "admission_failed_hardware_capacity"
    assert row["classification"] == (
        "plain_hardware_capacity_admission_failure_not_pra_failure"
    )
    assert row["admission"]["task"] == "django__django-15277"
    assert row["admission"]["model_calls"] == 0
    assert row["admission"]["plain_official_solve"] is False
    assert row["admission"]["pra_100_run"] is False
    assert row["admission"]["pra_90_run"] is False


def test_all_behavior_and_measurements_authorize_expansion(tmp_path: Path) -> None:
    paths = {}
    for engine in REQUIRED_ENGINES:
        path = tmp_path / f"{engine}.json"
        _gate(path)
        paths[engine] = path
    result = summarize(paths)
    assert result["behavioral_complete_engines"] == len(REQUIRED_ENGINES)
    assert result["measurement_complete_engines"] == len(REQUIRED_ENGINES)
    assert result["easy14_expansion_allowed"] is True


def test_published_cross_engine_gate_stays_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents"
        / "engine_agent_gates/cross_engine_gate.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["behavioral_complete_engines"] == 1
    assert result["measurement_complete_engines"] == 0
    assert result["engines"]["llama.cpp"]["behavioral_gate_complete"] is True
    assert result["engines"]["llama.cpp"]["outcome_family_checks"][
        "consumer_temporary_bytes_and_calls"
    ] is False
    assert result["easy14_expansion_allowed"] is False
