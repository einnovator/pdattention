from __future__ import annotations

import json
from pathlib import Path

import experiments.paper4_5_agent.prepare_easy3_direct_matrix as matrix_module
from experiments.paper4_5_agent.prepare_easy3_direct_matrix import (
    EXPECTED_ARMS,
    build_plan,
    validate_config,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "experiments/paper4_5_agent/configs/campaigns/"
    "swebench_easy3_direct_matrix.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_easy3_matrix_locks_tasks_factors_and_engine_gate() -> None:
    config = _config()
    qualification = validate_config(config, ROOT)

    assert qualification["valid"] is True
    assert qualification["engine_qualified"] is True
    assert config["locked_task_ids"] == [
        "django__django-15277",
        "django__django-15368",
        "scikit-learn__scikit-learn-13135",
    ]
    assert tuple(
        (row["label"], row["pra"], row["retention_fraction"], row["adaptor"])
        for row in config["arms"]
    ) == EXPECTED_ARMS
    assert config["execution"]["agent_host"].endswith("192.168.1.8:2222")
    assert config["execution"]["model_engine_host"].endswith("192.168.1.6")


def test_easy3_plan_is_task_major_and_never_executes_tasks(monkeypatch) -> None:
    config = _config()
    monkeypatch.setenv(
        "PRA_AGENT_NATIVE_ENGINE_CACHE_ON_URL", "http://127.0.0.1:18082/v1",
    )
    plan = build_plan(config, ROOT)

    assert plan["expensive_task_runs_started"] == 0
    assert plan["expected_cells"] == len(plan["cells"]) == 15
    assert [row["cell_id"] for row in plan["cells"][:5]] == [
        f"task-01-{arm[0]}" for arm in EXPECTED_ARMS
    ]
    assert [row["task_id"] for row in plan["cells"][:5]] == [
        "django__django-15277"
    ] * 5
    assert all(row["seed"] == 0 and row["temperature"] == 0 for row in plan["cells"])
    assert all(
        row["model_revision"] == plan["cells"][0]["model_revision"]
        for row in plan["cells"]
    )
    assert all(
        "--preflight-only" in row["qualification_argv"]
        for row in plan["cells"] if row["qualification_argv"]
    )
    for row in plan["cells"]:
        if row["run_argv"]:
            assert row["run_argv"][row["run_argv"].index("--sampling-seed") + 1] == "0"
            assert row["run_argv"][row["run_argv"].index("--top-p") + 1] == "1"
            assert "gateway" not in " ".join(row["run_argv"])
    assert all(
        row["locked_source_commit"] == "fcde2c74"
        and row["actual_git_head"] == plan["actual_git_head"]
        for row in plan["cells"]
    )


def test_source_commit_qualification_accepts_descendant_and_rejects_unrelated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(matrix_module, "_git_head", lambda root: "descendant-head")
    monkeypatch.setattr(
        matrix_module, "_git_contains",
        lambda root, ancestor, head: ancestor == "fcde2c74" and head == "descendant-head",
    )
    accepted = validate_config(_config(), ROOT)
    assert accepted["valid"] is True
    assert accepted["git_head"] == "descendant-head"

    monkeypatch.setattr(matrix_module, "_git_contains", lambda *args: False)
    rejected = validate_config(_config(), ROOT)
    assert rejected["valid"] is False
    assert any("does not contain locked source" in error for error in rejected["errors"])


def test_frozen_adaptor_cells_fail_closed_until_immutable_bundle_exists(monkeypatch) -> None:
    monkeypatch.setenv(
        "PRA_AGENT_NATIVE_ENGINE_CACHE_ON_URL", "http://127.0.0.1:18082/v1",
    )
    plan = build_plan(_config(), ROOT)
    adaptor = [row for row in plan["cells"] if row["arm"]["adaptor"] == "frozen_bundle"]
    no_adaptor = [row for row in plan["cells"] if row["arm"]["adaptor"] == "none"]

    assert len(adaptor) == 6
    assert all(row["status"] == "BLOCKED" and row["run_argv"] is None for row in adaptor)
    assert all(
        any("adaptor" in reason.lower() for reason in row["blockers"])
        for row in adaptor
    )
    assert all(row["status"] == "CONFIG_QUALIFIED" for row in no_adaptor)
    contract = plan["qualification"]["adaptor_qualification"]
    assert contract["semantic_role"] == "learned_routing_selection"
    assert contract[
        "certified_qwen3_coder_30b_memory_or_late_band_training_path"
    ] is False
    assert "frozen selected-record replay" in contract["explicitly_not"]
    assert any(
        "fails closed" in value
        for value in contract["required_injection_contract"]
    )


def test_plan_writes_one_incremental_manifest_per_example_cell(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PRA_AGENT_NATIVE_ENGINE_CACHE_ON_URL", "http://127.0.0.1:18082/v1",
    )
    plan = build_plan(_config(), ROOT)
    destination = tmp_path / "qualification.json"
    write_plan(plan, destination)

    manifests = sorted((tmp_path / "cells").glob("*/cell_manifest.json"))
    assert len(manifests) == 15
    assert json.loads(manifests[0].read_text())["expected_incremental_artifacts"] == [
        "cell_manifest.json", "run_manifest.json", "request_telemetry.jsonl",
        "interaction_history.jsonl", "selection_fixture.jsonl", "results.jsonl",
        "official_result.json", "divergence.json",
    ]
