from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.run_autonomous_multi_issue_campaign import (
    TARGET_SAVING_MAX,
    TARGET_SAVING_MIN,
    _control_cell_id,
    _episode_command,
    _aggregate,
    _divergence_accounting,
    _lost_paired_full_successes,
    _partial_aggregate,
    _requires_same_prefix_full_control,
    _same_prefix_full_cell,
    _shared_control_prefix_count,
    _validate_same_prefix_full_qualification,
    _validate_sequence_pairing_identity,
    campaign_cells,
    run_campaign,
    validate_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_multi_issue_frontier_v1.json"
)
BENCHMARK = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "benchmarks" /
    "easy14_longest_success5.json"
)
BASELINE_SUCCESS14_BENCHMARK = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "benchmarks" /
    "easy14_baseline_success14.json"
)
CONFIRMATION_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_persistent_spine_confirmation_v1.json"
)
N3_REPLACEMENT_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_persistent_spine_n3_replacement_v1.json"
)
N3_RELIABILITY_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_persistent_spine_n3_reliability_v1.json"
)
GLOBAL_N3_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_persistent_global_n3_all_user_r4m2v2p1_v2.json"
)
INSTRUCTION_EPOCH_N3_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_persistent_instruction_epoch_n3_e0_v1.json"
)
EASY14_ATOMIC_PROFILES_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_easy14_atomic_profiles_v1.json"
)
EASY14_ATOMIC_PROFILES_CTX131K_SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_easy14_atomic_profiles_v2_ctx131k.json"
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        spec=SPEC,
        output=tmp_path / "campaign",
        upstream_base_url="http://127.0.0.1:11435",
        tokenizer="locked-tokenizer",
        docker_executable=None,
        docker_platform=None,
        pythonpath=[],
        sequence_id=["independent-n2-a"],
        strategy_id=["S00_fresh_full", "S01_persistent_full"],
        strategy_config_id=None,
        max_cells=None,
        skip_grading=False,
        grade_auxiliary_workspace_state=False,
        health_probe_count=3,
        health_latency_ceiling_seconds=60.0,
        health_timeout_seconds=90.0,
        upstream_request_timeout_seconds=180,
        dry_run=True,
    )


def test_frozen_registry_has_unique_cells_and_explicit_primary_target():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)

    assert len(cells) == 48
    assert len({row["cell_id"] for row in cells}) == len(cells)
    assert spec["primary_target"]["minimum_saving_fraction"] == TARGET_SAVING_MIN
    assert spec["primary_target"]["maximum_saving_fraction"] == TARGET_SAVING_MAX
    assert spec["primary_target"]["official_resolution_delta_minimum"] == 0
    assert spec["primary_target"]["lost_persistent_full_successes"] == 0


def test_instruction_epoch_registry_is_boundary_free_and_uniquely_identified():
    spec = json.loads(INSTRUCTION_EPOCH_N3_SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)

    assert len(cells) == 2
    treatment = next(
        row for row in cells
        if row["policy"] == "persistent_instruction_epoch_retirement"
    )
    assert treatment["strategy"]["boundary_mode"] == "boundary_free"
    assert treatment["strategy"]["completed_recent_turns"] == 0
    assert "e0_all_user_active_epoch_full_v1" in treatment["cell_id"]


def test_shared_full_prefix_count_supports_legacy_and_exact_prefix_forks():
    assert _shared_control_prefix_count({}) == 0
    assert _shared_control_prefix_count({"share_full_first_episode": True}) == 1
    assert _shared_control_prefix_count({"share_full_prefix_episodes": 4}) == 4
    with pytest.raises(ValueError, match="nonnegative integer"):
        _shared_control_prefix_count({"share_full_prefix_episodes": True})
    with pytest.raises(ValueError, match="conflicts"):
        _shared_control_prefix_count({
            "share_full_first_episode": True,
            "share_full_prefix_episodes": 0,
        })


def test_every_persistent_treatment_requires_exact_prefix_full_control():
    treatment = {
        "cell_id": "candidate",
        "session_mode": "persistent",
        "policy": "frontier_dag_retirement",
        "strategy": {
            "policy": "frontier_dag_retirement",
            "budget_fraction": 0.5,
            "materialization_mode": "tool_structured_evidence",
            "keep_completed_task_statements": False,
        },
    }
    assert _requires_same_prefix_full_control(treatment) is True
    assert _requires_same_prefix_full_control({
        **treatment, "policy": "full"
    }) is False

    control = _same_prefix_full_cell(treatment)
    assert control["policy"] == "full"
    assert control["strategy"]["budget_fraction"] == 1.0
    assert control["strategy"]["materialization_mode"] == "whole_record"
    assert control["strategy"]["keep_completed_task_statements"] is True


def test_same_prefix_qualification_rejects_a_different_prefix(tmp_path):
    fields = {
        "instance_id": "repo__1",
        "pair_id": "pair",
        "model_revision": "model-r1",
        "tokenizer_revision": "tokenizer-r1",
        "dataset_revision": "dataset-r1",
        "benchmark_card_sha256": "benchmark",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_calls": 40,
        "max_completion_tokens": 2048,
        "scaffold_identity_sha256": "scaffold",
        "workspace_source_identity_sha256": "workspace",
    }
    control_output = tmp_path / "control"
    control_output.mkdir()
    control_manifest = {
        **fields,
        "created_at": "2026-09-17T10:00:00+00:00",
        "persistent_session": {
            "episode_index": 3,
            "prefix": {"sha256": "prefix-a", "episode_count": 2},
        },
        "selection": {"policy": "full"},
    }
    (control_output / "run_manifest.json").write_text(
        json.dumps(control_manifest), encoding="utf-8"
    )
    episode = {
        "same_prefix_full_control": {
            "status": "qualified", "output": str(control_output)
        }
    }
    candidate = {
        **fields,
        "created_at": "2026-09-17T10:01:00+00:00",
        "persistent_session": {
            "episode_index": 3,
            "prefix": {"sha256": "prefix-a", "episode_count": 2},
        },
    }
    _validate_same_prefix_full_qualification(
        candidate_episode=episode, candidate_manifest=candidate
    )

    candidate["persistent_session"]["prefix"]["sha256"] = "prefix-b"
    with pytest.raises(ValueError, match="did not consume the candidate prefix"):
        _validate_same_prefix_full_qualification(
            candidate_episode=episode, candidate_manifest=candidate
        )


def test_campaign_forwards_completed_instruction_epoch_floor(tmp_path):
    spec = json.loads(INSTRUCTION_EPOCH_N3_SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    spec["strategies"][1]["completed_instruction_epochs"] = 2
    cells = campaign_cells(spec)
    treatment = next(
        row for row in cells
        if row["policy"] == "persistent_instruction_epoch_retirement"
    )
    args = argparse.Namespace(
        upstream_base_url="http://127.0.0.1:1234",
        tokenizer="tokenizer",
        upstream_request_timeout_seconds=60,
        upstream_qualification_path=None,
        upstream_connect_attempts=1,
        upstream_connect_retry_seconds=0,
        upstream_relay_target=None,
        docker_executable="docker",
        docker_platform=None,
        pythonpath=[],
        skip_grading=False,
        grade_auxiliary_workspace_state=False,
    )
    command = _episode_command(
        spec=spec,
        benchmark=BENCHMARK,
        cell=treatment,
        instance_id=spec["sequences"][0]["instance_ids"][0],
        episode_number=1,
        output=tmp_path,
        prefix=None,
        session_id="session-test",
        args=args,
    )

    assert command[command.index("--completed-instruction-epochs") + 1] == "2"
    assert command[command.index("--docker-pull-timeout-seconds") + 1] == "900"


def test_campaign_registry_accepts_easy14_sequence():
    benchmark = json.loads(BASELINE_SUCCESS14_BENCHMARK.read_text(encoding="utf-8"))
    spec = json.loads(INSTRUCTION_EPOCH_N3_SPEC.read_text(encoding="utf-8"))
    spec["sequences"] = [{
        "sequence_id": "independent-n14-planned",
        "sequence_stratum": "independent_clean_workspace",
        "instance_ids": benchmark["instance_ids"],
    }]

    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)
    assert len(cells) == 2
    assert all(row["issue_count"] == 14 for row in cells)


def test_easy14_atomic_profile_registry_freezes_distinct_e1_e2_e3_cells():
    benchmark = json.loads(BASELINE_SUCCESS14_BENCHMARK.read_text(encoding="utf-8"))
    spec = json.loads(EASY14_ATOMIC_PROFILES_SPEC.read_text(encoding="utf-8"))

    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)

    assert len(cells) == 4
    assert all(row["issue_count"] == 14 for row in cells)
    assert cells[0]["strategy_id"] == "S01_persistent_full"
    candidates = cells[1:]
    assert [row["strategy"]["completed_instruction_epochs"] for row in candidates] == [
        2, 3, 1,
    ]
    assert all(row["strategy"]["retire_closed_instructions"] for row in candidates)
    assert all(row["strategy"]["boundary_mode"] == "boundary_free" for row in cells)


def test_easy14_ctx131k_registry_fails_closed_on_small_active_context():
    benchmark = json.loads(BASELINE_SUCCESS14_BENCHMARK.read_text(encoding="utf-8"))
    spec = json.loads(EASY14_ATOMIC_PROFILES_CTX131K_SPEC.read_text(encoding="utf-8"))

    validate_spec(spec, benchmark)
    assert spec["runtime_qualification"] == {
        "active_state_path": "/api/ps",
        "minimum_active_context_tokens": 131072,
        "reason": (
            "fail closed before a persistent FULL control can exceed the "
            "engine's active context"
        ),
    }
    assert all(row["issue_count"] == 14 for row in campaign_cells(spec))


def test_campaign_registry_rejects_more_than_twenty_issues():
    benchmark = json.loads(BASELINE_SUCCESS14_BENCHMARK.read_text(encoding="utf-8"))
    spec = json.loads(INSTRUCTION_EPOCH_N3_SPEC.read_text(encoding="utf-8"))
    spec["sequences"] = [{
        "sequence_id": "independent-n21-invalid",
        "sequence_stratum": "independent_clean_workspace",
        "instance_ids": [f"task-{index}" for index in range(21)],
    }]
    benchmark["instance_ids"] = list(spec["sequences"][0]["instance_ids"])

    with pytest.raises(ValueError, match="one to 20"):
        validate_spec(spec, benchmark)


def test_shared_first_episode_resolves_configured_control_cell_identity():
    treatment = {
        "sequence_id": "sequence-a",
        "repeat": 1,
        "strategy_id": "S03_instruction_epoch_retirement",
    }
    configured_control_id = (
        "sequence-a__S01_persistent_full-configured-control-v1__r01"
    )
    state_cells = {
        configured_control_id: {
            "sequence_id": "sequence-a",
            "repeat": 1,
            "strategy_id": "S01_persistent_full",
        },
        "sequence-a__S03_instruction_epoch_retirement-e0__r01": treatment,
    }

    assert _control_cell_id(treatment, state_cells) == configured_control_id


def test_confirmation_registry_freezes_repeats_and_held_out_sequence():
    spec = json.loads(CONFIRMATION_SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)

    assert len(cells) == 36
    assert len({row["cell_id"] for row in cells}) == 36
    r4 = [
        row for row in cells
        if row["strategy_config_id"] == "r4m2v2_tasks1"
        and row["sequence_id"] == "independent-n2-a"
    ]
    assert [row["repeat"] for row in r4] == [1, 2]
    held_out = next(
        row for row in spec["sequences"]
        if row["sequence_id"] == "independent-n2-b"
    )
    assert held_out["instance_ids"] == [
        "sphinx-doc__sphinx-8721", "scikit-learn__scikit-learn-13135"
    ]


def test_n3_replacement_uses_independent_baseline_success_third_issue():
    spec = json.loads(N3_REPLACEMENT_SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    validate_spec(spec, benchmark)
    sequence = spec["sequences"][0]

    assert sequence["sequence_id"] == "independent-n3-b"
    assert sequence["instance_ids"] == [
        "django__django-15277",
        "pytest-dev__pytest-7982",
        "scikit-learn__scikit-learn-13135",
    ]
    assert len(campaign_cells(spec)) == 9


def test_n3_reliability_spec_predeclares_three_full_and_two_r4_repeats():
    spec = json.loads(N3_RELIABILITY_SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)

    full = [row for row in cells if row["strategy_id"] == "S01_persistent_full"]
    r4 = [
        row for row in cells if row["strategy_config_id"] == "r4m2v2_tasks1"
    ]
    assert [row["repeat"] for row in full] == [1, 2, 3]
    assert [row["repeat"] for row in r4] == [1, 2]
    assert spec["qualification"]["full_reliability_gate"].startswith(
        "at least two of three"
    )


def test_boundary_free_n3_spec_hides_boundaries_in_both_arms(tmp_path):
    spec = json.loads(GLOBAL_N3_SPEC.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)
    assert len(cells) == 2
    assert all(row["strategy"]["boundary_mode"] == "boundary_free" for row in cells)
    assert not any(
        row["strategy"].get("share_full_first_episode", False) for row in cells
    )
    candidate = next(row for row in cells if row["policy"] != "full")
    assert candidate["strategy"]["keep_completed_task_statements"] is True

    args = _args(tmp_path)
    args.spec = GLOBAL_N3_SPEC
    args.sequence_id = None
    args.strategy_id = None
    state = run_campaign(args)
    for cell in state["cells"].values():
        commands = [
            row["command"] for row in cell["episodes"].values()
            if row["command"] is not None
        ]
        assert commands
        assert all(
            command[command.index("--boundary-mode") + 1] == "boundary_free"
            for command in commands
        )


def test_dry_run_distinguishes_fresh_from_persistent_prefixes(tmp_path):
    state = run_campaign(_args(tmp_path))
    fresh = state["cells"]["independent-n2-a__S00_fresh_full__r01"]
    persistent = state["cells"]["independent-n2-a__S01_persistent_full__r01"]

    fresh_commands = [row["command"] for row in fresh["episodes"].values()]
    persistent_commands = [row["command"] for row in persistent["episodes"].values()]
    assert all("--persistent-prefix" not in command for command in fresh_commands)
    assert "--persistent-prefix" not in persistent_commands[0]
    assert "--persistent-prefix" in persistent_commands[1]
    assert persistent_commands[1][persistent_commands[1].index("--episode-index") + 1] == "2"


def test_dry_run_can_select_one_registered_strategy_configuration(tmp_path):
    args = _args(tmp_path)
    args.strategy_id = ["S03_completed_episode_spine"]
    args.strategy_config_id = ["r4m2v2_tasks1"]
    state = run_campaign(args)

    assert list(state["cells"]) == [
        "independent-n2-a__S03_completed_episode_spine-r4m2v2_tasks1__r01"
    ]
    cell = next(iter(state["cells"].values()))
    episodes = list(cell["episodes"].values())
    assert episodes[0]["shared_control_episode"] is True
    assert episodes[0]["command"] is None
    command = episodes[1]["command"]
    assert command[command.index("--completed-recent-turns") + 1] == "4"
    qualifier = episodes[1]["same_prefix_full_control"]
    assert qualifier["status"] == "planned"
    qualifier_command = qualifier["command"]
    assert qualifier_command[qualifier_command.index("--policy") + 1] == "full"
    assert (
        qualifier_command[qualifier_command.index("--persistent-prefix") + 1]
        == episodes[1]["prefix"]
    )
    assert episodes[1]["heuristic_attribution_admissible"] is False


def test_dirty_persistent_attempt_is_preserved_in_numbered_retry(tmp_path):
    args = _args(tmp_path)
    args.strategy_id = ["S01_persistent_full"]
    first = run_campaign(args)
    cell_id = "independent-n2-a__S01_persistent_full__r01"
    episode_id = "episode_01_django-django-15277"
    base = Path(first["cells"][cell_id]["episodes"][episode_id]["output"])
    base.mkdir(parents=True)
    (base / "contaminated.txt").write_text("preserve me", encoding="utf-8")

    state = run_campaign(args)
    episode = state["cells"][cell_id]["episodes"][episode_id]

    assert episode["output"] == str(base.with_name(base.name + "__retry01"))
    assert episode["attempts"][0]["output"] == str(base)
    assert (base / "contaminated.txt").read_text(encoding="utf-8") == "preserve me"


def test_health_only_attempt_reserves_retry_path_without_creating_directory(tmp_path):
    args = _args(tmp_path)
    args.strategy_id = ["S01_persistent_full"]
    first = run_campaign(args)
    cell_id = "independent-n2-a__S01_persistent_full__r01"
    episode_id = "episode_01_django-django-15277"
    base = Path(first["cells"][cell_id]["episodes"][episode_id]["output"])
    retry01 = base.with_name(base.name + "__retry01")
    base.mkdir(parents=True)
    (base / "contaminated.txt").write_text("preserve me", encoding="utf-8")
    state_path = args.output / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    episode = state["cells"][cell_id]["episodes"][episode_id]
    episode["output"] = str(retry01)
    episode["status"] = "paused_upstream_unhealthy"
    episode["attempts"] = [
        {"output": str(base), "status": "infrastructure_error"},
        {"output": str(retry01), "status": "paused_upstream_unhealthy"},
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed = run_campaign(args)
    output = resumed["cells"][cell_id]["episodes"][episode_id]["output"]

    assert output == str(base.with_name(base.name + "__retry02"))


def test_sequence_aggregate_labels_unpaired_saving_and_defers_primary_metrics():
    cell = {"issue_count": 2}
    rows = [
        {
            "official_resolved": True,
            "calls": 10,
            "cumulative_full_tokens": 100,
            "cumulative_materialized_tokens": 60,
        },
        {
            "official_resolved": True,
            "calls": 12,
            "cumulative_full_tokens": 100,
            "cumulative_materialized_tokens": 60,
        },
    ]
    result = _aggregate(cell, rows)
    assert result["candidate_trajectory_gross_saving_fraction"] == 0.4
    assert result["failure_aware_saving_fraction"] is None
    assert result["in_primary_saving_target"] is None
    assert result["candidate_all_issues_resolved"] is True
    assert result["primary_target_met"] is None
    assert result["pairing_status"].startswith("requires contemporaneous")

    rows[1]["official_resolved"] = False
    failed = _aggregate(cell, rows)
    assert failed["unpaired_candidate_trajectory_target_region"] is True
    assert failed["in_primary_saving_target"] is None
    assert failed["candidate_all_issues_resolved"] is False
    assert failed["primary_target_met"] is None


def test_predeclared_stop_counts_only_losses_of_completed_full_successes():
    cell = {"sequence_id": "sequence", "repeat": 1}
    state_cells = {
        "control": {
            "sequence_id": "sequence", "repeat": 1,
            "strategy_id": "S01_persistent_full",
            "episodes": {
                "e1": {"status": "complete", "official_resolved": True},
                "e2": {"status": "complete", "official_resolved": False},
                "e3": {"status": "complete", "official_resolved": True},
            },
        }
    }
    row = {
        "episodes": {
            "e1": {"status": "complete", "official_resolved": False,
                   "instance_id": "lost", "cumulative_full_tokens": 100,
                   "cumulative_materialized_tokens": 80},
            "e2": {"status": "complete", "official_resolved": False,
                   "instance_id": "joint-failure", "cumulative_full_tokens": 100,
                   "cumulative_materialized_tokens": 80},
            "e3": {"status": "running", "official_resolved": False,
                   "instance_id": "incomplete", "cumulative_full_tokens": 100,
                   "cumulative_materialized_tokens": 80},
        }
    }

    assert _lost_paired_full_successes(
        cell=cell, row=row, state_cells=state_cells
    ) == ["lost"]
    assert _lost_paired_full_successes(
        cell=cell,
        row=row,
        state_cells=state_cells,
        observed_episode_ids=["e2"],
    ) == []


def test_predeclared_stop_does_not_charge_a_full_equivalent_repeat():
    cell = {"sequence_id": "sequence", "repeat": 1}
    state_cells = {
        "control": {
            "sequence_id": "sequence", "repeat": 1,
            "strategy_id": "S01_persistent_full",
            "episodes": {
                "e1": {"status": "complete", "official_resolved": True},
            },
        }
    }
    row = {"episodes": {"e1": {
        "status": "complete", "official_resolved": False,
        "instance_id": "backend-variance", "cumulative_full_tokens": 100,
        "cumulative_materialized_tokens": 100,
    }}}

    assert _lost_paired_full_successes(
        cell=cell, row=row, state_cells=state_cells
    ) == []


def test_stopped_prefix_summary_is_failure_aware_and_not_a_complete_cohort():
    rows = [
        {"official_resolved": True, "calls": 2,
         "cumulative_full_tokens": 100, "cumulative_materialized_tokens": 70},
        {"official_resolved": False, "calls": 3,
         "cumulative_full_tokens": 200, "cumulative_materialized_tokens": 100},
    ]

    result = _partial_aggregate(
        rows, status="stopped_predeclared_quality_gate"
    )

    assert result["status"] == "stopped_predeclared_quality_gate"
    assert result["observed_issue_count"] == 2
    assert result["official_resolved_count"] == 1
    assert result["candidate_trajectory_gross_saving_fraction"] == pytest.approx(
        1 - 170 / 300
    )
    assert result["failure_aware_saving_fraction"] == 0.0
    assert result["primary_target_met"] is False


def test_divergence_saving_excludes_the_divergent_request():
    baseline = [
        {"request_index": 1, "assistant_command_sha256": "a", "full_tokens": 100,
         "materialized_tokens": 70},
        {"request_index": 2, "assistant_command_sha256": "b", "full_tokens": 200,
         "materialized_tokens": 100},
    ]
    candidate = [
        {"request_index": 1, "assistant_command_sha256": "a", "full_tokens": 100,
         "materialized_tokens": 70},
        {"request_index": 2, "assistant_command_sha256": "different", "full_tokens": 200,
         "materialized_tokens": 80},
    ]

    result = _divergence_accounting(candidate, baseline)
    assert result["first_action_diverged"] is True
    assert result["first_action_divergence_request"] == 2
    assert result["selected_tokens_before_divergence_or_terminal"] == 70
    assert result["full_tokens_before_divergence_or_terminal"] == 100


def test_divergence_accounting_identifies_identical_input_backend_variance():
    baseline = [{
        "request_index": 1,
        "assistant_command_sha256": "baseline",
        "request_input_sha256": "same-canonical-input",
        "selected_messages_sha256": "same-materialized-input",
        "materialized_tokens": 100,
        "full_tokens": 100,
    }]
    candidate = [{
        "request_index": 1,
        "assistant_command_sha256": "candidate",
        "request_input_sha256": "same-canonical-input",
        "selected_messages_sha256": "same-materialized-input",
        "materialized_tokens": 100,
        "full_tokens": 100,
    }]

    result = _divergence_accounting(candidate, baseline)

    assert result["identical_input_first_action_divergence"] is True
    assert result["selection_active_at_first_action_divergence"] is False


def test_divergence_does_not_confuse_canonical_with_materialized_input():
    baseline = [{
        "request_index": 1,
        "assistant_command_sha256": "baseline",
        "request_input_sha256": "same-canonical-input",
        "selected_messages_sha256": "full-materialization",
        "materialized_tokens": 100,
        "full_tokens": 100,
    }]
    candidate = [{
        "request_index": 1,
        "assistant_command_sha256": "candidate",
        "request_input_sha256": "same-canonical-input",
        "selected_messages_sha256": "selected-materialization",
        "materialized_tokens": 80,
        "full_tokens": 100,
    }]

    result = _divergence_accounting(candidate, baseline)

    assert result["identical_input_first_action_divergence"] is False
    assert result["selection_active_at_first_action_divergence"] is True


def _pairing_manifest(instance_id: str, behavior: str) -> dict:
    return {
        "instance_id": instance_id,
        "pair_id": "campaign:sequence:r1",
        "agent_behavior_sha256": behavior,
        "scaffold_identity_sha256": "scaffold",
        "harness_version_observed": "2.4.6",
        "grader_version_observed": "4.1.0",
        "model_revision": "model",
        "tokenizer_revision": "tokenizer",
        "dataset_revision": "dataset",
        "benchmark_card_sha256": "benchmark",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_calls": 40,
        "max_completion_tokens": 1024,
    }


def test_pairing_identity_allows_task_specific_agent_behavior_digests():
    baseline = [
        _pairing_manifest("task-a", "behavior-a"),
        _pairing_manifest("task-b", "behavior-b"),
    ]
    candidate = [dict(row) for row in baseline]

    digest = _validate_sequence_pairing_identity(
        candidate_manifests=candidate,
        baseline_manifests=baseline,
        expected_pair_id="campaign:sequence:r1",
    )

    assert isinstance(digest, str) and len(digest) == 64


def test_pairing_identity_rejects_within_task_behavior_change():
    baseline = [_pairing_manifest("task-a", "behavior-a")]
    candidate = [dict(baseline[0], agent_behavior_sha256="changed")]

    try:
        _validate_sequence_pairing_identity(
            candidate_manifests=candidate,
            baseline_manifests=baseline,
            expected_pair_id="campaign:sequence:r1",
        )
    except ValueError as error:
        assert "agent_behavior_sha256" in str(error)
    else:
        raise AssertionError("pairing identity change was accepted")
