from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.run_autonomous_multi_issue_campaign import (
    TARGET_SAVING_MAX,
    TARGET_SAVING_MIN,
    _aggregate,
    _divergence_accounting,
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


def test_campaign_registry_accepts_ten_issue_asymptotic_sequence():
    benchmark = json.loads(BASELINE_SUCCESS14_BENCHMARK.read_text(encoding="utf-8"))
    spec = json.loads(INSTRUCTION_EPOCH_N3_SPEC.read_text(encoding="utf-8"))
    spec["sequences"] = [{
        "sequence_id": "independent-n10-planned",
        "sequence_stratum": "independent_clean_workspace",
        "instance_ids": benchmark["instance_ids"][:10],
    }]

    validate_spec(spec, benchmark)
    cells = campaign_cells(spec)
    assert len(cells) == 2
    assert all(row["issue_count"] == 10 for row in cells)


def test_campaign_registry_rejects_more_than_ten_issues():
    benchmark = json.loads(BASELINE_SUCCESS14_BENCHMARK.read_text(encoding="utf-8"))
    spec = json.loads(INSTRUCTION_EPOCH_N3_SPEC.read_text(encoding="utf-8"))
    spec["sequences"] = [{
        "sequence_id": "independent-n11-invalid",
        "sequence_stratum": "independent_clean_workspace",
        "instance_ids": benchmark["instance_ids"][:11],
    }]

    with pytest.raises(ValueError, match="one to ten"):
        validate_spec(spec, benchmark)


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
