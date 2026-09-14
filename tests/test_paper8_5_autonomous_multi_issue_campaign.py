from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper8_5_agent_memory.run_autonomous_multi_issue_campaign import (
    TARGET_SAVING_MAX,
    TARGET_SAVING_MIN,
    _aggregate,
    _divergence_accounting,
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
    commands = [episode["command"] for episode in cell["episodes"].values()]
    assert all(
        command[command.index("--completed-recent-turns") + 1] == "4"
        for command in commands
    )


def test_sequence_aggregate_marks_saving_region_but_defers_quality_to_pairing():
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
    assert result["failure_aware_saving_fraction"] == 0.4
    assert result["candidate_all_issues_resolved"] is True
    assert result["primary_target_met"] is None
    assert result["pairing_status"].startswith("requires contemporaneous")

    rows[1]["official_resolved"] = False
    failed = _aggregate(cell, rows)
    assert failed["in_primary_saving_target"] is True
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
