from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.run_failed_full_recovery_diagnostic import (
    build_diagnostic_command,
    validate_declaration,
)


def _declaration() -> dict:
    return {
        "schema_version": 1,
        "study": "paper8_5_failed_full_recovery_diagnostic",
        "diagnostic_id": "diagnostic",
        "source_campaign": {
            "campaign_id": "campaign",
            "campaign_state_sha256": "digest",
            "cell_id": "cell",
        },
        "policy": {
            "policy": "frontier_dag_retirement",
            "frontier_recent_user_prompts": 2,
            "frontier_protocol_exemplars": 1,
            "frontier_workflow_exemplars": 0,
            "boundary_mode": "boundary_free",
            "frontier_allow_heuristic": True,
        },
        "claim_boundary": "diagnostic only",
        "episodes": [{
            "episode_id": "episode_03_repo-task",
            "instance_id": "repo__task",
            "prefix_sha256": "prefix",
            "diagnostic_role": "cross_task_retirement_probe",
        }],
    }


def _state() -> dict:
    return {
        "campaign_id": "campaign",
        "cells": {"cell": {
            "strategy_id": "RECENT_FRONTIER_M2_P1_FROZEN",
            "episodes": {"episode_03_repo-task": {
                "instance_id": "repo__task",
                "same_prefix_full_control": {
                    "status": "full_control_unresolved",
                    "prefix_sha256": "prefix",
                    "command": ["python", "runner", "--instance-id", "repo__task", "--output", "old", "--run-id", "old-run", "--pair-id", "old-pair", "--policy", "full", "--frontier-recent-user-prompts", "2", "--frontier-protocol-exemplars", "1", "--frontier-workflow-exemplars", "0", "--boundary-mode", "boundary_free"],
                },
            }},
        }},
    }


def test_declaration_accepts_only_pinned_unresolved_full_control():
    declaration = _declaration()
    state = _state()
    assert validate_declaration(declaration, state, state_sha256="digest") == state["cells"]["cell"]
    state["cells"]["cell"]["episodes"]["episode_03_repo-task"]["same_prefix_full_control"]["status"] = "qualified"
    with pytest.raises(ValueError, match="not an unresolved FULL"):
        validate_declaration(declaration, state, state_sha256="digest")


def test_diagnostic_command_changes_policy_and_identity_not_prefix(tmp_path: Path):
    declaration = _declaration()
    source = _state()["cells"]["cell"]["episodes"]["episode_03_repo-task"]["same_prefix_full_control"]["command"]
    source.extend(("--persistent-prefix", "frozen-prefix.json", "--session-id", "session", "--episode-index", "3"))
    command = build_diagnostic_command(
        source_command=source,
        declaration=declaration,
        output=tmp_path / "out",
        episode_number=3,
        instance_id="repo__task",
    )
    assert command[command.index("--policy") + 1] == "frontier_dag_retirement"
    assert command[command.index("--persistent-prefix") + 1] == "frozen-prefix.json"
    assert command[command.index("--session-id") + 1] == "session"
    assert command[command.index("--episode-index") + 1] == "3"
    assert command[command.index("--pair-id") + 1] == "diagnostic"
    assert "--frontier-allow-heuristic" in command
