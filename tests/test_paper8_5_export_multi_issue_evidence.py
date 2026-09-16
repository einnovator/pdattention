from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.export_multi_issue_evidence import (
    build,
    write_bundle,
)


def _episode(
    path: Path, instance_id: str, calls: int, *, resolved: bool = True,
) -> dict:
    path.mkdir(parents=True)
    rows = [
        {
            "request_index": index,
            "upstream_status": 200,
            "assistant_command_sha256": f"action-{index}",
            "full_tokens": 100,
            "materialized_tokens": 80,
        }
        for index in range(1, calls + 1)
    ]
    (path / "request_selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (path / "run_manifest.json").write_text(json.dumps({
        "model_revision": "model", "tokenizer_revision": "tokenizer",
        "harness_version_observed": "agent", "agent_behavior_sha256": "behavior",
        "scaffold_identity_sha256": "scaffold",
        "workspace_source_identity_sha256": instance_id,
        "temperature": 0, "top_p": 1, "seed": 0,
        "max_completion_tokens": 1024,
    }), encoding="utf-8")
    (path / "official_result.json").write_text(
        json.dumps({
            "official_grader": True,
            "resolved": resolved,
            "error": not resolved,
            "failure_class": None if resolved else "patch_apply_failed",
        }),
        encoding="utf-8",
    )
    (path / "persistent_episode_export.json").write_text(
        json.dumps({"instance_id": instance_id}), encoding="utf-8"
    )
    return {
        "instance_id": instance_id, "output": str(path),
        "official_resolved": resolved, "calls": calls,
    }


def test_export_separates_candidate_gross_from_paired_failure_aware(tmp_path):
    root = tmp_path / "campaign"
    first = _episode(root / "full-first", "task-a", 2)
    full_second = _episode(root / "full-second", "task-b", 2)
    candidate_second = _episode(root / "candidate-second", "task-b", 1)
    state = {
        "campaign_id": "campaign-v1",
        "cells": {
            "sequence__S01_persistent_full__r01": {
                "status": "complete", "sequence_id": "sequence", "repeat": 1,
                "strategy_id": "S01_persistent_full", "strategy_config_id": "default",
                "official_resolved_count": 2, "issue_count": 2, "calls": 4,
                "cumulative_full_tokens": 1000,
                "cumulative_materialized_tokens": 1000,
                "episodes": {"first": first, "second": full_second},
            },
            "sequence__S03_completed_episode_spine-r4m2v2_tasks1__r01": {
                "status": "complete", "sequence_id": "sequence", "repeat": 1,
                "strategy_id": "S03_completed_episode_spine",
                "strategy_config_id": "r4m2v2_tasks1",
                "official_resolved_count": 2, "issue_count": 2, "calls": 3,
                "cumulative_full_tokens": 800,
                "cumulative_materialized_tokens": 600,
                "episodes": {"first": first, "second": candidate_second},
            },
        },
    }
    root.mkdir(exist_ok=True)
    (root / "campaign_state.json").write_text(json.dumps(state), encoding="utf-8")

    evidence = build(root)
    candidate = next(
        row for row in evidence["runs"]
        if row["strategy_id"] == "S03_completed_episode_spine"
    )
    assert candidate["candidate_trajectory_gross_saving_fraction"] == 0.25
    assert candidate["paired"]["failure_aware_saving_vs_persistent_full"] == 0.4
    assert candidate["calls_delta_vs_persistent_full"] == -1
    assert candidate["shared_first_episode_exact"] is True
    assert candidate["evidence_admissible"] is True
    assert candidate["paired_issues"][0]["raw_saving_vs_persistent_full"] == pytest.approx(0.2)
    assert candidate["paired_issues"][1]["raw_saving_vs_persistent_full"] == pytest.approx(0.6)
    assert candidate["paired_issues"][1]["calls_delta_vs_persistent_full"] == -1
    assert candidate["paired_issues"][1]["joint_success"] is True

    output = tmp_path / "bundle"
    write_bundle(evidence, output)
    assert "40.00%" in (output / "README.md").read_text(encoding="utf-8")
    assert (output / "runs.csv").is_file()
    issues = (output / "issues.csv").read_text(encoding="utf-8")
    assert "candidate_official_resolved" in issues
    assert "task-b" in issues


def test_export_pairs_control_with_nondefault_strategy_config_id(tmp_path):
    root = tmp_path / "campaign"
    full = _episode(root / "full", "task-a", 2)
    candidate_episode = _episode(root / "candidate", "task-a", 1)
    state = {
        "campaign_id": "campaign-configured-control-v1",
        "cells": {
            "sequence__S01_persistent_full-boundary_free_v2__r01": {
                "status": "complete", "sequence_id": "sequence", "repeat": 1,
                "strategy_id": "S01_persistent_full",
                "strategy_config_id": "boundary_free_v2",
                "official_resolved_count": 1, "issue_count": 1, "calls": 2,
                "cumulative_full_tokens": 1000,
                "cumulative_materialized_tokens": 1000,
                "episodes": {"first": full},
            },
            "sequence__S03_global-r4__r01": {
                "status": "complete", "sequence_id": "sequence", "repeat": 1,
                "strategy_id": "S03_global", "strategy_config_id": "r4",
                "official_resolved_count": 1, "issue_count": 1, "calls": 1,
                "cumulative_full_tokens": 800,
                "cumulative_materialized_tokens": 600,
                "episodes": {"first": candidate_episode},
            },
        },
    }
    root.mkdir(exist_ok=True)
    (root / "campaign_state.json").write_text(json.dumps(state), encoding="utf-8")

    evidence = build(root)
    candidate = next(row for row in evidence["runs"] if row["strategy_id"] == "S03_global")

    assert candidate["paired_persistent_full_cell_id"].endswith(
        "S01_persistent_full-boundary_free_v2__r01"
    )
    assert candidate["paired"]["failure_aware_saving_vs_persistent_full"] == 0.4


def test_export_keeps_predeclared_stopped_prefix_out_of_complete_denominator(tmp_path):
    root = tmp_path / "campaign"
    full_first = _episode(root / "full-first", "task-a", 2)
    full_second = _episode(root / "full-second", "task-b", 2)
    full_third = _episode(root / "full-third", "task-c", 2)
    candidate_first = _episode(root / "candidate-first", "task-a", 1)
    candidate_second = _episode(
        root / "candidate-second", "task-b", 1, resolved=False
    )
    candidate_first["status"] = "complete"
    candidate_second["status"] = "complete"
    state = {
        "campaign_id": "campaign-stopped-v1",
        "cells": {
            "control": {
                "status": "complete", "sequence_id": "sequence", "repeat": 1,
                "strategy_id": "S01_persistent_full", "strategy_config_id": "full",
                "official_resolved_count": 3, "issue_count": 3, "calls": 6,
                "cumulative_full_tokens": 600,
                "cumulative_materialized_tokens": 600,
                "episodes": {
                    "first": full_first, "second": full_second, "third": full_third,
                },
            },
            "candidate": {
                "status": "stopped_predeclared_quality_gate",
                "sequence_id": "sequence", "repeat": 1,
                "strategy_id": "S08_atomic_epoch_e2",
                "strategy_config_id": "e2", "official_resolved_count": 1,
                "issue_count": 3, "observed_issue_count": 2, "calls": 2,
                "cumulative_full_tokens": 200,
                "cumulative_materialized_tokens": 160,
                "episodes": {
                    "first": candidate_first, "second": candidate_second,
                    "third": {"status": "aborted_by_quality_gate"},
                },
                "stop_gate": {"threshold": 1, "observed": 1},
            },
        },
    }
    root.mkdir(exist_ok=True)
    (root / "campaign_state.json").write_text(json.dumps(state), encoding="utf-8")

    evidence = build(root)
    candidate = next(
        row for row in evidence["runs"]
        if row["strategy_id"] == "S08_atomic_epoch_e2"
    )

    assert candidate["campaign_cell_status"] == "stopped_predeclared_quality_gate"
    assert candidate["issue_count"] == 2
    assert candidate["planned_issue_count"] == 3
    assert candidate["paired"]["partial_stopped_prefix"] is True
    assert candidate["paired"]["lost_persistent_full_successes"] == 1
    assert candidate["paired"]["failure_aware_saving_vs_persistent_full"] == 0.0
    assert candidate["evidence_admissible"] is True
