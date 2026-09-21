from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.reduce_openhands_exact_cohort import reduce_cohort


def _pair(path: Path, task: str, full: int, candidate_full: int, candidate: int):
    path.write_text(json.dumps({
        "schema_version": 3, "instance_id": task, "qualification": "pass",
        "paired_input_saving": 1 - candidate / full,
        "full": {"official_resolved": True, "actions": 10,
                 "materialized_tokens": full, "provider_prompt_tokens": full * 2},
        "candidate": {"official_resolved": True, "actions": 8,
                      "full_tokens": candidate_full, "materialized_tokens": candidate,
                      "provider_prompt_tokens": candidate * 2},
    }))


def test_cohort_reducer_reports_own_and_paired_saving(tmp_path: Path):
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    _pair(one, "task-1", 100, 80, 60)
    _pair(two, "task-2", 200, 180, 140)

    result = reduce_cohort([one, two])

    assert result["qualification"] == "pass"
    assert result["full_official_resolved"] == 2
    assert result["candidate_official_resolved"] == 2
    assert result["own_logical_saving"] == pytest.approx(1 - 200 / 260)
    assert result["paired_input_saving"] == pytest.approx(1 - 200 / 300)
    assert result["action_delta"] == -4


def test_cohort_reducer_rejects_duplicate_task_identity(tmp_path: Path):
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    _pair(one, "task-1", 100, 80, 60)
    _pair(two, "task-1", 200, 180, 140)

    with pytest.raises(ValueError, match="overlapping task identities"):
        reduce_cohort([one, two])
