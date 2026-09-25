import json
from pathlib import Path

from experiments.paper8_5_agent_memory.reduce_cross_agent_session_matrix import (
    reduce_matrix,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_matrix_reports_own_and_paired_saving_separately(tmp_path: Path) -> None:
    full = tmp_path / "opencode" / "n3" / "full"
    candidate = tmp_path / "opencode" / "n3" / "recent_frontier"
    base = {
        "status": "execution_complete_pending_grade",
        "agent": "opencode",
        "task_count_completed": 3,
        "request_count": 30,
        "cumulative_full_history_tokens": 1000,
        "cumulative_materialized_history_tokens": 1000,
        "own_saving_fraction": 0.0,
        "policy": "full",
    }
    _write(full / "campaign_state.json", base)
    _write(full / "official_grade" / "report.json", {
        "resolved_ids": ["a", "b", "c"], "unresolved_ids": [],
    })
    changed = dict(base)
    changed.update(
        policy="frontier_dag_retirement",
        request_count=27,
        cumulative_full_history_tokens=1200,
        cumulative_materialized_history_tokens=600,
        own_saving_fraction=0.5,
    )
    _write(candidate / "campaign_state.json", changed)
    _write(candidate / "official_grade" / "report.json", {
        "resolved_ids": ["a", "b"], "unresolved_ids": ["c"],
    })

    rows = reduce_matrix(tmp_path)["rows"]
    selective = next(row for row in rows if row["arm"] == "recent_frontier")
    assert selective["own_saving_fraction"] == 0.5
    assert selective["paired_saving_fraction"] == 0.4
    assert selective["lost_full_success_ids"] == ["c"]


def test_matrix_reads_pra_agent_native_manifest(tmp_path: Path) -> None:
    root = tmp_path / "pra_agent" / "n1" / "full"
    _write(root / "run_manifest.json", {
        "task_count_completed": 1,
        "request_count": 19,
        "cumulative_full_history_tokens": 209642,
        "cumulative_materialized_history_tokens": 209642,
        "own_saving_fraction": 0.0,
        "history_policy": "full",
    })
    _write(root / "official_grade" / "report.json", {
        "resolved_ids": ["django__django-15277"], "unresolved_ids": [],
    })
    row = reduce_matrix(tmp_path)["rows"][0]
    assert row["agent"] == "pra_agent"
    assert row["n"] == 1
    assert row["resolved"] == 1
    assert row["calls"] == 19


def test_matrix_reads_native_tool_count_from_episode_manifest(tmp_path: Path) -> None:
    root = tmp_path / "opencode" / "n1" / "full"
    _write(root / "campaign_state.json", {
        "status": "execution_complete_pending_grade",
        "task_count_completed": 1,
        "request_count": 9,
        "cumulative_full_history_tokens": 100,
        "cumulative_materialized_history_tokens": 100,
        "own_saving_fraction": 0.0,
        "policy": "full",
    })
    _write(root / "episode-01" / "run_manifest.json", {
        "event_types": {"tool_use": 7, "text": 8},
    })
    row = reduce_matrix(tmp_path)["rows"][0]
    assert row["calls"] == 7


def test_matrix_separates_official_resolution_from_forced_termination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "kilo" / "n1" / "tail90"
    _write(root / "campaign_state.json", {
        "status": "execution_complete_pending_grade",
        "task_count_completed": 1,
        "request_count": 14,
        "cumulative_full_history_tokens": 100,
        "cumulative_materialized_history_tokens": 90,
        "own_saving_fraction": 0.1,
        "policy": "matched_token_tail",
    })
    _write(root / "official_grade" / "report.json", {
        "resolved_ids": ["org__task"], "unresolved_ids": [],
    })
    _write(root / "external_termination.json", {
        "semantic_completion_observed": False,
        "wall_time_valid": False,
    })
    row = reduce_matrix(tmp_path)["rows"][0]
    assert row["resolved"] == 1
    assert row["autonomous_completion"] is False
    assert row["wall_time_valid"] is False


def test_matrix_marks_post_patch_transport_error_non_autonomous(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pra_agent" / "n3" / "recent_frontier"
    _write(root / "run_manifest.json", {
        "task_count_completed": 3,
        "request_count": 39,
        "cumulative_full_history_tokens": 857587,
        "cumulative_materialized_history_tokens": 572963,
        "own_saving_fraction": 0.3318,
        "history_policy": "frontier_dag_retirement",
        "error_type": "HTTPError",
    })
    _write(root / "official_grade" / "report.json", {
        "resolved_ids": ["a", "b", "c"], "unresolved_ids": [],
    })
    row = reduce_matrix(tmp_path)["rows"][0]
    assert row["resolved"] == 3
    assert row["autonomous_completion"] is False
