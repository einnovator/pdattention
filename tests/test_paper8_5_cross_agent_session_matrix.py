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
