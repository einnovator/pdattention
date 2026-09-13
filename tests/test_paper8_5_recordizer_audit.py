import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.recordizer_audit import (
    score_rows,
    stratified_sample,
    write_audit,
)


def _trajectory(path: Path, task: str) -> Path:
    path.write_text(json.dumps({
        "instance_id": task,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix a.py"},
            {"role": "assistant", "content": "inspect\n```mswea_bash_command\nsed -n '1,2p' a.py\n```"},
            {"role": "user", "content": "one\ntwo\n<returncode>0</returncode>"},
            {"role": "assistant", "content": "done\n```mswea_bash_command\npytest -q\n```"},
            {"role": "user", "content": "1 passed\n<returncode>0</returncode>"},
        ],
    }), encoding="utf-8")
    return path


def test_stratified_audit_is_deterministic_and_writes_blinded_fields(tmp_path):
    paths = [
        _trajectory(tmp_path / f"t{i}.json", f"task-{i}") for i in range(2)
    ]
    rows, manifest = stratified_sample(paths, sample_size=6, seed=850)
    again, _ = stratified_sample(paths, sample_size=6, seed=850)
    assert [(row["task_id"], row["record_id"]) for row in rows] == [
        (row["task_id"], row["record_id"]) for row in again
    ]
    assert manifest["realized_sample_size"] == 6
    assert all(row["human_semantic_roles"] == "" for row in rows)
    output = tmp_path / "audit"
    write_audit(rows, manifest, output)
    assert (output / "recordizer_audit.csv").is_file()
    assert "awaiting_blinded_human_labels" in (
        output / "recordizer_audit_manifest.json"
    ).read_text()


def test_recordizer_audit_scores_multilabel_precision_and_recall():
    result = score_rows([
        {"auto_semantic_roles": "assistant_action;progress",
         "human_semantic_roles": "assistant_action;progress",
         "human_causal_group_correct": "yes"},
        {"auto_semantic_roles": "tool_observation;source_view",
         "human_semantic_roles": "tool_observation",
         "human_causal_group_correct": "no"},
        {"auto_semantic_roles": "tool_observation",
         "human_semantic_roles": "tool_observation;verification"},
    ])
    assert result["labelled_records"] == 3
    assert result["exact_semantic_role_set_accuracy"] == pytest.approx(1 / 3)
    assert result["per_role"]["source_view"]["precision"] == 0
    assert result["per_role"]["verification"]["recall"] == 0
    assert result["causal_group_boundary_accuracy"] == pytest.approx(.5)


def test_recordizer_audit_requires_human_labels():
    with pytest.raises(ValueError, match="no human"):
        score_rows([{"auto_semantic_roles": "task", "human_semantic_roles": ""}])
