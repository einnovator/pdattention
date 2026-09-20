from __future__ import annotations

from pathlib import Path
import subprocess

from pra_hf.execution_receipts import (
    build_execution_receipt,
    capture_execution_snapshot,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ("git", "-C", str(root), *args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_versioned_file_receipt_is_complete_only_for_declared_effect_scope(
    tmp_path: Path,
):
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.name", "PRA Test")
    _git(tmp_path, "config", "user.email", "pra-test@example.invalid")
    source = tmp_path / "source.py"
    source.write_text("old\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    resource_id = f"file://{source.resolve()}"

    before = capture_execution_snapshot(tmp_path, (resource_id,))
    source.write_text("new\n", encoding="utf-8")
    after = capture_execution_snapshot(tmp_path, (resource_id,))
    complete = build_execution_receipt(
        action_record_id="action-1",
        observation_record_ids=("observation-1",),
        tool_category="filesystem",
        operation_kind="write",
        transport_status="completed",
        semantic_status="succeeded",
        result_complete=True,
        effect_scope_complete=True,
        pre=before,
        post=after,
        resource_kinds={resource_id: "write"},
    )
    opaque = build_execution_receipt(
        **{
            key: value for key, value in complete.items()
            if key in {
                "action_record_id", "observation_record_ids", "tool_category",
                "operation_kind", "transport_status", "semantic_status",
                "result_complete",
            }
        },
        effect_scope_complete=False,
        pre=before,
        post=after,
        resource_kinds={resource_id: "write"},
    )

    assert before.workspace_complete is True
    assert after.workspace_complete is True
    assert complete["effect_trace_complete"] is True
    assert complete["workspace_generation_before"] != complete["workspace_generation"]
    assert complete["resources"][0]["version_before"] != (
        complete["resources"][0]["version_after"]
    )
    assert opaque["effect_trace_complete"] is False
