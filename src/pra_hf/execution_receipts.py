"""Agent-neutral execution-time resource receipt helpers.

Tool middleware may use these helpers to version explicit resources and the
current Git workspace.  The module knows neither agent protocols nor tool
names.  Opaque tools must still declare their effect scope incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resource_path(resource_id: str) -> Path | None:
    if resource_id.startswith("file://"):
        return Path(resource_id.removeprefix("file://"))
    return Path(resource_id) if "://" not in resource_id else None


def resource_version(resource_id: str) -> str:
    """Return a stable version for a file-like resource without following it."""

    path = _resource_path(resource_id)
    if path is None:
        return "unknown"
    try:
        if path.is_symlink():
            return "symlink:" + _sha256(os.readlink(path).encode())
        if path.is_file():
            return "sha256:" + _sha256(path.read_bytes())
        if path.is_dir():
            entries = sorted(
                (row.name, row.is_dir(), row.is_symlink())
                for row in path.iterdir()
            )
            return "directory:" + _sha256(
                json.dumps(entries, separators=(",", ":")).encode()
            )
        if not path.exists():
            return "missing"
    except OSError:
        return "unavailable"
    return "other"


def _git(root: Path, *arguments: str) -> tuple[bool, bytes]:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return False, b""
    return result.returncode == 0, result.stdout


def workspace_generation(root: Path) -> tuple[bool, str | None]:
    """Version the declared Git workspace including nonignored untracked data."""

    head_ok, head = _git(root, "rev-parse", "HEAD")
    index_ok, index = _git(root, "diff", "--cached", "--binary", "--full-index")
    work_ok, work = _git(root, "diff", "--binary", "--full-index")
    names_ok, names = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    complete = all((head_ok, index_ok, work_ok, names_ok))
    if not complete:
        return False, None
    untracked: list[tuple[str, str]] = []
    for encoded in names.split(b"\0"):
        if not encoded:
            continue
        relative = os.fsdecode(encoded)
        path = root / relative
        untracked.append((relative, resource_version(f"file://{path.resolve()}")))
    manifest = json.dumps(untracked, sort_keys=True, separators=(",", ":")).encode()
    return True, "sha256:" + _sha256(b"\0".join((head, index, work, manifest)))


@dataclass(frozen=True)
class ExecutionSnapshot:
    workspace_complete: bool
    workspace_generation: str | None
    resource_versions: Mapping[str, str]


def capture_execution_snapshot(
    workspace: Path,
    resource_ids: Iterable[str],
) -> ExecutionSnapshot:
    complete, generation = workspace_generation(workspace)
    versions = {
        str(resource_id): resource_version(str(resource_id))
        for resource_id in dict.fromkeys(resource_ids)
    }
    return ExecutionSnapshot(complete, generation, versions)


def build_execution_receipt(
    *,
    action_record_id: str,
    observation_record_ids: Iterable[str],
    tool_category: str,
    operation_kind: str,
    transport_status: str,
    semantic_status: str,
    result_complete: bool,
    effect_scope_complete: bool,
    pre: ExecutionSnapshot,
    post: ExecutionSnapshot,
    resource_kinds: Mapping[str, str],
    return_code: int | None = None,
    error_kind: str | None = None,
    cwd: str | None = None,
    session_id: str | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    resources = []
    versions_complete = True
    for resource_id, kind in resource_kinds.items():
        before = pre.resource_versions.get(resource_id, "unknown")
        after = post.resource_versions.get(resource_id, "unknown")
        if before in {"unknown", "unavailable"} or after in {"unknown", "unavailable"}:
            versions_complete = False
        resources.append({
            "resource_id": resource_id,
            "kind": kind,
            "version_before": before,
            "version_after": after,
        })
    effect_trace_complete = bool(
        effect_scope_complete
        and pre.workspace_complete
        and post.workspace_complete
        and versions_complete
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "action_record_id": action_record_id,
        "observation_record_ids": list(observation_record_ids),
        "tool_category": tool_category,
        "operation_kind": operation_kind,
        "transport_status": transport_status,
        "semantic_status": semantic_status,
        "return_code": return_code,
        "error_kind": error_kind,
        "result_complete": bool(result_complete),
        "effect_trace_complete": effect_trace_complete,
        "provenance": "runtime_traced",
        "cwd": cwd,
        "workspace_generation": post.workspace_generation,
        "workspace_generation_before": pre.workspace_generation,
        "resources": resources,
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    receipt["receipt_digest"] = _sha256(canonical)
    return receipt


__all__ = [
    "ExecutionSnapshot",
    "build_execution_receipt",
    "capture_execution_snapshot",
    "resource_version",
    "workspace_generation",
]
