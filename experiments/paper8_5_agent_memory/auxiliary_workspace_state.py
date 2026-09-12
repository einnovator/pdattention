"""Derive a separately labeled SWE-bench outcome from workspace checkpoints.

This module never changes the agent's submitted prediction.  It accepts a
checkpoint only when the final recorded action left a complete repository
state unchanged, all checkpoint artifacts match their receipts, and the final
tracked patch can be represented without composing staged and unstaged deltas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping


AUXILIARY_WORKSPACE_STATE_LABEL = "auxiliary_workspace_state"


class _Unavailable(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _Unavailable("invalid_json", f"invalid JSON in {path.name}: {error}") from error
    if not isinstance(value, Mapping):
        raise _Unavailable("invalid_json", f"{path.name} is not a JSON object")
    return dict(value)


def _single_session_paths(root: Path) -> tuple[list[Path], list[Path], Path]:
    if not root.is_dir():
        raise _Unavailable("instrumentation_root_missing", "instrumentation root is absent")
    decisions = sorted(root.rglob("decision_*.json"))
    executions = sorted(root.rglob("execution_*.json"))
    if not decisions or not executions:
        raise _Unavailable("checkpoint_receipts_missing", "decision or execution receipts are absent")
    parents = {path.parent.resolve() for path in (*decisions, *executions)}
    if len(parents) != 1:
        raise _Unavailable(
            "ambiguous_instrumentation_sessions",
            f"checkpoint receipts span {len(parents)} session directories",
        )
    return decisions, executions, parents.pop()


def _ordered_receipts(paths: list[Path], kind: str) -> list[tuple[Path, dict[str, Any]]]:
    rows = [(path, _read_object(path)) for path in paths]
    try:
        rows.sort(key=lambda item: int(item[1]["step"]))
        steps = [int(row["step"]) for _, row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise _Unavailable("invalid_receipt_step", f"{kind} receipt step is invalid") from error
    if steps != list(range(len(rows))):
        raise _Unavailable(
            "noncontiguous_receipts",
            f"{kind} receipt steps are not the contiguous range 0..{len(rows) - 1}",
        )
    return rows


def _checkpoint_artifact(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    field: str,
    digest_field: str,
) -> tuple[Path, bytes]:
    relative = receipt.get(field)
    expected_digest = receipt.get(digest_field)
    if not isinstance(relative, str) or not relative or not isinstance(expected_digest, str):
        raise _Unavailable("invalid_checkpoint_receipt", f"missing {field} provenance")
    source = (receipt_path.parent / relative).resolve()
    session = receipt_path.parent.resolve()
    if source.parent != session or source.is_symlink() or not source.is_file():
        raise _Unavailable(
            "unsafe_checkpoint_artifact",
            f"{field} is not a regular file directly inside the checkpoint session",
        )
    value = source.read_bytes()
    if _sha256_bytes(value) != expected_digest.lower():
        raise _Unavailable("checkpoint_digest_mismatch", f"{field} digest does not match")
    return source, value


def _validate_empty_untracked_archive(path: Path) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise _Unavailable("invalid_untracked_archive", str(error)) from error
    if members:
        raise _Unavailable(
            "untracked_files_not_representable",
            "the checkpoint contains non-ignored untracked files",
        )


def _primary_prediction(path: Path, instance_id: str) -> tuple[dict[str, Any], str]:
    payload = _read_object(path)
    if set(payload) != {instance_id} or not isinstance(payload.get(instance_id), Mapping):
        raise _Unavailable(
            "primary_prediction_mismatch",
            "primary predictions do not contain exactly the locked instance",
        )
    row = dict(payload[instance_id])
    if row.get("instance_id") != instance_id:
        raise _Unavailable("primary_prediction_mismatch", "prediction instance_id is inconsistent")
    model_name = row.get("model_name_or_path")
    if not isinstance(model_name, str) or not model_name:
        raise _Unavailable("primary_prediction_mismatch", "prediction model identity is absent")
    if not isinstance(row.get("model_patch"), str):
        raise _Unavailable("primary_prediction_mismatch", "prediction model_patch is not text")
    return row, model_name


def create_auxiliary_workspace_state_prediction(
    *,
    instrumentation_root: str | Path,
    output: str | Path,
    primary_predictions: str | Path,
    instance_id: str,
) -> dict[str, Any]:
    """Copy a verified final tracked patch and emit an auxiliary prediction.

    An unavailable auxiliary outcome is recorded rather than raised.  No patch
    or predictions file is emitted on that path, so callers cannot accidentally
    grade stale, incomplete, untracked, or ambiguously composed workspace state.
    """

    root = Path(instrumentation_root).resolve()
    destination = Path(output).resolve()
    provenance_path = destination / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.json"
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "outcome_label": AUXILIARY_WORKSPACE_STATE_LABEL,
        "evidence_role": "auxiliary_only",
        "replaces_official_submission": False,
        "instance_id": instance_id,
        "instrumentation_root": str(root),
    }
    try:
        primary_path = Path(primary_predictions).resolve()
        primary_row, model_name = _primary_prediction(primary_path, instance_id)
        primary_model_patch = str(primary_row["model_patch"])
        decision_paths, execution_paths, session = _single_session_paths(root)
        decisions = _ordered_receipts(decision_paths, "decision")
        executions = _ordered_receipts(execution_paths, "execution")
        if len(decisions) != len(executions):
            raise _Unavailable(
                "receipt_sequence_mismatch",
                "decision and execution receipt counts differ",
            )
        if any(
            decision_row.get("command_sha256")
            != execution_row.get("command_sha256")
            for (_, decision_row), (_, execution_row) in zip(decisions, executions)
        ):
            raise _Unavailable(
                "receipt_sequence_mismatch",
                "decision and execution command-digest sequences differ",
            )
        decision_path, decision = decisions[-1]
        execution_path, execution = executions[-1]
        step = len(decisions) - 1
        if (
            type(decision.get("schema_version")) is not int
            or decision["schema_version"] != 1
        ):
            raise _Unavailable("unsupported_checkpoint_schema", "decision schema is not version 1")
        if decision.get("scope") != "git_HEAD_plus_tracked_diff_plus_nonignored_untracked_files":
            raise _Unavailable("unsupported_checkpoint_scope", "checkpoint scope is not exact")
        if decision.get("checkpoint_complete") is not True or any(
            decision.get(field) is not True
            for field in (
                "index_capture_complete",
                "worktree_capture_complete",
                "untracked_capture_complete",
            )
        ):
            raise _Unavailable(
                "last_checkpoint_incomplete",
                "the chronologically last pre-action checkpoint is incomplete",
            )
        if (
            type(execution.get("schema_version")) is not int
            or execution["schema_version"] != 1
        ):
            raise _Unavailable("unsupported_execution_schema", "execution schema is not version 1")
        if decision.get("command_sha256") != execution.get("command_sha256"):
            raise _Unavailable("receipt_sequence_mismatch", "last command digests differ")
        pre_state = execution.get("pre_state")
        post_state = execution.get("post_state")
        if not isinstance(pre_state, Mapping) or not isinstance(post_state, Mapping):
            raise _Unavailable("invalid_execution_state", "last execution state is absent")
        pre_fingerprint = pre_state.get("workspace_version_fingerprint")
        post_fingerprint = post_state.get("workspace_version_fingerprint")
        if (
            pre_state.get("complete") is not True
            or post_state.get("complete") is not True
            or not isinstance(pre_fingerprint, str)
            or not pre_fingerprint
            or pre_fingerprint != post_fingerprint
            or pre_fingerprint != decision.get("workspace_version_fingerprint")
        ):
            raise _Unavailable(
                "final_action_changed_or_unverified_workspace",
                "the final action did not preserve one complete checkpointed workspace state",
            )

        index_source, index_patch = _checkpoint_artifact(
            decision_path, decision, "index_patch", "index_patch_sha256"
        )
        worktree_source, worktree_patch = _checkpoint_artifact(
            decision_path, decision, "worktree_patch", "worktree_patch_sha256"
        )
        untracked_source, _ = _checkpoint_artifact(
            decision_path, decision, "untracked_archive", "untracked_archive_sha256"
        )
        _validate_empty_untracked_archive(untracked_source)
        if index_patch and worktree_patch:
            raise _Unavailable(
                "staged_and_unstaged_composition_unsupported",
                "both staged and unstaged deltas are non-empty",
            )
        patch = index_patch or worktree_patch
        if not patch:
            raise _Unavailable("empty_workspace_patch", "checkpoint has no tracked workspace diff")
        if not patch.startswith(b"diff --git "):
            raise _Unavailable("invalid_workspace_patch", "captured bytes are not a Git patch")
        try:
            patch_text = patch.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _Unavailable("non_utf8_workspace_patch", str(error)) from error

        destination.mkdir(parents=True, exist_ok=True)
        index_output = destination / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.index.patch"
        worktree_output = destination / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.worktree.patch"
        patch_output = destination / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.patch"
        predictions_output = destination / f"{AUXILIARY_WORKSPACE_STATE_LABEL}_preds.json"
        index_output.write_bytes(index_patch)
        worktree_output.write_bytes(worktree_patch)
        patch_output.write_bytes(patch)
        predictions = {
            instance_id: {
                "model_name_or_path": model_name,
                "instance_id": instance_id,
                "model_patch": patch_text,
            }
        }
        predictions_output.write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
        provenance.update({
            "status": "available",
            "checkpoint_session": str(session),
            "checkpoint_step": step,
            "workspace_version_fingerprint": pre_fingerprint,
            "last_action_command_sha256": decision["command_sha256"],
            "last_action_workspace_stable": True,
            "source_decision_receipt": str(decision_path),
            "source_decision_receipt_sha256": _sha256_file(decision_path),
            "source_execution_receipt": str(execution_path),
            "source_execution_receipt_sha256": _sha256_file(execution_path),
            "source_index_patch": str(index_source),
            "source_index_patch_sha256": decision["index_patch_sha256"],
            "source_worktree_patch": str(worktree_source),
            "source_worktree_patch_sha256": decision["worktree_patch_sha256"],
            "source_untracked_archive": str(untracked_source),
            "source_untracked_archive_sha256": decision["untracked_archive_sha256"],
            "untracked_archive_members": 0,
            "composition": "single_nonempty_index_or_worktree_delta",
            "primary_predictions": str(primary_path),
            "primary_predictions_sha256": _sha256_file(primary_path),
            "primary_submission_model_patch_sha256": _sha256_bytes(
                primary_model_patch.encode("utf-8")
            ),
            "primary_submission_looks_like_git_diff": primary_model_patch.lstrip().startswith(
                "diff --git "
            ),
            "patch": str(patch_output),
            "patch_sha256": _sha256_bytes(patch),
            "patch_bytes": len(patch),
            "index_patch": str(index_output),
            "index_patch_sha256": _sha256_bytes(index_patch),
            "worktree_patch": str(worktree_output),
            "worktree_patch_sha256": _sha256_bytes(worktree_patch),
            "predictions": str(predictions_output),
            "predictions_sha256": _sha256_file(predictions_output),
            "official_grading_requested": False,
            "official_grading_completed": False,
            "official_result": None,
        })
    except (_Unavailable, OSError) as error:
        code = error.code if isinstance(error, _Unavailable) else "io_error"
        detail = error.detail if isinstance(error, _Unavailable) else str(error)
        destination.mkdir(parents=True, exist_ok=True)
        provenance.update({
            "status": "unavailable",
            "reason": code,
            "detail": detail,
            "official_grading_requested": False,
            "official_grading_completed": False,
            "official_result": None,
        })

    provenance_path.write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return provenance
