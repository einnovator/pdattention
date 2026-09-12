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
import re
import shlex
import tarfile
from typing import Any, Mapping


AUXILIARY_WORKSPACE_STATE_LABEL = "auxiliary_workspace_state"
_COMMAND = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
_SUBMIT_PREFIX = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "


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
    if not decisions:
        raise _Unavailable("checkpoint_receipts_missing", "decision receipts are absent")
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


def _read_only_terminal_pipeline(command: str) -> tuple[str, ...]:
    if not command.startswith(_SUBMIT_PREFIX):
        raise _Unavailable(
            "terminal_command_not_certified_read_only",
            "terminal command lacks the exact submission sentinel prefix",
        )
    pipeline = command[len(_SUBMIT_PREFIX):].strip()
    if not pipeline or any(value in pipeline for value in ("\n", ";", "&", "<", ">", "`", "$")):
        raise _Unavailable(
            "terminal_command_not_certified_read_only",
            "terminal command contains a non-pipeline shell control or redirection",
        )
    if "||" in pipeline:
        raise _Unavailable(
            "terminal_command_not_certified_read_only",
            "terminal command contains a conditional pipeline",
        )
    raw_segments = pipeline.split("|")
    if any(not segment.strip() for segment in raw_segments):
        raise _Unavailable(
            "terminal_command_not_certified_read_only",
            "terminal command contains an empty pipeline segment",
        )
    programs: list[str] = []
    for index, raw_segment in enumerate(raw_segments):
        try:
            tokens = shlex.split(raw_segment, posix=True)
        except ValueError as error:
            raise _Unavailable(
                "terminal_command_not_certified_read_only",
                f"terminal command is not valid shell text: {error}",
            ) from error
        if not tokens:
            raise _Unavailable(
                "terminal_command_not_certified_read_only",
                "terminal command contains an empty segment",
            )
        program = tokens[0]
        programs.append(program)
        if program == "git":
            if index != 0 or len(tokens) < 2 or tokens[1] not in {"diff", "show"}:
                raise _Unavailable(
                    "terminal_command_not_certified_read_only",
                    "only a leading git diff/show producer is permitted",
                )
            unsafe_git_options = {"--ext-diff", "--textconv", "--output"}
            if any(
                token in unsafe_git_options
                or any(token.startswith(option + "=") for option in unsafe_git_options)
                for token in tokens[2:]
            ):
                raise _Unavailable(
                    "terminal_command_not_certified_read_only",
                    "git producer requests an external or output-writing mode",
                )
            continue
        if program == "sed":
            if "-n" not in tokens[1:] or any(
                token == "-i" or token.startswith("--in-place")
                for token in tokens[1:]
            ):
                raise _Unavailable(
                    "terminal_command_not_certified_read_only",
                    "only non-editing sed -n is permitted",
                )
            continue
        if program in {"cat", "grep", "head", "tail"}:
            continue
        raise _Unavailable(
            "terminal_command_not_certified_read_only",
            f"unclassified terminal program: {program}",
        )
    if programs[0] not in {"cat", "grep", "head", "tail", "sed", "git"}:
        raise _Unavailable(
            "terminal_command_not_certified_read_only",
            "terminal pipeline has no read-only producer",
        )
    return tuple(programs)


def _terminal_submission_certificate(
    *,
    primary_path: Path,
    primary_model_patch: str,
    instance_id: str,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory_paths = sorted(primary_path.parent.rglob("*.traj.json"))
    if len(trajectory_paths) != 1:
        raise _Unavailable(
            "terminal_trajectory_ambiguous",
            f"expected one terminal trajectory, found {len(trajectory_paths)}",
        )
    trajectory_path = trajectory_paths[0]
    trajectory = _read_object(trajectory_path)
    info = trajectory.get("info")
    messages = trajectory.get("messages")
    if (
        trajectory.get("instance_id") != instance_id
        or not isinstance(info, Mapping)
        or info.get("exit_status") != "Submitted"
        or not isinstance(messages, list)
        or len(messages) < 2
    ):
        raise _Unavailable(
            "terminal_trajectory_not_submitted",
            "trajectory is not a unique Submitted outcome for the locked instance",
        )
    assistant = messages[-2]
    terminal = messages[-1]
    terminal_extra = terminal.get("extra") if isinstance(terminal, Mapping) else None
    if (
        not isinstance(assistant, Mapping)
        or assistant.get("role") != "assistant"
        or not isinstance(terminal, Mapping)
        or terminal.get("role") != "exit"
        or not isinstance(terminal_extra, Mapping)
        or terminal_extra.get("exit_status") != "Submitted"
        or not isinstance(terminal_extra.get("submission"), str)
    ):
        raise _Unavailable(
            "terminal_trajectory_not_submitted",
            "trajectory does not end with assistant then Submitted exit records",
        )
    assistant_content = assistant.get("content")
    matches = _COMMAND.findall(assistant_content) if isinstance(assistant_content, str) else []
    if len(matches) != 1:
        raise _Unavailable(
            "terminal_command_ambiguous",
            f"expected one terminal Bash command, found {len(matches)}",
        )
    command = matches[0].strip()
    command_digest = _sha256_bytes(command.encode("utf-8"))
    if command_digest != decision.get("command_sha256"):
        raise _Unavailable(
            "terminal_command_digest_mismatch",
            "terminal assistant command does not match the unmatched checkpoint",
        )
    submission = str(terminal_extra["submission"])
    submission_digest = _sha256_bytes(submission.encode("utf-8"))
    primary_digest = _sha256_bytes(primary_model_patch.encode("utf-8"))
    if submission_digest != primary_digest:
        raise _Unavailable(
            "terminal_submission_digest_mismatch",
            "terminal submission does not match the primary predictions model_patch",
        )
    programs = _read_only_terminal_pipeline(command)
    return {
        "terminal_trajectory": str(trajectory_path),
        "terminal_trajectory_sha256": _sha256_file(trajectory_path),
        "terminal_command": command,
        "terminal_command_sha256": command_digest,
        "terminal_command_programs": list(programs),
        "terminal_submission_sha256": submission_digest,
        "terminal_submission_matches_primary_prediction": True,
    }


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
        if len(decisions) not in {len(executions), len(executions) + 1}:
            raise _Unavailable(
                "receipt_sequence_mismatch",
                "decision and execution counts are neither N/N nor terminal N/N-1",
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
        execution_path: Path | None = None
        terminal_certificate: dict[str, Any] = {}
        if len(decisions) == len(executions):
            execution_path, execution = executions[-1]
            if (
                type(execution.get("schema_version")) is not int
                or execution["schema_version"] != 1
            ):
                raise _Unavailable(
                    "unsupported_execution_schema", "execution schema is not version 1"
                )
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
            final_state_certificate = "stable_complete_execution_pre_and_post"
        else:
            pre_fingerprint = decision.get("workspace_version_fingerprint")
            if not isinstance(pre_fingerprint, str) or not pre_fingerprint:
                raise _Unavailable(
                    "invalid_checkpoint_receipt",
                    "terminal checkpoint workspace fingerprint is absent",
                )
            terminal_certificate = _terminal_submission_certificate(
                primary_path=primary_path,
                primary_model_patch=primary_model_patch,
                instance_id=instance_id,
                decision=decision,
            )
            final_state_certificate = "submitted_read_only_terminal_checkpoint"

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
            "last_action_workspace_stable": execution_path is not None,
            "final_state_certificate": final_state_certificate,
            "decision_receipt_count": len(decisions),
            "execution_receipt_count": len(executions),
            "source_decision_receipt": str(decision_path),
            "source_decision_receipt_sha256": _sha256_file(decision_path),
            "source_execution_receipt": (
                str(execution_path) if execution_path is not None else None
            ),
            "source_execution_receipt_sha256": (
                _sha256_file(execution_path) if execution_path is not None else None
            ),
            **terminal_certificate,
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
