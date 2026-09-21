"""Agent-independent validation for SWE-bench terminal submissions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubmissionValidation:
    valid: bool
    reason: str
    file_sections: int = 0


def validate_unified_git_diff(payload: str) -> SubmissionValidation:
    """Require a syntactically recognizable, nonempty Git patch payload.

    This is deliberately structural. It does not claim that the patch applies
    or solves the task; the official grader remains authoritative for both.
    """

    text = payload.lstrip()
    if not text:
        return SubmissionValidation(False, "submission payload is empty")
    if not text.startswith("diff --git "):
        return SubmissionValidation(
            False, "submission payload does not begin with a 'diff --git' header"
        )
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        has_old = any(line.startswith("--- a/") or line == "--- /dev/null" for line in block)
        has_new = any(line.startswith("+++ b/") or line == "+++ /dev/null" for line in block)
        binary = any(
            line == "GIT binary patch" or line.startswith("Binary files ")
            for line in block
        )
        if not binary and not (has_old and has_new):
            return SubmissionValidation(
                False,
                f"diff section {position + 1} lacks unified old/new file headers",
                len(starts),
            )
    return SubmissionValidation(True, "recognized Git patch", len(starts))


def submission_recovery_observation(reason: str) -> str:
    """Return a model-visible, task-agnostic recovery instruction."""

    return (
        "PRA_SUBMISSION_PROTOCOL_ERROR: " + reason + ".\n"
        "The task remains active and no submission was accepted. Create patch.txt "
        "from `git diff -- <modified source files>`, inspect it in a separate "
        "command, then submit exactly with `echo "
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`."
    )
