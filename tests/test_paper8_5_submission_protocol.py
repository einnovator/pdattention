from experiments.paper8_5_agent_memory.submission_protocol import (
    PYTHON_WORKSPACE_CHECK_SOURCE,
    submission_recovery_observation,
    validate_unified_git_diff,
)


def test_python_workspace_check_rejects_invalid_changed_python(tmp_path):
    import subprocess
    import sys

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
            "commit", "-q", "-m", "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    source.write_text("if True:\nvalue = 2\n", encoding="utf-8")
    invalid = subprocess.run(
        [sys.executable, "-c", PYTHON_WORKSPACE_CHECK_SOURCE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "PRA_PYTHON_SUBMISSION_INVALID" in invalid.stdout

    source.write_text("if True:\n    value = 2\n", encoding="utf-8")
    valid = subprocess.run(
        [sys.executable, "-c", PYTHON_WORKSPACE_CHECK_SOURCE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert valid.returncode == 0
    assert "PRA_PYTHON_SUBMISSION_OK checked_files=1" in valid.stdout


VALID_PATCH = """diff --git a/foo.py b/foo.py
index 257cc56..5716ca5 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
"""


def test_submission_validator_accepts_a_unified_git_patch() -> None:
    result = validate_unified_git_diff(VALID_PATCH)
    assert result.valid is True
    assert result.file_sections == 1


def test_submission_validator_rejects_source_context_as_recoverable() -> None:
    source_context = """    def check(self):
        if self.max_length is not None:
            return []
--
    def deconstruct(self):
        pass
"""
    result = validate_unified_git_diff(source_context)
    assert result.valid is False
    visible = submission_recovery_observation(result.reason)
    assert "task remains active" in visible
    assert "git diff" in visible
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in visible


def test_submission_validator_rejects_header_only_false_positive() -> None:
    result = validate_unified_git_diff("diff --git a/foo.py b/foo.py\nsource excerpt\n")
    assert result.valid is False
    assert "old/new file headers" in result.reason
