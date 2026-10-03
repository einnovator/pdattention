from experiments.paper8_5_agent_memory.trajectory_diagnostics import (
    compare_miniswe_trajectories,
    extract_miniswe_steps,
    summarize_miniswe_steps,
)
from experiments.paper8_5_agent_memory.miniswe_semantics import bash_action_contract


def _assistant(command: str):
    return {"role": "assistant", "content": f"```mswea_bash_command\n{command}\n```"}


def _observation(code: int = 0, before: str | None = None, after: str | None = None):
    row = {
        "role": "user",
        "content": f"<returncode>{code}</returncode>\n<output>x</output>",
    }
    if before is not None and after is not None:
        row["extra"] = {
            "workspace_version_fingerprint": before,
            "post_workspace_version_fingerprint": after,
        }
    return row


def test_miniswe_phase_summary_counts_rework_and_failures():
    messages = [
        _assistant("find /testbed -name '*.py'"), _observation(),
        _assistant("cat /testbed/a.py"), _observation(),
        _assistant("cat /testbed/a.py"), _observation(),
        _assistant("sed -i 's/a/b/' /testbed/a.py"), _observation(1, "v1", "v1"),
        _assistant("pytest -q"), _observation(),
        _assistant("git diff > patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
        _observation(),
    ]
    steps = extract_miniswe_steps(messages)
    summary = summarize_miniswe_steps(steps)
    assert summary["calls"] == 6
    assert summary["calls_to_first_mutation"] == 4
    assert summary["calls_to_first_verification"] == 5
    assert summary["failed_tool_calls"] == 1
    assert summary["repeated_operation_resource_calls"] == 1
    assert summary["submission_calls"] == 1
    assert summary["calls_to_first_effective_mutation"] is None
    assert summary["ineffective_mutation_calls"] == 1
    assert summary["workspace_diff_submission_calls"] == 1
    assert summary["effective_source_mutation_calls"] == 0


def test_summary_distinguishes_effective_edit_from_manual_patch_construction():
    messages = [
        _assistant("sed -i 's/a/b/' /testbed/a.py"),
        _observation(0, "v1", "v2"),
        _assistant(
            "echo '--- a/a.py' > patch.txt && "
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        ),
        _observation(),
    ]
    summary = summarize_miniswe_steps(extract_miniswe_steps(messages))
    assert summary["calls_to_first_effective_mutation"] == 1
    assert summary["effective_mutation_calls"] == 1
    assert summary["calls_to_first_effective_source_mutation"] is None
    assert summary["manual_patch_construction_calls"] == 1


def test_source_mutation_excludes_patch_artifact():
    source_edit = _observation(0, "v1", "v2")
    source_edit["extra"].update({
        "resource_version_fingerprints": {"/testbed/a.py": "a"},
        "post_resource_version_fingerprints": {"/testbed/a.py": "b"},
    })
    patch_write = _observation(0, "v2", "v3")
    patch_write["extra"].update({
        "resource_version_fingerprints": {"patch.txt": "missing"},
        "post_resource_version_fingerprints": {"patch.txt": "patch"},
    })
    messages = [
        _assistant("sed -i 's/a/b/' /testbed/a.py"), source_edit,
        _assistant("echo x > patch.txt"), patch_write,
    ]
    summary = summarize_miniswe_steps(extract_miniswe_steps(messages))
    assert summary["calls_to_first_effective_source_mutation"] == 1
    assert summary["effective_source_mutation_calls"] == 1


def test_comparison_separates_shell_spelling_from_semantic_divergence():
    reference = extract_miniswe_steps([
        _assistant("find /testbed -name '*.py'"), _observation(),
        _assistant("cat /testbed/a.py"), _observation(),
    ])
    candidate = extract_miniswe_steps([
        _assistant("find /testbed -name '*.py' 2>/dev/null"), _observation(),
        _assistant("pytest -q"), _observation(),
        _assistant("cat /testbed/a.py"), _observation(),
    ])
    result = compare_miniswe_trajectories(reference, candidate)
    assert result["exact_common_prefix_calls"] == 0
    assert result["semantic_common_prefix_calls"] == 1
    assert result["first_semantic_divergence_call"] == 2
    assert result["call_delta"] == 1


def test_action_contract_detects_observation_bounding_change():
    bounded = bash_action_contract(
        "find /testbed -name '*.py' -type f | head -10"
    )
    unbounded = bash_action_contract("find /testbed -name '*.py' -type f")
    assert bounded["operation"] == unbounded["operation"] == "search_discovery"
    assert bounded["output_bounded"] is True
    assert unbounded["output_bounded"] is False
    assert bounded != unbounded
