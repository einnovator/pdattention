from experiments.paper8_5_agent_memory.trajectory_diagnostics import (
    compare_miniswe_trajectories,
    extract_miniswe_steps,
    summarize_miniswe_steps,
)
from experiments.paper8_5_agent_memory.miniswe_semantics import bash_action_contract


def _assistant(command: str):
    return {"role": "assistant", "content": f"```mswea_bash_command\n{command}\n```"}


def _observation(code: int = 0):
    return {"role": "user", "content": f"<returncode>{code}</returncode>\n<output>x</output>"}


def test_miniswe_phase_summary_counts_rework_and_failures():
    messages = [
        _assistant("find /testbed -name '*.py'"), _observation(),
        _assistant("cat /testbed/a.py"), _observation(),
        _assistant("cat /testbed/a.py"), _observation(),
        _assistant("sed -i 's/a/b/' /testbed/a.py"), _observation(1),
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
