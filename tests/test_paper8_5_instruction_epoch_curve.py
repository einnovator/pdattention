from experiments.paper8_5_agent_memory.instruction_epoch_curve import (
    instruction_epoch_curve,
)


def _trajectory(instance_id: str, payload: str) -> dict:
    return {
        "instance_id": instance_id,
        "info": {"exit_status": "Submitted", "submission": "diff --git a/a b/a"},
        "messages": [
            {"role": "system", "content": "Use bash."},
            {"role": "user", "content": f"Fix {instance_id}."},
            {
                "role": "assistant",
                "content": "```mswea_bash_command\ncat a.py\n```",
            },
            {"role": "user", "content": f"<output>{payload}</output>"},
            {
                "role": "assistant",
                "content": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
            {"role": "exit", "content": "diff --git a/a b/a"},
        ],
    }


def test_instruction_epoch_curve_separates_whole_request_and_interaction_saving():
    result = instruction_epoch_curve((
        _trajectory("repo-a__one", "old payload " * 100),
        _trajectory("repo-b__two", "middle payload " * 100),
        _trajectory("repo-c__three", "new payload " * 100),
    ))
    points = result["points"]

    assert [row["issue_count"] for row in points] == [1, 2, 3]
    assert points[0]["gross_saving_fraction"] == 0
    assert points[1]["gross_saving_fraction"] > 0
    assert points[2]["gross_saving_fraction"] > points[1]["gross_saving_fraction"]
    assert all(
        row["assistant_tool_saving_fraction"] >= row["gross_saving_fraction"]
        for row in points
    )
    assert all(row["instruction_floor_fraction"] > 0 for row in points)
    assert result["claim_scope"].startswith("logical opportunity")
