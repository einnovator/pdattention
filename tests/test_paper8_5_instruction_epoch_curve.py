from experiments.paper8_5_agent_memory.instruction_epoch_curve import (
    instruction_epoch_curve,
)
from experiments.paper8_5_agent_memory.plot_instruction_epoch_curve import curve_series
from experiments.paper8_5_agent_memory.selectors import (
    PersistentInstructionEpochRetirementConfig,
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


def test_scaling_curve_distinguishes_interaction_ideal_from_whole_request():
    evidence = instruction_epoch_curve((
        _trajectory("repo-a__one", "old payload " * 100),
        _trajectory("repo-b__two", "middle payload " * 100),
        _trajectory("repo-c__three", "new payload " * 100),
    ))
    series = curve_series(evidence, maximum_n=10)

    assert series["interaction_ideal"][-1] == 0.9
    assert series["fixed_instruction_fraction_illustration"][-1] < 0.9
    assert series["empirical_interaction"][-1] > series["empirical_whole"][-1]


def test_instruction_epoch_curve_records_and_applies_progress_spine_config():
    trajectories = (
        _trajectory("repo-a__one", "old payload " * 100),
        _trajectory("repo-b__two", "middle payload " * 100),
        _trajectory("repo-c__three", "new payload " * 100),
    )
    e0 = instruction_epoch_curve(trajectories)
    spine = instruction_epoch_curve(
        trajectories,
        config=PersistentInstructionEpochRetirementConfig(
            prior_protocol_turns=1,
        ),
    )

    assert spine["selector_configuration"] == {
        "prior_recent_turns": 0,
        "prior_mutation_turns": 0,
        "prior_verification_turns": 0,
        "prior_protocol_turns": 1,
        "prior_full_epochs": 0,
    }
    assert (
        spine["points"][-1]["selected_cumulative_tokens"]
        >= e0["points"][-1]["selected_cumulative_tokens"]
    )
