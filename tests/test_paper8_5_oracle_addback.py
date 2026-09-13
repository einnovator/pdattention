import pytest

from experiments.paper8_5_agent_memory.oracle_addback_queue import build_queue


def test_oracle_queue_uses_first_divergence_and_minimal_group_interventions():
    candidate = {
        "study": "paper8_5_agent_memory_frozen_replay",
        "instance_id": "task-1",
        "run_configuration_digest": "abc",
        "first_command_divergence": 7,
        "rows": [{
            "decision": 7,
            "action_valid": True,
            "exclusions": [
                {"causal_group_id": "turn:t1", "excluded_tokens": 90,
                 "rule_id": "H1", "resource_ids": ["a.py"]},
                {"causal_group_id": "turn:t0", "excluded_tokens": 10,
                 "rule_id": "H3", "resource_ids": ["b.py"]},
            ],
        }],
    }

    result = build_queue(candidate)

    assert result["evidence_class"].startswith("diagnostic_oracle")
    assert result["divergence_decision"] == 7
    assert [row["oracle_addback_causal_group_ids"] for row in result["trials"]] == [
        ["turn:t0"], ["turn:t1"],
    ]
    assert result["trials"][0]["cli_arguments"][-2:] == [
        "--oracle-addback-group", "turn:t0",
    ]


def test_oracle_queue_rejects_nondivergent_replay():
    with pytest.raises(ValueError, match="no command or validity divergence"):
        build_queue({
            "study": "paper8_5_agent_memory_frozen_replay",
            "rows": [{"decision": 1, "action_valid": True, "exclusions": []}],
        })
