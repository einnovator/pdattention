from experiments.paper8_5_agent_memory.summarize_oracle_headroom import reduce_trials


def _artifact(group, tokens, *, exact=False, generated="grep x a.py"):
    return {
        "study": "paper8_5_agent_memory_frozen_replay",
        "instance_id": "task",
        "reference_replay_digest": "reference",
        "run_configuration": {"oracle_omit_causal_group_ids": [group]},
        "rows": [{
            "decision": 4,
            "action_valid": True,
            "exact_command": exact,
            "exact_content": False,
            "reference_command": "grep x a.py",
            "generated_command": generated,
            "full_history_tokens": 1000,
            "oracle_omission": {
                "omitted_causal_group_ids": [group],
                "omitted_tokens": tokens,
            },
        }],
    }


def test_reducer_separates_exact_and_semantic_quality():
    rows, summary = reduce_trials([
        (_artifact("a", 100, exact=True), "a.json"),
        (_artifact("b", 200, generated="grep y a.py"), "b.json"),
        (_artifact("c", 300, generated="cat b.py"), "c.json"),
    ])
    assert len(rows) == 3
    assert summary == [{
        "instance_id": "task",
        "decision": 4,
        "depth": 1,
        "trials": 3,
        "valid_actions": 3,
        "exact_commands": 1,
        "semantic_transitions": 2,
        "exact_command_rate": 1 / 3,
        "semantic_transition_rate": 2 / 3,
        "max_exact_omitted_tokens": 100,
        "max_exact_omitted_fraction": 0.1,
        "max_semantic_omitted_tokens": 200,
        "max_semantic_omitted_fraction": 0.2,
    }]
