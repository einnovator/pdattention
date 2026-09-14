import pytest

from experiments.paper8_5_agent_memory.oracle_headroom_queue import build_queue


def _artifact(
    groups, *, exact=True, tokens=100, task="task", decision=10,
    reference="pytest -q a.py", generated=None,
):
    return {
        "study": "paper8_5_agent_memory_frozen_replay",
        "instance_id": task,
        "reference_replay_digest": "full",
        "run_configuration": {"oracle_omit_causal_group_ids": groups},
        "rows": [{
            "decision": decision,
            "action_valid": True,
            "exact_command": exact,
            "exact_content": exact,
            "reference_command": reference,
            "generated_command": generated or reference,
            "full_history_tokens": 1000,
            "oracle_omission": {
                "omitted_causal_group_ids": groups,
                "omitted_tokens": tokens,
            },
        }],
    }


def test_pair_queue_uses_only_exact_safe_singletons_and_ranks_saving():
    artifacts = [
        (_artifact(["a"], tokens=100), "a.json"),
        (_artifact(["b"], tokens=300), "b.json"),
        (_artifact(["c"], tokens=200), "c.json"),
        (_artifact(["d"], exact=False, tokens=900), "d.json"),
    ]
    result = build_queue(artifacts, target_depth=2, beam_width=2)
    assert result["safe_singletons"] == ["a", "b", "c"]
    assert [row["omitted_causal_group_ids"] for row in result["trials"]] == [
        ("b", "c"), ("a", "b")
    ]


def test_depth_three_expands_only_successful_pair_parents():
    artifacts = [
        (_artifact(["a"]), "a.json"),
        (_artifact(["b"]), "b.json"),
        (_artifact(["c"]), "c.json"),
        (_artifact(["a", "b"], exact=True, tokens=200), "ab.json"),
        (_artifact(["a", "c"], exact=False, tokens=200), "ac.json"),
    ]
    result = build_queue(artifacts, target_depth=3, beam_width=8)
    assert [row["omitted_causal_group_ids"] for row in result["trials"]] == [
        ("a", "b", "c")
    ]


def test_queue_rejects_mixed_decisions():
    with pytest.raises(ValueError, match="do not share"):
        build_queue([
            (_artifact(["a"], decision=10), "a.json"),
            (_artifact(["b"], decision=11), "b.json"),
        ], target_depth=2, beam_width=2)


def test_semantic_transition_gate_accepts_syntax_check_variants():
    artifacts = [
        (_artifact(
            ["a"], exact=False,
            reference="python3 -c \"import ast; ast.parse(open('a.py').read())\"",
            generated="python3 -m py_compile a.py",
        ), "a.json"),
        (_artifact(["b"]), "b.json"),
    ]
    exact = build_queue(
        artifacts, target_depth=2, beam_width=2,
        qualification="exact_command",
    )
    semantic = build_queue(
        artifacts, target_depth=2, beam_width=2,
        qualification="semantic_transition",
    )
    assert exact["trial_count"] == 0
    assert semantic["trial_count"] == 1
