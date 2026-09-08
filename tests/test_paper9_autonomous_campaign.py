"""Contracts for the real-model Paper 9 campaign harness."""

from __future__ import annotations

from experiments.paper9_subagents.run_autonomous_repository_campaign import (
    CampaignCondition,
    TASKS,
    _textual_tool_calls,
    run_condition,
)


class DeterministicToolClient:
    """Tiny model stand-in that autonomously chooses one read before answering."""

    def chat(self, messages, tools):
        if messages[-1]["role"] != "tool":
            return (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_text",
                                "arguments": {"path": "src/pra_hf/subagent_context.py"},
                            }
                        }
                    ],
                },
                {"prompt_tokens": 10, "generated_tokens": 3, "model_seconds": 0.01},
            )
        return (
            {
                "role": "assistant",
                "content": "The implementation is in src/pra_hf/subagent_context.py.",
            },
            {"prompt_tokens": 20, "generated_tokens": 8, "model_seconds": 0.01},
        )


def test_qwen_textual_function_markup_is_normalized() -> None:
    calls = _textual_tool_calls(
        "I will inspect it.\n<function=search_text>\n"
        "<parameter=query>UNKNOWN tool effects</parameter>\n"
        "<parameter=path>src</parameter>\n</function>\n</tool_call>"
    )
    assert calls == [
        {
            "function": {
                "name": "search_text",
                "arguments": {"query": "UNKNOWN tool effects", "path": "src"},
            }
        }
    ]


def test_autonomous_campaign_separates_peer_consumption_from_scheduling(tmp_path) -> None:
    isolated_rows, isolated = run_condition(
        DeterministicToolClient(),
        tmp_path.parents[0] / "missing",
        TASKS[:2],
        CampaignCondition("sequential_isolated", False, False),
        seed=11,
        max_steps=3,
        max_workers=2,
    )
    assert isolated["failures"] == 0
    assert isolated["path_accuracy"] == 1
    assert isolated["logical_tool_calls"] == 2
    assert isolated["physical_tool_calls"] == 2
    assert sum(int(row["routed_peer_records"]) for row in isolated_rows) == 0


def test_completed_peer_policy_reuses_autonomous_duplicate_read(tmp_path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src" / "pra_hf"
    source.mkdir(parents=True)
    (source / "subagent_context.py").write_text("class AgentContextGraph: pass\n")
    rows, summary = run_condition(
        DeterministicToolClient(),
        repo,
        TASKS[:2],
        CampaignCondition("sequential_completed", False, True),
        seed=11,
        max_steps=3,
        max_workers=2,
    )
    assert summary["failures"] == 0
    assert summary["path_accuracy"] == 1
    assert summary["logical_tool_calls"] == 2
    assert summary["physical_tool_calls"] == 1
    assert summary["reused_tool_calls"] == 1
    assert sum(int(row["routed_peer_records"]) for row in rows) == 1
