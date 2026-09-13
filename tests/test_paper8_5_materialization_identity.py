from experiments.paper8_5_agent_memory import recordize_minisweagent_messages
from experiments.paper8_5_agent_memory.materialization import (
    MaterializationMode,
    ToolObservationMaterializer,
)


def test_all_selected_lines_preserve_exact_observation_bytes():
    history = recordize_minisweagent_messages((
        {"role": "system", "content": "Use bash."},
        {"role": "user", "content": "Inspect src/example.py."},
        {
            "role": "assistant",
            "content": "```mswea_bash_command\nsed -n '1,4p' src/example.py\n```",
        },
        {
            "role": "user",
            "content": (
                "<returncode>0</returncode>\n<output>\n"
                "src/example.py:1: class Example:\n"
                "src/example.py:2:     pass\n"
                "</output>\n"
            ),
        },
    ))
    observation = history.record_by_id["m3"]
    row = ToolObservationMaterializer(
        mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
        threshold_tokens=1,
        match_context_lines=10,
        max_matched_lines=32,
    ).materialize(
        observation,
        query="Inspect src/example.py.",
    )

    assert row.mode == MaterializationMode.TOOL_STRUCTURED_EVIDENCE
    assert row.selected_line_spans == ((0, len(observation.content.splitlines())),)
    assert row.content == observation.content
    assert row.materialized_tokens == row.original_tokens
