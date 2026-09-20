from __future__ import annotations

import json
from pathlib import Path

from experiments.paper8_5_agent_memory.run_openhands_swebench import (
    _event_summary,
)


def test_openhands_event_summary_separates_actions_and_semantic_errors(
    tmp_path: Path,
):
    events = tmp_path / "events.jsonl"
    rows = [
        {
            "type": "ActionEvent",
            "event": {
                "llm_response_id": "response-1",
                "action": {"kind": "TerminalAction"},
            },
        },
        {
            "type": "ObservationEvent",
            "event": {
                "observation": {
                    "kind": "TerminalObservation",
                    "is_error": False,
                    "exit_code": 2,
                },
            },
        },
        {
            "type": "ActionEvent",
            "event": {
                "llm_response_id": "response-2",
                "action": {"kind": "FinishAction"},
            },
        },
        {
            "type": "paper85_run_summary",
            "event_count": 3,
            "execution_status": "finished",
        },
    ]
    events.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = _event_summary(events)

    assert summary["action_count"] == 2
    assert summary["observation_count"] == 1
    assert summary["distinct_llm_response_count"] == 2
    assert summary["semantic_failure_observations"] == 1
    assert summary["action_counts"] == {
        "FinishAction": 1,
        "TerminalAction": 1,
    }
    assert summary["run_summary"]["execution_status"] == "finished"
