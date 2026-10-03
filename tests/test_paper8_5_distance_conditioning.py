from experiments.paper8_5_agent_memory.reduce_distance_conditioning import (
    reduce_distance_conditioning,
)
import pytest


def _payload(full_commands=("same", "same")):
    arms = {
        "FULL": {"saving_fraction": 0.0, "exact_noop": True},
        "TASK_D3": {"saving_fraction": 0.25, "exact_noop": False},
    }
    rows = []
    for repeat, command in enumerate(full_commands, start=1):
        rows.append({
            "repeat": repeat,
            "arm": "FULL",
            "selected_messages_sha256": "full-request",
            "command_sha256": command,
            "response_content_sha256": f"response-{command}",
            "action_valid": True,
            "operation_class": "read",
        })
        rows.append({
            "repeat": repeat,
            "arm": "TASK_D3",
            "selected_messages_sha256": "selected-request",
            "command_sha256": "changed",
            "response_content_sha256": "changed-response",
            "action_valid": True,
            "operation_class": "search",
        })
    return {"study": "source", "arms": arms, "rows": rows}


def test_distance_reducer_admits_attribution_only_with_stable_full_command():
    result = reduce_distance_conditioning(_payload())

    assert result["full_control"]["causal_admission"] is True
    assert result["arms"]["TASK_D3"]["paired_command_match_fraction"] == 0.0
    assert result["arms"]["TASK_D3"]["paired_operation_match_fraction"] == 0.0
    assert result["arms"]["TASK_D3"]["causal_interpretation"] == (
        "admitted_against_stable_full"
    )


def test_distance_reducer_withholds_when_full_commands_vary():
    result = reduce_distance_conditioning(_payload(("first", "second")))

    assert result["full_control"]["causal_admission"] is False
    assert result["arms"]["TASK_D3"]["causal_interpretation"] == (
        "withheld_full_unstable_or_incomplete"
    )


def test_reducer_reports_first_logprob_token_divergence():
    payload = _payload()
    full_rows = [row for row in payload["rows"] if row["arm"] == "FULL"]
    for row, token, alternate in (
        (full_rows[0], "a", "b"),
        (full_rows[1], "b", "a"),
    ):
        row["choice_logprobs"] = {"content": [{
            "token": token,
            "logprob": -0.1,
            "top_logprobs": [
                {"token": token, "logprob": -0.1},
                {"token": alternate, "logprob": -0.2},
            ],
        }]}
        row["request_sha256"] = "same"
    result = reduce_distance_conditioning(payload)
    divergence = result["full_control"]["pairwise_token_divergences"][0]
    assert divergence["same_request_sha256"]
    assert divergence["first_divergence"]["token_index"] == 0
    assert divergence["first_divergence"]["left_top1_margin"] == pytest.approx(0.1)
