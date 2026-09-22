from __future__ import annotations

from experiments.paper4_5_agent.openai_response import completion_finish_reason
from pra_hf.deployment import PRAEngineResult, PRAWireRequest


def _request(max_tokens: int = 8) -> PRAWireRequest:
    return PRAWireRequest(
        model="m",
        messages=({"role": "user", "content": "task"},),
        max_new_tokens=max_tokens,
    )


def test_completion_ceiling_is_length_not_semantic_stop() -> None:
    result = PRAEngineResult(
        "truncated",
        raw={"usage": {"prompt_tokens": 10, "completion_tokens": 8}},
    )
    assert completion_finish_reason(_request(), result) == "length"


def test_short_completion_is_semantic_stop() -> None:
    result = PRAEngineResult(
        "done",
        raw={"usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    )
    assert completion_finish_reason(_request(), result) == "stop"


def test_explicit_provider_reason_wins() -> None:
    result = PRAEngineResult(
        "done",
        raw={
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        },
    )
    assert completion_finish_reason(_request(), result) == "tool_calls"
