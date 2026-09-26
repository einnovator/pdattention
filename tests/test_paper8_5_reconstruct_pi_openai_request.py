import json

import pytest

from experiments.paper8_5_agent_memory.reconstruct_pi_openai_request import (
    _digest,
    pi_message_to_openai,
    reconstruct_request,
)


def _events():
    return [
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "task"}],
                "timestamp": 1,
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "read",
                    "arguments": {"path": "b.py", "offset": 4},
                }],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": "result ✓"}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        },
    ]


def test_pi_projection_preserves_null_content_argument_order_and_unicode():
    projected = pi_message_to_openai(_events()[1]["message"])
    assert projected["content"] is None
    assert projected["tool_calls"][0]["function"]["arguments"] == (
        '{"path":"b.py","offset":4}'
    )
    tool = pi_message_to_openai(_events()[2]["message"])
    assert tool == {
        "role": "tool",
        "content": "result ✓",
        "tool_call_id": "call-1",
    }


def test_reconstruction_stops_before_indexed_assistant_and_validates_digest():
    template = {
        "model": "locked",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "older capture"},
        ],
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }
    expected_messages = [
        template["messages"][0],
        pi_message_to_openai(_events()[0]["message"]),
        pi_message_to_openai(_events()[1]["message"]),
        pi_message_to_openai(_events()[2]["message"]),
    ]
    result = reconstruct_request(
        template,
        _events(),
        native_request_index=2,
        expected_request_input_sha256=_digest(expected_messages),
    )
    assert result["messages"] == expected_messages
    assert result["tools"] == template["tools"]

    with pytest.raises(ValueError, match="frozen proxy identity"):
        reconstruct_request(
            template,
            _events(),
            native_request_index=2,
            expected_request_input_sha256="0" * 64,
        )
