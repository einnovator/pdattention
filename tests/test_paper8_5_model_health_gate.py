import pytest

from experiments.paper8_5_agent_memory.wait_for_model_health import (
    validate_chat_completion,
)


def test_health_completion_requires_exact_deterministic_text() -> None:
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": "PRA_HEALTH_OK"},
            "finish_reason": "stop",
        }]
    }
    assert validate_chat_completion(
        payload, expected_text="PRA_HEALTH_OK"
    ) == "PRA_HEALTH_OK"


def test_health_completion_rejects_alias_or_empty_behavior() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        validate_chat_completion(
            {"choices": [{"message": {"content": "almost"}}]},
            expected_text="PRA_HEALTH_OK",
        )
    with pytest.raises(ValueError, match="empty"):
        validate_chat_completion(
            {"choices": [{"message": {"content": ""}}]},
            expected_text="PRA_HEALTH_OK",
        )
