import pytest

from experiments.paper8_5_agent_memory.wait_for_model_health import (
    fetch_openai_model_identity,
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


def test_openai_catalog_identity_does_not_claim_revision_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"data":[{"id":"mlx-community/Qwen3-14B-4bit"}]}'

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: Response()
    )
    observed = fetch_openai_model_identity(
        "http://127.0.0.1:8081/v1/models",
        expected_model="mlx-community/Qwen3-14B-4bit",
        declared_revision="snapshot-a",
        timeout_seconds=1,
    )
    assert observed["revision"] == "snapshot-a"
    assert observed["revision_observed_by_endpoint"] is False
    assert observed["identity_basis"] == (
        "openai_model_catalog_plus_declared_revision"
    )


def test_openai_catalog_identity_rejects_missing_served_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"data":[{"id":"other"}]}'

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(ValueError, match="not found"):
        fetch_openai_model_identity(
            "http://127.0.0.1:8081/v1/models",
            expected_model="mlx-community/Qwen3-14B-4bit",
            declared_revision="snapshot-a",
            timeout_seconds=1,
        )
