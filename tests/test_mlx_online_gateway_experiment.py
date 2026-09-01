from experiments.paper6_2_mlx.run_online_native_gateway import _event_text, _payload
from pra_hf.deployment import PRAWireRequest


def test_online_payload_keeps_native_resource_in_pra_envelope() -> None:
    payload = _payload(
        model="model",
        session_id="session",
        request_id="request",
        question="question",
        resource_id="resource",
        source="evidence",
        version="v1",
        max_new_tokens=8,
    )

    assert payload["messages"] == [{"role": "user", "content": "question"}]
    assert payload["pra"]["pra_policy"]["selected_resource_ids"] == ["resource"]
    assert payload["pra"]["resources"][0]["metadata"]["shareable"] is True
    assert "evidence" not in payload["messages"][0]["content"]

    request = PRAWireRequest.from_openai(payload)
    assert request.request_id == "request"
    assert request.tenant_id == "benchmark"
    assert request.pra_policy["selected_resource_ids"] == ["resource"]


def test_event_text_accepts_openai_delta_envelope() -> None:
    assert _event_text({"choices": [{"delta": {"content": "answer"}}]}) == "answer"


def test_event_text_accepts_legacy_gateway_envelope() -> None:
    assert _event_text({"text": "answer"}) == "answer"


def test_event_text_ignores_non_content_events() -> None:
    assert _event_text({"choices": [{"delta": {}}]}) == ""
