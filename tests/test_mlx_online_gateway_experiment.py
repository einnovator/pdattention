from experiments.paper6_2_mlx.run_online_native_gateway import _payload


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
