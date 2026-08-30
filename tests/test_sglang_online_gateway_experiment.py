from experiments.paper6_2_mlx.run_online_native_gateway import _payload


def test_sglang_online_experiment_reuses_common_typed_wire_payload() -> None:
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

    assert payload["pra"]["required_capabilities"] == ["native_kv", "logical_refs"]
    assert payload["pra"]["resources"][0]["record_type"] == "qa_evidence"
