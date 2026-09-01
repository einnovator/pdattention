from experiments.paper6_2_mlx.run_online_native_gateway import _payload
from experiments.paper6_1_sglang.run_online_native_gateway import _optional_percentile


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


def test_sglang_gateway_preserves_missing_ttft_as_null() -> None:
    assert _optional_percentile([], 0.95) is None
    assert _optional_percentile([1.0, 3.0, 2.0], 0.50) == 2.0
