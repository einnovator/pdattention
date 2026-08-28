from __future__ import annotations

import json
import threading
import urllib.request

import pytest
from click.testing import CliRunner

from pra_hf.cli import cli as hf_cli
from pra_hf.deployment import (
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAGatewayMode,
    PRAWireRequest,
    PRAWireResource,
)
from pra_hf.gateway import PRACapabilityError, PRAGateway, create_gateway_server
from pra_torch.cli import cli as pra_cli


class RecordingAdapter:
    def __init__(self, *, logical_refs=False, native_kv=False, streaming=False):
        self.requests = []
        self.closed = []
        self._capabilities = PRAEngineCapabilities(
            adapter="recording",
            integration_level="E1" if native_kv else "E0",
            logical_refs=logical_refs,
            typed_records=logical_refs,
            text_fallback=True,
            native_kv=native_kv,
            streaming=streaming,
        )

    def capabilities(self):
        return self._capabilities

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request):
        self.requests.append(request)
        return PRAEngineResult("answer", {"ok": True})

    def stream(self, request):
        self.requests.append(request)
        yield {"type": "delta", "text": "ans", "request_id": request.request_id}
        yield {"type": "delta", "text": "wer", "request_id": request.request_id}
        yield {"type": "done", "request_id": request.request_id}

    def close_session(self, session_id):
        self.closed.append(session_id)


def _request(**overrides):
    values = {
        "model": "offline/model",
        "messages": ({"role": "user", "content": "question"},),
        "tenant_id": "tenant-a",
        "session_id": "session-a",
    }
    values.update(overrides)
    return PRAWireRequest(**values)


def _resource(name="facts"):
    return PRAWireResource(
        resource_id=name,
        uri=f"pra://tenant-a/{name}",
        text="alpha beta gamma",
        metadata={"tenant_id": "tenant-a"},
    )


def test_g00_pass_through_preserves_structured_messages():
    adapter = RecordingAdapter()
    request = _request(
        messages=(
            {"role": "user", "content": "question"},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        )
    )

    result = PRAGateway(adapter, mode="G00").generate(request)

    assert adapter.requests[0].messages == request.messages
    assert result.text == "answer"
    assert result.trace[1]["downgrades"] == []


def test_hf_capability_contract_names_supported_runtime_boundaries():
    capabilities = PRAEngineCapabilities(
        adapter="hf",
        integration_level="E1",
        native_kv=True,
        selected_interval_materialization=True,
        request_lifetime=True,
        host_device_residency=True,
        tenant_isolation=True,
    )

    assert capabilities.supports("selected_interval_materialization")
    assert capabilities.supports("request_lifetime")
    assert capabilities.supports("host_device_residency")
    assert capabilities.supports("tenant_isolation")
    assert not capabilities.supports("phase_selection")
    assert not capabilities.supports("scheduler_hints")


def test_g10_is_an_explicit_text_fallback_not_native_pra():
    adapter = RecordingAdapter()
    request = _request(
        resources=(_resource(),),
        required_capabilities=("native_kv",),
        allow_text_fallback=True,
    )

    result = PRAGateway(adapter, mode="G10").generate(request)

    transformed = adapter.requests[0]
    assert transformed.resources == ()
    assert "PRA text fallback context (not native K/V)" in transformed.messages[0]["content"]
    assert result.trace[1]["native_kv"] is False
    assert result.trace[1]["downgrades"]
    assert result.trace[1]["selected_resource_ids"] == ["facts"]


def test_capability_loss_never_silently_downgrades():
    request = _request(
        resources=(_resource(),),
        required_capabilities=("native_kv",),
    )
    with pytest.raises(PRACapabilityError, match="native_kv"):
        PRAGateway(RecordingAdapter(), mode="G00").generate(request)
    with pytest.raises(PRACapabilityError, match="logical_refs"):
        PRAGateway(RecordingAdapter(), mode="G11").generate(
            _request(resources=(_resource(),))
        )


def test_g01_derives_typed_resources_from_ordinary_agent_records():
    adapter = RecordingAdapter(logical_refs=True, native_kv=True)
    request = _request(
        messages=(
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
            {"role": "tool", "content": "large tool result", "tool_call_id": "c1"},
        )
    )

    result = PRAGateway(adapter, mode=PRAGatewayMode.G01_UPGRADE).generate(request)

    assert [resource.record_type for resource in adapter.requests[0].resources] == [
        "system_record",
        "tool_result",
    ]
    assert all(resource.provenance["inferred"] for resource in adapter.requests[0].resources)
    assert result.trace[1]["native_kv"] is True


def test_wire_contract_rejects_cross_tenant_resources_and_credentials():
    foreign = PRAWireResource(
        "foreign",
        "pra://tenant-b/facts",
        text="secret",
        metadata={"tenant_id": "tenant-b"},
    )
    with pytest.raises(PermissionError, match="another tenant"):
        _request(resources=(foreign,))
    with pytest.raises(ValueError, match="Credentials"):
        _request(metadata={"api_key": "must-not-enter-traces"})


def test_openai_envelope_and_http_health_boundary():
    adapter = RecordingAdapter()
    gateway = PRAGateway(adapter, mode="G00")
    server = create_gateway_server(gateway, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
            health = json.loads(response.read())
        assert health["status"] == "ok"
        payload = json.dumps(
            {
                "model": "offline/model",
                "messages": [{"role": "user", "content": "question"}],
                "temperature": 0,
                "pra": {"session_id": "session-a", "tenant_id": "tenant-a"},
            }
        ).encode()
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            completion = json.loads(response.read())
        assert completion["choices"][0]["message"]["content"] == "answer"
        assert completion["pra_trace"][0]["correlation_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_stream_preserves_ids_and_uses_the_same_mediation() -> None:
    adapter = RecordingAdapter(logical_refs=True, native_kv=True, streaming=True)
    request = _request(resources=(_resource(),))

    rows = list(PRAGateway(adapter, mode="G11").stream(request))

    assert rows[0]["type"] == "trace"
    assert rows[0]["request_id"] == request.request_id
    assert rows[0]["trace"]["native_kv"] is True
    assert "".join(row.get("text", "") for row in rows) == "answer"
    assert adapter.requests[0].session_id == request.session_id


def test_gateway_stream_rejects_unsupported_transport_before_iteration() -> None:
    with pytest.raises(PRACapabilityError, match="does not support streaming"):
        PRAGateway(RecordingAdapter(), mode="G00").stream(_request())


def test_openai_http_stream_uses_sse_and_terminates() -> None:
    gateway = PRAGateway(
        RecordingAdapter(streaming=True), mode="G00"
    )
    server = create_gateway_server(gateway, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps({
            "model": "offline/model",
            "messages": [{"role": "user", "content": "question"}],
            "stream": True,
            "pra": {"session_id": "session-a", "tenant_id": "tenant-a"},
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
            assert response.headers["Content-Type"] == "text/event-stream"
        assert '"text": "ans"' in body
        assert '"text": "wer"' in body
        assert "data: [DONE]" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("command", (pra_cli, hf_cli))
def test_gateway_cli_is_available_from_both_entrypoints(command):
    runner = CliRunner()
    result = runner.invoke(command, ["gateway", "serve", "--help"])
    assert result.exit_code == 0
    assert "--backend-url" in result.output
