from __future__ import annotations

import json
import threading
import urllib.request
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from pra_hf.cli import cli as hf_cli
from pra_hf.deployment import (
    HuggingFaceEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAGatewayMode,
    PRAWireRequest,
    PRAWireResource,
    OpenAICompatibleEngineAdapter,
)
from pra_hf.gateway import (
    PRACapabilityError,
    PRAGateway,
    _request_from_responses,
    create_gateway_server,
)
from pra_torch.cli import cli as pra_cli


class RecordingAdapter:
    def __init__(self, *, logical_refs=False, native_kv=False, streaming=False):
        self.requests = []
        self.closed = []
        self._capabilities = PRAEngineCapabilities(
            adapter="recording",
            integration_level="E2" if native_kv else "E0",
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


class UntypedNativeStreamAdapter(RecordingAdapter):
    """Model an engine executor that emits raw token rows."""

    def stream(self, request):
        self.requests.append(request)
        yield {"text": "raw", "native_kv_used": True}
        yield {"text": " token", "native_kv_used": True}


class ToolStreamingAdapter(RecordingAdapter):
    def __init__(self):
        super().__init__(streaming=True)

    def stream(self, request):
        self.requests.append(request)
        yield {
            "type": "tool_call_delta",
            "index": 0,
            "call_id": "call-7",
            "name": "shell",
            "arguments": '{"command":"echo ',
            "request_id": request.request_id,
        }
        yield {
            "type": "tool_call_delta",
            "index": 0,
            "arguments": 'ok"}',
            "request_id": request.request_id,
        }
        yield {
            "type": "done",
            "request_id": request.request_id,
            "finish_reason": "tool_calls",
            "raw": {
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 4,
                    "total_tokens": 25,
                }
            },
        }


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


def test_typed_decode_limit_precedes_legacy_hint_and_openai_max_tokens_is_mapped():
    legacy = _request(engine_hints={"max_new_tokens": 2})
    typed = _request(max_new_tokens=3, engine_hints={"max_new_tokens": 2})
    openai = PRAWireRequest.from_openai(
        {
            "model": "offline/model",
            "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 4,
        }
    )

    assert legacy.resolved_max_new_tokens == 2
    assert typed.resolved_max_new_tokens == 3
    assert openai.resolved_max_new_tokens == 4


def test_openai_adapter_does_not_invent_a_decode_limit() -> None:
    adapter = OpenAICompatibleEngineAdapter("http://engine")

    assert "max_tokens" not in adapter._payload(_request())
    assert adapter._payload(_request(max_new_tokens=7))["max_tokens"] == 7


def test_hf_non_streaming_adapter_forwards_resolved_decode_limit():
    class RecordingModel:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return SimpleNamespace(text="answer", stats={"generated_tokens": 2})

    model = RecordingModel()
    request = _request(max_new_tokens=2)

    result = HuggingFaceEngineAdapter(model).generate(request)

    assert result.text == "answer"
    assert model.calls[0][1]["max_new_tokens"] == 2


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
        integration_level="E2",
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
        assert completion["pra"]["native_kv"] is False
        assert completion["pra"]["trace_id"]
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


def test_gateway_normalizes_raw_native_token_rows_and_commits_session() -> None:
    adapter = UntypedNativeStreamAdapter(
        logical_refs=True, native_kv=True, streaming=True
    )
    request = _request(resources=(_resource(),))

    rows = list(PRAGateway(adapter, mode="G11").stream(request))

    assert [row["type"] for row in rows[1:4]] == ["delta", "delta", "done"]
    assert "".join(row.get("text", "") for row in rows) == "raw token"
    assert rows[1]["request_id"] == request.request_id
    assert rows[3]["native_kv_used"] is True


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
        assert '"content": "ans"' in body
        assert '"content": "wer"' in body
        assert '"object": "chat.completion.chunk"' in body
        assert "data: [DONE]" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_openai_http_stream_preserves_tool_calls_and_usage() -> None:
    server = create_gateway_server(
        PRAGateway(ToolStreamingAdapter(), mode="G00"), host="127.0.0.1", port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps({
            "model": "offline/model",
            "messages": [{"role": "user", "content": "run a command"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
        assert '"tool_calls": [' in body
        assert '"id": "call-7"' in body
        assert '"name": "shell"' in body
        assert '"arguments": "{\\\"command\\\":\\\"echo "' in body
        assert '"prompt_tokens": 21' in body
        assert '"completion_tokens": 4' in body
        assert '"finish_reason": "tool_calls"' in body
        assert "data: [DONE]" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_generic_openai_adapter_normalizes_sse_stream() -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            return iter(
                (
                    b'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"shell","arguments":"{\\"command\\":"}}]},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"echo ok\\"}"}}]},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n',
                )
            )

    adapter = OpenAICompatibleEngineAdapter("http://engine")
    with patch("urllib.request.urlopen", return_value=Response()):
        rows = list(adapter.stream(_request()))

    assert adapter.capabilities().streaming is True
    assert [row["type"] for row in rows] == [
        "delta", "tool_call_delta", "tool_call_delta", "delta", "done"
    ]
    assert "".join(row.get("text", "") for row in rows) == "hello"
    assert "".join(row.get("arguments", "") for row in rows) == '{"command":"echo ok"}'
    assert rows[-1]["raw"]["usage"]["prompt_tokens"] == 12
    assert rows[-1]["finish_reason"] == "tool_calls"


def test_responses_endpoint_preserves_function_output_pairing() -> None:
    adapter = RecordingAdapter()
    server = create_gateway_server(PRAGateway(adapter, mode="G00"), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps(
            {
                "model": "offline/model",
                "instructions": "Be concise.",
                "input": [
                    {"type": "message", "id": "item-1", "role": "user", "content": [{"type": "input_text", "text": "question"}]},
                    {"type": "function_call_output", "call_id": "call-7", "output": "tool result"},
                ],
                "pra": {"tenant_id": "tenant-a", "session_id": "session-a"},
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/responses",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["object"] == "response"
        assert result["output_text"] == "answer"
        assert adapter.requests[0].messages[1]["item_id"] == "item-1"
        assert adapter.requests[0].messages[2]["tool_call_id"] == "call-7"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_responses_tools_are_normalized_for_chat_upstreams() -> None:
    request = _request_from_responses(
        {
            "model": "offline/model",
            "input": "question",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "custom",
                    "name": "shell",
                    "description": "Run a shell command.",
                },
            ],
        }
    )

    assert request.tools[0]["function"]["name"] == "lookup"
    assert request.tools[1]["function"]["name"] == "shell"
    assert request.tools[1]["function"]["parameters"]["required"] == ["input"]
    assert request.metadata["responses_tool_types"] == {
        "lookup": "function",
        "shell": "custom",
    }


def test_responses_stream_emits_codex_text_event_sequence() -> None:
    server = create_gateway_server(
        PRAGateway(RecordingAdapter(streaming=True), mode="G00"),
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps(
            {"model": "offline/model", "input": "question", "stream": True}
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/responses",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
        assert "event: response.created" in body
        assert body.count("event: response.output_text.delta") == 2
        assert "event: response.completed" in body
        assert '"output_text": "answer"' in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_responses_stream_preserves_tool_calls_and_usage() -> None:
    server = create_gateway_server(
        PRAGateway(ToolStreamingAdapter(), mode="G00", models=("coder-model",)),
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/v1/models", timeout=5
        ) as response:
            models = json.loads(response.read())
        assert models["data"][0]["id"] == "coder-model"

        payload = json.dumps(
            {
                "model": "coder-model",
                "input": "run a command",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/responses",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
        assert "event: response.function_call_arguments.delta" in body
        assert "event: response.function_call_arguments.done" in body
        assert '"call_id": "call-7"' in body
        assert '\"arguments\": \"{\\\"command\\\":\\\"echo ok\\\"}\"' in body
        assert '"input_tokens": 21' in body
        assert '"output_tokens": 4' in body
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
