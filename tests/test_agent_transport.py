from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pra_hf.agent_transport import (
    AgentTransportCapabilities,
    AgentTurnContext,
    AgentWireMode,
    ContextTransportMode,
    NegotiatedRemoteBackend,
    PRAProtocolRequiredError,
    context_record_to_wire_resource,
    render_text_messages,
    resolve_wire_mode,
    wire_resource_identity,
)
from pra_hf.context_records import ContextRecord, RecordType, RecordView, RecordViewName
from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult, PRAWireRequest
from pra_hf.gateway import PRAGateway, create_gateway_server
from pra_hf.gateway_session import HistoryMode, ResourceOperation


class _Adapter:
    def __init__(self, *, pra: bool) -> None:
        self.requests = []
        self._capabilities = PRAEngineCapabilities(
            adapter="transport-test",
            integration_level="E2" if pra else "E0",
            logical_refs=pra,
            typed_records=pra,
            task_metadata=pra,
            resource_delta=pra,
            session_state=True,
            incremental_messages=True,
            native_kv=pra,
            text_fallback=True,
        )

    def capabilities(self):
        return self._capabilities

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request):
        self.requests.append(request)
        return PRAEngineResult("answer")

    def stream(self, request):
        raise NotImplementedError

    def close_session(self, session_id):
        return None


def _record(version: str = "v1", body: str = "alpha evidence") -> ContextRecord:
    return ContextRecord(
        "record-1",
        RecordType.RAG_RESULT,
        {"uri": "pra://tenant-a/evidence", "body": body},
        version=version,
        selection_provenance={
            "authorization_scope": "scope-a",
            "task": {"task_id": "task-1", "task_status": "active"},
        },
        views={
            RecordViewName.COMPACT: RecordView(
                RecordViewName.COMPACT, {"body": "alpha"}, ("body",)
            ),
            RecordViewName.FULL: RecordView(
                RecordViewName.FULL,
                {"uri": "pra://tenant-a/evidence", "body": body},
                ("uri", "body"),
            ),
        },
    )


def _turn(*, messages=None, record=None) -> AgentTurnContext:
    value = record or _record()
    return AgentTurnContext(
        messages=tuple(messages or ({"role": "user", "content": "question"},)),
        records=(value,),
        task_id="task-1",
        task_metadata={"status": "active"},
        selected_record_ids=(value.record_id,),
    )


def _serve(gateway: PRAGateway):
    server = create_gateway_server(gateway, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def test_context_record_projection_preserves_portable_semantics() -> None:
    resource = context_record_to_wire_resource(
        _record(), tenant_id="tenant-a", session_id="session-a"
    )

    assert resource.resource_id == "record-1"
    assert resource.uri == "pra://tenant-a/evidence"
    assert resource.task_id == "task-1"
    assert resource.task_status == "active"
    assert resource.authorization_scope == "scope-a"
    assert set(resource.available_views) == {"compact", "full"}
    assert resource.selected_view == "full"
    assert "alpha evidence" in resource.text
    assert not hasattr(resource, "native_kv")


def test_wire_resource_identity_changes_with_selected_view() -> None:
    compact = context_record_to_wire_resource(
        _record(), selected_view=RecordViewName.COMPACT
    )
    full = context_record_to_wire_resource(
        _record(), selected_view=RecordViewName.FULL
    )

    assert wire_resource_identity(compact) != wire_resource_identity(full)


def test_feature_based_transport_resolution() -> None:
    plain = AgentTransportCapabilities.ordinary_openai()
    typed = AgentTransportCapabilities(
        protocol_version="1", logical_refs=True, typed_records=True
    )
    delta = AgentTransportCapabilities(
        protocol_version="1", logical_refs=True, typed_records=True,
        resource_delta=True, incremental_messages=True,
    )

    assert resolve_wire_mode("auto", plain) == AgentWireMode.TEXT
    assert resolve_wire_mode("auto", typed) == AgentWireMode.PRA_FULL
    assert resolve_wire_mode("auto", delta) == AgentWireMode.PRA_DELTA
    assert resolve_wire_mode("text", delta) == AgentWireMode.TEXT
    with pytest.raises(PRAProtocolRequiredError):
        resolve_wire_mode("pra", plain, allow_text_fallback=False)


def test_auto_to_g10_keeps_typed_upstream_and_downgrades_at_gateway() -> None:
    adapter = _Adapter(pra=False)
    server, thread, endpoint = _serve(PRAGateway(adapter, mode="G10"))
    try:
        backend = NegotiatedRemoteBackend(endpoint, "model", transport="auto")
        assert backend.generate_turn(
            _turn(), tenant_id="tenant-a", session_id="session-a"
        ) == "answer"

        assert backend.inspect()["transport"]["negotiated_transport"] == "PRA_DELTA"
        assert adapter.requests[0].resources == ()
        assert "PRA text fallback context" in adapter.requests[0].messages[0]["content"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_auto_to_g11_preserves_resources_and_sends_body_only_on_change() -> None:
    adapter = _Adapter(pra=True)
    server, thread, endpoint = _serve(PRAGateway(adapter, mode="G11"))
    try:
        backend = NegotiatedRemoteBackend(endpoint, "model", transport="auto")
        first_messages = ({"role": "user", "content": "one"},)
        backend.generate_turn(
            _turn(messages=first_messages), tenant_id="tenant-a", session_id="session-a"
        )
        second_messages = (
            *first_messages,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "two"},
        )
        backend.generate_turn(
            _turn(messages=second_messages), tenant_id="tenant-a", session_id="session-a"
        )

        first, second = adapter.requests
        assert first.resources[0].resource_id == "record-1"
        assert first.resource_ops[0].operation == ResourceOperation.ADD
        assert second.history_mode == HistoryMode.DELTA
        assert second.messages == ({"role": "user", "content": "two"},)
        assert second.resources == ()
        assert second.resource_ops[0].operation == ResourceOperation.UNCHANGED

        backend.refresh_capabilities()
        backend.generate_turn(
            _turn(messages=second_messages), tenant_id="tenant-a", session_id="session-a"
        )
        resynced = adapter.requests[-1]
        assert resynced.history_mode == HistoryMode.FULL
        assert resynced.resource_ops[0].operation == ResourceOperation.ADD
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_forced_text_to_pra_endpoint_is_a_deterministic_baseline() -> None:
    adapter = _Adapter(pra=True)
    server, thread, endpoint = _serve(PRAGateway(adapter, mode="G11"))
    try:
        backend = NegotiatedRemoteBackend(endpoint, "model", transport="text")
        backend.generate_turn(_turn(), tenant_id="tenant-a", session_id="session-a")

        assert adapter.requests[0].resources == ()
        assert "PRA text fallback context" in adapter.requests[0].messages[0]["content"]
        assert backend.inspect()["transport"]["negotiated_transport"] == "TEXT"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_forced_pra_rejects_reachable_openai_server_without_pra() -> None:
    class OrdinaryHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), OrdinaryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        backend = NegotiatedRemoteBackend(
            endpoint, "model", transport=ContextTransportMode.PRA,
            allow_text_fallback=False,
        )
        with pytest.raises(PRAProtocolRequiredError):
            backend.generate_turn(_turn(), session_id="session-a")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_auto_uses_text_for_reachable_ordinary_openai_server() -> None:
    captured = []

    class OrdinaryHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            size = int(self.headers.get("Content-Length", "0"))
            captured.append(json.loads(self.rfile.read(size)))
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "plain"}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), OrdinaryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        backend = NegotiatedRemoteBackend(endpoint, "model", transport="auto")
        assert backend.generate_turn(_turn(), session_id="session-a") == "plain"
        assert "pra" not in captured[0]
        assert "PRA text fallback context" in captured[0]["messages"][0]["content"]
        assert backend.inspect()["transport"]["fallback"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_direct_text_and_g10_use_identical_execution_messages() -> None:
    adapter = _Adapter(pra=False)
    turn = _turn()
    resource = context_record_to_wire_resource(turn.records[0])
    request = PRAWireRequest(
        "model",
        turn.messages,
        resources=(resource,),
        allow_text_fallback=True,
    )

    PRAGateway(adapter, mode="G10").generate(request)

    assert adapter.requests[0].messages == render_text_messages(turn)


def test_gateway_preserves_provider_native_tool_schema() -> None:
    adapter = _Adapter(pra=False)
    tool = {"type": "function", "function": {"name": "lookup"}}
    request = PRAWireRequest(
        "model",
        ({"role": "user", "content": "question"},),
        tools=(tool,),
    )

    PRAGateway(adapter, mode="G10").generate(request)

    assert adapter.requests[0].tools == (tool,)


def test_wire_envelope_never_contains_credentials() -> None:
    resource = context_record_to_wire_resource(_record())
    value = json.dumps(resource.to_dict())
    assert "api_key" not in value
    assert "password" not in value
