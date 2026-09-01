from __future__ import annotations

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_ollama import (
    OllamaBackendHandshake,
    OllamaEngineAdapter,
    OllamaLlamaCppBackendExecutor,
)

from experiments.paper6_8_ollama.run_backend_handshake import run as run_handshake


class FakeOllama(OllamaEngineAdapter):
    def __init__(self, **kwargs):
        super().__init__("http://ollama.test", **kwargs)
        self.calls = []

    def _request_json(self, path, payload=None):
        self.calls.append((path, payload))
        if path == "/api/show":
            return {"modified_at": "now", "details": {"family": "qwen"}}
        if path == "/api/chat":
            return {
                "message": {"content": "answer"},
                "load_duration": 3,
                "prompt_eval_duration": 5,
                "eval_duration": 7,
            }
        if path == "/api/version":
            return {"version": "0.6.8"}
        if path == "/api/tags":
            return {"models": [{"name": "qwen3:0.6b"}]}
        if path == "/api/ps":
            return {"models": []}
        return {"done": True}


class NativeBackend:
    def negotiate(self, *, model, model_fingerprint):
        self.negotiated_model = model
        return OllamaBackendHandshake(
            protocol_version="pra-engine/1",
            backend="llama.cpp",
            backend_revision="458681e1d5d",
            model_fingerprint=model_fingerprint,
            model_artifact_digest="sha256:controlled-model",
            integration_level="E2",
            mechanisms=(
                "native_kv",
                "unified_kv_sequence_attach",
                "metadata_only_attach",
                "request_sequence_cleanup",
            ),
            resource_identity=True,
            tenant_isolation=True,
            request_cleanup=True,
        )

    def generate(self, request, handshake):
        self.handshake = handshake
        return PRAEngineResult("native")

    def invalidate_model(self, fingerprint):
        self.invalidated = fingerprint


def wire_request() -> PRAWireRequest:
    return PRAWireRequest(
        model="qwen3:0.6b",
        messages=({"role": "user", "content": "Question?"},),
        resources=(
            PRAWireResource(
                resource_id="doc-1", uri="memory://doc", text="Selected evidence."
            ),
        ),
        max_new_tokens=12,
    )


def test_endpoint_probe_reports_current_daemon_state() -> None:
    info = FakeOllama().inspect_endpoint()
    assert info.version == "0.6.8"
    assert info.installed_models == ("qwen3:0.6b",)


def test_e0_chat_injects_selected_resources_and_preserves_ollama_metrics() -> None:
    adapter = FakeOllama()
    result = adapter.generate(wire_request())
    payload = next(payload for path, payload in adapter.calls if path == "/api/chat")
    assert adapter.capabilities().integration_level.value == "E0"
    assert "Selected evidence" in payload["messages"][0]["content"]
    assert payload["options"]["num_predict"] == 12
    assert payload["think"] is False
    assert result.text == "answer"
    assert result.trace[0]["prompt_eval_ns"] == 5


def test_only_explicit_backend_delegation_upgrades_to_e2() -> None:
    backend = NativeBackend()
    adapter = FakeOllama(backend_executor=backend)
    assert adapter.capabilities().integration_level.value == "E0"
    assert adapter.generate(wire_request()).text == "native"
    assert adapter.capabilities().integration_level.value == "E2"
    assert backend.negotiated_model == "qwen3:0.6b"
    assert backend.handshake.backend_revision == "458681e1d5d"


class UnverifiedBackend(NativeBackend):
    def negotiate(self, *, model, model_fingerprint):
        receipt = super().negotiate(model=model, model_fingerprint=model_fingerprint)
        return OllamaBackendHandshake(
            **{
                **receipt.__dict__,
                "mechanisms": ("native_kv",),
            }
        )


def test_unverified_native_claim_falls_back_to_selected_text() -> None:
    adapter = FakeOllama(backend_executor=UnverifiedBackend())
    result = adapter.generate(wire_request())
    payload = next(payload for path, payload in adapter.calls if path == "/api/chat")
    assert result.text == "answer"
    assert "Selected evidence" in payload["messages"][0]["content"]
    assert adapter.capabilities().integration_level.value == "E0"
    assert result.trace[0]["native_handshake"] == "rejected"


def test_model_change_invalidates_and_renegotiates_backend_receipt() -> None:
    backend = NativeBackend()
    adapter = FakeOllama(backend_executor=backend)
    adapter.generate(wire_request())
    first = adapter._model_fingerprint
    adapter.calls.clear()

    request = wire_request()
    changed = PRAWireRequest(
        model="qwen3:1.7b",
        messages=request.messages,
        resources=request.resources,
        max_new_tokens=request.max_new_tokens,
    )
    adapter.generate(changed)
    assert backend.invalidated == first
    assert backend.negotiated_model == "qwen3:1.7b"
    assert adapter._model_fingerprint != first


def test_backend_handshake_artifact_closes_fallback_and_lifecycle_controls() -> None:
    payload = run_handshake()
    assert payload["stock_ollama_native_endpoint"] is False
    assert payload["summary"] == {
        "valid_receipts_upgrade": 1,
        "invalid_receipts_fallback": 2,
        "lifecycle_invalidations": 2,
    }
    assert payload["backend_receipt"]["schedule_matched_exact_logits"] == 10
    assert payload["backend_receipt"]["physical_kv_copy"] is False


def test_concrete_llamacpp_backend_invokes_negotiated_native_executor(monkeypatch) -> None:
    class Native:
        protocol = "pra.llama.cpp/v1"
        _capabilities = {
            "protocol": protocol,
            "native_sequence_attach": True,
        }

        def generate(self, request, block_store):
            self.generated = request.request_id
            return PRAEngineResult("native-live", {"pra": {"native_tokens": 8}})

        def close_session(self, session_id):
            self.closed = session_id

    native = Native()
    monkeypatch.setattr(
        "pra_ollama.adapter.LlamaCppNativeServerExecutor", lambda *args, **kwargs: native
    )
    backend = OllamaLlamaCppBackendExecutor(
        "http://native.test",
        backend_revision="458681e1d5d",
        model_artifact_digest="sha256:exact-model",
    )
    adapter = FakeOllama(backend_executor=backend)
    request = wire_request()
    result = adapter.generate(request)

    assert result.text == "native-live"
    assert adapter.capabilities().integration_level.value == "E2"
    assert native.generated == request.request_id
    adapter.close_session("session-a")
    assert native.closed == "session-a"
