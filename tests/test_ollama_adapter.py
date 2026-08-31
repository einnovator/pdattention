from __future__ import annotations

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_ollama import OllamaEngineAdapter


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
    integration_level = "E2"

    def generate(self, request):
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
    assert result.text == "answer"
    assert result.trace[0]["prompt_eval_ns"] == 5


def test_only_explicit_backend_delegation_upgrades_to_e2() -> None:
    adapter = FakeOllama(backend_executor=NativeBackend())
    assert adapter.capabilities().integration_level.value == "E2"
    assert adapter.generate(wire_request()).text == "native"
