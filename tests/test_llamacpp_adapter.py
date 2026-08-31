from __future__ import annotations

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_llamacpp import LlamaCppEngineAdapter, LlamaCppRuntimeProvider


class NativeExecutor:
    def generate(self, request, block_store):
        return PRAEngineResult("native", {"request_id": request.request_id})

    def stream(self, request, block_store):
        yield {"text": "native"}

    def close_session(self, session_id):
        self.closed = session_id


def request(text: str = "alpha") -> PRAWireRequest:
    return PRAWireRequest(
        request_id="request-1",
        tenant_id="tenant-a",
        session_id="session-a",
        model="model.gguf",
        messages=({"role": "user", "content": "question"},),
        resources=(
            PRAWireResource(
                resource_id="doc-1",
                uri="memory://doc",
                record_type="document",
                text=text,
            ),
        ),
    )


def test_upstream_http_and_slot_state_remain_e0() -> None:
    adapter = LlamaCppEngineAdapter(
        "http://localhost:8080", model_fingerprint="model-sha", slot_client=object()
    )
    capabilities = adapter.capabilities()
    assert capabilities.integration_level.value == "E0"
    assert capabilities.explicit_prefix_cache
    assert not capabilities.native_kv


def test_slot_identity_separates_model_tenant_session_and_resource() -> None:
    adapter = LlamaCppEngineAdapter(
        "http://localhost:8080", model_fingerprint="model-sha"
    )
    first = adapter.slot_state(request("alpha"), 2)
    second = adapter.slot_state(request("beta"), 2)
    assert first.filename != second.filename
    assert first.slot_id == second.slot_id == 2


def test_native_executor_is_the_only_e2_upgrade() -> None:
    adapter = LlamaCppEngineAdapter(
        "http://localhost:8080",
        model_fingerprint="model-sha",
        native_executor=NativeExecutor(),
    )
    assert adapter.capabilities().integration_level.value == "E2"
    assert adapter.capabilities().native_kv
    assert adapter.generate(request()).text == "native"


def test_runtime_provider_builds_real_llama_server_command() -> None:
    from pra_hf.runtime_providers import RuntimeConfig

    provider = LlamaCppRuntimeProvider()
    config = RuntimeConfig(
        engine="llama_cpp",
        model="model.gguf",
        engine_options={"executable": "llama-server", "n_gpu_layers": 99},
    )
    command = provider.build_command(config)
    assert command[:3] == ["llama-server", "--model", "model.gguf"]
    assert command[-2:] == ["--n-gpu-layers", "99"]
