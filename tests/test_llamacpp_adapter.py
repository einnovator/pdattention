from __future__ import annotations

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_llamacpp import (
    LlamaCppEngineAdapter,
    LlamaCppLivePrefixPlan,
    LlamaCppLivePrefixRange,
    LlamaCppNativeServerExecutor,
    LlamaCppRuntimeProvider,
)


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


def test_runtime_provider_enables_negotiated_native_server() -> None:
    from pra_hf.runtime_providers import RuntimeConfig

    provider = LlamaCppRuntimeProvider()
    config = RuntimeConfig(
        engine="llama_cpp",
        model="model.gguf",
        engine_options={"executable": "llama-server", "native_pra": True},
    )
    assert "--kv-unified" in provider.build_command(config)
    assert provider.capabilities(config).integration_level.value == "E2"
    assert provider.capabilities(config).native_kv
    assert provider.doctor(config).checks[-1]["status"] == "CONFIGURED"


class FakeNativeServerExecutor(LlamaCppNativeServerExecutor):
    def __init__(self):
        self.calls = []
        super().__init__("http://unused", resource_slot=2, request_slot=3)

    def _request_json(self, path, payload=None):
        self.calls.append((path, payload))
        if path == "/pra/capabilities":
            return {
                "protocol": "pra.llama.cpp/v1",
                "native_sequence_attach": True,
                "live_prefix_kv_subset": True,
            }
        if path == "/apply-template":
            return {"prompt": "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"}
        if payload and payload.get("pra_resource_slot") is not None:
            return {
                "content": "7319",
                "pra": {
                    "wire_tokens": 9,
                    "native_tokens": 16,
                    "physical_kv_copy": False,
                },
            }
        if payload and payload.get("pra_source_slot") is not None:
            selected = sum(
                row["end"] - row["start"]
                for row in payload["pra_selected_ranges"]
            )
            return {
                "content": "7319",
                "tokens": [1, 2],
                "pra": {
                    "kv_source": "live_prefix_capture",
                    "selected_kv_tokens": selected,
                    "reused_kv_tokens": selected,
                    "selected_text_reencoded_tokens": 0,
                    "commit_succeeded": payload["pra_commit_to_source"],
                    "physical_kv_copy": False,
                },
            }
        return {"content": ""}

    def _delete_resource(self, slot=None):
        resource_slot = self.resource_slot if slot is None else slot
        self.calls.append((f"/pra/resources/{resource_slot}", None))
        return {"success": True}


def test_native_server_executor_negotiates_encodes_once_and_attaches() -> None:
    executor = FakeNativeServerExecutor()
    adapter = LlamaCppEngineAdapter(
        "http://unused",
        model_fingerprint="model-sha",
        native_executor=executor,
    )
    first = adapter.generate(request("The code is 7319."))
    second = adapter.generate(request("The code is 7319."))

    assert first.text == second.text == "7319"
    assert [path for path, _ in executor.calls].count("/pra/capabilities") == 1
    assert len([call for call in executor.calls if call[1] and call[1].get("n_predict") == 0]) == 1
    attach = [call for call in executor.calls if call[1] and "pra_resource_slot" in call[1]]
    assert len(attach) == 2
    assert attach[0][1]["id_slot"] == 3
    assert attach[0][1]["pra_resource_slot"] == 2
    assert attach[0][1]["seed"] == 0
    assert first.trace[-1]["physical_kv_copy"] is False
    adapter.close_session("session-a")
    assert executor.calls[-1][0] == "/pra/resources/2"


def test_native_server_executor_rejects_non_native_server() -> None:
    class Unsupported(FakeNativeServerExecutor):
        def _request_json(self, path, payload=None):
            return {"protocol": "pra.llama.cpp/v1", "native_sequence_attach": False}

    import pytest

    with pytest.raises(RuntimeError, match="--kv-unified"):
        Unsupported()


def test_native_server_executor_never_reuses_resource_across_tenants() -> None:
    from dataclasses import replace

    executor = FakeNativeServerExecutor()
    adapter = LlamaCppEngineAdapter(
        "http://unused",
        model_fingerprint="model-sha",
        native_executor=executor,
    )
    tenant_a = request("The code is 7319.")
    tenant_b = replace(
        tenant_a,
        request_id="request-2",
        tenant_id="tenant-b",
        session_id="session-b",
    )
    adapter.generate(tenant_a)
    adapter.generate(tenant_b)

    encodes = [
        call for call in executor.calls if call[1] and call[1].get("n_predict") == 0
    ]
    assert len(encodes) == 2
    assert ("/pra/resources/2", None) in executor.calls
    assert executor.slot_allocator.snapshot()["tenant_scoped"] is True


def test_live_prefix_plan_requires_nonoverlap_and_the_resident_tail() -> None:
    import pytest

    with pytest.raises(ValueError, match="retain the source tail"):
        LlamaCppLivePrefixPlan(
            source_slot=0,
            source_tokens=16,
            ranges=(LlamaCppLivePrefixRange("r1", "p1", "g1", 0, 8),),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        LlamaCppLivePrefixPlan(
            source_slot=0,
            source_tokens=16,
            ranges=(
                LlamaCppLivePrefixRange("r1", "p1", "g1", 0, 10),
                LlamaCppLivePrefixRange("r2", "p2", "g2", 8, 16),
            ),
        )


def test_live_prefix_executor_sends_multiple_ranges_and_rejects_reencoding() -> None:
    executor = FakeNativeServerExecutor()
    plan = LlamaCppLivePrefixPlan(
        source_slot=0,
        source_tokens=16,
        ranges=(
            LlamaCppLivePrefixRange("task", "task", "task", 0, 4),
            LlamaCppLivePrefixRange("recent", "recent", "turn:2", 12, 16),
        ),
    )

    result = executor.generate_live_prefix(
        request(), prompt_suffix=[99, 100], plan=plan, request_slot=3
    )

    assert result.text == "7319"
    call = next(
        payload for _, payload in executor.calls
        if payload and payload.get("pra_source_slot") == 0
    )
    assert len(call["pra_selected_ranges"]) == 2
    assert call["prompt"] == [99, 100]
    assert result.trace[0]["selected_kv_tokens"] == 8
    assert result.trace[0]["selected_text_reencoded_tokens"] == 0
