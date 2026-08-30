from __future__ import annotations

import pytest

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_hf.engine_profiles import EngineType
from pra_hf.runtime_providers import RuntimeConfig, TensorRTLLMRuntimeProvider
from pra_tensorrt_llm import (
    InMemoryTensorRTConnector,
    TensorRTLLMBlockHandle,
    TensorRTLLMEngineAdapter,
    TensorRTLLMNativeAttachmentManager,
    TensorRTLLMTopology,
)


def _request(*, tenant: str = "tenant-a", resource_id: str = "r1") -> PRAWireRequest:
    return PRAWireRequest(
        model="org/model",
        tenant_id=tenant,
        messages=({"role": "user", "content": "question"},),
        resources=(
            PRAWireResource(
                resource_id=resource_id,
                uri=f"memory://{resource_id}",
                source_fingerprint="source-v1",
                metadata={"tenant_id": tenant},
            ),
        ),
    )


def _handle(topology: TensorRTLLMTopology, key: str = "r1") -> TensorRTLLMBlockHandle:
    return TensorRTLLMBlockHandle(
        logical_key=key,
        block_ids=(11, 12),
        token_count=48,
        byte_count=4096,
        topology_fingerprint=topology.fingerprint,
    )


def test_http_adapter_remains_e0_and_salts_resource_identity() -> None:
    adapter = TensorRTLLMEngineAdapter("http://localhost:8000", cache_salt_secret="s")
    first = _request(resource_id="first")
    second = _request(resource_id="second")

    assert adapter.capabilities().engine_type is EngineType.TENSORRT_LLM
    assert adapter.capabilities().integration_level.value == "E0"
    assert adapter.capabilities().native_kv is False
    assert adapter._payload(first)["cache_salt"] != adapter._payload(second)["cache_salt"]


def test_native_attachment_is_authorized_exactly_once_and_cleaned() -> None:
    topology = TensorRTLLMTopology("org/model", "rev")
    connector = InMemoryTensorRTConnector()
    manager = TensorRTLLMNativeAttachmentManager(connector, topology)
    manager.register(
        _handle(topology), tenant_id="tenant-a", authorization_scope="docs:read"
    )

    manager.open_request(
        "request-a",
        ("r1",),
        tenant_id="tenant-a",
        authorization_scopes=("docs:read",),
    )
    assert connector.pin_count("r1") == 1
    assert manager.attach_once("request-a")[0].block_ids == (11, 12)
    with pytest.raises(RuntimeError, match="already attached"):
        manager.attach_once("request-a")
    manager.close_request("request-a")

    assert connector.pin_count("r1") == 0
    assert manager.metrics()["active_requests"] == 0


def test_native_attachment_rejects_cross_tenant_and_prefix_pool_reuse() -> None:
    topology = TensorRTLLMTopology("org/model", "rev")
    manager = TensorRTLLMNativeAttachmentManager(InMemoryTensorRTConnector(), topology)
    manager.register(_handle(topology), tenant_id="tenant-a")

    with pytest.raises(PermissionError, match="another tenant"):
        manager.open_request("request-b", ("r1",), tenant_id="tenant-b")
    with pytest.raises(RuntimeError, match="ordinary sequential or prefix"):
        manager.assert_prefix_pool_safe(("r1",))


def test_native_attachment_rejects_topology_mismatch() -> None:
    runtime = TensorRTLLMTopology("org/model", "rev-a")
    other = TensorRTLLMTopology("org/model", "rev-b")
    manager = TensorRTLLMNativeAttachmentManager(InMemoryTensorRTConnector(), runtime)

    with pytest.raises(ValueError, match="topology"):
        manager.register(_handle(other), tenant_id="tenant-a")


def test_runtime_provider_uses_trtllm_serve_and_does_not_claim_e2() -> None:
    provider = TensorRTLLMRuntimeProvider()
    config = RuntimeConfig(
        engine="tensorrt_llm",
        model="org/model",
        host="0.0.0.0",
        port=9001,
        engine_options={"backend": "pytorch", "max_batch_size": 4},
    )

    command = provider.build_command(config)
    assert command[1:3] == ["serve", "org/model"]
    assert command[command.index("--backend") + 1] == "pytorch"
    assert command[command.index("--max-batch-size") + 1] == "4"
    assert provider.capabilities(config).integration_level.value == "E0"


class _NativeExecutor:
    def generate(self, request, block_store):
        return PRAEngineResult(text="native", raw={"finish_reason": "stop"})

    def stream(self, request, block_store):
        yield {"text": "native"}

    def close_session(self, session_id):
        return None


def test_adapter_only_claims_e2_with_explicit_native_executor() -> None:
    adapter = TensorRTLLMEngineAdapter(
        "http://localhost:8000", native_executor=_NativeExecutor()
    )
    assert adapter.capabilities().integration_level.value == "E2"
    assert adapter.capabilities().native_kv is True
    assert adapter.generate(_request()).text == "native"
