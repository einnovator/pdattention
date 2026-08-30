from __future__ import annotations

import pytest

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_hf.engine_profiles import EngineType
from pra_hf.runtime_providers import OpenVINORuntimeProvider, RuntimeConfig
from pra_openvino import (
    InMemoryOpenVINOStore,
    OpenVINOEngineAdapter,
    OpenVINOKVHandle,
    OpenVINONativeAttachmentManager,
    OpenVINOTopology,
)


def _request(*, tenant: str = "tenant-a") -> PRAWireRequest:
    return PRAWireRequest(
        model="org/model",
        tenant_id=tenant,
        messages=({"role": "user", "content": "question"},),
        resources=(
            PRAWireResource(
                resource_id="r1",
                uri="memory://r1",
                source_fingerprint="source-v1",
            ),
        ),
    )


def _handle(topology: OpenVINOTopology) -> OpenVINOKVHandle:
    return OpenVINOKVHandle(
        logical_key="r1",
        tensor_ids=("layer-0-key", "layer-0-value"),
        token_count=32,
        byte_count=4096,
        topology_fingerprint=topology.fingerprint,
    )


def test_http_adapter_and_provider_remain_e0() -> None:
    adapter = OpenVINOEngineAdapter("http://localhost:9000")
    provider = OpenVINORuntimeProvider()
    capabilities = provider.capabilities(RuntimeConfig(engine="openvino"))

    assert adapter.capabilities().engine_type is EngineType.OPENVINO
    assert adapter.capabilities().integration_level.value == "E0"
    assert not adapter.capabilities().native_kv
    assert capabilities.integration_level.value == "E0"
    assert not capabilities.native_kv


def test_native_handles_are_authorized_attached_once_and_cleaned() -> None:
    topology = OpenVINOTopology("org/model", device="GPU")
    store = InMemoryOpenVINOStore()
    manager = OpenVINONativeAttachmentManager(store, topology)
    manager.register(
        _handle(topology), tenant_id="tenant-a", authorization_scope="docs:read"
    )

    manager.open_request(
        "request-a",
        ("r1",),
        tenant_id="tenant-a",
        authorization_scopes=("docs:read",),
    )
    assert store.pin_count("r1") == 1
    assert manager.attach_once("request-a")[0].token_count == 32
    with pytest.raises(RuntimeError, match="already attached"):
        manager.attach_once("request-a")
    manager.close_request("request-a")

    assert store.pin_count("r1") == 0
    assert manager.metrics()["active_requests"] == 0


def test_native_handles_reject_topology_tenant_and_prefix_pool_mismatch() -> None:
    topology = OpenVINOTopology("org/model", revision="a")
    other = OpenVINOTopology("org/model", revision="b")
    manager = OpenVINONativeAttachmentManager(InMemoryOpenVINOStore(), topology)

    with pytest.raises(ValueError, match="topology"):
        manager.register(_handle(other), tenant_id="tenant-a")
    manager.register(_handle(topology), tenant_id="tenant-a")
    with pytest.raises(PermissionError, match="another tenant"):
        manager.open_request("request-b", ("r1",), tenant_id="tenant-b")
    with pytest.raises(RuntimeError, match="ordinary sequential or prefix"):
        manager.assert_prefix_pool_safe(("r1",))


class _NativeExecutor:
    def generate(self, request, block_store):
        return PRAEngineResult(text="native", raw={"finish_reason": "stop"})

    def stream(self, request, block_store):
        yield {"text": "native"}

    def close_session(self, session_id):
        return None


def test_adapter_claims_e2_only_with_explicit_native_executor() -> None:
    adapter = OpenVINOEngineAdapter(
        "http://localhost:9000", native_executor=_NativeExecutor()
    )
    assert adapter.capabilities().integration_level.value == "E2"
    assert adapter.capabilities().native_kv
    assert adapter.generate(_request()).text == "native"
