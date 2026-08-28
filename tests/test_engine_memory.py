from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pra_hf.deployment import PRAEngineResult, PRAWireRequest
from pra_hf.engine_memory import (
    LogicalPRABlock,
    LogicalPRABlockId,
    LogicalPRABlockStore,
    PRAResidencyState,
)
from pra_mlx import MLXEngineAdapter
from pra_sglang import SGLangEngineAdapter
from pra_vllm import VLLMEngineAdapter


def _identity(*, tenant: str = "tenant-a", scope: str | None = "team-a"):
    return LogicalPRABlockId(
        tenant_id=tenant,
        session_id="session-a",
        resource_id="resource-a",
        resource_version="v1",
        record_type="tool_result",
        token_start=4,
        token_end=20,
        layer=12,
        model_revision="model-revision-a",
        dtype="float16",
        layout="gqa:8x2x64",
        materialization_profile="last_12",
        position_policy="source_frame",
        security_scope=scope,
    )


def test_logical_block_lifecycle_separates_identity_from_physical_handles() -> None:
    store = LogicalPRABlockStore()
    key = store.register(LogicalPRABlock(_identity(), address_bytes=128, detail_bytes=4096))
    assert key == _identity().digest()
    assert store.snapshot().resident_detail_bytes == 0

    with pytest.raises(ValueError, match="physical engine handles"):
        store.transition(key, PRAResidencyState.RESIDENT)

    resident = store.transition(
        key,
        PRAResidencyState.RESIDENT,
        physical_handles=("vllm:block:7",),
        storage_tier="metal",
    )
    assert resident.identity.token_count == 16
    assert resident.physical_handles == ("vllm:block:7",)
    assert store.snapshot().resident_detail_bytes == 4096

    evictable = store.transition(key, PRAResidencyState.EVICTABLE)
    assert evictable.state == PRAResidencyState.EVICTABLE
    off_device = store.transition(key, PRAResidencyState.OFF_DEVICE, storage_tier="host")
    assert off_device.physical_handles == ()
    assert store.snapshot().resident_detail_bytes == 0


def test_logical_block_selection_enforces_tenant_and_scope_and_tracks_reuse() -> None:
    store = LogicalPRABlockStore()
    key = store.register(LogicalPRABlock(_identity(), address_bytes=32, detail_bytes=512))
    with pytest.raises(PermissionError, match="Cross-tenant"):
        store.select((key,), tenant_id="tenant-b", authorization_scopes=("team-a",))
    with pytest.raises(PermissionError, match="not authorized"):
        store.select((key,), tenant_id="tenant-a")

    first = store.select((key,), tenant_id="tenant-a", authorization_scopes=("team-a",))
    second = store.select((key,), tenant_id="tenant-a", authorization_scopes=("team-a",))
    assert first[0].selections == 1
    assert second[0].selections == 2
    assert second[0].reuses == 1


def test_resource_invalidation_clears_physical_realizations() -> None:
    store = LogicalPRABlockStore()
    key = store.register(
        LogicalPRABlock(
            _identity(scope=None),
            address_bytes=32,
            detail_bytes=512,
            state=PRAResidencyState.RESIDENT,
            physical_handles=("mlx:array:1",),
        )
    )
    assert store.invalidate_resource("tenant-a", "resource-a") == 1
    assert store.get(key).state == PRAResidencyState.INVALID
    assert store.get(key).physical_handles == ()


@pytest.mark.parametrize(
    ("factory", "engine_type", "adapter"),
    [
        (lambda: VLLMEngineAdapter("http://engine"), "vllm", "vllm_v1_http"),
        (lambda: SGLangEngineAdapter("http://engine"), "sglang", "sglang_http"),
        (lambda: MLXEngineAdapter("http://engine"), "mlx", "mlx_lm_http"),
    ],
)
def test_http_engine_adapters_do_not_claim_native_kv(factory, engine_type, adapter) -> None:
    capabilities = factory().capabilities()
    assert capabilities.engine_type.value == engine_type
    assert capabilities.adapter == adapter
    assert capabilities.integration_level.value == "E0"
    assert capabilities.automatic_prefix_cache
    assert not capabilities.logical_refs
    assert not capabilities.native_kv


def test_vllm_cache_salt_is_tenant_scoped_and_secret_derived() -> None:
    adapter = VLLMEngineAdapter(
        "http://engine", tenant_cache_salt_secret="deployment-secret"
    )
    request = PRAWireRequest(
        model="model",
        tenant_id="tenant-a",
        messages=({"role": "user", "content": "hello"},),
    )
    payload = adapter._payload(request)
    assert payload["cache_salt"]
    assert payload["cache_salt"] != "tenant-a"
    assert "deployment-secret" not in payload["cache_salt"]
    assert adapter.capabilities().tenant_isolation


class _NativeExecutor:
    def generate(self, request, block_store):
        return PRAEngineResult("native", {"logical_blocks": block_store.snapshot().logical_blocks})

    def stream(self, request, block_store):
        yield {"type": "done", "request_id": request.request_id}

    def close_session(self, session_id):
        self.closed = session_id


@pytest.mark.parametrize(
    "adapter",
    [
        VLLMEngineAdapter("http://engine", native_executor=_NativeExecutor()),
        SGLangEngineAdapter("http://engine", native_executor=_NativeExecutor()),
        MLXEngineAdapter("http://engine", native_executor=_NativeExecutor()),
    ],
)
def test_native_executor_is_required_before_advertising_e2(adapter) -> None:
    capabilities = adapter.capabilities()
    assert capabilities.integration_level.value == "E2"
    assert capabilities.logical_refs
    assert capabilities.native_kv
    result = adapter.generate(
        PRAWireRequest(model="model", messages=({"role": "user", "content": "hello"},))
    )
    assert result.text == "native"


def test_vllm_http_request_serializes_cache_salt() -> None:
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "ok"}}]}
    ).encode()
    response.__enter__.return_value = response
    adapter = VLLMEngineAdapter(
        "http://engine", tenant_cache_salt_secret="deployment-secret"
    )
    request = PRAWireRequest(
        model="model",
        tenant_id="tenant-a",
        messages=({"role": "user", "content": "hello"},),
    )
    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        assert adapter.generate(request).text == "ok"
    body = json.loads(urlopen.call_args.args[0].data)
    assert body["cache_salt"] == adapter._payload(request)["cache_salt"]

