"""Capability-honest vLLM adapter for facade and native-hook deployments."""

from __future__ import annotations

import hashlib
from typing import Iterator, Mapping, Protocol

from pra_hf.deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
)
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.engine_profiles import EngineType, PrefixCacheMode


class VLLMNativeExecutor(Protocol):
    """Installed vLLM hook that really consumes selected native K/V blocks."""

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult: ...

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]: ...

    def close_session(self, session_id: str) -> None: ...


class VLLMEngineAdapter(OpenAICompatibleEngineAdapter):
    """Use vLLM's OpenAI facade or an explicitly installed native executor.

    The HTTP-only form remains E0 even though vLLM internally uses paged K/V.
    Passing ``native_executor`` promotes the adapter to E2 because that object
    must implement selected logical-block materialization and request lifetime.
    """

    def __init__(
        self,
        base_url: str,
        *,
        tenant_cache_salt_secret: str | None = None,
        block_store: LogicalPRABlockStore | None = None,
        native_executor: VLLMNativeExecutor | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            base_url,
            timeout_seconds=timeout_seconds,
            name="vllm_v1",
            engine_type=EngineType.VLLM,
            pra_level="E2" if native_executor is not None else "E0",
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            session_state=native_executor is not None,
            incremental_messages=False,
            resource_delta=native_executor is not None,
            cache_affinity=native_executor is not None,
        )
        self.tenant_cache_salt_secret = tenant_cache_salt_secret
        self.block_store = block_store or LogicalPRABlockStore()
        self.native_executor = native_executor

    def capabilities(self) -> PRAEngineCapabilities:
        if self.native_executor is None:
            return PRAEngineCapabilities(
                adapter="vllm_v1_http",
                engine_type=EngineType.VLLM,
                integration_level="E0",
                prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
                automatic_prefix_cache=True,
                text_fallback=True,
                tenant_isolation=self.tenant_cache_salt_secret is not None,
            )
        return PRAEngineCapabilities(
            adapter="vllm_v1_pra",
            engine_type=EngineType.VLLM,
            integration_level="E2",
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            automatic_prefix_cache=True,
            session_state=True,
            resource_delta=True,
            cache_affinity=True,
            logical_refs=True,
            typed_records=True,
            text_fallback=True,
            native_kv=True,
            external_kv_residency=True,
            cpu_kv=True,
            gpu_kv=True,
            selected_interval_materialization=True,
            request_lifetime=True,
            host_device_residency=True,
            scheduler_hints=True,
            tenant_isolation=True,
        )

    def _payload(self, request: PRAWireRequest) -> dict[str, object]:
        payload = super()._payload(request)
        if self.tenant_cache_salt_secret:
            salt = hashlib.sha256(
                f"{self.tenant_cache_salt_secret}:{request.tenant_id}".encode("utf-8")
            ).hexdigest()
            payload["cache_salt"] = salt
        return payload

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        if self.native_executor is None:
            return super().generate(request)
        return self.native_executor.generate(request, self.block_store)

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, object]]:
        if self.native_executor is None:
            return super().stream(request)
        return self.native_executor.stream(request, self.block_store)

    def close_session(self, session_id: str) -> None:
        if self.native_executor is not None:
            self.native_executor.close_session(session_id)

