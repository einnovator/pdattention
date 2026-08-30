"""Capability-honest TensorRT-LLM HTTP and native-executor adapter."""

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


class TensorRTLLMNativeExecutor(Protocol):
    """Installed bridge that attaches selected PRA blocks to TRT-LLM attention."""

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult: ...

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]: ...

    def close_session(self, session_id: str) -> None: ...


class TensorRTLLMEngineAdapter(OpenAICompatibleEngineAdapter):
    """Use ``trtllm-serve`` as E0 or an explicit native bridge as E2.

    TensorRT-LLM's paged cache and prefix reuse do not by themselves implement
    non-prefix PRA memory.  Consequently, the adapter advertises native K/V
    only when the caller supplies a bridge that performs request-local block
    attachment and cleanup.
    """

    def __init__(
        self,
        base_url: str,
        *,
        cache_salt_secret: str | None = None,
        block_store: LogicalPRABlockStore | None = None,
        native_executor: TensorRTLLMNativeExecutor | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            base_url,
            timeout_seconds=timeout_seconds,
            name="tensorrt_llm",
            engine_type=EngineType.TENSORRT_LLM,
            pra_level="E2" if native_executor is not None else "E0",
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            session_state=native_executor is not None,
            incremental_messages=False,
            resource_delta=native_executor is not None,
            cache_affinity=native_executor is not None,
        )
        self.cache_salt_secret = cache_salt_secret
        self.block_store = block_store or LogicalPRABlockStore()
        self.native_executor = native_executor

    def capabilities(self) -> PRAEngineCapabilities:
        """Report only mechanisms implemented by this adapter instance."""

        if self.native_executor is None:
            return PRAEngineCapabilities(
                adapter="tensorrt_llm_http",
                engine_type=EngineType.TENSORRT_LLM,
                integration_level="E0",
                prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
                automatic_prefix_cache=True,
                text_fallback=True,
                streaming=True,
                tenant_isolation=self.cache_salt_secret is not None,
            )
        return PRAEngineCapabilities(
            adapter="tensorrt_llm_pra",
            engine_type=EngineType.TENSORRT_LLM,
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
            pinned_kv=True,
            gpu_kv=True,
            selected_interval_materialization=True,
            request_lifetime=True,
            streaming=True,
            host_device_residency=True,
            scheduler_hints=True,
            tenant_isolation=True,
        )

    def _payload(self, request: PRAWireRequest) -> dict[str, object]:
        """Salt ordinary prefix identity by tenant and selected PRA identity."""

        payload = super()._payload(request)
        if self.cache_salt_secret:
            identities = ",".join(
                f"{resource.resource_id}:{resource.version}:{resource.source_fingerprint}"
                for resource in request.resources
            )
            material = (
                f"{self.cache_salt_secret}:{request.tenant_id}:{identities}"
            ).encode("utf-8")
            payload["cache_salt"] = hashlib.sha256(material).hexdigest()
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
