"""MLX-LM facade plus optional in-process native PRA execution hook."""

from __future__ import annotations

from typing import Iterator, Mapping, Protocol

from pra_hf.deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
)
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.engine_profiles import EngineType, PrefixCacheMode


class MLXNativeExecutor(Protocol):
    """In-process MLX hook that consumes selected K/V arrays."""

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult: ...

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]: ...

    def close_session(self, session_id: str) -> None: ...


class MLXEngineAdapter(OpenAICompatibleEngineAdapter):
    """Treat MLX prompt reuse and PRA non-prefix memory as separate caches."""

    def __init__(
        self,
        base_url: str,
        *,
        block_store: LogicalPRABlockStore | None = None,
        native_executor: MLXNativeExecutor | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            base_url,
            timeout_seconds=timeout_seconds,
            name="mlx_lm",
            engine_type=EngineType.MLX,
            pra_level="E2" if native_executor is not None else "E0",
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            session_state=native_executor is not None,
            resource_delta=native_executor is not None,
            cache_affinity=native_executor is not None,
        )
        self.block_store = block_store or LogicalPRABlockStore()
        self.native_executor = native_executor

    def capabilities(self) -> PRAEngineCapabilities:
        if self.native_executor is None:
            return PRAEngineCapabilities(
                adapter="mlx_lm_http",
                engine_type=EngineType.MLX,
                integration_level="E0",
                prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
                automatic_prefix_cache=True,
                text_fallback=True,
            )
        return PRAEngineCapabilities(
            adapter="mlx_lm_native",
            engine_type=EngineType.MLX,
            integration_level="E2",
            prefix_cache_mode=PrefixCacheMode.EXPLICIT_PREFIX_HANDLE,
            explicit_prefix_cache=True,
            session_state=True,
            resource_delta=True,
            cache_affinity=True,
            prefix_cache_handle=True,
            logical_refs=True,
            typed_records=True,
            text_fallback=True,
            native_kv=True,
            external_kv_residency=True,
            cpu_kv=True,
            gpu_kv=True,
            selected_interval_materialization=True,
            request_lifetime=True,
            streaming=True,
            host_device_residency=False,
            tenant_isolation=True,
        )

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
