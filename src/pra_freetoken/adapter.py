"""FreeToken transport and optional native scheduler integration boundary."""

from __future__ import annotations

from typing import Protocol

from pra_hf.deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
)
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.engine_profiles import EngineType, PrefixCacheMode


class FreeTokenNativeExecutor(Protocol):
    """Native semantic-memory hook independent of FreeToken expert routing."""

    @property
    def integration_level(self) -> str: ...

    def generate(
        self, request: PRAWireRequest, semantic_blocks: LogicalPRABlockStore
    ) -> PRAEngineResult: ...


class FreeTokenEngineAdapter(OpenAICompatibleEngineAdapter):
    """Use FreeToken's OpenAI API at E0 or an explicit native executor at E2/E3."""

    def __init__(
        self,
        base_url: str,
        *,
        native_executor: FreeTokenNativeExecutor | None = None,
        semantic_blocks: LogicalPRABlockStore | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            base_url,
            timeout_seconds=timeout_seconds,
            name="freetoken",
            engine_type=EngineType.FREETOKEN,
            pra_level="E0",
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            session_state=True,
            cache_affinity=True,
        )
        self.native_executor = native_executor
        self.semantic_blocks = semantic_blocks or LogicalPRABlockStore()

    def capabilities(self) -> PRAEngineCapabilities:
        if self.native_executor is None:
            return super().capabilities()
        level = self.native_executor.integration_level
        if level not in {"E2", "E3"}:
            raise ValueError("A FreeToken native executor must advertise E2 or E3.")
        return PRAEngineCapabilities(
            adapter="freetoken_pra",
            engine_type=EngineType.FREETOKEN,
            integration_level=level,
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            automatic_prefix_cache=True,
            session_state=True,
            cache_affinity=True,
            logical_refs=True,
            typed_records=True,
            text_fallback=True,
            native_kv=True,
            external_kv_residency=True,
            selected_interval_materialization=True,
            request_lifetime=True,
            pra_scheduler=level == "E3",
            scheduler_hints=level == "E3",
            host_device_residency=True,
            tenant_isolation=True,
        )

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        if self.native_executor is None:
            result = super().generate(request)
            return PRAEngineResult(
                result.text,
                result.raw,
                (*result.trace, {"stage": "freetoken_e0", "native_kv": False}),
            )
        return self.native_executor.generate(request, self.semantic_blocks)
