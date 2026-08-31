"""Capability-honest llama.cpp facade and native-extension boundary."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Mapping, Protocol

from pra_hf.deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
)
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.engine_profiles import EngineType, PrefixCacheMode


@dataclass(frozen=True)
class LlamaCppSlotState:
    """Identity of a sequential llama-server slot checkpoint.

    Slot state is prefix-shaped conversational state.  It is useful for E0/E1
    reuse, but it is not detached PRA memory and must never be reported as E2.
    """

    slot_id: int
    filename: str
    model_fingerprint: str
    resource_digest: str


class LlamaCppSlotClient:
    """Small client for llama-server's explicit slot save/restore API."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def _action(self, slot_id: int, action: str, filename: str | None = None) -> Mapping[str, object]:
        query = urllib.parse.urlencode({"action": action})
        payload = b"{}" if filename is None else json.dumps({"filename": filename}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/slots/{int(slot_id)}?{query}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def save(self, state: LlamaCppSlotState) -> Mapping[str, object]:
        return self._action(state.slot_id, "save", state.filename)

    def restore(self, state: LlamaCppSlotState) -> Mapping[str, object]:
        return self._action(state.slot_id, "restore", state.filename)

    def erase(self, slot_id: int) -> Mapping[str, object]:
        return self._action(slot_id, "erase")


class LlamaCppNativeExecutor(Protocol):
    """Optional patched llama.cpp executor that truly consumes detached K/V."""

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult: ...

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]: ...

    def close_session(self, session_id: str) -> None: ...


class LlamaCppEngineAdapter(OpenAICompatibleEngineAdapter):
    """Use llama-server at E0/E1 or an explicit detached-memory extension at E2.

    Upstream sequence save/restore preserves ordinary positional slot state.
    Supplying a slot client therefore enables identity-aware reuse but never
    upgrades the attention integration.  Only ``native_executor`` may claim E2.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_fingerprint: str,
        slot_client: LlamaCppSlotClient | None = None,
        native_executor: LlamaCppNativeExecutor | None = None,
        block_store: LogicalPRABlockStore | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            base_url,
            timeout_seconds=timeout_seconds,
            name="llama_cpp",
            engine_type=EngineType.LLAMA_CPP,
            pra_level="E2" if native_executor is not None else "E0",
            prefix_cache_mode=(
                PrefixCacheMode.EXPLICIT_PREFIX_HANDLE
                if slot_client is not None
                else PrefixCacheMode.AUTOMATIC_PREFIX_CACHE
            ),
            session_state=slot_client is not None or native_executor is not None,
            cache_affinity=slot_client is not None or native_executor is not None,
        )
        self.model_fingerprint = str(model_fingerprint)
        self.slot_client = slot_client
        self.native_executor = native_executor
        self.block_store = block_store or LogicalPRABlockStore()

    def capabilities(self) -> PRAEngineCapabilities:
        if self.native_executor is not None:
            return PRAEngineCapabilities(
                adapter="llama_cpp_pra",
                engine_type=EngineType.LLAMA_CPP,
                integration_level="E2",
                prefix_cache_mode=PrefixCacheMode.EXPLICIT_PREFIX_HANDLE,
                explicit_prefix_cache=True,
                session_state=True,
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
                host_device_residency=True,
                tenant_isolation=True,
            )
        explicit = self.slot_client is not None
        return PRAEngineCapabilities(
            adapter="llama_cpp_http",
            engine_type=EngineType.LLAMA_CPP,
            integration_level="E0",
            prefix_cache_mode=(
                PrefixCacheMode.EXPLICIT_PREFIX_HANDLE
                if explicit
                else PrefixCacheMode.AUTOMATIC_PREFIX_CACHE
            ),
            automatic_prefix_cache=not explicit,
            explicit_prefix_cache=explicit,
            prefix_cache_handle=explicit,
            session_state=explicit,
            cache_affinity=explicit,
            text_fallback=True,
            streaming=False,
        )

    def slot_state(self, request: PRAWireRequest, slot_id: int) -> LlamaCppSlotState:
        """Derive a tenant/model/resource-bound name for sequential state reuse."""

        resources = [
            {
                "uri": resource.uri,
                "version": resource.metadata.get("version"),
                "text_sha256": hashlib.sha256((resource.text or "").encode()).hexdigest(),
            }
            for resource in request.resources
        ]
        identity = {
            "model": self.model_fingerprint,
            "tenant": request.tenant_id,
            "session": request.session_id,
            "resources": resources,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()
        return LlamaCppSlotState(
            slot_id=int(slot_id),
            filename=f"pra-{digest}.bin",
            model_fingerprint=self.model_fingerprint,
            resource_digest=digest,
        )

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        if self.native_executor is None:
            result = super().generate(request)
            return PRAEngineResult(
                result.text,
                result.raw,
                (*result.trace, {"stage": "llama_cpp", "native_kv": False}),
            )
        return self.native_executor.generate(request, self.block_store)

    def stream(self, request: PRAWireRequest) -> Iterator[Mapping[str, object]]:
        if self.native_executor is None:
            return super().stream(request)
        return self.native_executor.stream(request, self.block_store)

    def close_session(self, session_id: str) -> None:
        if self.native_executor is not None:
            self.native_executor.close_session(session_id)
