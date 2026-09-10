"""Capability-honest llama.cpp facade and native-extension boundary."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Mapping, Protocol, Sequence

from pra_hf.deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
)
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.engine_profiles import EngineType, PrefixCacheMode

from .resource_slots import (
    LlamaCppResourceIdentity,
    LlamaCppResourceSlotAllocator,
    LlamaCppSlotLease,
)


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


@dataclass(frozen=True)
class LlamaCppLivePrefixRange:
    """One stable record interval in an already evaluated source sequence."""

    record_id: str
    parent_record_id: str
    causal_group_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.record_id or not self.parent_record_id or not self.causal_group_id:
            raise ValueError("Live-prefix ranges require record, parent, and causal IDs.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Live-prefix ranges must be non-empty half-open intervals.")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "parent_record_id": self.parent_record_id,
            "causal_group_id": self.causal_group_id,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class LlamaCppLivePrefixPlan:
    """Validated zero-reencode selection from one live llama.cpp prefix."""

    source_slot: int
    source_tokens: int
    ranges: tuple[LlamaCppLivePrefixRange, ...]
    commit_to_source: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranges", tuple(self.ranges))
        if self.source_slot < 0 or self.source_tokens <= 0 or not self.ranges:
            raise ValueError("A live-prefix plan requires a source slot, tokens, and ranges.")
        cursor = -1
        for span in sorted(self.ranges, key=lambda row: (row.start, row.end)):
            if span.end > self.source_tokens:
                raise ValueError(f"Live-prefix range {span.record_id!r} exceeds its source.")
            if span.start < cursor:
                raise ValueError("Live-prefix ranges must not overlap.")
            cursor = span.end
        # llama.cpp currently requires the next decoded position to follow the
        # maximum resident K/V position. Agent policies already mandate the
        # active/recent tail; fail here instead of surfacing a decode-time 500.
        if max(span.end for span in self.ranges) != self.source_tokens:
            raise ValueError("Live-prefix selection must retain the source tail.")

    @property
    def selected_tokens(self) -> int:
        return sum(span.end - span.start for span in self.ranges)

    @property
    def full_retention(self) -> bool:
        cursor = 0
        for span in sorted(self.ranges, key=lambda row: (row.start, row.end)):
            if span.start != cursor:
                return False
            cursor = span.end
        return cursor == self.source_tokens


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


class LlamaCppNativeServerExecutor:
    """Invoke the PRA-aware llama-server sequence-attachment protocol.

    One idle slot owns an encoded resource sequence and a second slot owns the
    request.  The server adds request-sequence membership to the resource's
    unified-cache cells with ``llama_memory_seq_cp``; it does not copy K/V.
    A lock protects this explicit slot pair until a larger scheduler allocates
    pairs per concurrent request.
    """

    protocol = "pra.llama.cpp/v1"

    def __init__(
        self,
        base_url: str,
        *,
        resource_slot: int = 0,
        request_slot: int = 1,
        resource_slots: tuple[int, ...] | None = None,
        request_slots: tuple[int, ...] | None = None,
        model_fingerprint: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if resource_slot == request_slot:
            raise ValueError("PRA resource and request slots must differ.")
        self.base_url = base_url.rstrip("/")
        self.resource_slot = int(resource_slot)
        self.request_slot = int(request_slot)
        self.model_fingerprint = model_fingerprint
        self.timeout_seconds = float(timeout_seconds)
        self.slot_allocator = LlamaCppResourceSlotAllocator(
            resource_slots=resource_slots or (self.resource_slot,),
            request_slots=request_slots or (self.request_slot,),
        )
        self._slot_identities: dict[int, LlamaCppResourceIdentity] = {}
        self._lock = threading.RLock()
        self._capabilities = self._negotiate()

    def _request_json(
        self, path: str, payload: Mapping[str, object] | None = None
    ) -> Mapping[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _negotiate(self) -> Mapping[str, object]:
        capabilities = self._request_json("/pra/capabilities")
        if capabilities.get("protocol") != self.protocol:
            raise RuntimeError("llama-server does not expose the supported PRA protocol.")
        if not capabilities.get("native_sequence_attach"):
            raise RuntimeError("llama-server was not started with --kv-unified.")
        return capabilities

    def _delete_resource(self, slot: int | None = None) -> Mapping[str, object]:
        resource_slot = self.resource_slot if slot is None else int(slot)
        request = urllib.request.Request(
            f"{self.base_url}/pra/resources/{resource_slot}",
            method="DELETE",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _resource_text(request: PRAWireRequest) -> str:
        parts = [resource.text for resource in request.resources if resource.text]
        if not parts:
            raise ValueError("Native llama.cpp execution requires selected resource text.")
        return "\n".join(parts)

    def _query_text(self, request: PRAWireRequest) -> str:
        explicit = request.engine_hints.get("prompt")
        if explicit is not None:
            return str(explicit)
        rendered = self._request_json(
            "/apply-template",
            {
                "messages": list(request.messages),
                "tools": list(request.tools),
                "add_generation_prompt": True,
            },
        )
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("llama-server did not return a rendered chat prompt.")
        return prompt

    def _erase_request_slot(self, slot: int) -> None:
        self._request_json(f"/slots/{int(slot)}?action=erase", {})

    def _resource_identity(
        self, request: PRAWireRequest, digest: str
    ) -> LlamaCppResourceIdentity:
        return LlamaCppResourceIdentity(
            tenant_id=str(request.tenant_id),
            session_id=str(request.session_id),
            model_fingerprint=str(self.model_fingerprint or request.model),
            resource_digest=digest,
        )

    @staticmethod
    def _resource_digest(request: PRAWireRequest, text: str) -> str:
        identity = {
            "resources": [
                {
                    "resource_id": resource.resource_id,
                    "uri": resource.uri,
                    "version": resource.metadata.get("version"),
                    "text_sha256": hashlib.sha256(
                        (resource.text or "").encode("utf-8")
                    ).hexdigest(),
                }
                for resource in request.resources
            ],
            "composite_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _ensure_resource(
        self, request: PRAWireRequest, lease: LlamaCppSlotLease
    ) -> tuple[str, bool]:
        text = self._resource_text(request)
        digest = self._resource_digest(request, text)
        if lease.identity.resource_digest != digest:
            raise RuntimeError("Allocated resource identity does not match request content.")
        current = self._slot_identities.get(lease.resource_slot)
        if current == lease.identity:
            return digest, False
        if current is not None:
            self._delete_resource(lease.resource_slot)
        self._request_json(
            "/completion",
            {
                "prompt": text,
                "id_slot": lease.resource_slot,
                "n_predict": 0,
                "cache_prompt": True,
                "temperature": 0,
                "pra_pin_resource": True,
            },
        )
        self._slot_identities[lease.resource_slot] = lease.identity
        return digest, True

    def generate_live_prefix(
        self,
        request: PRAWireRequest,
        *,
        prompt_suffix: str | Sequence[int],
        plan: LlamaCppLivePrefixPlan,
        request_slot: int | None = None,
    ) -> PRAEngineResult:
        """Consume exact ranges from a live prefix without selected-text prefill."""

        if not bool(self._capabilities.get("live_prefix_kv_subset", False)):
            raise RuntimeError("llama-server does not expose live-prefix K/V selection.")
        slot = self.request_slot if request_slot is None else int(request_slot)
        if slot == plan.source_slot:
            raise ValueError("PRA source and request slots must differ.")
        raw = dict(
            self._request_json(
                "/completion",
                {
                    "prompt": prompt_suffix,
                    "id_slot": slot,
                    "pra_source_slot": plan.source_slot,
                    "pra_source_prefix_tokens": plan.source_tokens,
                    "pra_selected_ranges": [span.to_dict() for span in plan.ranges],
                    "pra_commit_to_source": plan.commit_to_source,
                    "n_predict": request.resolved_max_new_tokens,
                    "cache_prompt": True,
                    "temperature": float(
                        getattr(request, "openai_fields", {}).get(
                            "temperature", request.engine_hints.get("temperature", 0)
                        )
                    ),
                    "seed": int(
                        getattr(request, "openai_fields", {}).get(
                            "seed", request.engine_hints.get("seed", 0)
                        )
                    ),
                    "return_tokens": True,
                },
            )
        )
        pra = raw.get("pra") if isinstance(raw.get("pra"), Mapping) else {}
        if pra.get("kv_source") != "live_prefix_capture":
            raise RuntimeError("llama-server did not prove live-prefix K/V provenance.")
        if int(pra.get("selected_text_reencoded_tokens", -1)) != 0:
            raise RuntimeError("llama-server re-encoded selected live-prefix text.")
        if int(pra.get("selected_kv_tokens", -1)) != plan.selected_tokens:
            raise RuntimeError("llama-server selected-token telemetry does not match the plan.")
        if plan.commit_to_source and not bool(pra.get("commit_succeeded", False)):
            raise RuntimeError("llama-server did not commit newly evaluated live state.")
        return PRAEngineResult(
            str(raw.get("content", "")),
            raw,
            (
                {
                    "stage": "llama_cpp_live_prefix_subset",
                    "source_slot": plan.source_slot,
                    "request_slot": slot,
                    "selected_records": len(plan.ranges),
                    "source_tokens": plan.source_tokens,
                    "selected_kv_tokens": plan.selected_tokens,
                    "selected_text_reencoded_tokens": 0,
                    "full_retention": plan.full_retention,
                    "physical_kv_copy": False,
                },
            ),
        )

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult:
        del block_store
        with self._lock:
            text = self._resource_text(request)
            digest = self._resource_digest(request, text)
            identity = self._resource_identity(request, digest)
            lease = self.slot_allocator.acquire(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                identity=identity,
            )
            try:
                digest, encoded = self._ensure_resource(request, lease)
                prefix_caching = bool(
                    getattr(request, "openai_fields", {}).get(
                        "prefix_caching",
                        request.engine_hints.get("prefix_caching", True),
                    )
                )
                if not prefix_caching:
                    self._erase_request_slot(lease.request_slot)
                raw = self._request_json(
                    "/completion",
                    {
                        "prompt": self._query_text(request),
                        "id_slot": lease.request_slot,
                        "pra_resource_slot": lease.resource_slot,
                        "n_predict": request.resolved_max_new_tokens,
                        "cache_prompt": prefix_caching,
                        "temperature": float(
                            getattr(request, "openai_fields", {}).get(
                                "temperature", request.engine_hints.get("temperature", 0)
                            )
                        ),
                        "seed": int(
                            getattr(request, "openai_fields", {}).get(
                                "seed", request.engine_hints.get("seed", 0)
                            )
                        ),
                        "return_tokens": True,
                    },
                )
            finally:
                self.slot_allocator.release(request.request_id)
        raw = dict(raw)
        pra = dict(raw.get("pra", {}))
        engine_cached_tokens = (raw.get("timings") or {}).get("cache_n")
        native_tokens = pra.get("native_tokens")
        cached_tokens = (
            max(0, int(engine_cached_tokens) - int(native_tokens or 0))
            if engine_cached_tokens is not None else None
        )
        raw["prefix_cache_enabled"] = prefix_caching
        raw["prefix_cached_tokens"] = cached_tokens
        raw["engine_cached_tokens_total"] = engine_cached_tokens
        raw["prefix_cache_hit"] = bool(prefix_caching and (cached_tokens or 0) > 0)
        raw["native_attached_resources"] = [
            resource.resource_id for resource in request.resources
        ]
        trace = (
            {
                "stage": "llama_cpp_native_resource",
                "resource_digest": digest,
                "encoded": encoded,
                "resource_slot": lease.resource_slot,
                "tenant_scoped": True,
                "session_scoped": True,
            },
            {
                "stage": "llama_cpp_native_attach",
                "request_slot": lease.request_slot,
                "wire_tokens": pra.get("wire_tokens"),
                "native_tokens": pra.get("native_tokens"),
                "physical_kv_copy": pra.get("physical_kv_copy"),
                "prefix_cache_enabled": prefix_caching,
                "prefix_cached_tokens": cached_tokens,
                "engine_cached_tokens_total": engine_cached_tokens,
                "prefix_cache_hit": raw["prefix_cache_hit"],
            },
        )
        return PRAEngineResult(str(raw.get("content", "")), raw, trace)

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]:
        result = self.generate(request, block_store)
        yield {"text": result.text, "done": True, "raw": result.raw}

    def close_session(self, session_id: str) -> None:
        with self._lock:
            for slot in self.slot_allocator.invalidate_session(session_id):
                self._delete_resource(slot)
                self._slot_identities.pop(slot, None)


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
            negotiated = getattr(self.native_executor, "_capabilities", {})
            live_prefix = bool(negotiated.get("live_prefix_kv_subset", False))
            stable_records = bool(
                negotiated.get("stable_record_kv_identity", False)
            )
            agent_qualified = bool(
                negotiated.get("agent_history_kv_qualified", False)
            )
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
                live_prefix_kv_capture=live_prefix,
                live_prefix_kv_subset=live_prefix,
                zero_selected_text_reencoding=live_prefix,
                stable_record_kv_identity=stable_records,
                multiple_selected_records=live_prefix,
                source_positions_preserved=live_prefix,
                request_membership_attach=live_prefix,
                agent_history_kv_qualified=agent_qualified,
                external_kv_residency=True,
                cpu_kv=True,
                gpu_kv=True,
                selected_interval_materialization=True,
                request_lifetime=True,
                streaming=False,
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
