"""Capability-negotiated Ollama adapter with a conservative E0 fallback."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult, PRAWireRequest
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.engine_profiles import EngineType, PrefixCacheMode
from pra_llamacpp import LlamaCppNativeServerExecutor


@dataclass(frozen=True)
class OllamaEndpointInfo:
    """Observed endpoint state used for diagnostics and model invalidation."""

    version: str
    installed_models: tuple[str, ...]
    running_models: tuple[str, ...]


@dataclass(frozen=True)
class OllamaBackendHandshake:
    """Versioned proof that the active model runner can consume native PRA K/V.

    The fingerprint binds engine capability to the exact model package observed
    through Ollama.  Mechanism flags distinguish the narrow llama.cpp unified-
    cache sequence-attachment seam from an unverified generic ``E2`` label.
    """

    protocol_version: str
    backend: str
    backend_revision: str
    model_fingerprint: str
    model_artifact_digest: str | None
    integration_level: str
    mechanisms: tuple[str, ...]
    resource_identity: bool
    tenant_isolation: bool
    request_cleanup: bool

    def validates(self, model_fingerprint: str) -> bool:
        """Return whether this receipt is sufficient for an E2/E3 request."""

        required = {
            "native_kv",
            "unified_kv_sequence_attach",
            "metadata_only_attach",
            "request_sequence_cleanup",
        }
        return bool(
            self.protocol_version == "pra-engine/1"
            and self.backend == "llama.cpp"
            and self.backend_revision
            and self.model_fingerprint == model_fingerprint
            and self.model_artifact_digest
            and self.integration_level in {"E2", "E3"}
            and required.issubset(self.mechanisms)
            and self.resource_identity
            and self.tenant_isolation
            and self.request_cleanup
        )


class OllamaBackendExecutor(Protocol):
    """PRA-aware execution supplied by Ollama's active model backend."""

    def negotiate(
        self, *, model: str, model_fingerprint: str
    ) -> OllamaBackendHandshake | None: ...

    def generate(
        self, request: PRAWireRequest, handshake: OllamaBackendHandshake
    ) -> PRAEngineResult: ...

    def invalidate_model(self, model_fingerprint: str) -> None: ...


class OllamaLlamaCppBackendExecutor:
    """Delegate Ollama AUTO requests to a negotiated PRA-aware llama-server.

    Ollama remains the model catalog and default E0 serving surface. The
    sidecar is selected only after it proves the versioned sequence-attachment
    protocol; the returned receipt is bound to Ollama's current model
    fingerprint so model switches cannot reuse stale native state.
    """

    mechanisms = (
        "native_kv",
        "unified_kv_sequence_attach",
        "metadata_only_attach",
        "request_sequence_cleanup",
    )

    def __init__(
        self,
        native_base_url: str,
        *,
        backend_revision: str,
        model_artifact_digest: str,
        resource_slot: int = 0,
        request_slot: int = 1,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.backend_revision = str(backend_revision)
        self.model_artifact_digest = str(model_artifact_digest)
        self.native = LlamaCppNativeServerExecutor(
            native_base_url,
            resource_slot=resource_slot,
            request_slot=request_slot,
            timeout_seconds=timeout_seconds,
        )
        self.block_store = LogicalPRABlockStore()
        self._fingerprint: str | None = None

    def negotiate(
        self, *, model: str, model_fingerprint: str
    ) -> OllamaBackendHandshake | None:
        del model
        capabilities = self.native._capabilities
        if (
            capabilities.get("protocol") != self.native.protocol
            or not capabilities.get("native_sequence_attach")
        ):
            return None
        self._fingerprint = model_fingerprint
        return OllamaBackendHandshake(
            protocol_version="pra-engine/1",
            backend="llama.cpp",
            backend_revision=self.backend_revision,
            model_fingerprint=model_fingerprint,
            model_artifact_digest=self.model_artifact_digest,
            integration_level="E2",
            mechanisms=self.mechanisms,
            resource_identity=True,
            tenant_isolation=True,
            request_cleanup=True,
        )

    def generate(
        self, request: PRAWireRequest, handshake: OllamaBackendHandshake
    ) -> PRAEngineResult:
        if self._fingerprint is None or not handshake.validates(self._fingerprint):
            raise RuntimeError("The Ollama/native-backend receipt is stale.")
        return self.native.generate(request, self.block_store)

    def invalidate_model(self, model_fingerprint: str) -> None:
        if self._fingerprint == model_fingerprint:
            self.native.close_session("ollama-model")
            self._fingerprint = None


class OllamaEngineAdapter:
    """Expose Ollama as E0 unless an explicitly negotiated backend supplies E2.

    Ollama owns model distribution, runner lifecycle, and scheduling. PRA owns
    resource selection and identity. The adapter does not infer native support
    from an Ollama version or from historical llama.cpp ancestry.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        backend_executor: OllamaBackendExecutor | None = None,
        timeout_seconds: float = 300.0,
        keep_alive: str | int = "5m",
        thinking: bool | None = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend_executor = backend_executor
        self.timeout_seconds = float(timeout_seconds)
        self.keep_alive = keep_alive
        self.thinking = thinking
        self._model_fingerprint: str | None = None
        self._backend_handshake: OllamaBackendHandshake | None = None

    def _request_json(
        self, path: str, payload: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def inspect_endpoint(self) -> OllamaEndpointInfo:
        version = str(self._request_json("/api/version").get("version", "unknown"))
        installed = tuple(
            str(row.get("name", ""))
            for row in self._request_json("/api/tags").get("models", ())
            if row.get("name")
        )
        running = tuple(
            str(row.get("name", ""))
            for row in self._request_json("/api/ps").get("models", ())
            if row.get("name")
        )
        return OllamaEndpointInfo(version, installed, running)

    def capabilities(self) -> PRAEngineCapabilities:
        handshake = self._backend_handshake
        native = bool(
            handshake is not None
            and self._model_fingerprint is not None
            and handshake.validates(self._model_fingerprint)
        )
        return PRAEngineCapabilities(
            adapter="ollama_backend_pra" if native else "ollama_gateway",
            engine_type=EngineType.OLLAMA,
            integration_level=handshake.integration_level if native and handshake else "E0",
            prefix_cache_mode=PrefixCacheMode.SESSION_STATE,
            session_state=True,
            cache_affinity=True,
            logical_refs=native,
            typed_records=native,
            text_fallback=True,
            native_kv=native,
            external_kv_residency=native,
            selected_interval_materialization=native,
            request_lifetime=native,
            streaming=False,
            tenant_isolation=native,
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id

    @staticmethod
    def _e0_messages(request: PRAWireRequest) -> list[Mapping[str, Any]]:
        resources = [resource for resource in request.resources if resource.text]
        if not resources:
            return list(request.messages)
        selected = "\n\n".join(
            f"[PRA resource {resource.resource_id} | {resource.uri}]\n{resource.text}"
            for resource in resources
        )
        context = {
            "role": "system",
            "content": "Use the selected external context below when relevant.\n\n" + selected,
        }
        return [context, *request.messages]

    @staticmethod
    def model_fingerprint(model: str, show: Mapping[str, Any]) -> str:
        identity = {
            "model": model,
            "modified_at": show.get("modified_at"),
            "details": show.get("details", {}),
            "model_info": show.get("model_info", {}),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _refresh_model_identity(self, model: str) -> str:
        fingerprint = self.model_fingerprint(
            model, self._request_json("/api/show", {"model": model})
        )
        if self._model_fingerprint not in {None, fingerprint} and self.backend_executor:
            self.backend_executor.invalidate_model(self._model_fingerprint)
            self._backend_handshake = None
        self._model_fingerprint = fingerprint
        return fingerprint

    def negotiate_backend(
        self, model: str, model_fingerprint: str | None = None
    ) -> OllamaBackendHandshake | None:
        """Negotiate and validate native support for one observed model package.

        Negotiation failure is deliberately non-fatal: AUTO remains on the E0
        selected-text path.  A cached receipt is reused only while its model
        fingerprint remains valid.
        """

        fingerprint = model_fingerprint or self._refresh_model_identity(model)
        if self._backend_handshake and self._backend_handshake.validates(fingerprint):
            return self._backend_handshake
        self._backend_handshake = None
        if self.backend_executor is None:
            return None
        try:
            candidate = self.backend_executor.negotiate(
                model=model, model_fingerprint=fingerprint
            )
        except Exception:
            return None
        if candidate is not None and candidate.validates(fingerprint):
            self._backend_handshake = candidate
        return self._backend_handshake

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        fingerprint = self._refresh_model_identity(request.model)
        handshake = self.negotiate_backend(request.model, fingerprint)
        if self.backend_executor is not None and handshake is not None:
            result = self.backend_executor.generate(request, handshake)
            return PRAEngineResult(
                result.text,
                result.raw,
                (
                    *result.trace,
                    {
                        "stage": "ollama_backend",
                        "model_fingerprint": fingerprint,
                        "backend": handshake.backend,
                        "backend_revision": handshake.backend_revision,
                        "integration_level": handshake.integration_level,
                        "mechanisms": handshake.mechanisms,
                    },
                ),
            )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._e0_messages(request),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": request.resolved_max_new_tokens},
        }
        if self.thinking is not None:
            payload["think"] = self.thinking
        if request.tools:
            payload["tools"] = list(request.tools)
        started = time.perf_counter()
        raw = self._request_json("/api/chat", payload)
        elapsed = time.perf_counter() - started
        return PRAEngineResult(
            str(raw.get("message", {}).get("content", "")),
            raw,
            (
                {
                    "stage": "ollama_e0",
                    "seconds": elapsed,
                    "model_fingerprint": fingerprint,
                    "load_ns": int(raw.get("load_duration", 0)),
                    "prompt_eval_ns": int(raw.get("prompt_eval_duration", 0)),
                    "eval_ns": int(raw.get("eval_duration", 0)),
                    "native_handshake": (
                        "unavailable" if self.backend_executor is None else "rejected"
                    ),
                },
            ),
        )

    def unload(self, model: str) -> Mapping[str, Any]:
        result = self._request_json(
            "/api/generate", {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
        )
        if self._model_fingerprint and self.backend_executor:
            self.backend_executor.invalidate_model(self._model_fingerprint)
        self._model_fingerprint = None
        self._backend_handshake = None
        return result

    def stream(self, request: PRAWireRequest):
        raise NotImplementedError("The initial Ollama adapter exposes non-streaming generation.")

    def close_session(self, session_id: str) -> None:
        if self.backend_executor is not None:
            native = getattr(self.backend_executor, "native", None)
            if native is not None:
                native.close_session(session_id)
