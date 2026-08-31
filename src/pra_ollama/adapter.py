"""Capability-negotiated Ollama adapter with a conservative E0 fallback."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult, PRAWireRequest
from pra_hf.engine_profiles import EngineType, PrefixCacheMode


@dataclass(frozen=True)
class OllamaEndpointInfo:
    """Observed endpoint state used for diagnostics and model invalidation."""

    version: str
    installed_models: tuple[str, ...]
    running_models: tuple[str, ...]


class OllamaBackendExecutor(Protocol):
    """PRA-aware execution supplied by Ollama's active model backend."""

    @property
    def integration_level(self) -> str: ...

    def generate(self, request: PRAWireRequest) -> PRAEngineResult: ...

    def invalidate_model(self, model_fingerprint: str) -> None: ...


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend_executor = backend_executor
        self.timeout_seconds = float(timeout_seconds)
        self.keep_alive = keep_alive
        self._model_fingerprint: str | None = None

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
        native = self.backend_executor is not None and self.backend_executor.integration_level in {
            "E2",
            "E3",
        }
        return PRAEngineCapabilities(
            adapter="ollama_backend_pra" if native else "ollama_gateway",
            engine_type=EngineType.OLLAMA,
            integration_level=self.backend_executor.integration_level if native else "E0",
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
        self._model_fingerprint = fingerprint
        return fingerprint

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        fingerprint = self._refresh_model_identity(request.model)
        if self.backend_executor is not None:
            result = self.backend_executor.generate(request)
            return PRAEngineResult(
                result.text,
                result.raw,
                (*result.trace, {"stage": "ollama_backend", "model_fingerprint": fingerprint}),
            )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._e0_messages(request),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": request.resolved_max_new_tokens},
        }
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
        return result

    def stream(self, request: PRAWireRequest):
        raise NotImplementedError("The initial Ollama adapter exposes non-streaming generation.")

    def close_session(self, session_id: str) -> None:
        del session_id
