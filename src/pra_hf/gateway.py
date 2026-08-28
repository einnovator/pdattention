"""Standalone PRA gateway with deterministic mediation modes."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Mapping

from .deployment import (
    PRAEngineAdapter,
    PRAEngineResult,
    PRAGatewayMode,
    PRAWireRequest,
    inferred_resource,
)


class PRACapabilityError(RuntimeError):
    """Raised when requested semantics cannot be implemented or downgraded."""


class PRAGateway:
    """Translate logical requests without owning model-native K/V or tool effects."""

    def __init__(self, adapter: PRAEngineAdapter, *, mode: PRAGatewayMode | str) -> None:
        self.adapter = adapter
        self.mode = PRAGatewayMode(mode)

    def capabilities(self) -> dict[str, Any]:
        return {
            "gateway_mode": self.mode.value,
            "engine": self.adapter.capabilities().to_dict(),
            "streaming_implemented": self.adapter.capabilities().streaming,
        }

    def _negotiate(self, request: PRAWireRequest) -> tuple[list[str], list[str]]:
        capabilities = self.adapter.capabilities()
        unsupported = [
            name for name in request.required_capabilities if not capabilities.supports(name)
        ]
        downgrades = []
        if unsupported:
            can_text_fallback = (
                request.allow_text_fallback
                and capabilities.text_fallback
                and self.mode == PRAGatewayMode.G10_TEXT_FALLBACK
            )
            if not can_text_fallback:
                raise PRACapabilityError(
                    f"Engine {capabilities.adapter!r} lacks required capabilities: {unsupported}"
                )
            downgrades.append(
                "native/logical PRA request materialized as ordinary text context"
            )
        return unsupported, downgrades

    @staticmethod
    def _text_fallback(request: PRAWireRequest) -> tuple[PRAWireRequest, list[str]]:
        remaining = request.budget.max_selected_tokens
        selected = []
        blocks = []
        for resource in request.resources[: request.budget.max_resources]:
            if not resource.text or remaining <= 0:
                continue
            tokens = resource.text.split()
            materialized = " ".join(tokens[:remaining])
            remaining -= len(tokens[:remaining])
            selected.append(resource.resource_id)
            blocks.append(f"[PRA resource {resource.uri}]\n{materialized}")
        messages = list(request.messages)
        if blocks:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "PRA text fallback context (not native K/V):\n\n"
                    + "\n\n".join(blocks),
                },
            )
        return replace(request, messages=tuple(messages), resources=()), selected

    @staticmethod
    def _upgrade(request: PRAWireRequest) -> PRAWireRequest:
        inferred = tuple(
            resource
            for index, message in enumerate(request.messages)
            if (resource := inferred_resource(message, index)) is not None
        )
        return replace(request, resources=(*request.resources, *inferred))

    def generate(self, request: PRAWireRequest | Mapping[str, Any]) -> PRAEngineResult:
        if not isinstance(request, PRAWireRequest):
            request = PRAWireRequest.from_dict(request)
        started = time.perf_counter()
        unsupported, downgrades = self._negotiate(request)
        selected_ids: list[str] = []
        transformed = request
        if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK:
            transformed, selected_ids = self._text_fallback(request)
        elif self.mode == PRAGatewayMode.G01_UPGRADE:
            transformed = self._upgrade(request)
            if transformed.resources and not self.adapter.capabilities().logical_refs:
                raise PRACapabilityError("G01 upgrade requires an engine with logical_refs.")
        elif self.mode == PRAGatewayMode.G11_MEDIATION:
            if request.resources and not self.adapter.capabilities().logical_refs:
                raise PRACapabilityError(
                    "G11 mediation requires logical_refs; choose explicit G10 text fallback."
                )
        result = self.adapter.generate(transformed)
        trace = (
            {
                "stage": "gateway_parse",
                "gateway_mode": self.mode.value,
                "correlation_id": request.correlation_id,
            },
            {
                "stage": "protocol_translation",
                "unsupported_capabilities": unsupported,
                "downgrades": downgrades,
                "selected_resource_ids": selected_ids,
                "native_kv": self.adapter.capabilities().native_kv
                and self.mode != PRAGatewayMode.G10_TEXT_FALLBACK,
                "seconds": time.perf_counter() - started,
            },
            *result.trace,
        )
        return PRAEngineResult(result.text, result.raw, trace)

    def stream(
        self, request: PRAWireRequest | Mapping[str, Any]
    ) -> Iterator[Mapping[str, Any]]:
        """Stream portable deltas after applying the same deterministic mediation."""

        if not isinstance(request, PRAWireRequest):
            request = PRAWireRequest.from_dict(request)
        if not self.adapter.capabilities().streaming:
            raise PRACapabilityError(
                f"Engine {self.adapter.capabilities().adapter!r} does not support streaming."
            )
        unsupported, downgrades = self._negotiate(request)
        selected_ids: list[str] = []
        transformed = request
        if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK:
            transformed, selected_ids = self._text_fallback(request)
        elif self.mode == PRAGatewayMode.G01_UPGRADE:
            transformed = self._upgrade(request)
            if transformed.resources and not self.adapter.capabilities().logical_refs:
                raise PRACapabilityError("G01 upgrade requires an engine with logical_refs.")
        elif self.mode == PRAGatewayMode.G11_MEDIATION:
            if request.resources and not self.adapter.capabilities().logical_refs:
                raise PRACapabilityError(
                    "G11 mediation requires logical_refs; choose explicit G10 text fallback."
                )
        def rows() -> Iterator[Mapping[str, Any]]:
            yield {
                "type": "trace",
                "request_id": request.request_id,
                "trace": {
                    "stage": "protocol_translation",
                    "gateway_mode": self.mode.value,
                    "correlation_id": request.correlation_id,
                    "unsupported_capabilities": unsupported,
                    "downgrades": downgrades,
                    "selected_resource_ids": selected_ids,
                    "native_kv": self.adapter.capabilities().native_kv
                    and self.mode != PRAGatewayMode.G10_TEXT_FALLBACK,
                },
            }
            yield from self.adapter.stream(transformed)

        return rows()


def _handler(gateway: PRAGateway):
    class GatewayHandler(BaseHTTPRequestHandler):
        server_version = "PRA-Gateway/0.1"

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _sse(self, rows: Iterator[Mapping[str, Any]]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            iterator = iter(rows)
            try:
                for row in iterator:
                    payload = json.dumps(row, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            if self.path == "/health":
                self._json(200, {"status": "ok", **gateway.capabilities()})
            elif self.path == "/v1/pra/capabilities":
                self._json(200, gateway.capabilities())
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/v1/chat/completions":
                    request = PRAWireRequest.from_openai(payload)
                    if bool(payload.get("stream", False)):
                        self._sse(gateway.stream(request))
                    else:
                        result = gateway.generate(request)
                        self._json(
                            200,
                            {
                                "id": request.request_id,
                                "object": "chat.completion",
                                "choices": [
                                    {"index": 0, "message": {"role": "assistant", "content": result.text}}
                                ],
                                "pra_trace": list(result.trace),
                            },
                        )
                elif self.path == "/v1/pra/generate":
                    self._json(200, gateway.generate(payload).to_dict())
                else:
                    self._json(404, {"error": "not_found"})
            except (ValueError, TypeError, PermissionError, PRACapabilityError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})

        def log_message(self, format: str, *args) -> None:
            return None

    return GatewayHandler


def serve_gateway(
    gateway: PRAGateway,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the standalone reference gateway until interrupted."""

    create_gateway_server(gateway, host=host, port=port).serve_forever()


def create_gateway_server(
    gateway: PRAGateway,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadingHTTPServer:
    """Create a controllable server for embedding, tests, and process launchers."""

    return ThreadingHTTPServer((host, int(port)), _handler(gateway))
