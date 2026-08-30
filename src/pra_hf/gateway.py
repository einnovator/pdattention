"""Standalone PRA gateway with deterministic mediation modes."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, urlparse

from .deployment import (
    PRAEngineAdapter,
    PRAEngineResult,
    PRAGatewayMode,
    PRAWireRequest,
    inferred_resource,
)
from .gateway_session import (
    GatewaySessionRegistry,
    HistoryMode,
    ResourceOperation,
    ResolvedSessionTurn,
    canonical_digest,
)
from .session_service import SessionService


class PRACapabilityError(RuntimeError):
    """Raised when requested semantics cannot be implemented or downgraded."""


class FallbackInjectionPolicy(str, Enum):
    """Placement of selected text for engines without native PRA."""

    BEFORE_CURRENT_USER = "before_current_user"
    SYSTEM_SUFFIX = "system_suffix"
    TOOL_CONTEXT = "tool_context"
    APPEND_CONTEXT_RECORD = "append_context_record"
    ENGINE_NATIVE = "engine_native"


class PRAGateway:
    """Translate logical requests without owning model-native K/V or tool effects."""

    def __init__(
        self,
        adapter: PRAEngineAdapter,
        *,
        mode: PRAGatewayMode | str,
        session_service: SessionService | None = None,
        session_registry: GatewaySessionRegistry | None = None,
        fallback_injection: FallbackInjectionPolicy | str = FallbackInjectionPolicy.BEFORE_CURRENT_USER,
    ) -> None:
        self.adapter = adapter
        self.mode = PRAGatewayMode(mode)
        self.sessions = session_registry or GatewaySessionRegistry(session_service)
        self.fallback_injection = FallbackInjectionPolicy(fallback_injection)

    def capabilities(self) -> dict[str, Any]:
        engine = self.adapter.capabilities()
        accepts_typed = self.mode in {
            PRAGatewayMode.G10_TEXT_FALLBACK,
            PRAGatewayMode.G11_MEDIATION,
        } or engine.logical_refs
        effective = {
            "logical_refs": accepts_typed,
            "typed_records": accepts_typed,
            "task_metadata": accepts_typed,
            "resource_delta": accepts_typed,
            "session_state": True,
            "incremental_messages": True,
            "cache_affinity": engine.cache_affinity,
            "native_kv": bool(
                engine.native_kv and self.mode != PRAGatewayMode.G10_TEXT_FALLBACK
            ),
            "streaming": engine.streaming,
        }
        return {
            "protocol": "pra",
            "protocol_version": "1",
            "endpoint_type": "gateway",
            "gateway": {
                "mode": self.mode.value,
                "typed_records": accepts_typed,
                "resource_delta": accepts_typed,
                "session_state": True,
                "incremental_messages": True,
            },
            "effective_capabilities": effective,
            "gateway_mode": self.mode.value,
            "engine": {
                **engine.to_dict(),
                "type": engine.engine_type.value,
            },
            "fallback_injection": self.fallback_injection.value,
            "streaming_implemented": engine.streaming,
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

    def _text_fallback(self, request: PRAWireRequest) -> tuple[PRAWireRequest, list[str]]:
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
            context = "PRA text fallback context (not native K/V):\n\n" + "\n\n".join(blocks)
            messages = self._inject_fallback(messages, context)
        return replace(request, messages=tuple(messages), resources=()), selected

    def _inject_fallback(
        self, messages: list[Mapping[str, Any]], context: str
    ) -> list[Mapping[str, Any]]:
        """Inject current-turn evidence while retaining the prior message prefix."""

        if self.fallback_injection == FallbackInjectionPolicy.ENGINE_NATIVE:
            raise PRACapabilityError("ENGINE_NATIVE cannot be used for G10 text fallback.")
        values = [dict(row) for row in messages]
        if self.fallback_injection == FallbackInjectionPolicy.SYSTEM_SUFFIX:
            for index, message in enumerate(values):
                if message.get("role") == "system":
                    values[index]["content"] = f"{message.get('content', '')}\n\n{context}"
                    return values
            values.insert(0, {"role": "system", "content": context})
            return values
        user_index = next(
            (index for index in range(len(values) - 1, -1, -1) if values[index].get("role") == "user"),
            len(values),
        )
        if self.fallback_injection == FallbackInjectionPolicy.BEFORE_CURRENT_USER and user_index < len(values):
            # Folding evidence into the current user message is valid across
            # Qwen/Llama/Gemma and preserves every earlier serialized message.
            original = str(values[user_index].get("content", ""))
            values[user_index]["content"] = f"{context}\n\n{original}"
        elif self.fallback_injection == FallbackInjectionPolicy.TOOL_CONTEXT:
            values.insert(user_index, {"role": "tool", "content": context, "tool_call_id": "pra-context"})
        else:
            values.insert(user_index, {"role": "user", "content": context, "metadata": {"record_type": "pra_context"}})
        return values

    @staticmethod
    def _upgrade(request: PRAWireRequest) -> PRAWireRequest:
        inferred = tuple(
            resource
            for index, message in enumerate(request.messages)
            if (resource := inferred_resource(message, index)) is not None
        )
        return replace(request, resources=(*request.resources, *inferred))

    def _resolve_request(
        self, request: PRAWireRequest
    ) -> tuple[
        PRAWireRequest,
        ResolvedSessionTurn,
        str | None,
        list[str],
        list[str],
        list[str],
        str | None,
    ]:
        """Resolve session state, adapter lifecycle, and mediation exactly once."""

        capabilities = self.adapter.capabilities()
        unsupported, downgrades = self._negotiate(request)
        turn = self.sessions.resolve_turn(
            request,
            incremental_messages=capabilities.session_state and capabilities.incremental_messages,
            resource_delta=capabilities.logical_refs and capabilities.resource_delta,
        )
        invalidation = self._invalidation_reason(turn, request)
        if invalidation:
            if turn.state.engine_session_id:
                self.adapter.close_session(turn.state.engine_session_id)
            self.sessions.invalidate(request.tenant_id, request.session_id, request.model, invalidation)
            invalidated = self.sessions.get(request.tenant_id, request.session_id, request.model)
            if invalidated is not None:
                turn = replace(turn, state=invalidated)
        engine_session_id = request.engine_session_id or turn.state.engine_session_id
        session_needs_prepare = bool(request.session_id) and (
            turn.state.turns == 0 or invalidation is not None
        )
        base = replace(
            request,
            messages=turn.outbound_messages,
            resources=turn.active_resources,
            history_mode=turn.outbound_history_mode,
            engine_session_id=engine_session_id,
            prefix_cache_handle=request.prefix_cache_handle or turn.state.prefix_cache_handle,
            cache_affinity_key=request.cache_affinity_key or turn.state.cache_affinity_key,
            resource_ops=turn.resource_deltas,
        )
        if capabilities.resource_delta:
            changed = {
                row.resource_id
                for row in turn.resource_deltas
                if row.operation in {ResourceOperation.ADD, ResourceOperation.UPDATE}
            }
            base = replace(
                base,
                resources=tuple(
                    row for row in turn.active_resources if row.resource_id in changed
                ),
            )
        if session_needs_prepare:
            engine_session_id = self.adapter.prepare_session(base)
            base = replace(base, engine_session_id=engine_session_id)
        selected_ids: list[str] = []
        transformed = base
        if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK:
            # G10 selection uses the current logical resources even though an
            # E0 adapter never receives their detached PRA descriptors.
            fallback_base = base
            if (
                turn.outbound_history_mode == HistoryMode.FULL
                and turn.gateway_prefix_stable
                and turn.state.serialized_messages
            ):
                new_messages = turn.canonical_messages[len(turn.state.canonical_messages) :]
                fallback_base = replace(
                    base,
                    messages=(*turn.state.serialized_messages, *new_messages),
                )
            transformed, selected_ids = self._text_fallback(
                replace(fallback_base, resources=turn.active_resources)
            )
        elif self.mode == PRAGatewayMode.G01_UPGRADE:
            transformed = self._upgrade(base)
            if transformed.resources and not capabilities.logical_refs:
                raise PRACapabilityError("G01 upgrade requires an engine with logical_refs.")
        elif self.mode == PRAGatewayMode.G11_MEDIATION:
            if turn.active_resources and not capabilities.logical_refs:
                raise PRACapabilityError(
                    "G11 mediation requires logical_refs; choose explicit G10 text fallback."
                )
        return (
            transformed,
            turn,
            engine_session_id,
            unsupported,
            downgrades,
            selected_ids,
            invalidation,
        )

    @staticmethod
    def _invalidation_reason(turn: ResolvedSessionTurn, request: PRAWireRequest) -> str | None:
        state = turn.state
        if turn.prefix_changed_reason == "history_rewrite" and state.turns:
            return "system_prefix_or_history_rewrite"
        checks = (
            ("model_revision", "model_revision_changed"),
            ("chat_template_digest", "chat_template_changed"),
            ("visible_prefix_profile", "visible_prefix_profile_changed"),
        )
        for field_name, reason in checks:
            previous = getattr(state, field_name)
            current = request.metadata.get(field_name)
            if previous is not None and current is not None and previous != str(current):
                return reason
        if request.metadata.get("engine_restart"):
            return "engine_restart"
        if request.metadata.get("invalid_engine_session"):
            return "invalid_engine_session"
        return None

    def _commit_and_trace(
        self,
        request: PRAWireRequest,
        turn: ResolvedSessionTurn,
        engine_session_id: str | None,
        result: PRAEngineResult,
        transformed: PRAWireRequest,
        invalidation_reason: str | None,
    ) -> tuple[dict[str, Any], Any]:
        prefix_handle = result.raw.get("prefix_cache_handle") if result.raw else None
        committed_turn = turn
        if result.text:
            assistant = {"role": "assistant", "content": result.text}
            if not turn.canonical_messages or dict(turn.canonical_messages[-1]) != assistant:
                committed_turn = replace(
                    turn,
                    canonical_messages=(*turn.canonical_messages, assistant),
                )
        serialized = (
            (*turn.state.serialized_messages, *transformed.messages)
            if turn.outbound_history_mode == HistoryMode.DELTA
            else tuple(transformed.messages)
        )
        if result.text:
            serialized = (*serialized, {"role": "assistant", "content": result.text})
        previous_serialized = turn.state.serialized_messages
        if turn.outbound_history_mode == HistoryMode.DELTA:
            serialized_stable = bool(previous_serialized) and turn.gateway_prefix_stable
        else:
            serialized_stable = bool(previous_serialized) and tuple(
                serialized[: len(previous_serialized)]
            ) == previous_serialized
        state = self.sessions.commit(
            committed_turn,
            request,
            engine_session_id=engine_session_id,
            prefix_cache_handle=prefix_handle,
            serialized_messages=serialized,
        )
        physical_hit = result.raw.get("prefix_cache_hit") if result.raw else None
        message_bytes_sent = len(
            json.dumps(list(transformed.messages), default=str).encode("utf-8")
        )
        resource_bytes_sent = len(
            json.dumps(
                {
                    "resources": [row.to_dict() for row in transformed.resources],
                    "resource_ops": [
                        row.to_dict(include_resource=False)
                        for row in transformed.resource_ops
                    ],
                },
                default=str,
            ).encode("utf-8")
        ) if transformed.resources or transformed.resource_ops else 0
        trace = {
            "stage": "gateway_session",
            "history_mode": turn.outbound_history_mode.value,
            "gateway_prefix_stable": serialized_stable,
            "prefix_changed_reason": invalidation_reason or turn.prefix_changed_reason,
            "prefix_invalidations": int(invalidation_reason is not None),
            "prefix_digest": state.last_serialized_prefix_digest,
            "prefix_message_count": state.prefix_message_count,
            "prefix_token_count": state.prefix_token_count,
            "prefix_tokens_reusable": (
                len(previous_serialized) if serialized_stable else 0
            ),
            "prefix_reuse_fraction": (
                len(previous_serialized) / max(len(serialized), 1)
                if serialized_stable else 0.0
            ),
            "message_bytes_sent": message_bytes_sent,
            "resource_bytes_sent": resource_bytes_sent,
            "session_delta_bytes": (
                message_bytes_sent + resource_bytes_sent
                if turn.outbound_history_mode == HistoryMode.DELTA else 0
            ),
            "engine_prefix_cache_hit": physical_hit,
            "engine_session_reuse": bool(turn.state.engine_session_id),
            "engine_session_present": state.engine_session_id is not None,
            "cache_affinity_key": state.cache_affinity_key,
            "resource_ops": [row.operation.value for row in turn.resource_deltas],
        }
        return trace, state

    def generate(self, request: PRAWireRequest | Mapping[str, Any]) -> PRAEngineResult:
        if not isinstance(request, PRAWireRequest):
            request = PRAWireRequest.from_dict(request)
        started = time.perf_counter()
        (
            transformed,
            turn,
            engine_session_id,
            unsupported,
            downgrades,
            selected_ids,
            invalidation,
        ) = self._resolve_request(request)
        result = self.adapter.generate(transformed)
        session_trace, _ = self._commit_and_trace(
            request, turn, engine_session_id, result, transformed, invalidation
        )
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
            session_trace,
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
        (
            transformed,
            turn,
            engine_session_id,
            unsupported,
            downgrades,
            selected_ids,
            invalidation,
        ) = self._resolve_request(request)
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
            completed = False
            last: Mapping[str, Any] | None = None
            text_parts: list[str] = []
            for row in self.adapter.stream(transformed):
                last = row
                if row.get("type") == "delta":
                    text_parts.append(str(row.get("text", "")))
                if row.get("type") == "done":
                    completed = True
                yield row
            if completed:
                result = PRAEngineResult("".join(text_parts), dict(last or {}))
                session_trace, _ = self._commit_and_trace(
                    request,
                    turn,
                    engine_session_id,
                    result,
                    transformed,
                    invalidation,
                )
                yield {"type": "trace", "request_id": request.request_id, "trace": session_trace}

        return rows()

    def close_session(self, tenant_id: str, session_id: str, model: str) -> bool:
        """Close ephemeral engine state without deleting durable logical history."""

        state = self.sessions.close(tenant_id, session_id, model)
        if state is None:
            return False
        self.adapter.close_session(state.engine_session_id or session_id)
        return True

    def inspect_session(self, tenant_id: str, session_id: str, model: str) -> dict[str, Any] | None:
        state = self.sessions.inspect(tenant_id, session_id, model)
        if state is None:
            return None
        return {
            "engine_type": self.adapter.capabilities().engine_type.value,
            **state,
        }


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
                    if row.get("type") == "delta":
                        value = {
                            "id": row.get("request_id"),
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": row.get("text", "")},
                                "finish_reason": None,
                            }],
                        }
                    elif row.get("type") == "done":
                        value = {
                            "id": row.get("request_id"),
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                            "pra": row.get("trace", {}),
                        }
                    else:
                        value = {
                            "id": row.get("request_id"),
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "pra": row.get("trace", {}),
                        }
                    payload = json.dumps(value, default=str)
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
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json(200, {"status": "ok", **gateway.capabilities()})
            elif parsed.path == "/v1/pra/capabilities":
                self._json(200, gateway.capabilities())
            elif parsed.path.startswith("/v1/pra/sessions/"):
                session_id = parsed.path.rsplit("/", 1)[-1]
                query = parse_qs(parsed.query)
                tenant_id = query.get("tenant_id", ["default"])[0]
                model = query.get("model", [""])[0]
                state = gateway.inspect_session(tenant_id, session_id, model)
                self._json(200 if state else 404, state or {"error": "not_found"})
            else:
                self._json(404, {"error": "not_found"})

        def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/v1/pra/sessions/"):
                self._json(404, {"error": "not_found"})
                return
            session_id = parsed.path.rsplit("/", 1)[-1]
            query = parse_qs(parsed.query)
            tenant_id = query.get("tenant_id", ["default"])[0]
            model = query.get("model", [""])[0]
            closed = gateway.close_session(tenant_id, session_id, model)
            self._json(200 if closed else 404, {"closed": closed, "session_id": session_id})

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
                        protocol_trace = next(
                            (
                                row for row in result.trace
                                if row.get("stage") == "protocol_translation"
                            ),
                            {},
                        )
                        self._json(
                            200,
                            {
                                "id": request.request_id,
                                "object": "chat.completion",
                                "choices": [
                                    {"index": 0, "message": {"role": "assistant", "content": result.text}}
                                ],
                                "pra": {
                                    "selected_resource_ids": protocol_trace.get(
                                        "selected_resource_ids", []
                                    ),
                                    "materialized_tokens": result.raw.get(
                                        "materialized_tokens", 0
                                    ),
                                    "native_kv": protocol_trace.get("native_kv", False),
                                    "trace_id": request.correlation_id,
                                },
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
