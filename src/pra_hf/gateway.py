"""Standalone PRA gateway with deterministic mediation modes."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, urlparse

from .agent_transport import render_wire_resources_as_text
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
from .session_realization import (
    PrefixReuseObservation,
    RealizationDecision,
    RealizationPlanner,
)
from .session_service import SessionService
from .observability import DISABLED_OBSERVABILITY, Observability


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
        observability: Observability | None = None,
    ) -> None:
        self.adapter = adapter
        self.mode = PRAGatewayMode(mode)
        self.sessions = session_registry or GatewaySessionRegistry(session_service)
        self.fallback_injection = FallbackInjectionPolicy(fallback_injection)
        self.observability = observability or DISABLED_OBSERVABILITY

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

    @staticmethod
    def _budgeted_resources(request: PRAWireRequest) -> tuple[Any, ...]:
        """Freeze deterministic selected-text intervals before realization."""

        remaining = request.budget.max_selected_tokens
        rendered_resources = []
        for resource in request.resources[: request.budget.max_resources]:
            if not resource.text or remaining <= 0:
                continue
            tokens = resource.text.split()
            selected_tokens = tokens[:remaining]
            materialized = (
                resource.text
                if len(selected_tokens) == len(tokens)
                else " ".join(selected_tokens)
            )
            remaining -= len(selected_tokens)
            metadata = dict(resource.metadata)
            metadata.setdefault("selected_interval", (0, len(selected_tokens)))
            rendered_resources.append(
                replace(resource, text=materialized, metadata=metadata)
            )
        return tuple(rendered_resources)

    def _rendering_profile(self, request: PRAWireRequest) -> str:
        semantic = request.pra_policy.get("profile", "default")
        return f"{self.fallback_injection.value}/{semantic}"

    def _text_fallback(
        self, request: PRAWireRequest, rendered_resources: tuple[Any, ...]
    ) -> PRAWireRequest:
        messages = list(request.messages)
        if rendered_resources:
            if self.fallback_injection == FallbackInjectionPolicy.BEFORE_CURRENT_USER:
                messages = list(
                    render_wire_resources_as_text(messages, rendered_resources)
                )
            else:
                context = "PRA text fallback context (not native K/V):\n\n" + "\n\n".join(
                    f"[PRA resource {resource.uri}]\n{resource.text}"
                    for resource in rendered_resources
                )
                messages = self._inject_fallback(messages, context)
        return replace(request, messages=tuple(messages), resources=())

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
        rendering_profile = self._rendering_profile(request)
        selected_resources = self._budgeted_resources(
            replace(base, resources=turn.active_resources)
        )
        resolved_mode = (
            "selected-context"
            if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK
            else "native-serving"
            if capabilities.pra_scheduler
            else "native-memory"
            if capabilities.native_kv
            else "selected-context"
        )
        plan = RealizationPlanner().plan(
            selected_resources,
            turn.state.visible_materializations,
            requested_mode=str(request.metadata.get("requested_mode", resolved_mode)),
            resolved_mode=resolved_mode,
            rendering_profile=rendering_profile,
            tenant_id=request.tenant_id,
            native_capable=capabilities.native_kv,
            fallback_reason=(
                "engine_uses_selected_text"
                if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK
                and str(request.metadata.get("requested_mode", resolved_mode)) != resolved_mode
                else None
            ),
        )
        invalid = plan.resources_for(RealizationDecision.INVALID)
        if invalid:
            raise PermissionError("Selected resources failed realization authorization.")
        turn = replace(turn, realization_plan=plan)
        if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK:
            # G10 selection uses the current logical resources even though an
            # E0 adapter never receives their detached PRA descriptors.
            fallback_base = base
            if (
                turn.outbound_history_mode == HistoryMode.FULL
                and turn.gateway_prefix_stable
                and turn.state.serialized_messages
            ):
                prior_serialized = turn.state.serialized_messages
                active_count = request.metadata.get("active_context_message_count")
                if active_count is not None:
                    count = max(int(active_count), 0)
                    prior_serialized = prior_serialized[-count:] if count else ()
                new_messages = turn.canonical_messages[len(turn.state.canonical_messages) :]
                fallback_base = replace(
                    base,
                    messages=(*prior_serialized, *new_messages),
                )
            newly_materialized = plan.resources_for(
                RealizationDecision.MUST_MATERIALIZE,
                RealizationDecision.OPTIONAL_REFRESH,
            )
            transformed = self._text_fallback(fallback_base, newly_materialized)
            selected_ids = list(plan.diagnostics["selected_resources"])
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
        worker_identity = (
            result.raw.get("worker_identity", result.raw.get("runner_identity"))
            if result.raw else None
        )
        model_fingerprint = (
            result.raw.get("model_fingerprint") if result.raw else None
        ) or request.metadata.get("model_fingerprint") or request.model
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
            engine_worker_identity=(str(worker_identity) if worker_identity is not None else None),
            engine_model_fingerprint=str(model_fingerprint),
            serialized_messages=serialized,
            materialized_resources=(
                turn.realization_plan.resources_for(
                    RealizationDecision.MUST_MATERIALIZE,
                    RealizationDecision.OPTIONAL_REFRESH,
                )
                if self.mode == PRAGatewayMode.G10_TEXT_FALLBACK
                and turn.realization_plan is not None
                else ()
            ),
            rendering_profile=self._rendering_profile(request),
        )
        physical_hit = result.raw.get("prefix_cache_hit") if result.raw else None
        capabilities = self.adapter.capabilities()
        prefix_observation = PrefixReuseObservation.from_result(
            result.raw,
            prefix_cache_mode=capabilities.prefix_cache_mode.value,
            prior_engine_session=bool(turn.state.engine_session_id),
            prefix_digest=state.last_serialized_prefix_digest,
            prefix_token_count=state.prefix_token_count,
            model_fingerprint=str(model_fingerprint),
            prior_worker_identity=turn.state.engine_worker_identity,
            prior_model_fingerprint=turn.state.engine_model_fingerprint,
            engine_restarted=bool(request.metadata.get("engine_restart")),
        )
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
            "prefix_reuse_status": prefix_observation.status.value,
            "prefix_cached_tokens": prefix_observation.cached_token_count,
            "prefix_reuse_observation": prefix_observation.to_dict(),
            "engine_session_reuse": bool(turn.state.engine_session_id),
            "engine_session_present": state.engine_session_id is not None,
            "cache_affinity_key": state.cache_affinity_key,
            "resource_ops": [row.operation.value for row in turn.resource_deltas],
        }
        if turn.realization_plan is not None:
            trace.update(turn.realization_plan.diagnostics)
        attached = result.raw.get("native_attached_resources", ()) if result.raw else ()
        trace["native_attached_resources"] = list(attached or ())
        if result.raw and result.raw.get("native_attach_bytes") is not None:
            trace["native_attach_bytes"] = int(result.raw["native_attach_bytes"])
        elif not attached:
            trace["native_attach_bytes"] = 0
        return trace, state

    def _telemetry_attributes(self, request: PRAWireRequest) -> dict[str, Any]:
        capabilities = self.adapter.capabilities()
        return {
            "pra.request.id": request.request_id,
            "pra.tenant.id_hash": self.observability.hash_id(request.tenant_id),
            "pra.session.id_hash": self.observability.hash_id(request.session_id),
            "pra.task.id_hash": self.observability.hash_id(request.task_id),
            "pra.engine": capabilities.engine_type.value,
            "gen_ai.system": capabilities.engine_type.value,
            "gen_ai.request.model": request.model,
            "pra.execution_mode": self.mode.value,
            "pra.profile": str(request.pra_policy.get("profile", "default")),
        }

    def _record_metrics(
        self,
        request: PRAWireRequest,
        trace: Mapping[str, Any],
        seconds: float,
        *,
        status: str,
    ) -> None:
        if not self.observability.metrics_enabled:
            return
        capabilities = self.adapter.capabilities()
        engine = capabilities.engine_type.value
        mode = self.mode.value
        profile = str(request.pra_policy.get("profile", "default"))
        labels = {"engine": engine, "execution_mode": mode, "status": status}
        self.observability.increment("pra_gateway_requests_total", labels=labels)
        self.observability.observe("pra_gateway_request_duration_seconds", seconds, labels=labels)
        self.observability.set_gauge(
            "pra_gateway_active_sessions",
            len(self.sessions.inspect_all()),
            labels={"engine": engine},
        )
        message_bytes = float(trace.get("message_bytes_sent", 0))
        resource_bytes = float(trace.get("resource_bytes_sent", 0))
        delta_bytes = float(trace.get("session_delta_bytes", 0))
        for name, value in (
            ("pra_gateway_message_bytes_total", message_bytes),
            ("pra_gateway_resource_bytes_total", resource_bytes),
            ("pra_gateway_delta_bytes_total", delta_bytes),
            ("pra_gateway_transport_bytes_total", message_bytes + resource_bytes),
        ):
            self.observability.increment(name, value, labels={"engine": engine})
        prefix_status = str(trace.get("prefix_reuse_status", "unknown"))
        self.observability.increment(
            "pra_prefix_observations_total",
            labels={"engine": engine, "status": prefix_status},
        )
        self.observability.increment(
            "pra_prefix_cached_tokens_total",
            float(trace.get("prefix_cached_tokens", 0)),
            labels={"engine": engine},
        )
        context_labels = {
            "engine": engine,
            "profile": profile,
            "execution_mode": mode,
        }
        for name, keys in (
            ("pra_context_source_tokens_total", ("source_tokens", "context_source_tokens")),
            ("pra_context_selected_tokens_total", ("selected_tokens", "context_selected_tokens")),
            ("pra_context_new_materialized_tokens_total", ("new_materialized_tokens",)),
            ("pra_context_visible_reuse_tokens_total", ("visible_reuse_tokens", "prefix_tokens_reusable")),
        ):
            value = next((trace[key] for key in keys if trace.get(key) is not None), 0)
            self.observability.increment(name, float(value), labels=context_labels)
        attached = trace.get("native_attached_resources", ()) or ()
        if attached:
            self.observability.increment(
                "pra_native_attaches_total",
                len(attached),
                labels={"engine": engine, "status": "success"},
            )
        if trace.get("native_attach_bytes") is not None:
            self.observability.set_gauge(
                "pra_native_bytes",
                float(trace["native_attach_bytes"]),
                labels={"engine": engine, "storage_tier": "active"},
            )

    def generate(
        self,
        request: PRAWireRequest | Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
    ) -> PRAEngineResult:
        if not isinstance(request, PRAWireRequest):
            request = PRAWireRequest.from_dict(request)
        started = time.perf_counter()
        status = "success"
        session_trace: Mapping[str, Any] = {}
        try:
            with self.observability.span(
                "pra.gateway.request",
                lambda: self._telemetry_attributes(request),
                parent_headers=trace_headers,
            ):
                with self.observability.span("pra.gateway.session.resolve"):
                    (
                        transformed,
                        turn,
                        engine_session_id,
                        unsupported,
                        downgrades,
                        selected_ids,
                        invalidation,
                    ) = self._resolve_request(request)
                with self.observability.span(
                    "pra.gateway.translate",
                    lambda: {
                        "pra.routing.selected_records": len(selected_ids),
                        "pra.realization.fallback": bool(downgrades),
                    },
                ):
                    pass
                with self.observability.span(
                    "pra.engine.request",
                    lambda: self._telemetry_attributes(request),
                ):
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
        except BaseException:
            status = "error"
            if self.observability.metrics_enabled:
                self.observability.increment(
                    "pra_gateway_upstream_errors_total",
                    labels={"engine": self.adapter.capabilities().engine_type.value},
                )
            raise
        finally:
            self._record_metrics(
                request,
                session_trace,
                time.perf_counter() - started,
                status=status,
            )

    def stream(
        self,
        request: PRAWireRequest | Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
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
            started = time.perf_counter()
            status = "success"
            session_trace: Mapping[str, Any] = {}
            span_context = self.observability.span(
                "pra.gateway.request",
                lambda: self._telemetry_attributes(request),
                parent_headers=trace_headers,
            )
            span_context.__enter__()
            try:
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
                for emitted in self.adapter.stream(transformed):
                    row = emitted
                    # Normalize older native executors at the gateway boundary.
                    if not row.get("type") and "text" in row:
                        row = {
                            "type": "delta",
                            "request_id": request.request_id,
                            **row,
                        }
                    last = row
                    if row.get("type") == "delta":
                        text_parts.append(str(row.get("text", "")))
                    if row.get("type") == "done":
                        completed = True
                    yield row
                if not completed:
                    last = {
                        "type": "done",
                        "request_id": request.request_id,
                        "native_kv_used": bool(
                            (last or {}).get("native_kv_used", False)
                        ),
                    }
                    completed = True
                    yield last
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
                    yield {
                        "type": "trace",
                        "request_id": request.request_id,
                        "trace": session_trace,
                    }
            except BaseException as error:
                status = "error"
                span_context.__exit__(type(error), error, error.__traceback__)
                raise
            else:
                span_context.__exit__(None, None, None)
            finally:
                self._record_metrics(
                    request,
                    session_trace,
                    time.perf_counter() - started,
                    status=status,
                )

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
                        result = gateway.generate(request, trace_headers=dict(self.headers.items()))
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
                    self._json(
                        200,
                        gateway.generate(payload, trace_headers=dict(self.headers.items())).to_dict(),
                    )
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
