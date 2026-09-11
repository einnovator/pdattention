"""Serve matched plain and native-PRA llama.cpp endpoints for Easy-50.

The process can expose either a direct engine boundary or a G11 PRA gateway.
Both modes use the same patched llama-server, model, slot pair, chat template,
and explicit sequential-prefix-cache setting. Native resource attachment is
delegated to ``pra_llamacpp`` from Paper 6.7 and fails closed when its patched
server protocol is absent.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
import urllib.error
import urllib.parse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from pra_hf.deployment import PRAEngineResult, PRAWireRequest
from pra_hf.gateway import PRAGateway, serve_gateway


_TRAJECTORY_RESOURCE_ID = re.compile(r"m(?P<message>\d+)-(?P<segment>\d+)-(?P<role>.+)")


def _load_llamacpp_types():
    try:
        from pra_llamacpp import LlamaCppEngineAdapter, LlamaCppNativeServerExecutor
    except ImportError as error:
        raise RuntimeError(
            "pra_llamacpp is required; add the Paper 6.7 src directory to PYTHONPATH"
        ) from error
    return LlamaCppEngineAdapter, LlamaCppNativeServerExecutor


class CausalChatNativePromptMixin:
    """Make split-prefill resources an exact chat-template prefix.

    The generic llama.cpp adapter treats resources as unstructured text. That
    is appropriate for detached documents, but an agent trajectory contains
    ordered user/assistant turns. Encoding their raw text before a separately
    rendered query drops role boundaries and puts the model's system prompt in
    the middle of the token sequence. Here the resource slot owns the rendered
    system-plus-history prefix and the request slot receives only the exact
    rendered suffix containing the latest observation and generation marker.
    """

    def _render_chat(
        self, messages: list[dict[str, Any]], request: PRAWireRequest, *, generate: bool,
    ) -> str:
        rendered = self._request_json(
            "/apply-template",
            {
                "messages": messages,
                "tools": list(request.tools),
                "add_generation_prompt": generate,
            },
        )
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("llama-server did not return a rendered chat prompt.")
        return prompt

    @staticmethod
    def _trajectory_messages(request: PRAWireRequest) -> list[tuple[int, dict[str, Any]]]:
        grouped: dict[tuple[int, str], list[tuple[int, str]]] = {}
        for resource in request.resources:
            metadata = dict(resource.metadata)
            message_index = metadata.get("message_index")
            segment_index = metadata.get("segment_index")
            role = metadata.get("role")
            if message_index is None or segment_index is None or role is None:
                match = _TRAJECTORY_RESOURCE_ID.fullmatch(resource.resource_id)
                if match is None:
                    return []
                message_index = int(match.group("message"))
                segment_index = int(match.group("segment"))
                role = match.group("role")
            grouped.setdefault((int(message_index), str(role)), []).append(
                (int(segment_index), resource.text or "")
            )
        return [
            (
                message_index,
                {
                    "role": role,
                    "content": "".join(text for _, text in sorted(segments)),
                },
            )
            for (message_index, role), segments in sorted(grouped.items())
        ]

    def _causal_prompt_pair(self, request: PRAWireRequest) -> tuple[str, str] | None:
        # Client request IDs need not be globally unique.  Object identity
        # binds the cache to this immutable parsed request and prevents a
        # reused external ID from replaying another request's rendered chat.
        cache_key = (str(getattr(request, "request_id", "")), id(request))
        cached = getattr(self, "_causal_prompt_pair_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        history = self._trajectory_messages(request)
        if not history:
            self._causal_prompt_pair_cache = (cache_key, None)
            return None
        mandatory = [dict(message) for message in request.messages]
        system = [message for message in mandatory if message.get("role") == "system"]
        current = [message for message in mandatory if message.get("role") != "system"]
        prefix_messages = [*system, *(message for _, message in history)]
        full_messages = [*prefix_messages, *current]
        self._validate_causal_messages(full_messages)
        # Render the complete request first.  /apply-template is the model's
        # authoritative grammar check and must pass before any selected prefix
        # can be materialized into a resource slot.
        full = self._render_chat(full_messages, request, generate=True)
        prefix = self._render_chat(prefix_messages, request, generate=False)
        if not full.startswith(prefix):
            raise RuntimeError(
                "llama.cpp chat template is not prefix-separable for native agent history"
            )
        suffix = full[len(prefix):]
        if not suffix:
            raise RuntimeError("native agent chat suffix is empty")
        pair = (prefix, suffix)
        # The adapter asks for resource text, its digest, its prompt, and the
        # logical token IDs during one generation.  Rendering and validating
        # the same large chat for every accessor added several redundant
        # llama-server round trips per agent step.  Keep only the most recent
        # immutable wire request so validation still occurs exactly once.
        self._causal_prompt_pair_cache = (cache_key, pair)
        return pair

    @staticmethod
    def _validate_causal_messages(messages: list[dict[str, Any]]) -> None:
        """Fail before inference when selection fabricated an invalid chat."""

        prior_role: str | None = None
        seen_non_system = False
        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            if role == "system":
                if seen_non_system:
                    raise ValueError(
                        f"system message at index {index} follows conversation history"
                    )
            else:
                seen_non_system = True
            if role == "assistant" and prior_role == "assistant":
                raise ValueError(
                    "selected agent history contains adjacent assistant messages "
                    f"at indices {index - 1} and {index}"
                )
            prior_role = role
        if messages and str(messages[-1].get("role")) == "assistant":
            raise ValueError(
                "selected agent history ends with assistant before generation"
            )

    def _resource_text(self, request: PRAWireRequest) -> str:
        pair = self._causal_prompt_pair(request)
        if pair is not None:
            return pair[0]
        return super()._resource_text(request)

    def _query_text(self, request: PRAWireRequest) -> str:
        pair = self._causal_prompt_pair(request)
        if pair is not None:
            return pair[1]
        return super()._query_text(request)

    def _ensure_resource(self, request: PRAWireRequest, lease: Any) -> tuple[str, bool]:
        """Update a pinned selected prefix with llama.cpp's cache delta path.

        The generic adapter deletes a resident resource whenever its digest
        changes. Agent selection changes almost every turn, but its rendered
        system/task prefix is stable. Submitting the replacement prompt to the
        same explicitly pinned slot with ``cache_prompt`` lets llama.cpp trim
        the divergent tail and evaluate only the replacement suffix.
        """

        text = self._resource_text(request)
        digest = self._resource_digest(request, text)
        if lease.identity.resource_digest != digest:
            raise RuntimeError("Allocated resource identity does not match request content.")
        current = self._slot_identities.get(lease.resource_slot)
        metrics_by_digest = getattr(self, "_resource_update_metrics", None)
        if metrics_by_digest is None:
            metrics_by_digest = {}
            self._resource_update_metrics = metrics_by_digest
        if current == lease.identity:
            previous = metrics_by_digest.get(digest, {})
            metrics_by_digest[digest] = {
                "resource_update_mode": "identity_hit",
                "resource_prefix_cached_tokens": None,
                "resource_evaluated_tokens": 0,
                "resource_total_tokens": previous.get("resource_total_tokens"),
            }
            return digest, False

        raw = self._request_json(
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
        timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
        cached = timings.get("cache_n")
        evaluated = timings.get("prompt_n")
        total = raw.get("tokens_evaluated")
        if total is None and cached is not None and evaluated is not None:
            total = int(cached) + int(evaluated)
        metrics_by_digest[digest] = {
            "resource_update_mode": (
                "cold_prefill" if current is None else "prefix_delta"
            ),
            "resource_prefix_cached_tokens": (
                int(cached) if cached is not None else None
            ),
            "resource_evaluated_tokens": (
                int(evaluated) if evaluated is not None else None
            ),
            "resource_total_tokens": (
                int(total) if total is not None else None
            ),
        }
        self._slot_identities[lease.resource_slot] = lease.identity
        return digest, True

    def generate(self, request: PRAWireRequest, block_store: Any) -> PRAEngineResult:
        result = super().generate(request, block_store)
        resource_row = next(
            (
                row for row in result.trace
                if row.get("stage") == "llama_cpp_native_resource"
            ),
            {},
        )
        digest = resource_row.get("resource_digest")
        metrics = getattr(self, "_resource_update_metrics", {}).get(digest, {})
        if not metrics:
            return result
        raw = dict(result.raw)
        raw.update(metrics)
        return PRAEngineResult(
            result.text,
            raw,
            (*result.trace, {"stage": "llama_cpp_resource_delta", **metrics}),
        )


class PlainSlotExecutor:
    """Run the no-PRA control through one explicit llama-server request slot."""

    def __init__(self, native_executor: Any, *, prefix_caching: bool) -> None:
        self.native = native_executor
        self.prefix_caching = bool(prefix_caching)

    def _generate_prompt(self, request: PRAWireRequest, prompt: str) -> PRAEngineResult:
        if not self.prefix_caching:
            self.native._erase_request_slot(self.native.request_slot)
        raw = dict(self.native._request_json(
            "/completion",
            {
                "prompt": prompt,
                "id_slot": self.native.request_slot,
                "n_predict": request.resolved_max_new_tokens,
                "cache_prompt": self.prefix_caching,
                "temperature": float(request.openai_fields.get("temperature", 0)),
                "seed": int(request.openai_fields.get("seed", 0)),
                "return_tokens": True,
            },
        ))
        cached = (raw.get("timings") or {}).get("cache_n")
        raw["prefix_cache_enabled"] = self.prefix_caching
        raw["prefix_cached_tokens"] = cached
        raw["engine_cached_tokens_total"] = cached
        raw["prefix_cache_hit"] = bool(self.prefix_caching and (cached or 0) > 0)
        return PRAEngineResult(
            str(raw.get("content", "")),
            raw,
            ({
                "stage": "llama_cpp_plain",
                "prefix_cache_enabled": self.prefix_caching,
                "prefix_cached_tokens": cached,
                "engine_cached_tokens_total": cached,
                "prefix_cache_hit": raw["prefix_cache_hit"],
                "native_kv": False,
            },),
        )

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        return self._generate_prompt(request, self.native._query_text(request))

    def generate_complete_selection(self, request: PRAWireRequest) -> PRAEngineResult:
        """Consume a lossless selected trajectory as an ordinary full prompt.

        At 100% retention PRA is a semantic no-op.  Sending the selected prefix
        through the detached-resource path would change execution from ordinary
        full prefill into resource-slot materialization even when prefix caching
        is disabled, invalidating the cache-on/off control.
        """

        pair = self.native._causal_prompt_pair(request)
        prompt = "".join(pair) if pair is not None else self.native._query_text(request)
        return self._generate_prompt(request, prompt)


class HybridLlamaCppAdapter:
    """Dispatch plain controls and typed-resource requests on one engine target."""

    def __init__(self, native_adapter: Any, plain: PlainSlotExecutor, *, prefix_caching: bool):
        self.native_adapter = native_adapter
        self.plain = plain
        self.prefix_cache_enabled = bool(prefix_caching)
        self._live_session_slots: dict[str, int] = {}
        # G11 may send only ADD/UPDATE bodies after the first request.  The
        # native prompt renderer, however, requires the complete ordered
        # selected trajectory on every turn.  Retain that logical inventory
        # at the adapter boundary and reconstruct it before rendering.  The
        # wrapper previously advertised resource_delta without doing this,
        # so request three contained only the newly added action/observation
        # pair and silently discarded the pinned task prefix.
        self._session_resources: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
        # llama-server's /slots ``n_prompt_tokens`` is the prompt length of the
        # most recent request, not the number of tokens currently resident in
        # that sequence.  Keep the exact resident token IDs explicitly; using
        # n_prompt_tokens to validate generated history caused every request
        # after the first native hand-off to fall back to rematerialization,
        # while comparing only lengths missed boundary-token substitutions.
        self._live_session_tokens: dict[str, tuple[int, ...]] = {}
        # Complete accepted logical transcripts are reconstructed from a
        # hash-only manifest plus records already observed by this endpoint.
        # They are never reconstructed from omitted selected text.  This is
        # what lets record IDs be mapped to the K/V positions at which the
        # model originally evaluated them.
        self._logical_session_messages: dict[str, list[dict[str, Any]]] = {}

    def capabilities(self):
        return replace(
            self.native_adapter.capabilities(),
            resource_delta=True,
            cache_affinity=True,
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id

    @staticmethod
    def _resource_session_key(request: PRAWireRequest) -> tuple[str, str, str] | None:
        session_id = getattr(request, "session_id", None)
        if session_id is None:
            return None
        return (
            str(getattr(request, "tenant_id", "default")),
            str(session_id),
            str(getattr(request, "model", "default")),
        )

    def _hydrate_resource_delta(self, request: PRAWireRequest) -> PRAWireRequest:
        """Reconstruct the full ordered resource view from a G11 delta.

        Resource operations are ordered according to the current logical
        inventory.  ADD/UPDATE bodies may arrive in ``resources`` while
        UNCHANGED bodies must come from the adapter's prior session state.
        Fail closed if either side of that contract is incomplete.
        """

        key = self._resource_session_key(request)
        resource_ops = tuple(getattr(request, "resource_ops", ()))
        resources = tuple(getattr(request, "resources", ()))
        if resource_ops and key is None:
            raise ValueError("resource deltas require a session_id")
        if key is None:
            return request

        if not resource_ops:
            if resources:
                self._session_resources[key] = {
                    resource.resource_id: resource for resource in resources
                }
            return request

        prior = self._session_resources.get(key, {})
        supplied = {resource.resource_id: resource for resource in resources}
        active: list[Any] = []
        consumed: set[str] = set()
        for delta in resource_ops:
            resource_id = str(delta.resource_id)
            operation = str(getattr(delta.operation, "value", delta.operation)).upper()
            if operation == "REMOVE":
                continue
            resource = supplied.get(resource_id)
            if resource is None:
                resource = getattr(delta, "resource", None)
            if resource is None:
                resource = prior.get(resource_id)
            if resource is None:
                raise ValueError(
                    "resource delta cannot reconstruct active resource "
                    f"{resource_id!r}"
                )
            if operation in {"ADD", "UPDATE"} and resource_id not in supplied \
                    and getattr(delta, "resource", None) is None:
                raise ValueError(
                    f"resource delta {operation} for {resource_id!r} has no body"
                )
            active.append(resource)
            consumed.add(resource_id)

        unexpected = set(supplied) - consumed
        if unexpected:
            raise ValueError(
                "resource delta supplied bodies without active operations: "
                + ", ".join(sorted(unexpected))
            )
        self._session_resources[key] = {
            resource.resource_id: resource for resource in active
        }
        return replace(request, resources=tuple(active), resource_ops=())

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        request = self._hydrate_resource_delta(request)
        logical_messages = self._hydrate_logical_messages(request)
        openai_fields = dict(request.openai_fields)
        requested = openai_fields.get("prefix_caching")
        if requested is not None and bool(requested) != self.prefix_cache_enabled:
            raise ValueError(
                "request prefix_caching does not match the endpoint's locked cache mode"
            )
        request = PRAWireRequest.from_dict({
            **request.to_dict(),
            "openai_fields": {
                **openai_fields,
                "prefix_caching": self.prefix_cache_enabled,
            },
        })
        if request.resources:
            # The same request slot also serves resource-free first turns. Its
            # sequential cache is therefore stale as soon as selected history
            # is attached. Start every native request from an empty request
            # sequence; the pinned resource slot remains encoded and reusable.
            native = self.native_adapter.native_executor
            live_slot = self._live_session_slots.get(str(request.session_id))
            selection_complete = bool(request.metadata.get("selection_complete", False))
            if (
                self.prefix_cache_enabled
                and live_slot is not None
                and logical_messages is not None
            ):
                result = self._generate_from_live_records(
                    request, live_slot, logical_messages,
                )
                # The patched server commits the newly evaluated suffix back
                # into the source sequence.  The destination is disposable;
                # source remains the canonical full live transcript.
                self._live_session_slots[str(request.session_id)] = live_slot
                self._remember_resident_tokens(request, result, logical_messages)
                self._logical_session_messages[str(request.session_id)] = logical_messages
                return result
            if (
                not self.prefix_cache_enabled
                and logical_messages is not None
                and request.metadata.get("history_projection") == "live-agent-kv-v1"
            ):
                # Matched scientific control for the live selected-K/V path:
                # preserve the exact selected logical records and chat
                # presentation, but deliberately encode them from text into an
                # empty slot on every turn.  This is explicitly labelled
                # rematerialization and must never qualify as agent-history PRA.
                result = self._generate_fresh_selected_records(
                    request, logical_messages,
                )
                self._logical_session_messages[str(request.session_id)] = logical_messages
                return result
            if request.metadata.get("history_projection") == "live-agent-kv-v1":
                raise RuntimeError(
                    "live agent-history PRA requires an existing prefix-cache source, "
                    "a valid logical manifest, and prefix caching enabled"
                )
            if self.prefix_cache_enabled and live_slot is not None and selection_complete:
                # Backward-compatible control path for old fixtures that do
                # not claim the live-agent-kv contract.
                result, destination = self._generate_from_live_slot(request, live_slot)
                self._live_session_slots[str(request.session_id)] = destination
                self._remember_resident_tokens(request, result)
                return result
            if selection_complete:
                result = self.plain.generate_complete_selection(request)
                self._live_session_slots[str(request.session_id)] = (
                    self.plain.native.request_slot
                )
                if logical_messages is None:
                    self._remember_resident_tokens(request, result)
                else:
                    self._remember_resident_tokens(request, result, logical_messages)
                if logical_messages is not None:
                    self._logical_session_messages[str(request.session_id)] = logical_messages
                return result
            native._erase_request_slot(native.request_slot)
            result = self.native_adapter.generate(request)
            self._live_session_slots[str(request.session_id)] = native.request_slot
            self._remember_resident_tokens(request, result, logical_messages)
            if logical_messages is not None:
                self._logical_session_messages[str(request.session_id)] = logical_messages
            return result
        result = self.plain.generate(request)
        if request.session_id is not None:
            self._live_session_slots[str(request.session_id)] = self.plain.native.request_slot
            self._remember_resident_tokens(request, result, logical_messages)
            if logical_messages is not None:
                self._logical_session_messages[str(request.session_id)] = logical_messages
        return result

    def _generate_fresh_selected_records(
        self,
        request: PRAWireRequest,
        logical_messages: list[dict[str, Any]],
    ) -> PRAEngineResult:
        """Cold-prefill the same logical records chosen by live-K/V PRA."""

        selected_indices = {
            int(resource.metadata["message_index"])
            for resource in request.resources
        }
        selected_indices.update(
            int(index) for index in request.metadata["mandatory_message_indices"]
        )
        if not selected_indices:
            raise RuntimeError("fresh selected-record control has no selected messages")
        if min(selected_indices) < 0 or max(selected_indices) >= len(logical_messages):
            raise RuntimeError("selected message index is outside the logical transcript")
        selected_messages = [
            logical_messages[index] for index in sorted(selected_indices)
        ]
        physical_request = PRAWireRequest.from_dict({
            **request.to_dict(),
            "messages": selected_messages,
            "resources": [],
            "resource_ops": [],
        })
        result = self.plain.generate(physical_request)
        raw = dict(result.raw)
        timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
        evaluated = int(
            raw.get("tokens_evaluated", timings.get("prompt_n", 0)) or 0
        )
        raw["pra"] = {
            "native_kv": False,
            "kv_source": "fresh_selected_prefill",
            "selected_text_reencoded_tokens": evaluated,
            "physical_kv_copy": False,
        }
        return PRAEngineResult(
            result.text,
            raw,
            (*result.trace, {
                "stage": "llama_cpp_fresh_selected_prefill_control",
                "selected_records": len(selected_indices),
                "selected_text_reencoded_tokens": evaluated,
                "physical_kv_copy": False,
                "native_kv": False,
            }),
        )

    @staticmethod
    def _content_sha256(message: Mapping[str, Any]) -> str:
        return __import__("hashlib").sha256(
            str(message.get("content", "")).encode("utf-8")
        ).hexdigest()

    def _hydrate_logical_messages(
        self, request: PRAWireRequest,
    ) -> list[dict[str, Any]] | None:
        """Recover the full logical transcript without omitted text fallback."""

        metadata = getattr(request, "metadata", {})
        manifest = metadata.get("logical_message_manifest")
        mandatory_indices = metadata.get("mandatory_message_indices")
        if manifest is None and mandatory_indices is None:
            return None
        if not isinstance(manifest, list) or not isinstance(mandatory_indices, list):
            raise ValueError("live agent-history metadata requires manifest and indices")
        if len(mandatory_indices) != len(request.messages):
            raise ValueError("mandatory message indices do not match physical messages")

        known: dict[int, dict[str, Any]] = {}
        prior = self._logical_session_messages.get(str(request.session_id), [])
        for index, message in enumerate(prior):
            known[index] = dict(message)
        for index, message in zip(mandatory_indices, request.messages):
            known[int(index)] = dict(message)

        resource_parts: dict[int, list[tuple[int, str, str]]] = {}
        for resource in request.resources:
            metadata = dict(resource.metadata)
            if "message_index" not in metadata or "segment_index" not in metadata:
                raise ValueError(
                    f"live history resource {resource.resource_id!r} lacks record coordinates"
                )
            resource_parts.setdefault(int(metadata["message_index"]), []).append(
                (
                    int(metadata["segment_index"]),
                    str(metadata.get("role", "")),
                    str(resource.text or ""),
                )
            )
        for index, parts in resource_parts.items():
            parts.sort()
            roles = {role for _, role, _ in parts}
            if len(roles) != 1:
                raise ValueError(f"record m{index} has inconsistent child roles")
            known.setdefault(index, {
                "role": next(iter(roles)),
                "content": "".join(text for _, _, text in parts),
            })

        logical: list[dict[str, Any]] = []
        for expected_index, row in enumerate(manifest):
            if int(row.get("message_index", -1)) != expected_index:
                raise ValueError("logical message manifest is not contiguous")
            message = known.get(expected_index)
            if message is None:
                raise ValueError(
                    f"logical record m{expected_index} was neither resident nor supplied"
                )
            if str(message.get("role", "")) != str(row.get("role", "")):
                raise ValueError(f"logical record m{expected_index} changed role")
            if self._content_sha256(message) != str(row.get("content_sha256", "")):
                raise ValueError(f"logical record m{expected_index} changed content")
            logical.append(message)
        return logical

    def _logical_prompt_tokens(self, request: PRAWireRequest) -> tuple[int, ...]:
        native = self.native_adapter.native_executor
        pair = native._causal_prompt_pair(request) if request.resources else None
        prompt = "".join(pair) if pair is not None else native._query_text(request)
        result = native._request_json(
            "/tokenize", {"content": prompt, "add_special": True}
        )
        return tuple(int(token) for token in result.get("tokens", ()))

    def _remember_resident_tokens(
        self,
        request: PRAWireRequest,
        result: PRAEngineResult,
        logical_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        generated = tuple(int(token) for token in result.raw.get("tokens", ()))
        # The final sampled token is returned but has not yet been evaluated
        # into KV.  It must be sent as a bridge on the following request.
        prompt_tokens = (
            self._full_logical_tokens(request, logical_messages)
            if logical_messages is not None
            else self._logical_prompt_tokens(request)
        )
        resident = prompt_tokens + (
            generated[:-1] if generated else ()
        )
        self._live_session_tokens[str(request.session_id)] = resident

    def _full_logical_tokens(
        self, request: PRAWireRequest, messages: list[dict[str, Any]],
    ) -> tuple[int, ...]:
        native = self.native_adapter.native_executor
        prompt = native._render_chat(messages, request, generate=True)
        return tuple(int(token) for token in native._request_json(
            "/tokenize", {"content": prompt, "add_special": True}
        ).get("tokens", ()))

    def _message_token_boundaries(
        self, request: PRAWireRequest, messages: list[dict[str, Any]],
        full_tokens: tuple[int, ...],
    ) -> list[int]:
        native = self.native_adapter.native_executor
        boundaries: list[int] = []
        for end in range(1, len(messages) + 1):
            prompt = native._render_chat(messages[:end], request, generate=False)
            tokens = tuple(int(token) for token in native._request_json(
                "/tokenize", {"content": prompt, "add_special": True}
            ).get("tokens", ()))
            if full_tokens[:len(tokens)] != tokens:
                raise RuntimeError(
                    "chat template is not record-prefix-separable at "
                    f"logical message {end - 1}"
                )
            boundaries.append(len(tokens))
        return boundaries

    def _generate_from_live_records(
        self,
        request: PRAWireRequest,
        source: int,
        logical_messages: list[dict[str, Any]],
    ) -> PRAEngineResult:
        """Attach selected record K/V ranges from the canonical live source."""

        from pra_llamacpp import (
            LlamaCppLivePrefixPlan,
            LlamaCppLivePrefixRange,
        )

        native = self.native_adapter.native_executor
        resident = self._live_session_tokens.get(str(request.session_id))
        if not resident:
            raise RuntimeError("live history source has no recorded resident tokens")
        full_tokens = self._full_logical_tokens(request, logical_messages)
        common = 0
        for cached_token, prompt_token in zip(resident, full_tokens):
            if cached_token != prompt_token:
                break
            common += 1
        if common <= 0:
            raise RuntimeError("logical transcript shares no K/V prefix with its source")
        boundaries = self._message_token_boundaries(
            request, logical_messages, full_tokens,
        )
        selected_indices = sorted({
            int(resource.metadata["message_index"])
            for resource in request.resources
        })
        mandatory = [int(index) for index in request.metadata["mandatory_message_indices"]]
        system_indices = [
            index for index, message in enumerate(logical_messages)
            if str(message.get("role")) == "system"
        ]
        active_non_system = [
            index for index in mandatory
            if str(logical_messages[index].get("role")) != "system"
        ]
        active_start = min(active_non_system) if active_non_system else len(logical_messages)

        ranges: list[Any] = []
        for index in [*system_indices, *selected_indices]:
            start = 0 if index == 0 else boundaries[index - 1]
            end = min(boundaries[index], common)
            if end <= start:
                continue
            metadata = next(
                (
                    dict(resource.metadata) for resource in request.resources
                    if int(resource.metadata.get("message_index", -1)) == index
                ),
                {},
            )
            ranges.append(LlamaCppLivePrefixRange(
                record_id=("system-prefix" if index in system_indices else f"m{index}"),
                parent_record_id=str(metadata.get("parent_record_id", f"m{index}")),
                causal_group_id=str(metadata.get("causal_group_id", f"record:m{index}")),
                start=start,
                end=end,
            ))
        active_token_start = (
            0 if active_start == 0 else boundaries[active_start - 1]
        )
        if active_token_start < common:
            ranges.append(LlamaCppLivePrefixRange(
                record_id=f"active-tail:m{active_start}",
                parent_record_id=f"active-tail:m{active_start}",
                causal_group_id=f"active-tail:m{active_start}",
                start=active_token_start,
                end=common,
            ))
        ranges.sort(key=lambda row: (row.start, row.end))
        for left, right in zip(ranges, ranges[1:]):
            if right.start < left.end:
                raise RuntimeError(
                    f"selected live record ranges overlap: {left.record_id}, {right.record_id}"
                )
        if not ranges or ranges[-1].end < common:
            # llama.cpp requires the newest retained position to be the
            # predecessor of the suffix. A single boundary cell is enough;
            # it does not pull an omitted historical record back into view.
            ranges.append(LlamaCppLivePrefixRange(
                record_id=f"source-tail:p{common}",
                parent_record_id=f"source-tail:p{common}",
                causal_group_id=f"source-tail:p{common}",
                start=common - 1,
                end=common,
            ))
        plan = LlamaCppLivePrefixPlan(
            source_slot=source,
            source_tokens=common,
            ranges=tuple(ranges),
            commit_to_source=True,
        )
        if plan.selected_tokens == common:
            # A lossless selection is ordinary prefix-cache continuation. Do
            # not add all source cells to a second sequence and repeatedly
            # copy sequence membership back: that changes the long-context
            # numerical path and is slower than continuing the canonical
            # sequence directly. Sparse selections still use cross-sequence
            # live K/V attachment below.
            result, _ = self._generate_from_live_slot(request, source)
            raw = dict(result.raw)
            pra = dict(raw.get("pra", {}))
            pra.update(
                kv_source="live_prefix_capture",
                source_slot=source,
                source_tokens=common,
                selected_records=len(ranges),
                selected_kv_tokens=common,
                selected_text_reencoded_tokens=0,
                physical_kv_copy=False,
                full_retention=True,
                exact_live_prefix_continuation=True,
            )
            raw["pra"] = pra
            return PRAEngineResult(
                result.text,
                raw,
                (*result.trace, {
                    "stage": "llama_cpp_live_prefix_full_continue",
                    "source_slot": source,
                    "request_slot": source,
                    "selected_kv_tokens": common,
                    "selected_records": len(ranges),
                    "selected_text_reencoded_tokens": 0,
                    "physical_kv_copy": False,
                    "full_retention": True,
                    "exact_live_prefix_continuation": True,
                }),
            )
        destination = (
            native.request_slot if native.request_slot != source else native.resource_slot
        )
        result = native.generate_live_prefix(
            request,
            prompt_suffix=list(full_tokens[common:]),
            plan=plan,
            request_slot=destination,
        )
        raw = dict(result.raw)
        raw.update(
            prefix_cache_enabled=True,
            prefix_cached_tokens=plan.selected_tokens,
            engine_cached_tokens_total=plan.selected_tokens,
            prefix_cache_hit=bool(plan.selected_tokens),
            native_attached_resources=[resource.resource_id for resource in request.resources],
        )
        return PRAEngineResult(result.text, raw, result.trace)

    @staticmethod
    def _resident_token_count(raw: Mapping[str, Any]) -> int:
        """Return tokens resident after generation, including attached K/V.

        llama.cpp reports the sampled stop/last token in ``predicted_n`` even
        though that token has not itself been evaluated into the KV sequence.
        Native requests report only the wire suffix as ``prompt_n``; their PRA
        metadata supplies the already-resident prefix length.
        """

        timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
        prompt = int(raw.get("tokens_evaluated", timings.get("prompt_n", 0)) or 0)
        predicted = int(raw.get("tokens_predicted", timings.get("predicted_n", 0)) or 0)
        pra = raw.get("pra") if isinstance(raw.get("pra"), Mapping) else {}
        native = int(pra.get("native_tokens", 0) or 0)
        return native + prompt + max(predicted - 1, 0)

    def _live_prefix_matches(self, request: PRAWireRequest, slot: int) -> bool:
        native = self.native_adapter.native_executor
        pair = native._causal_prompt_pair(request)
        if pair is None:
            return False
        tokens = tuple(int(token) for token in native._request_json(
            "/tokenize", {"content": pair[0], "add_special": True}
        ).get("tokens", ()))
        resident = self._live_session_tokens.get(str(request.session_id))
        if resident is None or tokens[:len(resident)] != resident:
            return False
        states = native._request_json("/slots")
        state = next((row for row in states if int(row["id"]) == int(slot)), None)
        return state is not None and not bool(state.get("is_processing", True))

    def _generate_from_live_slot(
        self, request: PRAWireRequest, source: int,
    ) -> tuple[PRAEngineResult, int]:
        """Continue the same live slot when selection retains its full prefix.

        Cross-slot ``seq_cp`` is a useful detached-resource primitive, but the
        quantized 30B agent can diverge after repeated slot hand-offs.  A 100%
        selection has no reason to move: submitting the complete logical prompt
        to the source slot lets llama.cpp reuse the exact live prefix-cache
        state and is the behaviorally matched pass-through implementation.
        """

        native = self.native_adapter.native_executor
        full_tokens = self._logical_prompt_tokens(request)
        resident = self._live_session_tokens[str(request.session_id)]
        # A normal turn extends the resident sequence and this tail includes
        # the client-visible boundary token that llama.cpp returned without
        # evaluating into KV. Agent parsers can also reject a sampled response
        # and append only a format-error message. In that case the logical
        # transcript backtracks before the rejected assistant output, so its
        # full prompt is not an extension of the live sequence. llama.cpp's
        # cache_prompt path can trim that stale suffix safely; account for the
        # exact common prefix instead of treating an empty append as fatal.
        common_prefix = 0
        for cached_token, prompt_token in zip(resident, full_tokens):
            if cached_token != prompt_token:
                break
            common_prefix += 1
        wire_tokens = list(full_tokens[common_prefix:])
        raw = dict(native._request_json(
            "/completion",
            {
                "prompt": list(full_tokens),
                "id_slot": source,
                "n_predict": request.resolved_max_new_tokens,
                "cache_prompt": True,
                "temperature": float(request.openai_fields.get("temperature", 0)),
                "seed": int(request.openai_fields.get("seed", 0)),
                "return_tokens": True,
            },
        ))
        cached = (raw.get("timings") or {}).get("cache_n")
        pra = {
            "native_tokens": cached,
            "wire_tokens": len(wire_tokens),
            "physical_kv_copy": False,
        }
        raw["pra"] = pra
        raw.update(
            prefix_cache_enabled=True,
            prefix_cached_tokens=cached,
            engine_cached_tokens_total=cached,
            prefix_cache_hit=bool(cached),
            native_attached_resources=[r.resource_id for r in request.resources],
        )
        return PRAEngineResult(
            str(raw.get("content", "")), raw,
            ({
                "stage": "llama_cpp_live_prefix_continue",
                "resource_slot": source,
                "request_slot": source,
                "native_tokens": pra.get("native_tokens"),
                "wire_tokens": pra.get("wire_tokens"),
                "physical_kv_copy": pra.get("physical_kv_copy"),
                "rematerialized": False,
            },),
        ), source

    def stream(self, request: PRAWireRequest):
        raise RuntimeError("Easy-50 llama.cpp endpoint is intentionally non-streaming")

    def close_session(self, session_id: str) -> None:
        slot = self._live_session_slots.pop(str(session_id), None)
        self._live_session_tokens.pop(str(session_id), None)
        self._logical_session_messages.pop(str(session_id), None)
        if slot is not None:
            self.native_adapter.native_executor._erase_request_slot(slot)
        self.native_adapter.close_session(session_id)


def _usage(raw: Mapping[str, Any]) -> dict[str, int] | None:
    existing = raw.get("usage")
    if isinstance(existing, Mapping):
        return dict(existing)
    timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
    prompt = raw.get("tokens_evaluated", timings.get("prompt_n"))
    completion = raw.get("tokens_predicted", timings.get("predicted_n"))
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _completion(
    request: PRAWireRequest, result: PRAEngineResult, *, native: bool,
) -> dict[str, Any]:
    raw = dict(result.raw)
    response = raw if isinstance(raw.get("choices"), list) else {
        "id": request.request_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
    }
    usage = _usage(raw)
    if usage is not None:
        response["usage"] = usage
    native_raw = raw.get("pra") if isinstance(raw.get("pra"), Mapping) else {}
    response["pra"] = {
        "native_kv": bool(native_raw.get("native_kv", native)),
        "prefix_cache_hit": raw.get("prefix_cache_hit"),
        "prefix_cached_tokens": raw.get("prefix_cached_tokens"),
        "engine_cached_tokens_total": raw.get("engine_cached_tokens_total"),
        "native_tokens": native_raw.get("native_tokens"),
        "wire_tokens": native_raw.get("wire_tokens"),
        "physical_kv_copy": native_raw.get("physical_kv_copy"),
        "resource_update_mode": raw.get("resource_update_mode"),
        "resource_prefix_cached_tokens": raw.get("resource_prefix_cached_tokens"),
        "resource_evaluated_tokens": raw.get("resource_evaluated_tokens"),
        "resource_total_tokens": raw.get("resource_total_tokens"),
    }
    response["pra_trace"] = list(result.trace)
    return response


def _direct_handler(adapter: HybridLlamaCppAdapter, model: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                capabilities = adapter.capabilities().to_dict()
                capabilities["prefix_cache_enabled"] = adapter.prefix_cache_enabled
                self._json(200, {
                    "status": "ok",
                    "endpoint_type": "engine",
                    "prefix_cache_enabled": adapter.prefix_cache_enabled,
                    "effective_capabilities": capabilities,
                    "engine": capabilities,
                })
            elif self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": model, "object": "model", "owned_by": "llama.cpp-pra"}],
                })
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not_found"})
                return
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                if payload.get("stream"):
                    raise ValueError("streaming is disabled for the locked Easy-50 endpoint")
                request = PRAWireRequest.from_openai(payload)
                native = bool(request.resources)
                result = adapter.generate(request)
                completion = _completion(request, result, native=native)
                if bool(request.metadata.get("ephemeral_session", False)):
                    adapter.close_session(str(request.session_id))
                self._json(200, completion)
            except (ValueError, TypeError, PermissionError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})
            except urllib.error.HTTPError as error:
                upstream_body = error.read().decode("utf-8", errors="replace")
                traceback.print_exc()
                self._json(error.code, {
                    "error": "upstream_http_error",
                    "message": str(error),
                    "upstream_body": upstream_body,
                })
            except Exception as error:
                traceback.print_exc()
                self._json(500, {"error": "engine_internal_error", "message": str(error)})

        def do_DELETE(self) -> None:  # noqa: N802
            prefix = "/v1/pra/sessions/"
            path = urllib.parse.urlsplit(self.path).path
            if not path.startswith(prefix):
                self._json(404, {"error": "not_found"})
                return
            session_id = urllib.parse.unquote(path[len(prefix):])
            if not session_id or "/" in session_id:
                self._json(400, {
                    "error": "invalid_session_id",
                    "message": "session id must be a non-empty path segment",
                })
                return
            try:
                adapter.close_session(session_id)
                self._json(200, {"closed": True, "session_id": session_id})
            except Exception as error:  # noqa: BLE001 - diagnostic server boundary
                traceback.print_exc()
                self._json(500, {
                    "error": "engine_internal_error", "message": str(error),
                })

        def log_message(self, format: str, *args: object) -> None:
            return None

    return Handler


def serve(args: argparse.Namespace) -> None:
    LlamaCppEngineAdapter, LlamaCppNativeServerExecutor = _load_llamacpp_types()
    AgentNativeExecutor = type(
        "AgentNativeExecutor",
        (CausalChatNativePromptMixin, LlamaCppNativeServerExecutor),
        {},
    )
    native_executor = AgentNativeExecutor(
        args.llama_url,
        resource_slot=args.resource_slot,
        request_slot=args.request_slot,
        model_fingerprint=args.model_fingerprint,
        timeout_seconds=args.timeout_seconds,
    )
    if args.reset_slots:
        try:
            native_executor._delete_resource(args.resource_slot)
        except urllib.error.HTTPError:
            pass
        native_executor._erase_request_slot(args.request_slot)
    native_adapter = LlamaCppEngineAdapter(
        args.llama_url,
        model_fingerprint=args.model_fingerprint,
        native_executor=native_executor,
        timeout_seconds=args.timeout_seconds,
    )
    adapter = HybridLlamaCppAdapter(
        native_adapter,
        PlainSlotExecutor(native_executor, prefix_caching=args.prefix_caching),
        prefix_caching=args.prefix_caching,
    )
    if args.mode in {"g00", "g11"}:
        serve_gateway(
            PRAGateway(adapter, mode=args.mode.upper(), models=(args.model,)),
            host=args.host,
            port=args.port,
        )
        return
    ThreadingHTTPServer(
        (args.host, args.port), _direct_handler(adapter, args.model)
    ).serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-url", default="http://127.0.0.1:18082")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("direct", "g00", "g11"), default="direct")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--resource-slot", type=int, default=0)
    parser.add_argument("--request-slot", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument(
        "--prefix-caching", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--reset-slots", action=argparse.BooleanOptionalAction, default=True,
        help="Clear the resource/request slots at arm startup to prevent cross-arm reuse.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
