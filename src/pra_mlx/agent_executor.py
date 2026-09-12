"""Direct no-gateway mlx-lm executor for resident agent-history K/V.

The executor keeps one canonical, post-RoPE K/V history per agent session.
Selected history is addressed as original-position views and is never
tokenized or evaluated again.  Request-local prompt/response K/V is grafted
back into the canonical history after the request releases its borrow; that
real device-side copy is reported separately from selection.

Sparse requests are fail-closed: before decoding, their first-step logits are
compared with a dense packed reference consuming the identical selected K/V
and original query position.  This is a mechanism gate, not an assertion that
the current portable MLX segmented attention kernel is already qualified.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pra_hf.agent_executor import (
    AgentHistoryLedger,
    _common_prefix,
    _render,
    enforce_retention_floor,
    selected_record_plan,
    split_generation_prompt,
)
from pra_hf.deployment import PRAEngineResult, PRAWireRequest

from .mlx_live_kv import MLXLiveKVRuntime
from .native import (
    MLXNativeLayerKV,
    MLXNativeMemory,
    capture_live_native_memory,
    make_native_prompt_cache,
)
from .qwen3_segmented import (
    install_qwen3_segmented_attention,
    set_qwen3_segmented_attention_active,
)


def _array_bytes(value: object) -> int:
    amount = getattr(value, "nbytes", None)
    if amount is not None:
        return int(amount)
    size = int(getattr(value, "size", 0))
    itemsize = int(getattr(getattr(value, "dtype", None), "itemsize", 0))
    return size * itemsize


def _max_abs_delta(left: object, right: object) -> float:
    """Synchronously reduce one same-subset logit comparison."""

    import mlx.core as mx

    delta = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(delta)
    item = getattr(delta, "item", None)
    return float(item() if callable(item) else delta)


@dataclass(frozen=True)
class MLXKVGraftMetrics:
    local_tokens: int
    canonical_suffix_graft_d2d_bytes: int
    canonical_reallocation_d2d_bytes: int

    @property
    def total_kv_copy_bytes(self) -> int:
        return (
            self.canonical_suffix_graft_d2d_bytes
            + self.canonical_reallocation_d2d_bytes
        )


@dataclass
class _Session:
    tenant_id: str
    session_id: str
    source_id: str
    ledger: AgentHistoryLedger = field(default_factory=AgentHistoryLedger)
    canonical_tokens: list[int] = field(default_factory=list)
    canonical_memory: MLXNativeMemory | None = None
    generation: int = 0
    calls: int = 0
    selected_history_reencoded_tokens: int = 0
    total_kv_copy_bytes: int = 0


class MLXAgentHistoryExecutor:
    """Stateful direct mlx-lm execution for the frozen Paper 4.5 gate."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        model_id: str,
        model_revision: str,
        wire_tail_tokens: int = 32,
        chat_template_profile: str = "native",
        chat_template_digest: str | None = None,
        max_abs_logit_delta: float = 0.005,
        require_same_subset_reference: bool = True,
        agent_history_qualified: bool = False,
        prefill_step_size: int = 2048,
        fused_disjoint_attention: bool = True,
    ) -> None:
        if wire_tail_tokens <= 0:
            raise ValueError("wire_tail_tokens must be positive.")
        if max_abs_logit_delta < 0:
            raise ValueError("max_abs_logit_delta cannot be negative.")
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be positive.")
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.wire_tail_tokens = int(wire_tail_tokens)
        self.chat_template_profile = str(chat_template_profile)
        selected = str(getattr(tokenizer, "chat_template", "") or "")
        observed = hashlib.sha256(selected.encode("utf-8")).hexdigest()
        if chat_template_digest is not None and observed != str(chat_template_digest):
            raise ValueError("Configured chat template digest does not match the tokenizer.")
        self.chat_template_digest = observed
        self.max_abs_logit_delta = float(max_abs_logit_delta)
        self.require_same_subset_reference = bool(require_same_subset_reference)
        self.agent_history_qualified = bool(agent_history_qualified)
        self.prefill_step_size = int(prefill_step_size)
        self.fused_disjoint_attention = bool(fused_disjoint_attention)
        self.runtime = MLXLiveKVRuntime()
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self.prefix_cache_enabled = False
        self.patched_layers = install_qwen3_segmented_attention(
            model, compiled=False, active=False
        )
        self._segmented_attention_active = False
        if self.agent_history_qualified and (
            self.patched_layers <= 0 or not self.require_same_subset_reference
        ):
            raise ValueError(
                "Qualified MLX agent history requires segmented attention and "
                "the per-request same-subset correctness gate."
            )

    def capabilities(self) -> Mapping[str, object]:
        return {
            "adapter": "mlx_lm_direct_agent",
            "integration_level": "E2",
            "engine_type": "mlx-lm",
            "native_kv": True,
            "session_state": True,
            "live_prefix_kv_capture": True,
            "live_prefix_kv_subset": True,
            "zero_selected_text_reencoding": True,
            "stable_record_kv_identity": True,
            "multiple_selected_records": True,
            "source_positions_preserved": True,
            "request_membership_attach": True,
            "request_lifetime": True,
            "streaming": False,
            "prefix_cache_enabled": False,
            "same_subset_reference_required": self.require_same_subset_reference,
            "same_subset_max_abs_logit_delta": self.max_abs_logit_delta,
            # Selection-time interval packing is zero in the disjoint path.
            # MLX slice aliasing and attention temporaries remain measured,
            # rather than being promoted to a blanket zero-copy claim.
            "selected_history_physical_copy_known": self.agent_history_qualified,
            "agent_history_kv_qualified": self.agent_history_qualified,
            "qualification_status": (
                "qualified model/profile with per-request same-subset enforcement"
                if self.agent_history_qualified
                else "per-request same-subset gate required"
            ),
            "segmented_attention_layers": self.patched_layers,
            "segmented_attention_dispatch": "sparse_only",
            "segmented_attention_active": self._segmented_attention_active,
            "fused_disjoint_attention": self.fused_disjoint_attention,
            "chat_template_profile": self.chat_template_profile,
            "chat_template_digest": self.chat_template_digest,
        }

    @staticmethod
    def _session_id(request: PRAWireRequest) -> str:
        if request.session_id:
            return str(request.session_id)
        seed = json.dumps(
            [dict(row) for row in request.messages[:2]],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return "plain-" + hashlib.sha256(seed).hexdigest()[:24]

    def _session(self, request: PRAWireRequest) -> _Session:
        session_id = self._session_id(request)
        state = self._sessions.get(session_id)
        if state is None:
            digest = hashlib.sha256(session_id.encode()).hexdigest()[:20]
            state = _Session(
                str(request.tenant_id),
                session_id,
                source_id=f"mlx-agent-source-{digest}",
            )
            self._sessions[session_id] = state
        elif state.tenant_id != str(request.tenant_id):
            raise RuntimeError("MLX agent session crossed tenant scope.")
        return state

    def _template_kwargs(self, request: PRAWireRequest) -> dict[str, Any]:
        value = request.openai_fields.get("chat_template_kwargs") or {}
        if not isinstance(value, Mapping):
            raise ValueError("chat_template_kwargs must be a mapping.")
        options = dict(value)
        if self.chat_template_profile == "qwen3-stable-no-thinking":
            if options.get("enable_thinking") not in {None, False}:
                raise ValueError("qwen3-stable-no-thinking rejects enable_thinking=true.")
            options["enable_thinking"] = False
        expected = request.metadata.get("chat_template_digest")
        if expected is not None and str(expected) != self.chat_template_digest:
            raise ValueError("Request chat template digest does not match the session.")
        return options

    def _eos_ids(self) -> set[int]:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            return set()
        if isinstance(value, (tuple, list, set)):
            return set(map(int, value))
        return {int(value)}

    @staticmethod
    def _selected_message_indices(request: PRAWireRequest) -> tuple[int, ...]:
        missing = [
            row.resource_id
            for row in request.resources
            if row.metadata.get("message_index") is None
        ]
        if missing:
            raise ValueError(
                "Live agent-history resources require message_index metadata: "
                + ", ".join(missing)
            )
        return tuple(sorted({
            int(row.metadata["message_index"])
            for row in request.resources
            if row.metadata.get("message_index") is not None
        }))

    def _new_cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def _set_segmented_attention(self, active: bool) -> None:
        enabled = bool(active)
        if enabled and self.patched_layers <= 0:
            raise RuntimeError("Sparse MLX attention is not installed.")
        set_qwen3_segmented_attention_active(self.model, enabled)
        self._segmented_attention_active = enabled

    def _evaluate(self, token_ids: Sequence[int], cache: Sequence[object]):
        import mlx.core as mx

        if not token_ids:
            raise ValueError("MLX model evaluation requires at least one token.")
        logits = self.model(mx.array([list(map(int, token_ids))], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        return logits

    def _prefill(self, token_ids: Sequence[int], cache: Sequence[object]) -> int:
        """Populate cache in bounded chunks without running the prompt LM head.

        mlx-lm decoder models expose their transformer backbone as ``model``.
        Calling that backbone directly updates the same prompt caches but avoids
        constructing vocabulary-sized logits for every prompt token.  Small
        test doubles and unusual integrations may not expose a callable
        backbone, so retain the full-model fallback for compatibility.
        """

        import mlx.core as mx

        calls = 0
        values = list(map(int, token_ids))
        backbone = getattr(self.model, "model", None)
        prefill_model = backbone if callable(backbone) else self.model
        for start in range(0, len(values), self.prefill_step_size):
            chunk = values[start : start + self.prefill_step_size]
            activations = prefill_model(
                mx.array([chunk], dtype=mx.int32), cache=cache
            )
            states = [getattr(row, "state", None) for row in cache]
            states = [row for row in states if row is not None]
            if states:
                mx.eval(states)
            else:
                # Test doubles and unusual cache implementations may not expose
                # state.  Keep their behavior correct without using this fallback
                # for normal mlx-lm prompt caches.
                mx.eval(activations)
            clear = getattr(mx, "clear_cache", None)
            if callable(clear):
                clear()
            calls += 1
        return calls

    def _capture(self, cache: Sequence[object], tokens: int) -> MLXNativeMemory:
        from pra_hf.live_history import LiveKVSelectionPlan

        return capture_live_native_memory(
            cache, LiveKVSelectionPlan.full(tokens)
        ).memory

    @staticmethod
    def _local_states(caches: Sequence[object]) -> tuple[tuple[object, object], ...]:
        result = []
        for wrapped in caches:
            local = getattr(wrapped, "local_cache", wrapped)
            state = getattr(local, "state", None)
            if not isinstance(state, tuple) or len(state) < 2:
                raise RuntimeError("MLX request-local attention cache has no K/V state.")
            result.append((state[0], state[1]))
        return tuple(result)

    def _graft(
        self,
        memory: MLXNativeMemory,
        request_caches: Sequence[object],
        *,
        expected_local_tokens: int,
    ) -> tuple[MLXNativeMemory, MLXKVGraftMetrics]:
        """Append local K/V and account for MLX concatenate materialization."""

        import mlx.core as mx

        states = self._local_states(request_caches)
        if len(states) != len(memory.layers):
            raise RuntimeError("MLX request/canonical layer counts disagree.")
        layers = []
        suffix_bytes = 0
        old_bytes = memory.nbytes
        for prior, (keys, values) in zip(memory.layers, states):
            local_tokens = int(keys.shape[2])
            if local_tokens != int(values.shape[2]):
                raise RuntimeError("MLX request-local K/V lengths disagree.")
            if local_tokens != expected_local_tokens:
                raise RuntimeError(
                    "MLX local K/V length disagrees with the committed token suffix: "
                    f"cache={local_tokens}, expected={expected_local_tokens}."
                )
            suffix_bytes += _array_bytes(keys) + _array_bytes(values)
            layers.append(MLXNativeLayerKV(
                mx.concatenate((prior.keys, keys), axis=2),
                mx.concatenate((prior.values, values), axis=2),
            ))
        result = MLXNativeMemory(
            tuple(layers), memory.source_tokens + expected_local_tokens
        )
        mx.eval(*(value for layer in result.layers for value in (layer.keys, layer.values)))
        return result, MLXKVGraftMetrics(
            expected_local_tokens,
            suffix_bytes,
            old_bytes,
        )

    def _ensure_source(
        self, state: _Session, source: list[int]
    ) -> tuple[int, int, MLXKVGraftMetrics | None]:
        """Build once, then extend canonical K/V using only the new suffix."""

        # Canonical history construction and extension must use MLX-LM's
        # original attention path.  Segmented attention is a sparse consumer,
        # not a replacement for ordinary prefill or PRA-100.
        self._set_segmented_attention(False)
        if state.canonical_memory is None:
            cache = self._new_cache()
            self._prefill(source, cache)
            state.canonical_memory = self._capture(cache, len(source))
            state.canonical_tokens = list(source)
            return len(source), 0, None

        common = _common_prefix(state.canonical_tokens, source)
        rewritten = min(len(state.canonical_tokens), len(source)) - common
        if rewritten or common != len(state.canonical_tokens):
            raise RuntimeError(
                "The chat template rewrote or truncated already evaluated agent "
                f"history ({max(rewritten, len(state.canonical_tokens) - common)} tokens). "
                "Resident live-K/V requires an append-stable template and will "
                "not fall back to re-prefill."
            )
        delta = source[common:]
        if not delta:
            return 0, 0, None
        caches = make_native_prompt_cache(
            self.model,
            state.canonical_memory,
            segmented=False,
            query_position_base=len(state.canonical_tokens),
        )
        self._prefill(delta, caches)
        updated, metrics = self._graft(
            state.canonical_memory, caches, expected_local_tokens=len(delta)
        )
        state.canonical_memory = updated
        state.canonical_tokens = list(source)
        state.total_kv_copy_bytes += metrics.total_kv_copy_bytes
        return len(delta), 0, metrics

    def _ordinary_generate(
        self, prompt: list[int], max_tokens: int
    ) -> tuple[str, list[int], float, int]:
        started = time.perf_counter()
        cache = self._new_cache()
        calls = self._prefill(prompt[:-1], cache)
        logits = self._evaluate(prompt[-1:], cache)
        generated: list[int] = []
        eos = self._eos_ids()
        calls += 1
        while len(generated) < max_tokens:
            token = int(__import__("mlx.core", fromlist=["argmax"]).argmax(
                logits[0, -1]
            ).item())
            if token in eos:
                break
            generated.append(token)
            if len(generated) >= max_tokens:
                break
            logits = self._evaluate([token], cache)
            calls += 1
        text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
        return text, generated, time.perf_counter() - started, calls

    def _native_generate(
        self, request: PRAWireRequest, state: _Session
    ) -> PRAEngineResult:
        import mlx.core as mx

        live_projection = request.metadata.get("history_projection") == "live-agent-kv-v1"
        if live_projection and not request.metadata.get("logical_message_manifest"):
            raise ValueError("Live agent-history PRA requires a complete logical message manifest.")
        messages = state.ledger.reconcile(request)
        template_kwargs = self._template_kwargs(request)
        prompt, source, wire = split_generation_prompt(
            self.tokenizer,
            messages,
            chat_template_kwargs=template_kwargs,
        )
        newly_encoded, reencoded, extension_graft = self._ensure_source(state, source)
        assert state.canonical_memory is not None

        requested = float(request.metadata.get(
            "target_retention_fraction",
            request.metadata.get("budget_fraction", 1.0),
        ))
        mandatory = tuple(map(int, request.metadata.get("mandatory_message_indices", ())))
        selected = self._selected_message_indices(request) if live_projection else ()
        plan = selected_record_plan(
            self.tokenizer,
            messages,
            prompt,
            source_tokens=len(source),
            retention_fraction=requested,
            mandatory_message_indices=mandatory,
            selected_message_indices=selected,
            chat_template_kwargs=template_kwargs,
            round_up_to_retention_floor=(
                request.metadata.get("selection_budget_policy")
                == "causal_bundle_round_up_v1"
            ),
        )
        contract = request.metadata.get("selection_contract")
        enforce_retention_floor(
            plan,
            requested,
            selection_contract=None if contract is None else str(contract),
        )
        if not plan.full_retention and self.patched_layers <= 0:
            raise RuntimeError(
                "Sparse MLX agent history requires an installed segmented "
                "original-position attention consumer."
            )

        state.generation += 1
        self.runtime.register_source(
            state.source_id,
            state.canonical_memory,
            tenant_id=request.tenant_id,
            session_id=state.session_id,
            generation=state.generation,
        )
        request_id = str(request.request_id)
        use_segmented = bool(self.patched_layers) and not plan.full_retention
        candidate = self.runtime.begin_request(
            request_id,
            state.source_id,
            plan,
            tenant_id=request.tenant_id,
            session_id=state.session_id,
            expected_generation=state.generation,
            segmented=use_segmented,
            disjoint_selection=use_segmented,
        )
        candidate_cache = make_native_prompt_cache(
            self.model,
            candidate.selection.memory,
            segmented=use_segmented,
            fused_disjoint_attention=self.fused_disjoint_attention,
            query_position_base=plan.source_position_base,
        )
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
        active_before = int(getattr(mx, "get_active_memory", lambda: 0)())
        started = time.perf_counter()
        reference = None
        reference_pack_bytes = 0
        logit_delta: float | None = None
        generated: list[int] = []
        terminal: int | None = None
        calls = 0
        outcome = "error"
        updated_memory: MLXNativeMemory | None = None
        graft_metrics: MLXKVGraftMetrics | None = None
        try:
            self._set_segmented_attention(use_segmented)
            logits = self._evaluate(wire, candidate_cache)
            calls += 1
            if not plan.full_retention and self.require_same_subset_reference:
                reference = self.runtime.begin_request(
                    request_id + "-same-subset-reference",
                    state.source_id,
                    plan,
                    tenant_id=request.tenant_id,
                    session_id=state.session_id,
                    expected_generation=state.generation,
                    segmented=False,
                    disjoint_selection=False,
                )
                reference_cache = make_native_prompt_cache(
                    self.model,
                    reference.selection.memory,
                    segmented=False,
                    query_position_base=plan.source_position_base,
                )
                reference_logits = self._evaluate(wire, reference_cache)
                reference_pack_bytes = (
                    reference.selection.memory.nbytes
                    if reference.selection.physical_kv_copy
                    else 0
                )
                logit_delta = _max_abs_delta(
                    logits[0, -1], reference_logits[0, -1]
                )
                reference.finish()
                reference = None
                if logit_delta > self.max_abs_logit_delta:
                    raise RuntimeError(
                        "MLX sparse same-subset correctness gate failed: "
                        f"max_abs_logit_delta={logit_delta:.9g}, "
                        f"limit={self.max_abs_logit_delta:.9g}."
                    )

            eos = self._eos_ids()
            while len(generated) < request.resolved_max_new_tokens:
                token = int(mx.argmax(logits[0, -1]).item())
                if token in eos:
                    terminal = token
                    logits = self._evaluate([token], candidate_cache)
                    calls += 1
                    break
                generated.append(token)
                logits = self._evaluate([token], candidate_cache)
                calls += 1
                if len(generated) >= request.resolved_max_new_tokens:
                    break

            committed = len(wire) + len(generated) + int(terminal is not None)
            updated_memory, graft_metrics = self._graft(
                state.canonical_memory,
                candidate_cache,
                expected_local_tokens=committed,
            )
            outcome = "finished"
        finally:
            try:
                if reference is not None:
                    reference.fail()
                if outcome == "finished":
                    candidate.finish()
                else:
                    candidate.fail()
            finally:
                self._set_segmented_attention(False)
        assert updated_memory is not None and graft_metrics is not None
        state.canonical_memory = updated_memory
        state.canonical_tokens = [*source, *wire, *generated]
        if terminal is not None:
            state.canonical_tokens.append(terminal)
        state.generation += 1
        self.runtime.register_source(
            state.source_id,
            state.canonical_memory,
            tenant_id=request.tenant_id,
            session_id=state.session_id,
            generation=state.generation,
        )
        state.selected_history_reencoded_tokens += reencoded
        state.total_kv_copy_bytes += graft_metrics.total_kv_copy_bytes
        state.calls += 1
        text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
        state.ledger.append_assistant(text)

        active_after = int(getattr(mx, "get_active_memory", lambda: 0)())
        peak = int(getattr(mx, "get_peak_memory", lambda: 0)())
        consumer_active_delta = max(active_after - active_before, 0)
        consumer_peak_delta = max(peak - active_before, 0)
        elapsed = time.perf_counter() - started
        extension_copy = (
            0 if extension_graft is None else extension_graft.total_kv_copy_bytes
        )
        selected_pack = (
            candidate.selection.memory.nbytes
            if candidate.selection.physical_kv_copy
            else 0
        )
        trace = {
            "stage": "native_attach",
            "engine": "mlx-lm",
            "native_kv_used": True,
            "source_tokens": len(source),
            "selected_kv_tokens": plan.selected_tokens,
            "wire_tokens": len(wire),
            "logical_prompt_tokens": len(prompt),
            "effective_attention_prompt_tokens": plan.selected_tokens + len(wire),
            "completion_tokens": len(generated),
            "requested_retention_fraction": requested,
            "realized_retention_fraction": (
                plan.selected_tokens / max(len(source), 1)
            ),
            "realized_historical_kv_retention_fraction": (
                plan.selected_tokens / max(len(source), 1)
            ),
            "engine_reported_history_kv_retention_fraction": (
                plan.selected_tokens / max(len(source), 1)
            ),
            "full_retention": bool(plan.full_retention),
            "selection_contract": contract or "minimum-retention-floor",
            "pra_100_semantic_noop": bool(plan.full_retention),
            "exact_trajectory_eligible": bool(plan.full_retention),
            "selected_history_reencoded_tokens": reencoded,
            "new_history_encoded_tokens": newly_encoded,
            # The selection object can prove whether PRA explicitly packed
            # intervals, but it cannot prove whether MLX materialized a lazy
            # slice or an attention-sized transient.  Keep the end-to-end
            # physical-copy verdict unknown until hardware allocation
            # qualification covers both stages.
            "physical_kv_copy": (
                False
                if plan.full_retention or self.agent_history_qualified
                else None
            ),
            "selection_physical_kv_copy": candidate.selection.physical_kv_copy,
            "selected_interval_pack_bytes": selected_pack,
            "selected_history_kv_copy_bytes": (
                0
                if plan.full_retention or self.agent_history_qualified
                else None
            ),
            "physical_kv_copy_bytes": (
                0
                if plan.full_retention or self.agent_history_qualified
                else None
            ),
            "canonical_extension_copy_bytes": extension_copy,
            "canonical_suffix_graft_d2d_bytes": (
                graft_metrics.canonical_suffix_graft_d2d_bytes
            ),
            "canonical_reallocation_d2d_bytes": (
                graft_metrics.canonical_reallocation_d2d_bytes
            ),
            "known_total_kv_copy_bytes": (
                extension_copy + graft_metrics.total_kv_copy_bytes + selected_pack
            ),
            "total_kv_copy_bytes": (
                extension_copy
                + graft_metrics.total_kv_copy_bytes
                + selected_pack
                + reference_pack_bytes
                if plan.full_retention or self.agent_history_qualified
                else None
            ),
            "host_to_device_bytes": 0,
            "consumer_temporary_active_delta_bytes": consumer_active_delta,
            "consumer_temporary_bytes": consumer_peak_delta,
            "consumer_temporary_peak_bytes": consumer_peak_delta,
            "consumer_temporary_measurement_scope": (
                "whole native request including same-subset reference and canonical graft"
            ),
            "same_subset_reference_required": (
                bool(self.require_same_subset_reference) and not plan.full_retention
            ),
            "same_subset_reference_pack_bytes": reference_pack_bytes,
            "same_subset_max_abs_logit_delta": logit_delta,
            "same_subset_gate_limit": self.max_abs_logit_delta,
            "fused_disjoint_attention": self.fused_disjoint_attention,
            "segmented_attention_dispatch": "sparse_only",
            "attention_path": (
                "segmented_sparse" if use_segmented else "canonical_native"
            ),
            "segmented_attention_active": use_segmented,
            "same_subset_gate_passed": (
                None if plan.full_retention else logit_delta is not None
                and logit_delta <= self.max_abs_logit_delta
            ),
            "source_position_base": plan.source_position_base,
            "selection_plan": plan.to_dict(),
            "model_calls": calls,
            "elapsed_seconds": elapsed,
            "chat_template_profile": self.chat_template_profile,
            "chat_template_digest": self.chat_template_digest,
        }
        return PRAEngineResult(
            text=text,
            raw={
                "native_attach_bytes": 0,
                "prefix_cache_hit": bool(len(source) - newly_encoded),
                "prefix_cached_tokens": len(source) - newly_encoded,
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(generated),
                    "total_tokens": len(prompt) + len(generated),
                },
                "pra": dict(trace),
            },
            trace=(trace,),
        )

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        with self._lock:
            # Recover the canonical engine path even after a failed sparse
            # request; this also keeps plain requests free of wrapper overhead.
            self._set_segmented_attention(False)
            native = (
                request.metadata.get("history_projection") == "live-agent-kv-v1"
                or bool(request.resources)
            )
            if not native:
                prompt = _render(
                    self.tokenizer,
                    request.messages,
                    generation_prompt=True,
                    chat_template_kwargs=self._template_kwargs(request),
                )
                text, generated, elapsed, calls = self._ordinary_generate(
                    prompt, request.resolved_max_new_tokens
                )
                return PRAEngineResult(
                    text=text,
                    raw={
                        "usage": {
                            "prompt_tokens": len(prompt),
                            "completion_tokens": len(generated),
                            "total_tokens": len(prompt) + len(generated),
                        },
                        "pra": {"native_kv_used": False},
                    },
                    trace=({
                        "stage": "plain_generation",
                        "engine": "mlx-lm",
                        "native_kv_used": False,
                        "prompt_tokens": len(prompt),
                        "completion_tokens": len(generated),
                        "model_calls": calls,
                        "elapsed_seconds": elapsed,
                        "segmented_attention_dispatch": "sparse_only",
                        "attention_path": "canonical_native",
                        "segmented_attention_active": False,
                        "chat_template_profile": self.chat_template_profile,
                        "chat_template_digest": self.chat_template_digest,
                    },),
                )
            return self._native_generate(request, self._session(request))

    def stream(self, request: PRAWireRequest):
        raise RuntimeError("Frozen MLX agent gate is intentionally non-streaming.")

    def close_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(str(session_id), None)
            if state is not None:
                self.runtime.terminate_session(state.tenant_id, state.session_id)

    def close(self) -> None:
        with self._lock:
            for state in tuple(self._sessions.values()):
                self.runtime.terminate_session(state.tenant_id, state.session_id)
            self._sessions.clear()
            self.runtime.registry.close()
