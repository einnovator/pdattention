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
    positioned_materialized_history,
    selected_record_plan,
)
from pra_hf.deployment import PRAEngineResult, PRAWireRequest
from pra_hf.live_history import LiveKVInterval

from .mlx_live_kv import MLXLiveKVRuntime, _set_cache_query_start
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


def _argmax_token(logits: object) -> int:
    """Synchronously resolve one next-token decision for gate diagnostics."""

    import mlx.core as mx

    token = mx.argmax(logits)
    mx.eval(token)
    item = getattr(token, "item", None)
    return int(item() if callable(item) else token)


def _render_text(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    *,
    generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    rendered = tokenizer.apply_chat_template(
        [dict(row) for row in messages],
        tokenize=False,
        add_generation_prompt=generation_prompt,
        **dict(chat_template_kwargs or {}),
    )
    if not isinstance(rendered, str):
        raise RuntimeError("MLX live history requires a textual chat-template rendering.")
    return rendered


def _encode_text(tokenizer: object, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(map(int, encoded))


def _decode_exact(tokenizer: object, token_ids: Sequence[int]) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(map(int, token_ids)), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(map(int, token_ids)), **kwargs))


def _incremental_generation_prompt(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    *,
    canonical_text: str,
    canonical_tokens: Sequence[int],
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int], list[int], str]:
    """Extend live generated tokens from text without re-tokenizing them.

    Decode followed by encode is not an identity for every tokenizer.  A live
    cache owns the token IDs actually sampled by the model, so later requests
    append only the template text following that exact decoded prefix.
    """

    source_text = _render_text(
        tokenizer,
        messages,
        generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_text = _render_text(
        tokenizer,
        messages,
        generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not prompt_text.startswith(source_text):
        raise RuntimeError("The chat template generation prompt rewrites completed text.")
    wire = _encode_text(tokenizer, prompt_text[len(source_text) :])
    if not wire:
        raise RuntimeError("The chat template produced no generation-prompt suffix.")
    if canonical_text:
        if not source_text.startswith(canonical_text):
            raise RuntimeError(
                "The append-stable template rewrote resident logical history text."
            )
        source = [
            *map(int, canonical_tokens),
            *_encode_text(tokenizer, source_text[len(canonical_text) :]),
        ]
    else:
        source = _encode_text(tokenizer, source_text)
    return [*source, *wire], source, wire, source_text


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
    canonical_text: str = ""
    message_spans: dict[int, LiveKVInterval] = field(default_factory=dict)
    canonical_memory: MLXNativeMemory | None = None
    full_reference_cache: Sequence[object] | None = None
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
        require_full_retention_reference: bool = False,
        agent_history_qualified: bool = False,
        prefill_step_size: int = 2048,
        fused_disjoint_attention: bool = True,
        max_model_len: int = 8192,
    ) -> None:
        if wire_tail_tokens <= 0:
            raise ValueError("wire_tail_tokens must be positive.")
        if max_abs_logit_delta < 0:
            raise ValueError("max_abs_logit_delta cannot be negative.")
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be positive.")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive.")
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
        self.require_full_retention_reference = bool(
            require_full_retention_reference
        )
        self.agent_history_qualified = bool(agent_history_qualified)
        self.prefill_step_size = int(prefill_step_size)
        self.fused_disjoint_attention = bool(fused_disjoint_attention)
        self.max_model_len = int(max_model_len)
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
            "full_retention_prefix_cache_reference_required": (
                self.require_full_retention_reference
            ),
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
            "max_model_len": self.max_model_len,
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
        tenant_id = str(request.tenant_id)
        self.runtime.registry.assert_session_active(tenant_id, session_id)
        state = self._sessions.get(session_id)
        if state is None:
            digest = hashlib.sha256(session_id.encode()).hexdigest()[:20]
            state = _Session(
                tenant_id,
                session_id,
                source_id=f"mlx-agent-source-{digest}",
            )
            self._sessions[session_id] = state
        elif state.tenant_id != tenant_id:
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

    def _enforce_context_window(
        self, prompt_tokens: int, max_new_tokens: int
    ) -> None:
        """Reject an over-limit logical request before allocating or mutating K/V."""

        requested = int(prompt_tokens) + int(max_new_tokens)
        if requested > self.max_model_len:
            raise ValueError(
                "MLX request exceeds the configured context window: "
                f"prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}, "
                f"requested_total={requested}, max_model_len={self.max_model_len}."
            )

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

    @staticmethod
    def _causal_group(
        messages: Sequence[Mapping[str, Any]], index: int
    ) -> str:
        if index <= 1:
            return "preamble"
        for prior in range(index, -1, -1):
            if str(messages[prior].get("role", "")) == "assistant":
                return f"turn:{prior}"
        return "preamble"

    def _update_message_spans(
        self,
        state: _Session,
        messages: Sequence[Mapping[str, Any]],
        source: Sequence[int],
        *,
        source_text: str,
        chat_template_kwargs: Mapping[str, Any],
    ) -> None:
        existing = len(state.message_spans)
        if existing > len(messages):
            raise RuntimeError("Logical history truncated resident message spans.")
        prior_end = max(
            (span.end for span in state.message_spans.values()), default=0
        )
        base_text = state.canonical_text if existing else ""
        base_tokens = len(state.canonical_tokens) if existing else 0
        for index in range(existing, len(messages)):
            boundary_text = _render_text(
                self.tokenizer,
                messages[: index + 1],
                generation_prompt=False,
                chat_template_kwargs=chat_template_kwargs,
            )
            if not boundary_text.startswith(base_text):
                raise RuntimeError("Chat template rewrote resident message text.")
            boundary = base_tokens + len(
                _encode_text(self.tokenizer, boundary_text[len(base_text) :])
            )
            state.message_spans[index] = LiveKVInterval(
                prior_end,
                boundary,
                record_id=f"message:{index}:{messages[index].get('role', 'unknown')}",
                causal_group_id=self._causal_group(messages, index),
            )
            prior_end = boundary
        if prior_end != len(source) or not source_text:
            raise RuntimeError(
                "Incremental MLX message spans disagree with canonical source tokens."
            )

    def _fresh_message_spans(
        self,
        messages: Sequence[Mapping[str, Any]],
        source: Sequence[int],
        *,
        chat_template_kwargs: Mapping[str, Any],
    ) -> tuple[LiveKVInterval, ...]:
        """Resolve immutable record spans for a fresh full-history prefill.

        MLX kernels may take slightly different numerical paths for different
        query batch shapes.  Splitting at stable message boundaries makes a
        fresh replay and an append-only resident session evaluate every closed
        record with the same batch partition.
        """

        spans: list[LiveKVInterval] = []
        prior_end = 0
        for index in range(len(messages)):
            boundary_text = _render_text(
                self.tokenizer,
                messages[: index + 1],
                generation_prompt=False,
                chat_template_kwargs=chat_template_kwargs,
            )
            boundary = len(_encode_text(self.tokenizer, boundary_text))
            if boundary < prior_end:
                raise RuntimeError("Chat-template message boundary moved backwards.")
            spans.append(LiveKVInterval(
                prior_end,
                boundary,
                record_id=f"message:{index}:{messages[index].get('role', 'unknown')}",
                causal_group_id=self._causal_group(messages, index),
            ))
            prior_end = boundary
        if prior_end != len(source):
            raise RuntimeError(
                "Fresh MLX message spans disagree with canonical source tokens."
            )
        return tuple(spans)

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

    def _materialized_prefill_step(self) -> int:
        """Stay on MLX's vector-SDPA path for positioned compact records."""

        args = getattr(self.model, "args", None)
        query_heads = int(getattr(args, "num_attention_heads", 0) or 0)
        kv_heads = int(getattr(args, "num_key_value_heads", 0) or 0)
        group_limit = 8
        if query_heads > 0 and kv_heads > 0 and query_heads % kv_heads == 0:
            group_limit = max(1, 32 // (query_heads // kv_heads))
        # MLX routes query widths above eight to its full-attention kernel.
        # The interval-addressed consumer mirrors vector SDPA, so both the
        # candidate and packed oracle intentionally prefill receipts in the
        # same vector-sized chunks.
        return max(1, min(self.prefill_step_size, 8, group_limit))

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

    def _prefill_record_aligned(
        self,
        token_ids: Sequence[int],
        cache: Sequence[object],
        spans: Sequence[LiveKVInterval],
        *,
        start: int = 0,
    ) -> int:
        """Prefill closed records with append-stable batching.

        Chunking restarts at every logical record.  A later record therefore
        cannot change the query shape used to compute K/V for an earlier one.
        Gaps are retained fail-closed as their own segment rather than dropped.
        """

        values = list(map(int, token_ids))
        cursor = int(start)
        if cursor < 0 or cursor > len(values):
            raise ValueError("Record-aligned prefill start is outside the source.")
        calls = 0
        for span in sorted(spans, key=lambda row: (row.start, row.end)):
            if span.end <= cursor:
                continue
            if span.start > cursor:
                calls += self._prefill(values[cursor:span.start], cache)
                cursor = span.start
            begin = max(cursor, span.start)
            end = min(int(span.end), len(values))
            if end > begin:
                calls += self._prefill(values[begin:end], cache)
                cursor = end
        if cursor < len(values):
            calls += self._prefill(values[cursor:], cache)
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
        ignored_leading_local_tokens: int = 0,
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
            expected_total = (
                int(ignored_leading_local_tokens) + int(expected_local_tokens)
            )
            if local_tokens != expected_total:
                raise RuntimeError(
                    "MLX local K/V length disagrees with the committed token suffix: "
                    f"cache={local_tokens}, expected={expected_total}."
                )
            if ignored_leading_local_tokens:
                keys = keys[:, :, int(ignored_leading_local_tokens) :, :]
                values = values[:, :, int(ignored_leading_local_tokens) :, :]
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
            self._prefill_record_aligned(
                source, cache, tuple(state.message_spans.values())
            )
            state.canonical_memory = self._capture(cache, len(source))
            if self.require_full_retention_reference:
                # Keep the original mlx-lm cache as the zero-selection
                # prefix-cache control.  PRA borrows an immutable view of the
                # same source K/V; subsequent request-local tokens advance the
                # two consumers independently.
                state.full_reference_cache = cache
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
        self._prefill_record_aligned(
            source,
            caches,
            tuple(state.message_spans.values()),
            start=common,
        )
        updated, metrics = self._graft(
            state.canonical_memory, caches, expected_local_tokens=len(delta)
        )
        if state.full_reference_cache is not None:
            reference_offset = int(state.full_reference_cache[0].offset)
            if reference_offset != common:
                raise RuntimeError(
                    "MLX prefix-cache reference lost canonical alignment: "
                    f"offset={reference_offset}, expected={common}."
                )
            self._prefill_record_aligned(
                source,
                state.full_reference_cache,
                tuple(state.message_spans.values()),
                start=common,
            )
        state.canonical_memory = updated
        state.canonical_tokens = list(source)
        state.total_kv_copy_bytes += metrics.total_kv_copy_bytes
        return len(delta), 0, metrics

    def _ordinary_generate(
        self, prompt: list[int], max_tokens: int
    ) -> tuple[str, list[int], float, int]:
        self._enforce_context_window(len(prompt), max_tokens)
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

    def _ordinary_generate_record_aligned(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        chat_template_kwargs: Mapping[str, Any],
        max_tokens: int,
    ) -> tuple[str, list[int], float, int]:
        """Fresh FULL control using the same immutable record partition."""

        import mlx.core as mx

        prompt, source, wire, _ = _incremental_generation_prompt(
            self.tokenizer,
            messages,
            canonical_text="",
            canonical_tokens=(),
            chat_template_kwargs=chat_template_kwargs,
        )
        self._enforce_context_window(len(source) + len(wire), max_tokens)
        spans = self._fresh_message_spans(
            messages, source, chat_template_kwargs=chat_template_kwargs
        )
        started = time.perf_counter()
        cache = self._new_cache()
        calls = self._prefill_record_aligned(source, cache, spans)
        logits = self._evaluate(wire, cache)
        calls += 1
        generated: list[int] = []
        eos = self._eos_ids()
        while len(generated) < max_tokens:
            token = int(mx.argmax(logits[0, -1]).item())
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
        prompt, source, wire, source_text = _incremental_generation_prompt(
            self.tokenizer,
            messages,
            canonical_text=state.canonical_text,
            canonical_tokens=state.canonical_tokens,
            chat_template_kwargs=template_kwargs,
        )
        self._enforce_context_window(
            len(source) + len(wire), request.resolved_max_new_tokens
        )
        self._update_message_spans(
            state,
            messages,
            source,
            source_text=source_text,
            chat_template_kwargs=template_kwargs,
        )
        newly_encoded, reencoded, extension_graft = self._ensure_source(state, source)
        state.canonical_text = source_text
        assert state.canonical_memory is not None

        source_bootstrap = bool(
            state.calls == 0
            and request.metadata.get("source_bootstrap_contract")
            == "full-logical-history-once-v1"
        )
        requested = float(request.metadata.get(
            "target_retention_fraction",
            request.metadata.get("budget_fraction", 1.0),
        ))
        effective_requested = requested
        mandatory = tuple(map(int, request.metadata.get("mandatory_message_indices", ())))
        selected = self._selected_message_indices(request) if live_projection else ()
        materialized = ()
        if live_projection:
            materialized = positioned_materialized_history(
                self.tokenizer,
                request,
                messages,
                prompt,
                source_tokens=len(source),
                chat_template_kwargs=template_kwargs,
            )
            replaced = {row.message_index for row in materialized}
            selected = tuple(index for index in selected if index not in replaced)
        plan = selected_record_plan(
            self.tokenizer,
            messages,
            prompt,
            source_tokens=len(source),
            retention_fraction=effective_requested,
            mandatory_message_indices=mandatory,
            selected_message_indices=selected,
            chat_template_kwargs=template_kwargs,
            round_up_to_retention_floor=(
                request.metadata.get("selection_budget_policy")
                == "causal_bundle_round_up_v1"
            ),
            resident_spans=tuple(state.message_spans.values()),
        )
        contract = request.metadata.get("selection_contract")
        enforce_retention_floor(
            plan,
            effective_requested,
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
        same_subset_reference_kind: str | None = None
        logit_delta: float | None = None
        same_subset_candidate_token: int | None = None
        same_subset_reference_token: int | None = None
        full_reference_cache: Sequence[object] | None = None
        full_reference_logits: object | None = None
        full_reference_max_delta: float | None = None
        full_reference_compared_tokens = 0
        full_reference_calls = 0
        generated: list[int] = []
        terminal: int | None = None
        calls = 0
        materialized_calls = 0
        materialized_tokens = sum(row.tokens for row in materialized)
        materialized_prefill_step = self._materialized_prefill_step()
        outcome = "error"
        updated_memory: MLXNativeMemory | None = None
        graft_metrics: MLXKVGraftMetrics | None = None
        try:
            self._set_segmented_attention(use_segmented)
            for row in materialized:
                for offset in range(
                    0, row.tokens, materialized_prefill_step
                ):
                    _set_cache_query_start(
                        candidate_cache, row.position_start + offset
                    )
                    self._evaluate(
                        row.token_ids[
                            offset : offset + materialized_prefill_step
                        ],
                        candidate_cache,
                    )
                    calls += 1
                    materialized_calls += 1
            _set_cache_query_start(candidate_cache, plan.source_position_base)
            logits = self._evaluate(wire, candidate_cache)
            calls += 1
            if plan.full_retention and self.require_full_retention_reference:
                # Qualification-only control: advance a standard mlx-lm
                # prefix cache built from the identical record-aligned source
                # beside resident PRA K/V for every decoded token.
                full_reference_cache = state.full_reference_cache
                if full_reference_cache is None:
                    raise RuntimeError(
                        "MLX full-retention prefix-cache reference is missing."
                    )
                full_reference_logits = self._evaluate(
                    wire, full_reference_cache
                )
                full_reference_calls += 1
            elif self.require_full_retention_reference:
                raise RuntimeError(
                    "MLX full-retention reference mode cannot execute a "
                    "reduced-history plan."
                )
            if not plan.full_retention and self.require_same_subset_reference:
                # A positioned compact record must not attend selected source
                # K/V that originally followed it.  Preserve those original
                # causal coordinates in an independently packed identical-
                # subset reference consumed by MLX's native dense attention.
                # Requests without positioned materialization retain the
                # ordinary packed reference used by the original gate.
                positioned_reference = bool(materialized)
                same_subset_reference_kind = (
                    "packed_original_position_mlx_prompt_cache"
                    if positioned_reference
                    else "packed_mlx_prompt_cache"
                )
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
                    causal_intervals=(
                        tuple((row.start, row.end) for row in plan.intervals)
                        if positioned_reference
                        else ()
                    ),
                )
                for row in materialized:
                    for offset in range(
                        0, row.tokens, materialized_prefill_step
                    ):
                        _set_cache_query_start(
                            reference_cache, row.position_start + offset
                        )
                        self._evaluate(
                            row.token_ids[
                                offset : offset + materialized_prefill_step
                            ],
                            reference_cache,
                        )
                _set_cache_query_start(reference_cache, plan.source_position_base)
                reference_logits = self._evaluate(wire, reference_cache)
                reference_pack_bytes = (
                    reference.selection.memory.nbytes
                    if reference.selection.physical_kv_copy
                    else 0
                )
                logit_delta = _max_abs_delta(
                    logits[0, -1], reference_logits[0, -1]
                )
                same_subset_candidate_token = _argmax_token(logits[0, -1])
                same_subset_reference_token = _argmax_token(
                    reference_logits[0, -1]
                )
                reference.finish()
                reference = None
                if logit_delta > self.max_abs_logit_delta:
                    raise RuntimeError(
                        "MLX sparse same-subset correctness gate failed: "
                        f"max_abs_logit_delta={logit_delta:.9g}, "
                        f"limit={self.max_abs_logit_delta:.9g}, "
                        f"candidate_token={same_subset_candidate_token}, "
                        f"reference_token={same_subset_reference_token}."
                    )

            eos = self._eos_ids()
            while len(generated) < request.resolved_max_new_tokens:
                token = int(mx.argmax(logits[0, -1]).item())
                if full_reference_logits is not None:
                    reference_token = int(
                        mx.argmax(full_reference_logits[0, -1]).item()
                    )
                    step_delta = _max_abs_delta(
                        logits[0, -1], full_reference_logits[0, -1]
                    )
                    full_reference_max_delta = max(
                        full_reference_max_delta or 0.0, step_delta
                    )
                    output_index = full_reference_compared_tokens
                    full_reference_compared_tokens += 1
                    if (
                        reference_token != token
                        or step_delta > self.max_abs_logit_delta
                    ):
                        raise RuntimeError(
                            "MLX full-retention prefix-cache correctness gate "
                            "failed: "
                            f"output_token_index={output_index}, "
                            f"resident_token={token}, "
                            f"fresh_token={reference_token}, "
                            f"max_abs_logit_delta={step_delta:.9g}, "
                            f"limit={self.max_abs_logit_delta:.9g}."
                        )
                if token in eos:
                    terminal = token
                    logits = self._evaluate([token], candidate_cache)
                    calls += 1
                    if full_reference_cache is not None:
                        self._evaluate([token], full_reference_cache)
                        full_reference_calls += 1
                    break
                generated.append(token)
                logits = self._evaluate([token], candidate_cache)
                calls += 1
                if full_reference_cache is not None:
                    full_reference_logits = self._evaluate(
                        [token], full_reference_cache
                    )
                    full_reference_calls += 1
                if len(generated) >= request.resolved_max_new_tokens:
                    break

            committed = len(wire) + len(generated) + int(terminal is not None)
            updated_memory, graft_metrics = self._graft(
                state.canonical_memory,
                candidate_cache,
                expected_local_tokens=committed,
                ignored_leading_local_tokens=materialized_tokens,
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
        if state.full_reference_cache is not None:
            reference_offset = int(state.full_reference_cache[0].offset)
            if reference_offset != len(state.canonical_tokens):
                raise RuntimeError(
                    "MLX prefix-cache reference ended at the wrong token: "
                    f"offset={reference_offset}, "
                    f"expected={len(state.canonical_tokens)}."
                )
        text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
        assistant_index = len(messages)
        state.message_spans[assistant_index] = LiveKVInterval(
            len(source),
            len(state.canonical_tokens),
            record_id=f"message:{assistant_index}:assistant",
            causal_group_id=f"turn:{assistant_index}",
        )
        state.canonical_text = _decode_exact(
            self.tokenizer, state.canonical_tokens
        )
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
            # Canonical source capture does not generate. Request one already
            # consumes the requested resident selection.
            "native_kv_used": True,
            "source_tokens": len(source),
            "selected_kv_tokens": plan.selected_tokens,
            "materialized_history_encoded_tokens": materialized_tokens,
            "materialized_history_model_calls": materialized_calls,
            "materialized_history_prefill_step": materialized_prefill_step,
            "wire_tokens": len(wire),
            "logical_prompt_tokens": len(prompt),
            "effective_attention_prompt_tokens": (
                plan.selected_tokens + materialized_tokens + len(wire)
            ),
            "completion_tokens": len(generated),
            "output_token_ids": list(generated),
            "terminal_token_id": terminal,
            "requested_retention_fraction": requested,
            "effective_requested_retention_fraction": effective_requested,
            "realized_retention_fraction": (
                (plan.selected_tokens + materialized_tokens) / max(len(source), 1)
            ),
            "realized_historical_kv_retention_fraction": (
                (plan.selected_tokens + materialized_tokens) / max(len(source), 1)
            ),
            "engine_reported_history_kv_retention_fraction": (
                (plan.selected_tokens + materialized_tokens) / max(len(source), 1)
            ),
            "full_retention": bool(plan.full_retention),
            "source_bootstrap": source_bootstrap,
            "source_bootstrap_tokens": len(prompt) if source_bootstrap else None,
            "source_bootstrap_cached_tokens": 0 if source_bootstrap else None,
            "source_bootstrap_evaluated_tokens": (
                len(prompt) if source_bootstrap else None
            ),
            "source_bootstrap_generated_tokens": 0 if source_bootstrap else None,
            "selection_deferred_until_source_resident": False,
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
            "same_subset_reference_kind": same_subset_reference_kind,
            "same_subset_max_abs_logit_delta": logit_delta,
            "same_subset_gate_limit": self.max_abs_logit_delta,
            "same_subset_candidate_first_token_id": same_subset_candidate_token,
            "same_subset_reference_first_token_id": same_subset_reference_token,
            "same_subset_first_token_match": (
                None
                if same_subset_candidate_token is None
                else same_subset_candidate_token == same_subset_reference_token
            ),
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
            "full_retention_prefix_cache_reference_required": (
                bool(self.require_full_retention_reference)
                and plan.full_retention
            ),
            "full_retention_prefix_cache_reference_kind": (
                "live_mlx_prefix_cache"
                if plan.full_retention and self.require_full_retention_reference
                else None
            ),
            "full_retention_prefix_cache_compared_tokens": (
                full_reference_compared_tokens
                if plan.full_retention and self.require_full_retention_reference
                else None
            ),
            "full_retention_prefix_cache_model_calls": (
                full_reference_calls
                if plan.full_retention and self.require_full_retention_reference
                else None
            ),
            "full_retention_prefix_cache_max_abs_logit_delta": (
                full_reference_max_delta
                if plan.full_retention and self.require_full_retention_reference
                else None
            ),
            "full_retention_prefix_cache_gate_passed": (
                True
                if plan.full_retention and self.require_full_retention_reference
                else None
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
                template_kwargs = self._template_kwargs(request)
                prompt = _render(
                    self.tokenizer,
                    request.messages,
                    generation_prompt=True,
                    chat_template_kwargs=template_kwargs,
                )
                text, generated, elapsed, calls = (
                    self._ordinary_generate_record_aligned(
                        request.messages,
                        chat_template_kwargs=template_kwargs,
                        max_tokens=request.resolved_max_new_tokens,
                    )
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
