"""Direct, no-gateway HF executor for resident agent-history K/V.

The executor owns one canonical ``DynamicCache`` per agent session.  Sparse
requests borrow original-position tensor views from that cache; selected
history is never decoded, tokenized, packed, or copied.  The newly evaluated
wire/assistant suffix is grafted back into the canonical cache after the
request finishes.  DynamicCache growth copies existing K/V internally, so
that lifecycle cost is measured separately from the zero-copy selection.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .deployment import PRAEngineResult, PRAWireRequest
from .hf_live_kv import (
    HFLiveKVRuntime,
    HFSparseDynamicCache,
    _layer_pair,
    enable_qwen_sparse_live_kv,
)
from .live_history import LiveKVInterval, LiveKVSelectionPlan


PURE_CHATML_APPEND_STABLE_TEMPLATE = """\
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def qwen3_append_stable_no_thinking_template(native_template: str) -> str:
    """Retain Qwen3's empty no-think block when an answer becomes history."""

    needle = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    replacement = """\
{%- if enable_thinking is defined and enable_thinking is false %}
                    {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\\n' + content }}
                {%- endif %}"""
    occurrences = native_template.count(needle)
    if occurrences != 2:
        raise ValueError(
            "Pinned Qwen3 template shape changed: expected exactly two "
            f"historical assistant branches, found {occurrences}."
        )
    return native_template.replace(needle, replacement)


def configure_append_stable_template(tokenizer: object, profile: str) -> str:
    """Install one frozen template profile and return its SHA-256 digest."""

    native = str(getattr(tokenizer, "chat_template", "") or "")
    if not native:
        raise ValueError("Tokenizer does not expose a chat template.")
    if profile == "native":
        selected = native
    elif profile == "qwen3-stable-no-thinking":
        selected = qwen3_append_stable_no_thinking_template(native)
    elif profile == "pure-chatml-stable":
        selected = PURE_CHATML_APPEND_STABLE_TEMPLATE
    else:
        raise ValueError(f"Unknown chat template profile: {profile!r}")
    tokenizer.chat_template = selected
    return hashlib.sha256(selected.encode("utf-8")).hexdigest()


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    matched = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        matched += 1
    return matched


def _token_ids(value: object) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids", ())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    rows = list(value) if isinstance(value, Iterable) else []
    if rows and isinstance(rows[0], (list, tuple)):
        if len(rows) != 1:
            raise ValueError("Tokenizer returned more than one prompt batch.")
        rows = list(rows[0])
    return list(map(int, rows))


def _render(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    *,
    generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [dict(row) for row in messages],
        tokenize=True,
        add_generation_prompt=generation_prompt,
        **dict(chat_template_kwargs or {}),
    )
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered, add_special_tokens=False)
    return _token_ids(rendered)


def split_generation_prompt(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Split completed records from the template's generation-only suffix.

    A fixed token tail is not a valid record boundary: a short tool result can
    make that tail move backwards into already resident assistant K/V. Render
    both template states so the reusable source always ends after the current
    completed record and the wire contains only the generation prompt.
    """

    source = _render(
        tokenizer,
        messages,
        generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt = _render(
        tokenizer,
        messages,
        generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    if prompt[: len(source)] != source:
        raise RuntimeError(
            "The chat template generation prompt rewrites completed record tokens."
        )
    wire = prompt[len(source) :]
    if not wire:
        raise RuntimeError("The chat template produced no generation-prompt suffix.")
    return prompt, source, wire


def validate_append_stable_template(
    tokenizer: object,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Fail before model load if an assistant turn rewrites its live prefix."""

    base = (
        {"role": "system", "content": "PRA template stability system"},
        {"role": "user", "content": "PRA template stability user"},
    )
    active = _render(
        tokenizer,
        base,
        generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    completed = (*base, {"role": "assistant", "content": "PRA_STABLE_PROBE"})
    historical = _render(
        tokenizer,
        completed,
        generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if _common_prefix(active, historical) != len(active):
        raise RuntimeError(
            "Configured chat template rewrites the active assistant prefix when "
            "the response becomes history."
        )
    next_turn = _render(
        tokenizer,
        (*completed, {"role": "user", "content": "PRA_STABLE_OBSERVATION"}),
        generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    if _common_prefix(historical, next_turn) != len(historical):
        raise RuntimeError(
            "Configured chat template rewrites completed history on the next turn."
        )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class AgentHistoryLedger:
    """Reconstruct the full logical transcript without tokenizing resources."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def reconcile(self, request: PRAWireRequest) -> tuple[dict[str, Any], ...]:
        manifest = request.metadata.get("logical_message_manifest")
        if request.metadata.get("history_projection") == "live-agent-kv-v1" and not manifest:
            raise ValueError("Live agent-history PRA requires a complete logical message manifest.")
        if not manifest:
            self.messages = [dict(row) for row in request.messages]
            return tuple(dict(row) for row in self.messages)

        rows = [dict(row) for row in manifest]
        if [int(row.get("message_index", -1)) for row in rows] != list(range(len(rows))):
            raise ValueError("Logical message manifest indices must be contiguous.")
        mandatory = list(map(int, request.metadata.get("mandatory_message_indices", ())))
        if len(mandatory) != len(request.messages):
            raise ValueError("Mandatory message indices do not match wire messages.")
        values: list[dict[str, Any] | None] = [None] * len(rows)
        for index, prior in enumerate(self.messages[: len(values)]):
            values[index] = dict(prior)
        for index, message in zip(mandatory, request.messages):
            if index < 0 or index >= len(values):
                raise ValueError("Mandatory message index lies outside the manifest.")
            values[index] = dict(message)
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            raise RuntimeError(
                "Live agent history is missing resident records: "
                + ", ".join(map(str, missing))
            )
        complete = [dict(value) for value in values if value is not None]
        for item, message in zip(rows, complete):
            if str(message.get("role", "")) != str(item.get("role", "")):
                raise RuntimeError("Resident message role disagrees with the manifest.")
            if _digest_text(str(message.get("content", ""))) != str(
                item.get("content_sha256", "")
            ):
                raise RuntimeError("Resident message content disagrees with the manifest.")
        # Bodies are witnesses for cache-resident spans.  Never pass them to
        # the tokenizer; validation is against the in-memory logical ledger.
        for resource in request.resources:
            index = resource.metadata.get("message_index")
            if index is None:
                continue
            index = int(index)
            if index >= len(complete) or str(resource.text or "") not in str(
                complete[index].get("content", "")
            ):
                raise RuntimeError(
                    f"Selected resource {resource.resource_id!r} is not a resident record span."
                )
        self.messages = complete
        return tuple(dict(row) for row in complete)

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": str(content)})


def causal_message_spans(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[int],
    *,
    source_tokens: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[LiveKVInterval, ...]:
    if source_tokens <= 0 or source_tokens > len(prompt_ids):
        raise ValueError("source_tokens must fit the rendered prompt.")
    boundaries = [0]
    for index in range(len(messages)):
        prefix = _render(
            tokenizer,
            messages[: index + 1],
            generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs,
        )
        boundary = min(source_tokens, _common_prefix(prefix, prompt_ids))
        boundaries.append(max(boundaries[-1], boundary))
    boundaries[-1] = source_tokens
    spans: list[LiveKVInterval] = []
    causal_group = "preamble"
    for index, message in enumerate(messages):
        start, end = boundaries[index], boundaries[index + 1]
        if end <= start:
            continue
        role = str(message.get("role", "unknown"))
        if role == "assistant":
            causal_group = f"turn:{index}"
        elif index <= 1:
            causal_group = "preamble"
        spans.append(
            LiveKVInterval(
                start,
                end,
                record_id=f"message:{index}:{role}",
                causal_group_id=causal_group,
            )
        )
    return tuple(spans)


def selected_record_plan(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[int],
    *,
    source_tokens: int,
    retention_fraction: float,
    mandatory_message_indices: Sequence[int],
    selected_message_indices: Sequence[int],
    chat_template_kwargs: Mapping[str, Any] | None = None,
    round_up_to_retention_floor: bool = False,
) -> LiveKVSelectionPlan:
    fraction = float(retention_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("retention_fraction must be in (0, 1].")
    if fraction == 1:
        return LiveKVSelectionPlan.full(source_tokens)
    keep = set(map(int, mandatory_message_indices)) | set(map(int, selected_message_indices))
    invalid = sorted(index for index in keep if index < 0 or index >= len(messages))
    if invalid:
        raise ValueError(
            "Selected or mandatory record indices lie outside resident history: "
            + ", ".join(map(str, invalid))
        )
    spans = causal_message_spans(
        tokenizer,
        messages,
        prompt_ids,
        source_tokens=source_tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    chosen = tuple(
        span for span in spans if int(span.record_id.split(":", 2)[1]) in keep
    )
    if not chosen:
        raise RuntimeError("Sparse live-history request selected no resident records.")
    represented = {int(span.record_id.split(":", 2)[1]) for span in chosen}
    missing = sorted(keep - represented)
    if missing:
        raise RuntimeError(
            "Selected resident records have no chat-template token span: "
            + ", ".join(map(str, missing))
        )
    if round_up_to_retention_floor:
        minimum = math.ceil(fraction * source_tokens)
        chosen_ids = {span.record_id for span in chosen}
        # The proxy uses a portable whitespace estimate, whereas this layer
        # has the model's authoritative tokenizer.  Complete any partially
        # selected causal groups first, then restore the newest omitted whole
        # groups until the exact token floor is met.  No text is re-tokenized:
        # all added intervals refer to the resident canonical K/V owner.
        groups: dict[str, list[LiveKVInterval]] = {}
        for span in spans:
            groups.setdefault(span.causal_group_id or span.record_id, []).append(span)
        selected_groups = {
            span.causal_group_id or span.record_id for span in chosen
        }
        for group_id in selected_groups:
            chosen_ids.update(span.record_id for span in groups[group_id])
        for group_id in reversed(tuple(groups)):
            selected_tokens = sum(
                span.tokens for span in spans if span.record_id in chosen_ids
            )
            if selected_tokens >= minimum:
                break
            chosen_ids.update(span.record_id for span in groups[group_id])
        chosen = tuple(span for span in spans if span.record_id in chosen_ids)
    return LiveKVSelectionPlan.create(
        source_tokens, chosen, source_position_base=source_tokens
    )


def enforce_retention_floor(
    plan: LiveKVSelectionPlan,
    requested_fraction: float,
    *,
    selection_contract: str | None = None,
) -> None:
    """Reject a selected record set that underfills its declared retention arm.

    Record-aligned selectors may round above a fractional target, but silently
    rounding below it changes the experiment.  The sole exception is an
    explicitly labelled arbitrary-subset mechanism probe, where the requested
    fraction is descriptive rather than a minimum budget contract.
    """

    requested = float(requested_fraction)
    minimum = math.ceil(requested * plan.source_tokens)
    if plan.selected_tokens >= minimum:
        return
    if selection_contract == "arbitrary-subset-mechanism-probe":
        return
    raise RuntimeError(
        "Selected live-history records underfill the requested retention floor: "
        f"selected={plan.selected_tokens}, required={minimum}, "
        f"source={plan.source_tokens}, requested={requested:.6f}. "
        "Record-aligned selection must round up; label a deliberately arbitrary "
        "subset with selection_contract='arbitrary-subset-mechanism-probe'."
    )


def _tensor_bytes(tensor: object) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _cache_bytes(cache: object) -> int:
    total = 0
    for layer in getattr(cache, "layers", ()):
        try:
            keys, values, _key_name, _value_name = _layer_pair(layer)
        except TypeError:  # DynamicCache starts with lazy, empty layers.
            continue
        total += _tensor_bytes(keys) + _tensor_bytes(values)
    return total


def _cache_storage_pointers(cache: object) -> tuple[tuple[int, int], ...]:
    rows = []
    for layer in getattr(cache, "layers", ()):
        try:
            keys, values, _key_name, _value_name = _layer_pair(layer)
        except TypeError:
            continue
        rows.append(
            (
                int(keys.untyped_storage().data_ptr()),
                int(values.untyped_storage().data_ptr()),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class HFKVGraftMetrics:
    local_tokens: int
    request_tail_pack_d2d_bytes: int
    canonical_suffix_graft_d2d_bytes: int
    canonical_reallocation_d2d_bytes: int

    @property
    def total_kv_copy_bytes(self) -> int:
        return (
            self.request_tail_pack_d2d_bytes
            + self.canonical_suffix_graft_d2d_bytes
            + self.canonical_reallocation_d2d_bytes
        )


@dataclass(frozen=True)
class _HFDecodeResult:
    generated: list[int]
    terminal: int | None
    cache: object
    calls: int
    canonical_suffix_d2d_bytes: int = 0
    canonical_reallocation_d2d_bytes: int = 0


@dataclass
class _Session:
    tenant_id: str
    session_id: str
    source_id: str
    ledger: AgentHistoryLedger = field(default_factory=AgentHistoryLedger)
    canonical_tokens: list[int] = field(default_factory=list)
    canonical_cache: object | None = None
    generation: int = 1
    calls: int = 0
    selected_history_reencoded_tokens: int = 0
    total_kv_copy_bytes: int = 0


class HFAgentHistoryExecutor:
    """Persistent DynamicCache executor for direct HF agent requests."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        model_id: str,
        model_revision: str,
        wire_tail_tokens: int = 32,
        prefill_step_size: int = 256,
        chat_template_profile: str = "native",
        chat_template_digest: str | None = None,
    ) -> None:
        if wire_tail_tokens <= 0:
            raise ValueError("wire_tail_tokens must be positive.")
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be positive.")
        self.model = model
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.wire_tail_tokens = int(wire_tail_tokens)
        self.prefill_step_size = int(prefill_step_size)
        self.chat_template_profile = str(chat_template_profile)
        selected = str(getattr(tokenizer, "chat_template", "") or "")
        observed = hashlib.sha256(selected.encode("utf-8")).hexdigest()
        if chat_template_digest is not None and observed != str(chat_template_digest):
            raise ValueError("Configured chat template digest does not match the tokenizer.")
        self.chat_template_digest = observed
        self.runtime = HFLiveKVRuntime[object]()
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self.prefix_cache_enabled = False
        self.patched_layers = enable_qwen_sparse_live_kv(model)

    def capabilities(self) -> Mapping[str, object]:
        return {
            "adapter": "huggingface_direct_agent",
            "integration_level": "E2",
            "engine_type": "huggingface",
            "native_kv": True,
            "session_state": True,
            "live_prefix_kv_capture": True,
            "live_prefix_kv_subset": True,
            "zero_selected_text_reencoding": True,
            "stable_record_kv_identity": True,
            "multiple_selected_records": True,
            "source_positions_preserved": True,
            "request_membership_attach": True,
            "agent_history_kv_qualified": True,
            "selected_interval_materialization": True,
            "request_lifetime": True,
            "resource_delta": False,
            "streaming": False,
            "physical_kv_copy_reported": True,
            "prefix_cache_enabled": False,
            "prefill_step_size": self.prefill_step_size,
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
                str(request.tenant_id), session_id, f"hf-agent-source-{digest}"
            )
            self._sessions[session_id] = state
        elif state.tenant_id != str(request.tenant_id):
            raise PermissionError("HF agent session crossed tenant scope.")
        return state

    @staticmethod
    def _template_kwargs(request: PRAWireRequest) -> dict[str, Any]:
        value = request.openai_fields.get("chat_template_kwargs") or {}
        if not isinstance(value, Mapping):
            raise ValueError("chat_template_kwargs must be a mapping.")
        return dict(value)

    def _checked_template_kwargs(self, request: PRAWireRequest) -> dict[str, Any]:
        options = self._template_kwargs(request)
        if self.chat_template_profile == "qwen3-stable-no-thinking":
            if options.get("enable_thinking") not in {None, False}:
                raise ValueError("qwen3-stable-no-thinking rejects enable_thinking=true.")
            options["enable_thinking"] = False
        expected = request.metadata.get("chat_template_digest")
        if expected is not None and str(expected) != self.chat_template_digest:
            raise ValueError("Request chat template digest does not match the resident session.")
        return options

    def _device(self):
        import torch

        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            return next(parameters()).device
        return torch.device("cpu")

    def _new_cache(self):
        from transformers import DynamicCache

        return DynamicCache()

    def _forward(self, cache: object, token_ids: Sequence[int], start: int):
        import torch

        if not token_ids:
            raise ValueError("HF cache extension requires at least one token.")
        device = self._device()
        ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
        positions = torch.arange(start, start + len(token_ids), device=device)
        last_logit_kwargs = getattr(self, "_last_logit_kwargs", None)
        if last_logit_kwargs is None:
            forward = getattr(self.model, "forward", self.model)
            try:
                parameters = inspect.signature(forward).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "logits_to_keep" in parameters:
                last_logit_kwargs = {"logits_to_keep": 1}
            elif "num_logits_to_keep" in parameters:
                last_logit_kwargs = {"num_logits_to_keep": 1}
            else:
                last_logit_kwargs = {}
            self._last_logit_kwargs = last_logit_kwargs

        # Every call is inference, including single-token decode and canonical
        # cache extension.  Without this guard PyTorch retains an autograd
        # graph through successive DynamicCache concatenations; agent decode
        # then grows memory per token and can OOM even a small model.
        # Models that expose a last-logit control must also avoid projecting
        # every prompt position into vocabulary space: the agent only consumes
        # the final row, and the full [prompt, vocab] tensor can dominate VRAM.
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                past_key_values=cache,
                cache_position=positions,
                use_cache=True,
                return_dict=True,
                **last_logit_kwargs,
            )
        return outputs, getattr(outputs, "past_key_values", cache)

    def _extend_canonical(
        self, state: _Session, tokens: Sequence[int]
    ) -> tuple[int, int, int]:
        """Append tokens and account DynamicCache reallocations honestly."""

        if not tokens:
            return 0, 0, 0
        assert state.canonical_cache is not None
        suffix_bytes = 0
        reallocation = 0
        base_position = len(state.canonical_tokens)
        values = list(map(int, tokens))
        for offset in range(0, len(values), self.prefill_step_size):
            step = values[offset : offset + self.prefill_step_size]
            before_bytes = _cache_bytes(state.canonical_cache)
            before_pointers = _cache_storage_pointers(state.canonical_cache)
            _outputs, cache = self._forward(
                state.canonical_cache, step, base_position + offset
            )
            state.canonical_cache = cache
            after_bytes = _cache_bytes(cache)
            after_pointers = _cache_storage_pointers(cache)
            reallocated = bool(
                before_pointers and before_pointers != after_pointers
            )
            if reallocated:
                suffix_bytes += max(after_bytes - before_bytes, 0)
                reallocation += before_bytes
        return suffix_bytes, reallocation, suffix_bytes + reallocation

    def _ensure_owner(self, state: _Session, source: list[int]) -> tuple[int, int, int]:
        common = _common_prefix(state.canonical_tokens, source)
        reencoded = max(0, min(len(state.canonical_tokens), len(source)) - common)
        if reencoded:
            raise RuntimeError(
                "The chat template rewrote already evaluated agent history "
                f"({reencoded} tokens). Resident live-K/V requires an append-stable "
                "template and will not fall back to re-prefill."
            )
        delta = list(source[common:])
        if state.canonical_cache is None:
            state.canonical_cache = self._new_cache()
            state.canonical_tokens = []
        prior_cache = state.canonical_cache
        suffix, reallocation, total = self._extend_canonical(state, delta)
        state.canonical_tokens = list(source)
        state.selected_history_reencoded_tokens += reencoded
        view = self.runtime.registry.view(state.source_id)
        if view is not None and state.canonical_cache is not prior_cache:
            state.generation += 1
            self.runtime.register_source(
                state.source_id,
                state.canonical_cache,
                tenant_id=state.tenant_id,
                session_id=state.session_id,
                generation=state.generation,
            )
        elif view is None:
            self.runtime.register_source(
                state.source_id,
                state.canonical_cache,
                tenant_id=state.tenant_id,
                session_id=state.session_id,
                generation=state.generation,
            )
        return len(delta), reencoded, total

    @staticmethod
    def _selected_message_indices(request: PRAWireRequest) -> tuple[int, ...]:
        missing = [
            resource.resource_id
            for resource in request.resources
            if resource.metadata.get("message_index") is None
        ]
        if missing:
            raise ValueError(
                "Live agent-history resources require message_index metadata: "
                + ", ".join(missing)
            )
        return tuple(
            sorted(
                {
                    int(resource.metadata["message_index"])
                    for resource in request.resources
                }
            )
        )

    def _eos_ids(self) -> set[int]:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            return set()
        if isinstance(value, (tuple, list, set)):
            return set(map(int, value))
        return {int(value)}

    def _decode_with_cache(
        self,
        cache: object,
        wire: Sequence[int],
        *,
        start: int,
        max_tokens: int,
    ) -> _HFDecodeResult:
        import torch

        calls = 0
        suffix_copy = 0
        reallocation_copy = 0

        def advance(active_cache: object, values: Sequence[int], position: int):
            nonlocal suffix_copy, reallocation_copy
            before = _cache_bytes(active_cache)
            pointers = _cache_storage_pointers(active_cache)
            result, updated = self._forward(active_cache, values, position)
            after = _cache_bytes(updated)
            updated_pointers = _cache_storage_pointers(updated)
            if pointers and pointers != updated_pointers:
                reallocation_copy += before
                suffix_copy += max(after - before, 0)
            return result, updated

        values = list(map(int, wire))
        if not values:
            raise ValueError("HF generation wire must contain at least one token.")
        outputs = None
        for offset in range(0, len(values), self.prefill_step_size):
            step = values[offset : offset + self.prefill_step_size]
            outputs, cache = advance(cache, step, start + offset)
            calls += 1
        assert outputs is not None
        token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        generated: list[int] = []
        eos = self._eos_ids()
        while len(generated) < max_tokens and token not in eos:
            generated.append(token)
            outputs, cache = advance(
                cache,
                (token,),
                start + len(wire) + len(generated) - 1,
            )
            calls += 1
            token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        terminal = token if token in eos else None
        if terminal is not None:
            # Commit EOS into K/V so the next append-stable prompt is an exact
            # token prefix of the resident cache.
            _outputs, cache = advance(
                cache, (terminal,), start + len(wire) + len(generated)
            )
            calls += 1
        return _HFDecodeResult(
            generated,
            terminal,
            cache,
            calls,
            suffix_copy,
            reallocation_copy,
        )

    def _graft_sparse_tail(
        self,
        canonical: object,
        sparse: HFSparseDynamicCache,
        *,
        source_tokens: int,
    ) -> HFKVGraftMetrics:
        import torch

        local_tokens: int | None = None
        pack_bytes = 0
        suffix_bytes = 0
        reallocation_bytes = 0
        for layer_index, (owner_layer, sparse_layer) in enumerate(
            zip(getattr(canonical, "layers", ()), sparse.layers)
        ):
            rows = tuple(sparse_layer.tail_segments)
            count = sum(row.tokens for row in rows)
            if local_tokens is None:
                local_tokens = count
            elif count != local_tokens:
                raise RuntimeError("HF request-local cache lengths disagree.")
            if not rows:
                continue
            if len(rows) == 1:
                keys, values = rows[0].keys, rows[0].values
            else:
                keys = torch.cat([row.keys for row in rows], dim=-2)
                values = torch.cat([row.values for row in rows], dim=-2)
                pack_bytes += _tensor_bytes(keys) + _tensor_bytes(values)
            before_keys, before_values, _kn, _vn = _layer_pair(owner_layer)
            before = _tensor_bytes(before_keys) + _tensor_bytes(before_values)
            before_pointer = (
                int(before_keys.untyped_storage().data_ptr()),
                int(before_values.untyped_storage().data_ptr()),
            )
            canonical.update(
                keys,
                values,
                layer_index,
                cache_kwargs={
                    "cache_position": torch.arange(
                        source_tokens,
                        source_tokens + count,
                        device=keys.device,
                    )
                },
            )
            after_keys, after_values, _kn, _vn = _layer_pair(
                getattr(canonical, "layers")[layer_index]
            )
            after_pointer = (
                int(after_keys.untyped_storage().data_ptr()),
                int(after_values.untyped_storage().data_ptr()),
            )
            suffix_bytes += _tensor_bytes(keys) + _tensor_bytes(values)
            if after_pointer != before_pointer:
                reallocation_bytes += before
        if local_tokens is None:
            raise RuntimeError("No HF attention cache was available for canonical K/V graft.")
        return HFKVGraftMetrics(
            local_tokens,
            pack_bytes,
            suffix_bytes,
            reallocation_bytes,
        )

    def _generate_native(self, request: PRAWireRequest, state: _Session) -> PRAEngineResult:
        live_projection = request.metadata.get("history_projection") == "live-agent-kv-v1"
        messages = state.ledger.reconcile(request)
        template_kwargs = self._checked_template_kwargs(request)
        prompt, source, wire = split_generation_prompt(
            self.tokenizer,
            messages,
            chat_template_kwargs=template_kwargs,
        )
        newly_encoded, reencoded, owner_copy = self._ensure_owner(state, source)
        requested = float(
            request.metadata.get(
                "target_retention_fraction",
                request.metadata.get("budget_fraction", 1.0),
            )
        )
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
        selection_contract = request.metadata.get("selection_contract")
        enforce_retention_floor(
            plan,
            requested,
            selection_contract=(
                None if selection_contract is None else str(selection_contract)
            ),
        )
        started = time.perf_counter()
        selected_cache = None
        lease = None
        outcome = "error"
        try:
            if plan.full_retention:
                # A 100% request is the ordinary live prefix-cache continuation,
                # not a numerically different sparse-attention implementation.
                prior_cache = state.canonical_cache
                decoded = self._decode_with_cache(
                    state.canonical_cache,
                    wire,
                    start=len(source),
                    max_tokens=request.resolved_max_new_tokens,
                )
                generated, terminal, cache, calls = (
                    decoded.generated,
                    decoded.terminal,
                    decoded.cache,
                    decoded.calls,
                )
                state.canonical_cache = cache
                if cache is not prior_cache:
                    state.generation += 1
                    self.runtime.register_source(
                        state.source_id,
                        cache,
                        tenant_id=state.tenant_id,
                        session_id=state.session_id,
                        generation=state.generation,
                    )
                graft = HFKVGraftMetrics(
                    len(wire) + len(generated) + int(terminal is not None),
                    0,
                    decoded.canonical_suffix_d2d_bytes,
                    decoded.canonical_reallocation_d2d_bytes,
                )
                mode = "dense_semantic_noop"
            else:
                request_id = str(request.request_id)
                lease = self.runtime.begin_request(
                    request_id,
                    state.source_id,
                    plan,
                    tenant_id=request.tenant_id,
                    session_id=state.session_id,
                    expected_generation=state.generation,
                )
                selected_cache = lease.selection.cache
                decoded = self._decode_with_cache(
                    selected_cache,
                    wire,
                    start=plan.source_position_base,
                    max_tokens=request.resolved_max_new_tokens,
                )
                generated, terminal, calls = (
                    decoded.generated,
                    decoded.terminal,
                    decoded.calls,
                )
                # Release the immutable source borrow before mutating the
                # canonical cache with the request-local suffix.
                lease.finish()
                lease = None
                graft = self._graft_sparse_tail(
                    state.canonical_cache,
                    selected_cache,
                    source_tokens=len(source),
                )
                expected = len(wire) + len(generated) + int(terminal is not None)
                if graft.local_tokens != expected:
                    raise RuntimeError(
                        "Canonical K/V graft length disagrees with generated history: "
                        f"cache={graft.local_tokens}, expected={expected}."
                    )
                mode = "sparse_original_position"
            state.canonical_tokens = [*source, *wire, *generated]
            if terminal is not None:
                state.canonical_tokens.append(terminal)
            text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
            state.ledger.append_assistant(text)
            state.calls += 1
            state.total_kv_copy_bytes += owner_copy + graft.total_kv_copy_bytes
            outcome = "finished"
        finally:
            if lease is not None:
                if outcome == "finished":
                    lease.finish()
                else:
                    lease.fail()

        elapsed = time.perf_counter() - started
        transient = int(
            getattr(getattr(selected_cache, "metrics", None), "transient_attention_bytes", 0)
        )
        transient_kv = int(
            getattr(getattr(selected_cache, "metrics", None), "transient_kv_copy_bytes", 0)
        )
        trace = {
            "stage": "native_attach",
            "engine": "huggingface",
            "native_kv_used": True,
            "consumption_mode": mode,
            "source_tokens": len(source),
            "selected_kv_tokens": plan.selected_tokens,
            "wire_tokens": len(wire),
            "logical_prompt_tokens": len(prompt),
            "effective_attention_prompt_tokens": plan.selected_tokens + len(wire),
            "completion_tokens": len(generated),
            "requested_retention_fraction": requested,
            "realized_retention_fraction": plan.selected_tokens / max(len(source), 1),
            "engine_reported_history_kv_retention_fraction": (
                plan.selected_tokens / max(len(source), 1)
            ),
            "full_retention": bool(plan.full_retention),
            "selection_contract": selection_contract or "minimum-retention-floor",
            "selected_history_reencoded_tokens": reencoded,
            "new_history_encoded_tokens": newly_encoded,
            "selected_history_kv_copy_bytes": 0,
            "selected_interval_copy_bytes": 0,
            "physical_kv_copy_bytes": 0,
            "physical_kv_copy": False,
            "physical_kv_copy_scope": "selected_history_attachment_only",
            "request_tail_pack_d2d_bytes": graft.request_tail_pack_d2d_bytes,
            "canonical_suffix_graft_d2d_bytes": graft.canonical_suffix_graft_d2d_bytes,
            "canonical_reallocation_d2d_bytes": graft.canonical_reallocation_d2d_bytes,
            "owner_extension_copy_bytes": owner_copy,
            "total_kv_copy_bytes": owner_copy + graft.total_kv_copy_bytes,
            "cumulative_total_kv_copy_bytes": state.total_kv_copy_bytes,
            "host_to_device_bytes": 0,
            "selected_history_host_to_device_bytes": 0,
            "host_to_device_scope": "selected_history_kv_only",
            "consumer_temporary_bytes": transient + transient_kv,
            "consumer_temporary_attention_bytes": transient,
            "consumer_temporary_kv_copy_bytes": transient_kv,
            "consumer_temporary_measurement_scope": (
                "logical sparse-attention allocation counters; excludes model "
                "weights and canonical cache"
            ),
            "fused_attention_calls": int(
                getattr(getattr(selected_cache, "metrics", None), "fused_attention_calls", 0)
            ),
            "model_forward_calls": calls,
            "session_call_index": state.calls,
            "source_position_base": plan.source_position_base,
            "selection_plan": plan.to_dict(),
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

    def _ordinary_generate(self, request: PRAWireRequest) -> PRAEngineResult:
        prompt = _render(
            self.tokenizer,
            request.messages,
            generation_prompt=True,
            chat_template_kwargs=self._checked_template_kwargs(request),
        )
        cache = self._new_cache()
        started = time.perf_counter()
        decoded = self._decode_with_cache(
            cache, prompt, start=0, max_tokens=request.resolved_max_new_tokens
        )
        generated, calls = decoded.generated, decoded.calls
        text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
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
            trace=(
                {
                    "stage": "plain_generation",
                    "native_kv_used": False,
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(generated),
                    "model_forward_calls": calls,
                    "temperature": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "chat_template_profile": self.chat_template_profile,
                    "chat_template_digest": self.chat_template_digest,
                },
            ),
        )

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        with self._lock:
            native = (
                request.metadata.get("history_projection") == "live-agent-kv-v1"
                or bool(request.resources)
            )
            if not native:
                return self._ordinary_generate(request)
            return self._generate_native(request, self._session(request))

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
