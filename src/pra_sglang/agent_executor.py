"""Session-aware SGLang-MLX executor for resident agent-history K/V.

Unlike :mod:`pra_sglang.native_executor`, this boundary never constructs
selected agent history by encoding detached resource text.  It keeps one
canonical full-history request per agent session, reconciles the typed record
manifest against that history, and attaches original-position views over the
already evaluated cache to each generation request.

The current SGLang MLX request cache owns its decode K/V independently from the
canonical source.  After a response finishes, the executor therefore grafts
the request-local suffix/response K/V into the canonical preallocated cache.
That is a device-side K/V copy, not token re-evaluation, and is reported
separately from selected-interval materialization (which remains zero-copy).
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pra_hf.deployment import PRAEngineResult, PRAWireRequest
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan

from .mlx_native import SGLangMLXLiveKVRuntime, SGLangMLXNativeBridge


PURE_CHATML_APPEND_STABLE_TEMPLATE = """\
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def qwen3_append_stable_no_thinking_template(native_template: str) -> str:
    """Preserve Qwen3's no-think first turn while making history append-only.

    The stock Qwen3 template emits an empty think block at the active
    generation prefix, then drops that block when the assistant becomes
    history.  We only rewrite the two historical/content-only assistant
    branches to retain the same empty block.  The first-turn token stream is
    consequently identical to stock ``enable_thinking=false``.
    """

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
    """Install a frozen template profile and return its SHA-256 digest."""

    profile = str(profile)
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


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    matched = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        matched += 1
    return matched


def _repetition_token_ids(
    token_history: Sequence[int], repeat_last_n: int
) -> tuple[int, ...]:
    """Return unique ids in the llama/Ollama repetition window."""

    window = int(repeat_last_n)
    if window < -1:
        raise ValueError("repeat_last_n must be -1 or non-negative.")
    if window == 0:
        return ()
    values = token_history if window == -1 else token_history[-window:]
    return tuple(sorted({int(token_id) for token_id in values}))


def _fixed_shape_repetition_token_ids(
    token_history: Sequence[int], repeat_last_n: int
) -> tuple[int, ...]:
    """Deduplicate semantically, then pad with identical benign writes.

    A fixed index-array shape lets MLX reuse the compiled decode graph. The
    padding repeats one already-deduplicated id with the same replacement
    value; it therefore cannot compound the penalty.
    """

    unique = _repetition_token_ids(token_history, repeat_last_n)
    if not unique:
        return ()
    window = int(repeat_last_n)
    width = len(token_history) if window == -1 else min(window, len(token_history))
    return unique + (unique[0],) * (width - len(unique))


def _token_digest(token_ids: Sequence[int]) -> str:
    encoded = b"".join(
        int(token_id).to_bytes(4, "little", signed=False)
        for token_id in token_ids
    )
    return hashlib.sha256(encoded).hexdigest()


def _execution_plan(plan: LiveKVSelectionPlan) -> LiveKVSelectionPlan:
    """Collapse complete record coverage to one direct canonical K/V view."""

    return LiveKVSelectionPlan.full(plan.source_tokens) if plan.full_retention else plan


def _selected_sampler_prompt(
    source: Sequence[int], wire: Sequence[int], plan: LiveKVSelectionPlan
) -> list[int]:
    """Build logical sampler history without materializing selected K/V."""

    selected = [
        int(token_id)
        for interval in sorted(plan.intervals, key=lambda row: row.start)
        for token_id in source[interval.start:interval.end]
    ]
    if len(selected) != plan.selected_tokens:
        raise RuntimeError(
            "Sampler history disagrees with selected K/V token extent."
        )
    return selected + list(map(int, wire))


class _MlxRepetitionPenaltyController:
    """Apply the missing sign-aware repetition transform on lazy MLX logits.

    The pinned SGLang MLX backend explicitly ignores repetition penalties.
    This wrapper sits at its native token-selection boundary, before greedy
    argmax/sampling, and uses the runner's authoritative live token history.
    """

    _MARKER = "_pra_repetition_penalty_controller"

    def __init__(self, runner: object) -> None:
        if getattr(runner, self._MARKER, None) is not None:
            raise RuntimeError("The MLX runner already has a PRA penalty controller.")
        original = getattr(runner, "_select_tokens_with_logprobs", None)
        if not callable(original):
            raise RuntimeError(
                "Pinned SGLang MLX runner lacks its token-selection boundary."
            )
        self.runner = runner
        self._original = original
        self._settings: dict[str, tuple[float, int]] = {}
        self._prefill_history: dict[str, tuple[int, ...]] = {}
        self._metrics: dict[str, dict[str, int]] = {}

        def wrapped(
            last_logits: object,
            req_ids: list[str],
            caches: list[list[Any]],
            edit_rows: object | None = None,
            logprob_spec: object | None = None,
        ):
            if (
                not bool(getattr(self.runner, "_enable_sampling", False))
                and edit_rows is None
                and logprob_spec is None
            ):
                return self._exact_greedy(last_logits, req_ids), None
            adjusted = self._apply(last_logits, req_ids)
            return self._original(
                adjusted, req_ids, caches, edit_rows, logprob_spec
            )

        setattr(runner, "_select_tokens_with_logprobs", wrapped)
        setattr(runner, self._MARKER, self)

    def register(
        self,
        request_id: str,
        token_history: Sequence[int],
        *,
        penalty: float,
        repeat_last_n: int,
    ) -> None:
        value = float(penalty)
        window = int(repeat_last_n)
        if not value > 0.0:
            raise ValueError("repetition_penalty must be positive.")
        _repetition_token_ids((), window)
        request_id = str(request_id)
        self._settings[request_id] = (value, window)
        self._prefill_history[request_id] = tuple(map(int, token_history))
        initial_ids = _repetition_token_ids(token_history, window)
        self._metrics[request_id] = {
            "application_calls": 0,
            "history_tokens_first_step": len(token_history),
            "unique_tokens_first_step": len(initial_ids),
            "unique_tokens_max": len(initial_ids),
        }

    def unregister(self, request_id: str) -> dict[str, int]:
        request_id = str(request_id)
        self._settings.pop(request_id, None)
        self._prefill_history.pop(request_id, None)
        return self._metrics.pop(request_id, {})

    def _history(self, request_id: str) -> Sequence[int]:
        live = getattr(self.runner, "_req_token_ids", {}).get(request_id)
        if live is not None:
            return live
        return self._prefill_history.get(request_id, ())

    def _record_application(self, request_id: str, unique_count: int) -> None:
        metrics = self._metrics[request_id]
        metrics["application_calls"] += 1
        metrics["unique_tokens_max"] = max(
            metrics["unique_tokens_max"], int(unique_count)
        )

    def _exact_greedy(self, last_logits: object, req_ids: Sequence[str]):
        """Return the exact penalized argmax without scattering full logits.

        For a last-N window containing U unique ids, the winner must be among
        the original top U+1 logits: at least one candidate is unpenalized and
        dominates every logit outside that set. Selecting top N+1 keeps the
        graph shape fixed for the frozen last-64 campaign.
        """

        import mlx.core as mx

        chosen = []
        for row_index, request_id in enumerate(req_ids):
            request_id = str(request_id)
            row = last_logits[row_index]
            penalty, window = self._settings.get(request_id, (1.0, 0))
            history = self._history(request_id)
            unique_ids = _repetition_token_ids(history, window)
            if penalty == 1.0 or not unique_ids:
                chosen.append(mx.argmax(row, axis=-1))
                continue
            candidate_bound = window + 1 if window > 0 else len(unique_ids) + 1
            width = min(candidate_bound, int(row.shape[-1]))
            partition = mx.argpartition(
                row, int(row.shape[-1]) - width, axis=-1
            )
            candidate_ids = partition[-width:]
            candidate_logits = mx.take(row, candidate_ids, axis=-1)
            seen = mx.array(unique_ids, dtype=mx.uint32)
            is_seen = mx.any(candidate_ids[:, None] == seen[None, :], axis=-1)
            penalized = mx.where(
                candidate_logits <= 0,
                candidate_logits * penalty,
                candidate_logits / penalty,
            )
            adjusted = mx.where(is_seen, penalized, candidate_logits)
            best = mx.max(adjusted, axis=-1)
            # Full-vocabulary argmax resolves equal logits to the lowest token
            # id. ``argpartition`` candidate order is undefined, so restore
            # that tie contract explicitly.
            tied_ids = mx.where(
                adjusted == best, candidate_ids, int(row.shape[-1])
            )
            chosen.append(mx.min(tied_ids, axis=-1))
            self._record_application(request_id, len(unique_ids))
        return mx.stack(chosen, axis=0)

    def _apply(self, last_logits: object, req_ids: Sequence[str]):
        import mlx.core as mx

        rows = []
        changed = False
        for row_index, request_id in enumerate(req_ids):
            request_id = str(request_id)
            row = last_logits[row_index]
            penalty, window = self._settings.get(request_id, (1.0, 0))
            history = self._history(request_id)
            unique_ids = _repetition_token_ids(history, window)
            ids = _fixed_shape_repetition_token_ids(history, window)
            if penalty != 1.0 and ids:
                indices = mx.array(ids, dtype=mx.int32)
                values = mx.take(row, indices, axis=-1)
                values = mx.where(values <= 0, values * penalty, values / penalty)
                row = mx.put_along_axis(row, indices, values, axis=-1)
                self._record_application(request_id, len(unique_ids))
                changed = True
            rows.append(row)
        if not changed:
            return last_logits
        return mx.stack(rows, axis=0)


def _live_kv_copy_metrics(
    *, selected_kv_tokens: int, canonical_suffix_graft_d2d_bytes: int
) -> dict[str, int | bool]:
    """Keep zero-copy selection distinct from canonical lifecycle copying.

    ``native_attach_bytes`` is a byte counter, not a selected-token counter.
    Attaching the selected interval descriptors transfers no payload bytes.
    The request-local suffix graft is a real device-to-device copy and is
    therefore exposed through its own counter and the total-copy counter.
    """

    selected = int(selected_kv_tokens)
    graft = int(canonical_suffix_graft_d2d_bytes)
    if selected < 0 or graft < 0:
        raise ValueError("K/V accounting counters cannot be negative.")
    return {
        "selected_kv_tokens": selected,
        "native_attach_bytes": 0,
        "physical_kv_copy": False,
        "selected_interval_copy_bytes": 0,
        "selected_history_kv_copy_bytes": 0,
        "physical_kv_copy_bytes": 0,
        "canonical_suffix_graft_d2d_bytes": graft,
        "total_kv_copy_bytes": graft,
        "host_to_device_bytes": 0,
    }


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
    options = dict(chat_template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        [dict(row) for row in messages],
        tokenize=True,
        add_generation_prompt=generation_prompt,
        **options,
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
    """Keep the source boundary on completed records, never a fixed tail."""

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


@dataclass
class AgentHistoryLedger:
    """Reconstruct a logical transcript from mandatory rows and stable hashes."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def reconcile(self, request: PRAWireRequest) -> tuple[dict[str, Any], ...]:
        manifest = request.metadata.get("logical_message_manifest")
        if (
            request.metadata.get("history_projection") == "live-agent-kv-v1"
            and not manifest
        ):
            raise ValueError(
                "Live agent-history PRA requires a complete logical message manifest."
            )
        if not manifest:
            # Plain admission and the first native request carry a complete
            # ordinary message list.  Copy it so caller-owned values cannot
            # mutate the resident history silently.
            self.messages = [dict(row) for row in request.messages]
            return tuple(dict(row) for row in self.messages)

        rows = [dict(row) for row in manifest]
        expected = list(range(len(rows)))
        observed = [int(row.get("message_index", -1)) for row in rows]
        if observed != expected:
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
            role = str(message.get("role", ""))
            if role != str(item.get("role", "")):
                raise RuntimeError("Resident message role disagrees with the manifest.")
            content = str(message.get("content", ""))
            if _digest_text(content) != str(item.get("content_sha256", "")):
                raise RuntimeError("Resident message content disagrees with the manifest.")

        # Selected resource bodies are validation witnesses only.  They are
        # never tokenized by this executor.
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
    """Map chat-template records into the canonical prompt token frame."""

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

    result: list[LiveKVInterval] = []
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
        result.append(
            LiveKVInterval(
                start,
                end,
                record_id=f"message:{index}:{role}",
                causal_group_id=causal_group,
            )
        )
    return tuple(result)


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
    """Create a record-rounded original-position plan for one request."""

    fraction = float(retention_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("retention_fraction must be in (0, 1].")
    if fraction == 1:
        return LiveKVSelectionPlan.full(source_tokens)
    invalid = sorted(
        index
        for index in set(map(int, mandatory_message_indices))
        | set(map(int, selected_message_indices))
        if index < 0 or index >= len(messages)
    )
    if invalid:
        raise ValueError(
            "Selected or mandatory record indices lie outside resident history: "
            + ", ".join(map(str, invalid))
        )
    keep = set(map(int, mandatory_message_indices)) | set(
        map(int, selected_message_indices)
    )
    spans = causal_message_spans(
        tokenizer,
        messages,
        prompt_ids,
        source_tokens=source_tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    chosen = tuple(
        span
        for span in spans
        if int(span.record_id.split(":", 2)[1]) in keep
    )
    if not chosen:
        raise RuntimeError("Sparse live-history request selected no resident records.")
    represented = {
        int(span.record_id.split(":", 2)[1]) for span in chosen
    }
    missing_spans = sorted(keep - represented)
    if missing_spans:
        raise RuntimeError(
            "Selected resident records have no chat-template token span: "
            + ", ".join(map(str, missing_spans))
        )
    if round_up_to_retention_floor:
        minimum = math.ceil(fraction * source_tokens)
        chosen_ids = {span.record_id for span in chosen}
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
        source_tokens,
        chosen,
        source_position_base=source_tokens,
    )


def enforce_retention_floor(
    plan: LiveKVSelectionPlan,
    requested_fraction: float,
    *,
    selection_contract: str | None = None,
) -> None:
    """Fail closed when a nominal retention arm is silently underfilled.

    Record-aligned selection may round above the requested fraction, but an
    end-to-end arm must never materialize less history than it declares.  The
    explicit exceptions are an arbitrary-subset mechanism probe and a frozen
    agent-memory plan.  Both own their exact logical subset, making the nominal
    fraction descriptive rather than permission for the engine to add records.
    """

    requested = float(requested_fraction)
    minimum = math.ceil(requested * plan.source_tokens)
    if plan.selected_tokens >= minimum:
        return
    if selection_contract in {
        "arbitrary-subset-mechanism-probe",
        "frozen-agent-memory-plan-v1",
    }:
        return
    raise RuntimeError(
        "Selected live-history records underfill the requested retention floor: "
        f"selected={plan.selected_tokens}, required={minimum}, "
        f"source={plan.source_tokens}, requested={requested:.6f}. "
        "Record-aligned floor arms must round up; only an explicit mechanism "
        "probe or frozen agent-memory plan may own an underfilled subset."
    )


@dataclass
class _Session:
    tenant_id: str
    session_id: str
    owner_request_id: str
    source_id: str
    ledger: AgentHistoryLedger = field(default_factory=AgentHistoryLedger)
    canonical_tokens: list[int] = field(default_factory=list)
    generation: int = 0
    calls: int = 0
    selected_history_reencoded_tokens: int = 0
    canonical_append_copy_bytes: int = 0


class SGLangMLXAgentHistoryExecutor:
    """Execute plain or resident-live-K/V agent requests on one MLX runner."""

    def __init__(
        self,
        runner: object,
        tokenizer: object,
        *,
        model_id: str,
        model_revision: str,
        block_store: LogicalPRABlockStore,
        wire_tail_tokens: int = 32,
        chat_template_profile: str = "native",
        chat_template_digest: str | None = None,
        additional_stop_token_ids: Sequence[int] = (),
        default_repetition_penalty: float = 1.0,
        default_repeat_last_n: int = 64,
        prefill_step_size: int = 2048,
    ) -> None:
        if wire_tail_tokens <= 0:
            raise ValueError("wire_tail_tokens must be positive.")
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be positive.")
        self.runner = runner
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.block_store = block_store
        self.wire_tail_tokens = int(wire_tail_tokens)
        self.chat_template_profile = str(chat_template_profile)
        selected_template = str(getattr(tokenizer, "chat_template", "") or "")
        observed_digest = hashlib.sha256(
            selected_template.encode("utf-8")
        ).hexdigest()
        if chat_template_digest is not None and observed_digest != str(
            chat_template_digest
        ):
            raise ValueError(
                "Configured chat template digest does not match the tokenizer."
            )
        self.chat_template_digest = observed_digest
        self.additional_stop_token_ids = frozenset(
            int(token_id) for token_id in additional_stop_token_ids
        )
        self.default_repetition_penalty = float(default_repetition_penalty)
        self.default_repeat_last_n = int(default_repeat_last_n)
        self.prefill_step_size = int(prefill_step_size)
        if not self.default_repetition_penalty > 0.0:
            raise ValueError("default_repetition_penalty must be positive.")
        _repetition_token_ids((), self.default_repeat_last_n)
        self._repetition = _MlxRepetitionPenaltyController(runner)
        self.bridge = SGLangMLXNativeBridge(runner)
        self.runtime = SGLangMLXLiveKVRuntime(self.bridge)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self.prefix_cache_enabled = False

    def capabilities(self) -> Mapping[str, object]:
        return {
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
            "chat_template_profile": self.chat_template_profile,
            "chat_template_digest": self.chat_template_digest,
            "additional_stop_token_ids": tuple(
                sorted(self.additional_stop_token_ids)
            ),
            "effective_stop_token_ids": tuple(sorted(self._eos_ids())),
            "default_repetition_penalty": self.default_repetition_penalty,
            "default_repeat_last_n": self.default_repeat_last_n,
            "prefill_step_size": self.prefill_step_size,
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
                owner_request_id=f"agent-owner-{digest}",
                source_id=f"agent-source-{digest}",
            )
            self._sessions[session_id] = state
        elif state.tenant_id != tenant_id:
            raise RuntimeError("SGLang agent session crossed tenant scope.")
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
                raise ValueError(
                    "qwen3-stable-no-thinking rejects enable_thinking=true."
                )
            options["enable_thinking"] = False
        expected = request.metadata.get("chat_template_digest")
        if expected is not None and str(expected) != self.chat_template_digest:
            raise ValueError(
                "Request chat template digest does not match the resident session."
            )
        return options

    def _eos_ids(self) -> set[int]:
        values = getattr(self.tokenizer, "eos_token_id", None)
        if values is None:
            return set(self.additional_stop_token_ids)
        if isinstance(values, (tuple, list, set)):
            return set(map(int, values)) | set(self.additional_stop_token_ids)
        return {int(values)} | set(self.additional_stop_token_ids)

    def _repetition_settings(self, request: PRAWireRequest) -> tuple[float, int]:
        penalty = float(
            request.openai_fields.get(
                "repetition_penalty", self.default_repetition_penalty
            )
        )
        repeat_last_n = int(
            request.openai_fields.get(
                "repeat_last_n", self.default_repeat_last_n
            )
        )
        if not penalty > 0.0:
            raise ValueError("repetition_penalty must be positive.")
        _repetition_token_ids((), repeat_last_n)
        return penalty, repeat_last_n

    def _eval(self, pending: object) -> None:
        self.runner.eval_pending(pending)

    def _prefill_tokens(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        needs_final_logits: bool,
    ) -> int:
        """Evaluate a prompt in bounded chunks and return its final token.

        Non-final chunks use the runner's headless trunk.  This avoids
        materializing a prompt-by-vocabulary logit tensor merely to populate
        K/V, while preserving the runner's normal cache and token bookkeeping.
        """

        values = list(map(int, token_ids))
        if not values:
            raise ValueError("SGLang-MLX prefill requires at least one token.")
        chunks = [
            values[start : start + self.prefill_step_size]
            for start in range(0, len(values), self.prefill_step_size)
        ]
        first = chunks[0]
        pending = self.runner.prefill_start(
            request_id,
            first,
            first,
            [],
            [],
            0,
            needs_logits=needs_final_logits and len(chunks) == 1,
        )
        self._eval(pending)
        token = int(self.runner.prefill_finalize(pending))
        for index, chunk in enumerate(chunks[1:], start=1):
            pending = self.runner.extend_start(
                request_id,
                chunk,
                [],
                needs_logits=needs_final_logits and index == len(chunks) - 1,
            )
            self._eval(pending)
            token = int(self.runner.extend_finalize(pending))
        return token

    def _ordinary_generate(
        self,
        request_id: str,
        prompt: list[int],
        max_tokens: int,
        *,
        repetition_penalty: float,
        repeat_last_n: int,
    ) -> tuple[str, list[int], float, dict[str, int]]:
        started = time.perf_counter()
        generated: list[int] = []
        penalty_metrics: dict[str, int] = {}
        self._repetition.register(
            request_id,
            prompt,
            penalty=repetition_penalty,
            repeat_last_n=repeat_last_n,
        )
        try:
            token = self._prefill_tokens(
                request_id, prompt, needs_final_logits=True
            )
            eos = self._eos_ids()
            while len(generated) < max_tokens:
                if token in eos:
                    break
                generated.append(token)
                if len(generated) >= max_tokens:
                    break
                decode = self.runner.decode_batch_start([request_id])
                self._eval(decode)
                token = int(self.runner.decode_batch_finalize(decode)[0])
            text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
        finally:
            if self.runner.has_request(request_id):
                self.runner.remove_request(request_id)
            penalty_metrics = self._repetition.unregister(request_id)
        return text, generated, time.perf_counter() - started, penalty_metrics

    def _ensure_owner(
        self, state: _Session, source: list[int]
    ) -> tuple[int, int]:
        """Reuse the exact canonical prefix and evaluate only its new suffix."""

        common = _common_prefix(state.canonical_tokens, source)
        reencoded = max(0, min(len(state.canonical_tokens), len(source)) - common)
        if reencoded:
            raise RuntimeError(
                "The chat template rewrote already evaluated agent history "
                f"({reencoded} tokens). Resident live-K/V requires an "
                "append-stable template and will not fall back to re-prefill."
            )
        if not self.runner.has_request(state.owner_request_id):
            self._prefill_tokens(
                state.owner_request_id, source, needs_final_logits=False
            )
            state.canonical_tokens = list(source)
            return len(source), 0

        caches = self.runner._req_caches[state.owner_request_id]
        for cache in caches:
            if hasattr(cache, "offset"):
                cache.offset = common
        # extend_finalize removes one stale sampled token before appending.
        self.runner._req_token_ids[state.owner_request_id] = (
            list(state.canonical_tokens[:common]) + [0]
        )
        delta = list(source[common:])
        if delta:
            pending = self.runner.extend_start(
                state.owner_request_id, delta, [], needs_logits=False
            )
            self._eval(pending)
            self.runner.extend_finalize(pending)
        state.canonical_tokens = list(source)
        state.selected_history_reencoded_tokens += reencoded
        return len(delta), reencoded

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
        values = {
            int(resource.metadata["message_index"])
            for resource in request.resources
            if resource.metadata.get("message_index") is not None
        }
        return tuple(sorted(values))

    def _graft_local_kv(
        self,
        owner_caches: Sequence[object],
        selected_caches: Sequence[object],
        *,
        source_tokens: int,
    ) -> tuple[int, int]:
        """Append request-local K/V to the canonical preallocated cache."""

        copied = 0
        local_tokens: int | None = None
        for owner, wrapped in zip(owner_caches, selected_caches):
            local = getattr(wrapped, "local_cache", None)
            if local is None or not hasattr(local, "keys"):
                continue
            count = int(local.offset)
            if local_tokens is None:
                local_tokens = count
            elif count != local_tokens:
                raise RuntimeError("SGLang request-local cache lengths disagree.")
            end = source_tokens + count
            grow = getattr(owner, "_grow", None)
            if end > int(getattr(owner, "max_seq_len", end)):
                if grow is None:
                    raise RuntimeError("Canonical cache cannot grow for K/V graft.")
                grow(end)
            keys = local.keys[:, :, :count, :]
            values = local.values[:, :, :count, :]
            owner.keys[:, :, source_tokens:end, :] = keys
            owner.values[:, :, source_tokens:end, :] = values
            owner.offset = end
            copied += int(keys.nbytes + values.nbytes)
        if local_tokens is None:
            raise RuntimeError("No attention cache was available for canonical K/V graft.")
        return local_tokens, copied

    def _native_generate(
        self, request: PRAWireRequest, state: _Session
    ) -> PRAEngineResult:
        import mlx.core as mx

        live_projection = (
            request.metadata.get("history_projection") == "live-agent-kv-v1"
        )
        if live_projection and not request.metadata.get("logical_message_manifest"):
            raise ValueError(
                "Live agent-history PRA requires a complete logical message manifest."
            )
        messages = state.ledger.reconcile(request)
        if (
            not live_projection
            and request.resources
        ):
            # Native readiness probes must consume their resource rather than
            # merely acknowledge it.  The autonomous-agent path never enters
            # this branch: its resource bodies are validation witnesses for
            # already resident K/V.
            context = "\n\n".join(
                str(resource.text or "") for resource in request.resources
            )
            messages = (
                *messages[:-1],
                {"role": "user", "content": context},
                messages[-1],
            )
        template_kwargs = self._checked_template_kwargs(request)
        prompt, source, wire = split_generation_prompt(
            self.tokenizer,
            messages,
            chat_template_kwargs=template_kwargs,
        )
        newly_encoded, reencoded = self._ensure_owner(state, source)

        requested = float(
            request.metadata.get(
                "target_retention_fraction",
                request.metadata.get("budget_fraction", 1.0),
            )
        )
        mandatory = tuple(
            map(int, request.metadata.get("mandatory_message_indices", ()))
        )
        # Readiness probes carry an ordinary resource without an agent-record
        # coordinate.  Only the live-history projection interprets resources
        # as resident message witnesses and therefore requires message_index.
        selected_indices = (
            self._selected_message_indices(request) if live_projection else ()
        )
        plan = selected_record_plan(
            self.tokenizer,
            messages,
            prompt,
            source_tokens=len(source),
            retention_fraction=requested,
            mandatory_message_indices=mandatory,
            selected_message_indices=selected_indices,
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
        if plan.full_retention:
            # Preserve logical record identities in the selector audit, but
            # normalize complete coverage to one direct canonical view for
            # execution. Record boundaries alone must not route PRA-100 (or a
            # rounded-up PRA-90 request) through segmented attention.
            plan = _execution_plan(plan)
        repetition_penalty, repeat_last_n = self._repetition_settings(request)
        sampler_prompt = _selected_sampler_prompt(source, wire, plan)

        state.generation += 1
        self.runtime.register_source(
            state.source_id,
            self.runner._req_caches[state.owner_request_id],
            owner_request_id=state.owner_request_id,
            tenant_id=request.tenant_id,
            session_id=state.session_id,
            generation=state.generation,
            source_tokens=len(source),
        )
        native_id = str(request.request_id)
        lease = self.runtime.begin_request(
            native_id,
            state.source_id,
            plan,
            tenant_id=request.tenant_id,
            session_id=state.session_id,
            expected_generation=state.generation,
            disjoint=not plan.full_retention,
        )
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if reset_peak is not None:
            reset_peak()
        active_before = int(getattr(mx, "get_active_memory", lambda: 0)())
        started = time.perf_counter()
        outcome = "error"
        generated: list[int] = []
        extra_prediction = 0
        evaluations = 0
        penalty_metrics: dict[str, int] = {}
        self._repetition.register(
            native_id,
            sampler_prompt,
            penalty=repetition_penalty,
            repeat_last_n=repeat_last_n,
        )
        try:
            # ``new_token_ids`` remains only the wire suffix: selected history
            # is already resident K/V. ``full_token_ids`` is logical sampler
            # state, so repetition sees exactly the same selected subset.
            pending = self.runner.prefill_start(
                native_id, wire, sampler_prompt, [], [], 0
            )
            self._eval(pending)
            token = int(self.runner.prefill_finalize(pending))
            evaluations += 1
            eos = self._eos_ids()
            max_tokens = request.resolved_max_new_tokens
            while len(generated) < max_tokens:
                if token in eos:
                    break
                generated.append(token)
                if len(generated) >= max_tokens:
                    break
                decode = self.runner.decode_batch_start([native_id])
                self._eval(decode)
                token = int(self.runner.decode_batch_finalize(decode)[0])
                evaluations += 1

            # Commit the final emitted token (or EOS) into local K/V.  The
            # newly sampled value is stale look-ahead and is not grafted.
            decode = self.runner.decode_batch_start([native_id])
            self._eval(decode)
            extra_prediction = int(self.runner.decode_batch_finalize(decode)[0])
            evaluations += 1
            local_tokens, graft_bytes = self._graft_local_kv(
                self.runner._req_caches[state.owner_request_id],
                self.runner._req_caches[native_id],
                source_tokens=len(source),
            )
            expected_local = len(wire) + len(generated) + int(token in eos)
            if local_tokens != expected_local:
                raise RuntimeError(
                    "Canonical K/V graft length disagrees with generated token history: "
                    f"cache={local_tokens}, expected={expected_local}."
                )
            state.canonical_tokens = [*source, *wire, *generated]
            if token in eos:
                state.canonical_tokens.append(token)
            self.runner._req_token_ids[state.owner_request_id] = [
                *state.canonical_tokens,
                extra_prediction,
            ]
            state.canonical_append_copy_bytes += graft_bytes
            text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
            state.ledger.append_assistant(text)
            outcome = "finished"
        finally:
            penalty_metrics = self._repetition.unregister(native_id)
            if outcome == "finished":
                lease.finish()
            else:
                lease.fail()

        mx.eval(
            *(
                value
                for cache in self.runner._req_caches[state.owner_request_id]
                for value in getattr(cache, "state", ())
            )
        )
        active_after = int(getattr(mx, "get_active_memory", lambda: 0)())
        peak = int(getattr(mx, "get_peak_memory", lambda: 0)())
        active_delta = max(active_after - active_before, 0)
        peak_delta = max(peak - active_before, 0)
        state.calls += 1
        elapsed = time.perf_counter() - started
        copy_metrics = _live_kv_copy_metrics(
            selected_kv_tokens=plan.selected_tokens,
            canonical_suffix_graft_d2d_bytes=graft_bytes,
        )
        trace = {
            "stage": "native_attach",
            "engine": "sglang-mlx",
            "native_kv_used": True,
            "native_attached_resources": [
                resource.resource_id for resource in request.resources
            ],
            "source_tokens": len(source),
            "selected_kv_tokens": copy_metrics["selected_kv_tokens"],
            "wire_tokens": len(wire),
            "requested_retention_fraction": requested,
            "realized_retention_fraction": plan.selected_tokens / max(len(source), 1),
            "engine_reported_history_kv_retention_fraction": (
                plan.selected_tokens / max(len(source), 1)
            ),
            "full_retention": bool(plan.full_retention),
            "selection_contract": selection_contract or "minimum-retention-floor",
            "pra_100_semantic_noop": bool(plan.full_retention),
            "exact_trajectory_eligible": bool(plan.full_retention),
            "realized_historical_kv_retention_fraction": (
                plan.selected_tokens / max(len(source), 1)
            ),
            "selected_history_reencoded_tokens": reencoded,
            "new_history_encoded_tokens": newly_encoded,
            "physical_kv_copy": copy_metrics["physical_kv_copy"],
            "selected_interval_copy_bytes": copy_metrics[
                "selected_interval_copy_bytes"
            ],
            "selected_history_kv_copy_bytes": copy_metrics[
                "selected_history_kv_copy_bytes"
            ],
            # The stable public counter denotes selected-history
            # materialization only. Canonical growth has a separate lifecycle
            # counter so it cannot be mistaken for a zero-copy selection.
            "physical_kv_copy_bytes": copy_metrics["physical_kv_copy_bytes"],
            "canonical_suffix_graft_d2d_bytes": copy_metrics[
                "canonical_suffix_graft_d2d_bytes"
            ],
            "total_kv_copy_bytes": copy_metrics["total_kv_copy_bytes"],
            "host_to_device_bytes": copy_metrics["host_to_device_bytes"],
            "consumer_temporary_active_delta_bytes": active_delta,
            "consumer_temporary_bytes": peak_delta,
            "consumer_temporary_peak_bytes": peak_delta,
            "consumer_temporary_measurement_scope": (
                "total request peak above pre-attach active allocation"
            ),
            "fused_attention_calls": evaluations * self.bridge.patched_layers,
            "source_position_base": plan.source_position_base,
            "selection_plan": plan.to_dict(),
            "local_tokens_grafted": local_tokens,
            "repetition_penalty": repetition_penalty,
            "repeat_last_n": repeat_last_n,
            "repetition_penalty_application_calls": penalty_metrics.get(
                "application_calls", 0
            ),
            "repetition_penalty_history_tokens_first_step": penalty_metrics.get(
                "history_tokens_first_step", 0
            ),
            "repetition_penalty_unique_tokens_first_step": penalty_metrics.get(
                "unique_tokens_first_step", 0
            ),
            "repetition_penalty_unique_tokens_max": penalty_metrics.get(
                "unique_tokens_max", 0
            ),
            "repetition_penalty_lazy_mlx": True,
            "repetition_penalty_cpu_syncs": 0,
            "sampler_prompt_tokens": len(sampler_prompt),
            "sampler_prompt_selected_history_tokens": plan.selected_tokens,
            "sampler_prompt_wire_tokens": len(wire),
            "logical_prompt_tokens": len(prompt),
            "effective_attention_prompt_tokens": plan.selected_tokens + len(wire),
            "completion_tokens": len(generated),
            "sampler_prompt_sha256": _token_digest(sampler_prompt),
            "elapsed_seconds": elapsed,
            "chat_template_profile": self.chat_template_profile,
            "chat_template_digest": self.chat_template_digest,
        }
        raw = {
            "native_attached_resources": trace["native_attached_resources"],
            "native_attach_bytes": copy_metrics["native_attach_bytes"],
            "prefix_cache_hit": bool(len(source) - newly_encoded),
            "prefix_cached_tokens": len(source) - newly_encoded,
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt) + len(generated),
            },
            "pra": dict(trace),
        }
        return PRAEngineResult(text=text, raw=raw, trace=(trace,))

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult:
        if block_store is not self.block_store:
            raise ValueError("SGLang executor and adapter must share one block store.")
        with self._lock:
            native = (
                request.metadata.get("history_projection") == "live-agent-kv-v1"
                or bool(request.resources)
            )
            if not native:
                repetition_penalty, repeat_last_n = self._repetition_settings(request)
                prompt = _render(
                    self.tokenizer,
                    request.messages,
                    generation_prompt=True,
                    chat_template_kwargs=self._checked_template_kwargs(request),
                )
                text, generated, elapsed, penalty_metrics = self._ordinary_generate(
                    str(request.request_id),
                    prompt,
                    request.resolved_max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    repeat_last_n=repeat_last_n,
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
                        "native_kv_used": False,
                        "prompt_tokens": len(prompt),
                        "completion_tokens": len(generated),
                        "elapsed_seconds": elapsed,
                        "repetition_penalty": repetition_penalty,
                        "repeat_last_n": repeat_last_n,
                        "repetition_penalty_application_calls": penalty_metrics.get(
                            "application_calls", 0
                        ),
                        "repetition_penalty_history_tokens_first_step": (
                            penalty_metrics.get("history_tokens_first_step", 0)
                        ),
                        "repetition_penalty_unique_tokens_first_step": (
                            penalty_metrics.get("unique_tokens_first_step", 0)
                        ),
                        "repetition_penalty_unique_tokens_max": (
                            penalty_metrics.get("unique_tokens_max", 0)
                        ),
                        "repetition_penalty_lazy_mlx": True,
                        "repetition_penalty_cpu_syncs": 0,
                        "sampler_prompt_tokens": len(prompt),
                        "sampler_prompt_sha256": _token_digest(prompt),
                        "chat_template_profile": self.chat_template_profile,
                        "chat_template_digest": self.chat_template_digest,
                    },),
                )
            state = self._session(request)
            return self._native_generate(request, state)

    def stream(self, request: PRAWireRequest, block_store: LogicalPRABlockStore):
        raise RuntimeError("Frozen SGLang agent gate is intentionally non-streaming.")

    def close_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(str(session_id), None)
            if state is None:
                return
            self.runtime.terminate_session(state.tenant_id, state.session_id)
            if self.runner.has_request(state.owner_request_id):
                self.runner.remove_request(state.owner_request_id)

    def close(self) -> None:
        with self._lock:
            for state in tuple(self._sessions.values()):
                self.runtime.terminate_session(state.tenant_id, state.session_id)
                if self.runner.has_request(state.owner_request_id):
                    self.runner.remove_request(state.owner_request_id)
            self._sessions.clear()
            self.runtime.close()
            self.bridge.close()
