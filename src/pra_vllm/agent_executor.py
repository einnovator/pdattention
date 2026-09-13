"""Stateful mini-swe-agent bridge for vLLM V1 scheduler page aliases.

Only complete pages are canonical.  The external request contributes a newly
rendered suffix; compact selected token IDs exist solely as scheduler geometry
and are acknowledged as computed by the authoritative alias callbacks.  They
are never sent through the model or copied to the worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pra_hf.agent_executor import AgentHistoryLedger
from pra_hf.deployment import PRAEngineResult, PRAWireRequest

from .cuda_sparse_protocol import SparseCudaConnectorCommand


def _encode(tokenizer: object, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    return list(map(int, values))


def _render_text(
    tokenizer: object,
    messages: Sequence[Mapping[str, Any]],
    *,
    generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    value = tokenizer.apply_chat_template(
        [dict(row) for row in messages],
        tokenize=False,
        add_generation_prompt=generation_prompt,
        **dict(template_kwargs),
    )
    if not isinstance(value, str):
        raise TypeError("vLLM agent bridge requires a text-rendering chat template.")
    return value


def _page_indices_for_spans(
    spans: Mapping[int, tuple[int, int]],
    selected_message_indices: Sequence[int],
    *,
    source_tokens: int,
    block_size: int,
) -> tuple[int, ...]:
    selected: set[int] = set()
    for message_index in selected_message_indices:
        if int(message_index) not in spans:
            raise RuntimeError(
                f"Selected message {message_index} has no resident token span."
            )
        start, end = spans[int(message_index)]
        start, end = max(0, start), min(source_tokens, end)
        for page in range(start // block_size, (end + block_size - 1) // block_size):
            if (page + 1) * block_size <= source_tokens:
                selected.add(page)
    return tuple(sorted(selected))


def _enforce_page_retention_floor(
    *,
    selected_tokens: int,
    source_tokens: int,
    requested_fraction: float,
    selection_contract: str | None,
) -> None:
    """Fail closed when page-rounded selection underfills its declared arm."""

    minimum = math.ceil(float(requested_fraction) * int(source_tokens))
    if int(selected_tokens) >= minimum:
        return
    if selection_contract == "arbitrary-subset-mechanism-probe":
        return
    raise RuntimeError(
        "Selected vLLM pages underfill the requested retention floor: "
        f"selected={selected_tokens}, required={minimum}, "
        f"source={source_tokens}, requested={requested_fraction:.6f}. "
        "Record/block-aligned selection must round up; label a deliberately "
        "arbitrary subset with selection_contract="
        "'arbitrary-subset-mechanism-probe'."
    )


def validate_retention_fractions(values: Sequence[float]) -> tuple[float, ...]:
    """Validate an ordered bridge-smoke arm set anchored by PRA-100."""

    fractions = tuple(float(value) for value in values)
    if not fractions:
        raise ValueError("At least one retention fraction is required.")
    if len(set(fractions)) != len(fractions):
        raise ValueError("Retention fractions must be unique.")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("Retention fractions must be in (0, 1].")
    if 1.0 not in fractions:
        raise ValueError("The frozen qualification must include PRA-100.")
    return fractions


def record_rounded_selected_indices(
    messages: Sequence[Mapping[str, Any]],
    spans: Mapping[int, tuple[int, int]],
    *,
    source_tokens: int,
    block_size: int,
    retention_fraction: float,
    required_message_indices: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Round a logical selection upward with complete causal groups/pages.

    ``required_message_indices`` is the selector's decision. Page rounding may
    add older causal groups, but it must never remove a selected record merely
    to obtain a numerically closer physical retention fraction.
    """

    groups: list[list[int]] = (
        [[0, 1]] if len(messages) >= 2 else [list(range(len(messages)))]
    )
    for index, message in enumerate(messages[2:], start=2):
        if str(message.get("role", "")) == "assistant" or len(groups) == 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    required_indices = {int(index) for index in required_message_indices}
    mandatory_groups = {
        0,
        *range(max(1, len(groups) - 2), len(groups)),
        *(
            group_index
            for group_index, group in enumerate(groups)
            if required_indices.intersection(group)
        ),
    }
    eligible = [index for index in range(len(groups)) if index not in mandatory_groups]
    required = math.ceil(source_tokens * float(retention_fraction))
    selected_groups = set(mandatory_groups)

    def materialize() -> tuple[tuple[int, ...], tuple[int, ...]]:
        selected = tuple(
            message_index
            for group_index, group in enumerate(groups)
            if group_index in selected_groups
            for message_index in group
        )
        return selected, _page_indices_for_spans(
            spans,
            selected,
            source_tokens=source_tokens,
            block_size=block_size,
        )

    selected, pages = materialize()
    # Prefer the most recent omitted causal state when physical page rounding
    # needs more K/V than the logical selector requested.
    for group_index in reversed(eligible):
        if len(pages) * block_size >= required:
            break
        selected_groups.add(group_index)
        selected, pages = materialize()
    if len(pages) * block_size < required:
        raise RuntimeError(
            "No whole old causal-group combination yields a sparse page-rounded "
            f"selection at or above {retention_fraction:.3f}."
        )
    return selected, pages


@dataclass(frozen=True)
class VLLMGenerationReceipt:
    request_id: str
    text: str
    token_ids: tuple[int, ...]
    callback_delta: Mapping[str, int]
    committed_source: tuple[str, int, int] | None = None
    consumer_temporary_bytes: int | None = None
    elapsed_seconds: float = 0.0


class VLLMAgentSchedulerDriver(Protocol):
    block_size: int

    def generate_plain(
        self, prompt_token_ids: Sequence[int], *, max_tokens: int
    ) -> VLLMGenerationReceipt: ...

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        command: SparseCudaConnectorCommand,
        *,
        max_tokens: int,
        commit_source: tuple[str, int] | None = None,
        selected_page_indices: Sequence[int] = (),
        parent_source_key: str | None = None,
    ) -> VLLMGenerationReceipt: ...

    def evict_source(self, logical_key: str, generation: int) -> tuple[int, ...]: ...

    def terminate_source(self, logical_key: str, generation: int) -> bool: ...


class VLLMInProcessSchedulerDriver:
    """Thin vLLM 0.28 driver that requires all scheduler alias callbacks."""

    REQUIRED_CALLBACKS = (
        "alias_hit_events",
        "alias_prepare_events",
        "alias_commit_events",
        "alias_release_events",
    )

    def __init__(self, llm: object) -> None:
        self.llm = llm
        try:
            engine = llm.llm_engine.engine_core.engine_core
            self.connector = engine.scheduler.connector
            self.block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
        except AttributeError as error:
            raise RuntimeError(
                "Stateful vLLM PRA requires the in-process 0.28 V1 scheduler."
            ) from error
        required = (
            "_scheduler_alias_registry",
            "_directory",
            "pop_scheduler_committed_source",
            "evict_scheduler_source",
            "terminate_scheduler_source",
        )
        missing = [name for name in required if not hasattr(self.connector, name)]
        if missing:
            raise RuntimeError(
                "vLLM scheduler PRA callbacks are absent: " + ", ".join(missing)
            )
        telemetry = self.connector._scheduler_alias_registry.telemetry().__dict__
        missing_metrics = [name for name in self.REQUIRED_CALLBACKS if name not in telemetry]
        if missing_metrics:
            raise RuntimeError(
                "vLLM scheduler callback receipts are absent: "
                + ", ".join(missing_metrics)
            )

    def _telemetry(self) -> dict[str, int]:
        values = self.connector._scheduler_alias_registry.telemetry().__dict__
        return {key: int(value) for key, value in values.items()}

    def _manifest(
        self,
        command: SparseCudaConnectorCommand,
        *,
        parent_source_key: str,
        selected_page_indices: Sequence[int],
        commit_source: tuple[str, int],
    ) -> None:
        directory = Path(self.connector._directory(command.logical_key))
        directory.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": "pra-vllm-cuda-agent-alias-v1",
            "logical_key": command.logical_key,
            "source_tokens": command.source_tokens,
            "source_generation": command.source_generation,
            "parent_logical_key": str(parent_source_key),
            "selected_page_indices": list(map(int, selected_page_indices)),
            "commit_source_logical_key": str(commit_source[0]),
            "commit_source_generation": int(commit_source[1]),
            "physical_kv_copy_bytes": 0,
            "host_to_device_bytes": 0,
        }
        (directory / "manifest.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        command: SparseCudaConnectorCommand,
        *,
        max_tokens: int,
        commit_source: tuple[str, int] | None = None,
        selected_page_indices: Sequence[int] = (),
        parent_source_key: str | None = None,
    ) -> VLLMGenerationReceipt:
        if command.mode == "load":
            if commit_source is None or parent_source_key is None:
                raise ValueError("Sparse load requires an explicit next source generation.")
            self._manifest(
                command,
                parent_source_key=parent_source_key,
                selected_page_indices=selected_page_indices,
                commit_source=commit_source,
            )
        from vllm import SamplingParams

        before = self._telemetry()
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            active = int(torch.cuda.memory_allocated()) if cuda else 0
            if cuda:
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover - optional measurement only
            torch, cuda, active = None, False, 0
        started = time.perf_counter()
        outputs = self.llm.generate(
            {
                "prompt_token_ids": list(map(int, prompt_token_ids)),
                "cache_salt": command.cache_salt(),
            },
            SamplingParams(temperature=0, max_tokens=int(max_tokens)),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        if len(outputs) != 1 or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no completion for the agent request.")
        row, candidate = outputs[0], outputs[0].outputs[0]
        after = self._telemetry()
        delta = {key: after[key] - before.get(key, 0) for key in after}
        if command.mode == "load":
            missing = [name for name in self.REQUIRED_CALLBACKS if delta.get(name) != 1]
            if missing:
                raise RuntimeError(
                    "vLLM sparse request missed scheduler callbacks: "
                    + ", ".join(missing)
                )
        elif delta.get("source_pin_events") != 1:
            raise RuntimeError("vLLM store request did not publish its source pages.")
        committed = self.connector.pop_scheduler_committed_source(row.request_id)
        if command.mode == "load" and committed is None:
            available = sorted(
                map(
                    str,
                    getattr(
                        self.connector, "_scheduler_committed_sources", {}
                    ),
                )
            )
            snapshot = self.connector._scheduler_alias_registry.snapshot()
            raise RuntimeError(
                "vLLM load request did not commit its next source generation: "
                f"output_request_id={row.request_id!s}, "
                f"available_commit_receipts={available}, "
                f"active_aliases={snapshot['active_requests']}, "
                f"pending_aliases={snapshot['pending_requests']}."
            )
        temporary = None
        if cuda and torch is not None:
            temporary = max(int(torch.cuda.max_memory_allocated()) - active, 0)
        return VLLMGenerationReceipt(
            str(row.request_id),
            str(candidate.text),
            tuple(map(int, candidate.token_ids)),
            delta,
            committed,
            temporary,
            elapsed,
        )

    def generate_plain(
        self, prompt_token_ids: Sequence[int], *, max_tokens: int
    ) -> VLLMGenerationReceipt:
        """Generate without scheduler aliases for the matched no-PRA control."""

        from vllm import SamplingParams

        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            active = int(torch.cuda.memory_allocated()) if cuda else 0
            if cuda:
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover - optional measurement only
            torch, cuda, active = None, False, 0
        started = time.perf_counter()
        outputs = self.llm.generate(
            {"prompt_token_ids": list(map(int, prompt_token_ids))},
            SamplingParams(temperature=0, max_tokens=int(max_tokens)),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        if len(outputs) != 1 or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no completion for the plain request.")
        row, candidate = outputs[0], outputs[0].outputs[0]
        temporary = None
        if cuda and torch is not None:
            temporary = max(int(torch.cuda.max_memory_allocated()) - active, 0)
        return VLLMGenerationReceipt(
            str(row.request_id),
            str(candidate.text),
            tuple(map(int, candidate.token_ids)),
            {},
            None,
            temporary,
            elapsed,
        )

    def evict_source(self, logical_key: str, generation: int) -> tuple[int, ...]:
        return tuple(self.connector.evict_scheduler_source(
            logical_key, source_generation=int(generation)
        ))

    def terminate_source(self, logical_key: str, generation: int) -> bool:
        return bool(self.connector.terminate_scheduler_source(
            logical_key, source_generation=int(generation)
        ))


@dataclass
class _Session:
    tenant_id: str
    session_id: str
    source_key: str
    ledger: AgentHistoryLedger = field(default_factory=AgentHistoryLedger)
    history_text: str = ""
    history_tokens: list[int] = field(default_factory=list)
    message_spans: dict[int, tuple[int, int]] = field(default_factory=dict)
    source_tokens: int = 0
    generation: int = 1
    calls: int = 0


class VLLMCudaAgentHistoryExecutor:
    """Translate OpenAI agent turns into scheduler-owned page generations."""

    def __init__(
        self,
        driver: VLLMAgentSchedulerDriver,
        tokenizer: object,
        *,
        model_id: str,
        chat_template_digest: str,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if int(driver.block_size) <= 0:
            raise ValueError("vLLM driver block_size must be positive.")
        self.driver = driver
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.chat_template_digest = str(chat_template_digest)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.chat_template_profile = "append-stable"
        self.prefix_cache_enabled = True
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> Mapping[str, object]:
        return {
            "adapter": "vllm_cuda_agent_scheduler_alias",
            "integration_level": "E2",
            "engine_type": "vllm",
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
            "streaming": False,
            "prefix_cache_enabled": True,
            "automatic_prefix_cache": True,
            "prefix_cache_mode": "automatic_prefix_cache",
            "complete_pages_only": True,
            "multiprocess_qualified": False,
            "hybrid_kv_groups_qualified": False,
        }

    def _session(self, request: PRAWireRequest) -> _Session:
        if not request.session_id:
            raise ValueError("Stateful vLLM agent requests require session_id.")
        key = str(request.session_id)
        state = self._sessions.get(key)
        if state is None:
            digest = hashlib.sha256(key.encode()).hexdigest()[:20]
            state = _Session(str(request.tenant_id), key, f"agent-{digest}-g1")
            self._sessions[key] = state
        elif state.tenant_id != str(request.tenant_id):
            raise PermissionError("vLLM agent session crossed tenant scope.")
        return state

    def _update_new_message_spans(
        self,
        state: _Session,
        messages: Sequence[Mapping[str, Any]],
        prompt_text: str,
        prompt_tokens: Sequence[int],
    ) -> None:
        existing = len(state.message_spans)
        prior_end = max((end for _start, end in state.message_spans.values()), default=0)
        base_text = state.history_text
        base_tokens = len(state.history_tokens)
        if existing == 0:
            base_text, base_tokens = "", 0
        for index in range(existing, len(messages)):
            boundary_text = _render_text(
                self.tokenizer,
                messages[: index + 1],
                generation_prompt=False,
                template_kwargs=self.chat_template_kwargs,
            )
            if not boundary_text.startswith(base_text):
                raise RuntimeError("Chat template rewrote resident message history.")
            boundary = base_tokens + len(_encode(self.tokenizer, boundary_text[len(base_text) :]))
            state.message_spans[index] = (prior_end, boundary)
            prior_end = boundary
        if len(prompt_tokens) < prior_end or not prompt_text:
            raise RuntimeError("Incremental prompt geometry is inconsistent.")

    def _incremental_prompt(
        self, state: _Session, messages: Sequence[Mapping[str, Any]]
    ) -> tuple[str, list[int]]:
        text = _render_text(
            self.tokenizer,
            messages,
            generation_prompt=True,
            template_kwargs=self.chat_template_kwargs,
        )
        if state.history_text:
            if not text.startswith(state.history_text):
                raise RuntimeError("Chat template is not append-stable for this session.")
            tokens = [
                *state.history_tokens,
                *_encode(self.tokenizer, text[len(state.history_text) :]),
            ]
        else:
            tokens = _encode(self.tokenizer, text)
        self._update_new_message_spans(state, messages, text, tokens)
        return text, tokens

    def _commit_logical_response(
        self,
        state: _Session,
        prompt_text: str,
        prompt_tokens: list[int],
        text: str,
        generated: Sequence[int],
    ) -> None:
        assistant_index = len(state.ledger.messages)
        completed_messages = (
            *state.ledger.messages,
            {"role": "assistant", "content": str(text)},
        )
        completed = _render_text(
            self.tokenizer,
            completed_messages,
            generation_prompt=False,
            template_kwargs=self.chat_template_kwargs,
        )
        if not completed.startswith(prompt_text):
            raise RuntimeError("Completed assistant record rewrote its generation prefix.")
        delta = _encode(self.tokenizer, completed[len(prompt_text) :])
        generated = tuple(map(int, generated))
        if tuple(delta[: len(generated)]) != generated:
            raise RuntimeError(
                "vLLM output tokens do not match the append-stable assistant record."
            )
        start = max((end for _start, end in state.message_spans.values()), default=0)
        state.ledger.append_assistant(text)
        state.history_tokens = [*prompt_tokens, *delta]
        state.history_text = completed
        state.message_spans[assistant_index] = (start, len(state.history_tokens))

    @staticmethod
    def _selected_indices(request: PRAWireRequest) -> tuple[int, ...]:
        selected = {
            int(resource.metadata["message_index"])
            for resource in request.resources
            if resource.metadata.get("message_index") is not None
        }
        selected.update(map(int, request.metadata.get("mandatory_message_indices", ())))
        return tuple(sorted(selected))

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        with self._lock:
            if request.metadata.get("history_projection") != "live-agent-kv-v1":
                prompt_text = _render_text(
                    self.tokenizer,
                    request.messages,
                    generation_prompt=True,
                    template_kwargs=self.chat_template_kwargs,
                )
                prompt = _encode(self.tokenizer, prompt_text)
                receipt = self.driver.generate_plain(
                    prompt, max_tokens=request.resolved_max_new_tokens
                )
                trace = {
                    "stage": "plain_generation",
                    "engine": "vllm-cuda",
                    "native_kv_used": False,
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(receipt.token_ids),
                    "consumer_temporary_bytes": receipt.consumer_temporary_bytes,
                    "elapsed_seconds": receipt.elapsed_seconds,
                    "chat_template_digest": self.chat_template_digest,
                }
                return PRAEngineResult(
                    receipt.text,
                    raw={
                        "usage": {
                            "prompt_tokens": len(prompt),
                            "completion_tokens": len(receipt.token_ids),
                            "total_tokens": len(prompt) + len(receipt.token_ids),
                        },
                        "pra": {"native_kv_used": False},
                    },
                    trace=(trace,),
                )
            if str(request.metadata.get("chat_template_digest", self.chat_template_digest)) != self.chat_template_digest:
                raise ValueError("Request chat template digest does not match the engine.")
            state = self._session(request)
            messages = state.ledger.reconcile(request)
            prior_history_tokens = len(state.history_tokens)
            prompt_text, prompt = self._incremental_prompt(state, messages)
            new_request_tokens = len(prompt) - prior_history_tokens
            block = int(self.driver.block_size)
            old_source = state.source_key
            old_generation = state.generation
            old_source_tokens = state.source_tokens
            uncached_history_tokens = max(
                len(state.history_tokens) - state.source_tokens, 0
            )
            next_source_receipt: tuple[str, int, int] | None = None
            requested = float(request.metadata.get("target_retention_fraction", 1.0))
            selected_indices = tuple(range(len(messages)))
            requested_selected_indices = selected_indices
            retention_rounded_up = False
            if state.source_tokens == 0:
                source_tokens = (len(prompt) // block) * block
                if source_tokens <= 0:
                    raise RuntimeError("Initial agent prompt has no complete vLLM KV page.")
                command = SparseCudaConnectorCommand(
                    "store",
                    state.source_key,
                    source_tokens,
                    source_tokens,
                    source_generation=state.generation,
                    residency="warm",
                    request_scope=f"{state.session_id}-call1",
                )
                receipt = self.driver.generate(
                    prompt,
                    command,
                    max_tokens=request.resolved_max_new_tokens,
                )
                if int(receipt.callback_delta.get("source_pin_events", 0)) != 1:
                    raise RuntimeError("vLLM store request lacks its source-pin callback.")
                state.source_tokens = source_tokens
                page_indices = tuple(range(source_tokens // block))
                selected_tokens = source_tokens
                submitted_suffix_tokens = len(prompt)
                mode = "initial_store"
            else:
                selected_indices = self._selected_indices(request)
                requested_selected_indices = selected_indices
                selection_contract = request.metadata.get("selection_contract")
                if requested >= 1:
                    page_indices = tuple(range(state.source_tokens // block))
                    mode = "dense_semantic_noop"
                else:
                    page_indices = _page_indices_for_spans(
                        state.message_spans,
                        selected_indices,
                        source_tokens=state.source_tokens,
                        block_size=block,
                    )
                    mode = "sparse_original_position_pages"
                    selected_tokens = len(page_indices) * block
                    required_tokens = math.ceil(state.source_tokens * requested)
                    if (
                        selected_tokens < required_tokens
                        and selection_contract != "arbitrary-subset-mechanism-probe"
                    ):
                        selected_indices, page_indices = record_rounded_selected_indices(
                            messages,
                            state.message_spans,
                            source_tokens=state.source_tokens,
                            block_size=block,
                            retention_fraction=requested,
                            required_message_indices=selected_indices,
                        )
                        retention_rounded_up = True
                        mode = "sparse_original_position_pages_rounded_up"
                if not page_indices:
                    raise RuntimeError("PRA selection contains no complete resident page.")
                selected_tokens = len(page_indices) * block
                _enforce_page_retention_floor(
                    selected_tokens=selected_tokens,
                    source_tokens=state.source_tokens,
                    requested_fraction=requested,
                    selection_contract=(
                        None
                        if selection_contract is None
                        else str(selection_contract)
                    ),
                )
                compact = [
                    token
                    for page in page_indices
                    for token in state.history_tokens[page * block : (page + 1) * block]
                ]
                suffix = prompt[state.source_tokens :]
                if not suffix:
                    raise RuntimeError("vLLM alias request requires a new query suffix.")
                selected_key = f"{state.session_id}-selection-{state.calls + 1}"
                next_generation = state.generation + 1
                next_key = f"{state.session_id}-source-g{next_generation}"
                command = SparseCudaConnectorCommand(
                    "load",
                    selected_key,
                    selected_tokens,
                    state.source_tokens,
                    source_generation=state.generation,
                    residency="hot",
                    request_scope=f"{state.session_id}-call{state.calls + 1}",
                )
                receipt = self.driver.generate(
                    [*compact, *suffix],
                    command,
                    max_tokens=request.resolved_max_new_tokens,
                    commit_source=(next_key, next_generation),
                    selected_page_indices=page_indices,
                    parent_source_key=state.source_key,
                )
                required_callbacks = (
                    "alias_hit_events",
                    "alias_prepare_events",
                    "alias_commit_events",
                    "alias_release_events",
                )
                missing_callbacks = [
                    name
                    for name in required_callbacks
                    if int(receipt.callback_delta.get(name, 0)) != 1
                ]
                if missing_callbacks:
                    raise RuntimeError(
                        "vLLM sparse request lacks scheduler callbacks: "
                        + ", ".join(missing_callbacks)
                    )
                if receipt.committed_source is None:
                    raise RuntimeError("vLLM scheduler omitted its source-commit receipt.")
                committed_key, committed_generation, committed_tokens = receipt.committed_source
                if (committed_key, committed_generation) != (next_key, next_generation):
                    raise RuntimeError("vLLM committed a different source generation.")
                next_source_receipt = (
                    committed_key,
                    committed_generation,
                    committed_tokens,
                )
                submitted_suffix_tokens = len(suffix)

            self._commit_logical_response(
                state, prompt_text, prompt, receipt.text, receipt.token_ids
            )
            if next_source_receipt is not None:
                state.source_key, state.generation, state.source_tokens = (
                    next_source_receipt
                )
                self.driver.evict_source(old_source, old_generation)
            if state.source_tokens > len(state.history_tokens):
                raise RuntimeError("Committed K/V extent exceeds logical agent history.")
            state.calls += 1
            trace = {
                "stage": "vllm_scheduler_agent_alias",
                "engine": "vllm-cuda",
                "native_kv_used": True,
                "consumption_mode": mode,
                "source_logical_key": state.source_key,
                "source_generation": state.generation,
                "source_tokens": command.source_position_base,
                "canonical_source_tokens_after": state.source_tokens,
                "selected_source_generation": old_generation,
                "selected_page_indices": list(page_indices),
                "requested_selected_message_indices": list(
                    requested_selected_indices
                ),
                "realized_selected_message_indices": list(selected_indices),
                "retention_rounded_up": retention_rounded_up,
                "selected_kv_tokens": selected_tokens,
                "requested_retention_fraction": requested,
                "realized_retention_fraction": selected_tokens
                / max(command.source_position_base, 1),
                "realized_historical_kv_retention": selected_tokens
                / max(command.source_position_base, 1),
                "engine_reported_history_kv_retention_fraction": selected_tokens
                / max(command.source_position_base, 1),
                "full_retention": bool(
                    selected_tokens == command.source_position_base
                ),
                "selection_contract": request.metadata.get(
                    "selection_contract", "minimum-retention-floor"
                ),
                "selected_history_reencoded_tokens": 0,
                "uncached_partial_history_reencoded_tokens": uncached_history_tokens,
                "selected_history_kv_copy_bytes": 0,
                "selected_interval_copy_bytes": 0,
                "physical_kv_copy_bytes": 0,
                "host_to_device_bytes": 0,
                "total_kv_copy_bytes": 0,
                "consumer_temporary_bytes": receipt.consumer_temporary_bytes,
                "scheduler_geometry_token_ids": selected_tokens,
                "new_suffix_tokens_submitted": submitted_suffix_tokens,
                "new_request_suffix_tokens": new_request_tokens,
                "logical_prompt_tokens": len(prompt),
                "effective_attention_prompt_tokens": (
                    selected_tokens + submitted_suffix_tokens
                ),
                "completion_tokens": len(receipt.token_ids),
                "uncommitted_partial_page_tokens": len(state.history_tokens) - state.source_tokens,
                "scheduler_callback_delta": dict(receipt.callback_delta),
                "output_token_ids": list(receipt.token_ids),
                "session_call_index": state.calls,
                "elapsed_seconds": receipt.elapsed_seconds,
                "chat_template_digest": self.chat_template_digest,
            }
            return PRAEngineResult(
                receipt.text,
                raw={
                    "native_attach_bytes": 0,
                    "prefix_cache_hit": mode != "initial_store",
                    "prefix_cached_tokens": selected_tokens,
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": len(receipt.token_ids),
                        "total_tokens": len(prompt) + len(receipt.token_ids),
                    },
                    "pra": dict(trace),
                },
                trace=(trace,),
            )

    def close_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(str(session_id), None)
            if state is not None and state.source_tokens:
                self.driver.terminate_source(state.source_key, state.generation)
