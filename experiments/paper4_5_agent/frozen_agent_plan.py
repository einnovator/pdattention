"""Validate Paper 8.5 requests and map their records into live-K/V geometry.

The logical selector has already run.  This module never scores or changes a
record.  It binds an exact full request to its exported selected message IDs,
then expresses the historical subset as original-position token intervals and
the current causal tail as ordinary wire tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan

from .context_treatment import _load_selection_fixture, _selection_input_digest
from .sparse_gate_common import causal_message_spans


@dataclass(frozen=True)
class FrozenMaterializedMessage:
    record_id: str
    message_index: int
    role: str
    content: str
    content_sha256: str


@dataclass(frozen=True)
class FrozenAgentDecision:
    request_index: int
    request_input_sha256: str
    session_id: str
    logical_payload: Mapping[str, Any]
    expected_assistant_content: str
    expected_assistant_message: Mapping[str, Any] | None
    selected_message_indices: tuple[int, ...]
    mandatory_message_indices: tuple[int, ...]
    source_policy: str
    source_plan_digest: str
    materialized_message_replacements: tuple[FrozenMaterializedMessage, ...]


@dataclass(frozen=True)
class FrozenLiveKVGeometry:
    prompt_ids: tuple[int, ...]
    source_ids: tuple[int, ...]
    wire_tail_ids: tuple[int, ...]
    plan: LiveKVSelectionPlan
    selected_message_indices: tuple[int, ...]
    mandatory_message_indices: tuple[int, ...]
    materialized_history_spans: tuple["FrozenMaterializedSpan", ...] = ()

    @property
    def realized_retention_fraction(self) -> float:
        if not self.prompt_ids:
            return 1.0
        materialized = sum(len(span.token_ids) for span in self.materialized_history_spans)
        return (self.plan.selected_tokens + materialized + len(self.wire_tail_ids)) / len(
            self.prompt_ids
        )


@dataclass(frozen=True)
class FrozenMaterializedSpan:
    record_id: str
    message_index: int
    role: str
    token_ids: tuple[int, ...]
    position_start: int

    @property
    def position_end(self) -> int:
        return self.position_start + len(self.token_ids)


def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _chat_template_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract model-visible template inputs from the frozen request."""

    result = dict(payload.get("chat_template_kwargs") or {})
    for key in ("tools", "documents"):
        if key in payload:
            result[key] = payload[key]
    return result


def _template_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize OpenAI text parts without discarding native tool fields."""

    result: list[dict[str, Any]] = []
    for index, source in enumerate(messages):
        message = dict(source)
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "text":
                    raise ValueError(
                        "direct engine bridge supports only text content parts; "
                        f"message {index} contains an unsupported part"
                    )
                parts.append(str(part.get("text", "")))
            message["content"] = "".join(parts)
        elif content is None:
            message["content"] = ""
        elif not isinstance(content, str):
            raise ValueError(f"message {index} has unsupported content")
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            normalized_calls: list[dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, Mapping):
                    raise ValueError(f"message {index} has malformed tool call")
                normalized_call = dict(call)
                function = normalized_call.get("function")
                if not isinstance(function, Mapping):
                    raise ValueError(f"message {index} has malformed tool function")
                normalized_function = dict(function)
                arguments = normalized_function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"message {index} has non-JSON tool arguments"
                        ) from error
                if not isinstance(arguments, Mapping):
                    raise ValueError(
                        f"message {index} tool arguments are not an object"
                    )
                normalized_function["arguments"] = dict(arguments)
                normalized_call["function"] = normalized_function
                normalized_calls.append(normalized_call)
            message["tool_calls"] = normalized_calls
        result.append(message)
    return result


def _render_prompt(
    tokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered, add_special_tokens=False)
    elif isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("chat template returned a batched prompt")
        rendered = rendered[0]
    return [int(token) for token in rendered]


def load_frozen_agent_decisions(
    request_replay_path: Path,
    selection_fixture_path: Path,
) -> list[FrozenAgentDecision]:
    """Join exact requests to exact plans and reject every identity mismatch."""

    fixture = _load_selection_fixture(selection_fixture_path)
    plan_rows = [
        json.loads(line)
        for line in selection_fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan_by_digest = {
        str(row["request_input_sha256"]): row for row in plan_rows
    }
    if len(plan_by_digest) != len(plan_rows):
        raise ValueError("selection fixture repeats a request identity")
    request_rows = [
        json.loads(line)
        for line in request_replay_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not request_rows:
        raise ValueError("request replay is empty")
    decisions: list[FrozenAgentDecision] = []
    for ordinal, row in enumerate(request_rows, 1):
        if int(row.get("request_index", -1)) != ordinal:
            raise ValueError("request replay is not contiguous and one-based")
        payload = row.get("logical_payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"request {ordinal} has no logical payload")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"request {ordinal} has no logical messages")
        payload_sha256 = row.get("logical_payload_sha256")
        if payload_sha256 is not None and _json_sha256(payload) != payload_sha256:
            raise ValueError(f"request {ordinal} failed logical-payload identity")
        request_digest = _selection_input_digest(messages)
        if request_digest != row.get("request_input_sha256"):
            raise ValueError(f"request {ordinal} failed full-request identity")
        expected = str(row.get("expected_assistant_content", ""))
        if _content_sha256(expected) != row.get(
            "expected_assistant_content_sha256"
        ):
            raise ValueError(f"request {ordinal} failed response identity")
        expected_message = row.get("expected_assistant_message")
        if expected_message is not None:
            if not isinstance(expected_message, Mapping):
                raise ValueError(f"request {ordinal} has malformed assistant message")
            if _json_sha256(expected_message) != row.get(
                "expected_assistant_message_sha256"
            ):
                raise ValueError(f"request {ordinal} failed assistant-action identity")
        plan_row = plan_by_digest.get(request_digest)
        plan_entry = fixture.get(request_digest)
        if plan_row is None or plan_entry is None:
            raise ValueError(f"request {ordinal} has no frozen selection")
        selected = tuple(int(index) for index in plan_row["selected_message_indices"])
        if tuple(sorted(set(selected))) != selected:
            raise ValueError(f"request {ordinal} selected indices are not ordered unique")
        if any(index < 0 or index >= len(messages) for index in selected):
            raise ValueError(f"request {ordinal} selected index is outside the request")
        mandatory = plan_entry.mandatory_message_indices
        if mandatory is None:
            raise ValueError(f"request {ordinal} has no frozen mandatory floor")
        if not set(mandatory).issubset(selected):
            raise ValueError(f"request {ordinal} omits mandatory selected state")
        raw_replacements = plan_row.get("materialized_message_replacements", [])
        if not isinstance(raw_replacements, list):
            raise ValueError(f"request {ordinal} materialized replacements are not a list")
        replacements: list[FrozenMaterializedMessage] = []
        replacement_record_ids: set[str] = set()
        resource_text_by_index: dict[int, str] = {}
        for resource in plan_row.get("resources", ()):
            if not isinstance(resource, Mapping):
                continue
            resource_id = str(resource.get("resource_id", ""))
            prefix = resource_id.split("-", 1)[0]
            if not prefix.startswith("m") or not prefix[1:].isdigit():
                continue
            index = int(prefix[1:])
            resource_text_by_index[index] = (
                resource_text_by_index.get(index, "") + str(resource.get("text", ""))
            )
        for replacement in raw_replacements:
            if not isinstance(replacement, Mapping):
                raise ValueError(f"request {ordinal} has a malformed replacement")
            index = int(replacement.get("message_index", -1))
            record_id = str(replacement.get("record_id", ""))
            role = str(replacement.get("role", ""))
            content = str(replacement.get("content", ""))
            content_sha256 = str(replacement.get("content_sha256", ""))
            if not record_id or record_id in replacement_record_ids:
                raise ValueError(f"request {ordinal} replacement identity mismatch")
            replacement_record_ids.add(record_id)
            if index not in selected or index in mandatory:
                raise ValueError(f"request {ordinal} replacement is not selected history")
            if role != str(messages[index].get("role", "")):
                raise ValueError(f"request {ordinal} replacement role mismatch")
            if _content_sha256(content) != content_sha256:
                raise ValueError(f"request {ordinal} replacement content hash mismatch")
            if resource_text_by_index.get(index) != content:
                raise ValueError(f"request {ordinal} replacement resource mismatch")
            replacements.append(FrozenMaterializedMessage(
                record_id=record_id,
                message_index=index,
                role=role,
                content=content,
                content_sha256=content_sha256,
            ))
        decisions.append(FrozenAgentDecision(
            request_index=ordinal,
            request_input_sha256=request_digest,
            session_id=str(row.get("session_id", "")),
            logical_payload=payload,
            expected_assistant_content=expected,
            expected_assistant_message=(
                dict(expected_message) if expected_message is not None else None
            ),
            selected_message_indices=selected,
            mandatory_message_indices=mandatory,
            source_policy=str(plan_entry.source_policy or ""),
            source_plan_digest=str(plan_entry.source_plan_digest or ""),
            materialized_message_replacements=tuple(replacements),
        ))
    if set(plan_by_digest) != {row.request_input_sha256 for row in decisions}:
        raise ValueError("request replay and selection fixture have different cohorts")
    return decisions


def frozen_live_kv_geometry(
    tokenizer,
    decision: FrozenAgentDecision,
    *,
    full_retention: bool = False,
    prefix_ids_cache: MutableMapping[str, tuple[int, ...]] | None = None,
) -> FrozenLiveKVGeometry:
    """Split one exact request into resident historical K/V and a wire tail."""

    messages = _template_messages(decision.logical_payload["messages"])
    template_kwargs = _chat_template_kwargs(decision.logical_payload)
    prompt_ids = _render_prompt(
        tokenizer, messages, chat_template_kwargs=template_kwargs
    )
    if not prompt_ids:
        raise ValueError("chat template produced an empty prompt")
    active = [
        index for index in decision.mandatory_message_indices
        if str(messages[index].get("role", "")) != "system"
    ]
    if not active:
        raise ValueError("frozen plan has no current non-system causal tail")
    active_start = min(active)
    full_spans = causal_message_spans(
        tokenizer,
        messages,
        prompt_ids,
        source_tokens=len(prompt_ids),
        prefix_ids_cache=prefix_ids_cache,
        chat_template_kwargs=template_kwargs,
    )
    span_by_index: dict[int, LiveKVInterval] = {}
    for span in full_spans:
        parts = str(span.record_id).split(":", 2)
        if len(parts) == 3 and parts[0] == "message":
            span_by_index[int(parts[1])] = span
    if active_start not in span_by_index:
        raise ValueError("chat template did not expose the active-tail boundary")
    source_tokens = span_by_index[active_start].start
    if source_tokens <= 0 or source_tokens >= len(prompt_ids):
        raise ValueError("active causal tail does not split the rendered prompt")
    unexpectedly_visible = [
        index for index in range(active_start, len(messages))
        if index not in decision.selected_message_indices
    ]
    if unexpectedly_visible:
        raise ValueError(
            "wire tail would restore messages excluded by the frozen plan: "
            + ", ".join(str(index) for index in unexpectedly_visible)
        )
    source_spans = tuple(
        span for span in full_spans if span.end <= source_tokens
    )
    source_span_by_index = {
        int(str(span.record_id).split(":", 2)[1]): span for span in source_spans
    }
    replacement_indices = {
        replacement.message_index
        for replacement in decision.materialized_message_replacements
    }
    selected_source_indices = (
        set(range(active_start))
        if full_retention
        else {
            index for index in decision.selected_message_indices
            if index < active_start and index not in replacement_indices
        }
    )
    intervals = tuple(
        span for span in source_spans
        if int(str(span.record_id).split(":", 2)[1]) in selected_source_indices
    )
    plan = LiveKVSelectionPlan.create(
        source_tokens,
        intervals,
        source_position_base=source_tokens,
    )
    materialized_spans: list[FrozenMaterializedSpan] = []
    if not full_retention:
        for replacement in decision.materialized_message_replacements:
            source_span = source_span_by_index.get(replacement.message_index)
            if source_span is None:
                raise ValueError("materialized replacement is outside historical source")
            variant = [dict(message) for message in messages]
            variant[replacement.message_index]["content"] = replacement.content
            variant_prompt_ids = _render_prompt(
                tokenizer, variant, chat_template_kwargs=template_kwargs
            )
            variant_spans = causal_message_spans(
                tokenizer,
                variant,
                variant_prompt_ids,
                source_tokens=len(variant_prompt_ids),
                prefix_ids_cache=prefix_ids_cache,
                chat_template_kwargs=template_kwargs,
            )
            variant_span = next(
                (
                    span for span in variant_spans
                    if str(span.record_id).startswith(
                        f"message:{replacement.message_index}:"
                    )
                ),
                None,
            )
            if variant_span is None:
                raise ValueError("chat template did not expose replacement span")
            token_ids = tuple(
                int(value)
                for value in variant_prompt_ids[variant_span.start:variant_span.end]
            )
            if not token_ids:
                raise ValueError("materialized replacement produced no tokens")
            if len(token_ids) > source_span.tokens:
                raise ValueError(
                    "materialized replacement exceeds its original logical position span"
                )
            materialized_spans.append(FrozenMaterializedSpan(
                record_id=replacement.record_id,
                message_index=replacement.message_index,
                role=replacement.role,
                token_ids=token_ids,
                position_start=source_span.start,
            ))
    return FrozenLiveKVGeometry(
        prompt_ids=tuple(prompt_ids),
        source_ids=tuple(prompt_ids[:source_tokens]),
        wire_tail_ids=tuple(prompt_ids[source_tokens:]),
        plan=plan,
        selected_message_indices=decision.selected_message_indices,
        mandatory_message_indices=decision.mandatory_message_indices,
        materialized_history_spans=tuple(materialized_spans),
    )
