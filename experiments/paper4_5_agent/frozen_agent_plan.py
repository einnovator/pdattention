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
class FrozenAgentDecision:
    request_index: int
    request_input_sha256: str
    session_id: str
    logical_payload: Mapping[str, Any]
    expected_assistant_content: str
    selected_message_indices: tuple[int, ...]
    mandatory_message_indices: tuple[int, ...]
    source_policy: str
    source_plan_digest: str


@dataclass(frozen=True)
class FrozenLiveKVGeometry:
    prompt_ids: tuple[int, ...]
    source_ids: tuple[int, ...]
    wire_tail_ids: tuple[int, ...]
    plan: LiveKVSelectionPlan
    selected_message_indices: tuple[int, ...]
    mandatory_message_indices: tuple[int, ...]

    @property
    def realized_retention_fraction(self) -> float:
        if not self.prompt_ids:
            return 1.0
        return (self.plan.selected_tokens + len(self.wire_tail_ids)) / len(
            self.prompt_ids
        )


def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _render_prompt(tokenizer, messages: Sequence[Mapping[str, Any]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True
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
        request_digest = _selection_input_digest(messages)
        if request_digest != row.get("request_input_sha256"):
            raise ValueError(f"request {ordinal} failed full-request identity")
        expected = str(row.get("expected_assistant_content", ""))
        if _content_sha256(expected) != row.get(
            "expected_assistant_content_sha256"
        ):
            raise ValueError(f"request {ordinal} failed response identity")
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
        decisions.append(FrozenAgentDecision(
            request_index=ordinal,
            request_input_sha256=request_digest,
            session_id=str(row.get("session_id", "")),
            logical_payload=payload,
            expected_assistant_content=expected,
            selected_message_indices=selected,
            mandatory_message_indices=mandatory,
            source_policy=str(plan_entry.source_policy or ""),
            source_plan_digest=str(plan_entry.source_plan_digest or ""),
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

    messages = list(decision.logical_payload["messages"])
    prompt_ids = _render_prompt(tokenizer, messages)
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
    selected_source_indices = (
        set(range(active_start))
        if full_retention
        else {
            index for index in decision.selected_message_indices
            if index < active_start
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
    return FrozenLiveKVGeometry(
        prompt_ids=tuple(prompt_ids),
        source_ids=tuple(prompt_ids[:source_tokens]),
        wire_tail_ids=tuple(prompt_ids[source_tokens:]),
        plan=plan,
        selected_message_indices=decision.selected_message_indices,
        mandatory_message_indices=decision.mandatory_message_indices,
    )
