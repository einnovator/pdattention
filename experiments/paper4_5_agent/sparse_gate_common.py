"""Shared causal-record geometry for live agent-history sparse gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan


def _ids(tokenizer, messages: Sequence[Mapping[str, object]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=False
    )
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered, add_special_tokens=False)
    elif isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def causal_message_spans(
    tokenizer,
    messages: Sequence[Mapping[str, object]],
    prompt_ids: Sequence[int],
    *,
    source_tokens: int,
) -> tuple[LiveKVInterval, ...]:
    """Map complete chat records into the canonical prompt-token frame.

    Prefix rendering is used instead of independently tokenizing message text,
    so role delimiters and the model's exact chat template remain part of the
    owning record. Assistant records and following observations share a causal
    group identity.
    """

    if source_tokens <= 0 or source_tokens > len(prompt_ids):
        raise ValueError("source_tokens must fit the rendered agent prompt.")
    boundaries = [0]
    for index in range(len(messages)):
        prefix = _ids(tokenizer, messages[: index + 1])
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


def sparse_causal_plan(
    tokenizer,
    messages: Sequence[Mapping[str, object]],
    prompt_ids: Sequence[int],
    *,
    source_tokens: int,
    retention_fraction: float,
) -> LiveKVSelectionPlan:
    """Drop whole old causal groups while pinning preamble and active state."""

    fraction = float(retention_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("retention_fraction must be in (0, 1].")
    if fraction == 1:
        return LiveKVSelectionPlan.full(source_tokens)
    spans = causal_message_spans(
        tokenizer, messages, prompt_ids, source_tokens=source_tokens
    )
    groups: dict[str, list[LiveKVInterval]] = {}
    order: list[str] = []
    for span in spans:
        if span.causal_group_id not in groups:
            groups[span.causal_group_id] = []
            order.append(span.causal_group_id)
        groups[span.causal_group_id].append(span)

    # The system/task preamble and two most recent completed causal groups are
    # mandatory. Retention is a floor: remove only whole old groups that fit
    # inside the requested drop allowance. A group that would cross the
    # allowance is skipped rather than turning nominal 90% retention into, for
    # example, 60%. This is a mechanism gate, not the efficacy selector.
    mandatory = {"preamble", *order[-2:]}
    drop_budget = max(0, source_tokens - math.ceil(source_tokens * fraction))
    dropped: set[str] = set()
    dropped_tokens = 0
    for group in order:
        if group in mandatory:
            continue
        cost = sum(span.tokens for span in groups[group])
        if dropped_tokens + cost <= drop_budget:
            dropped.add(group)
            dropped_tokens += cost
    if not dropped:
        raise RuntimeError("Agent prefix has no old causal group eligible for sparse selection.")
    selected = tuple(span for span in spans if span.causal_group_id not in dropped)
    return LiveKVSelectionPlan.create(
        source_tokens,
        selected,
        source_position_base=source_tokens,
    )
