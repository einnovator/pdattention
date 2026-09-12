"""Within-record realization policies, separate from record selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Callable

from .model import AgentMemoryPlan, AgentRecord, AgentRecordRole, CanonicalAgentHistory
from .selectors import TokenCounter, whitespace_tokens


class MaterializationMode(str, Enum):
    WHOLE_RECORD = "whole_record"
    TOOL_HEAD_TAIL = "tool_head_tail"
    TOOL_MATCHED_SPAN = "tool_matched_span"
    MATCHED_TOKEN_TAIL = "matched_token_tail"


@dataclass(frozen=True)
class MaterializedRecord:
    record_id: str
    content: str
    mode: MaterializationMode
    original_tokens: int
    materialized_tokens: int
    selected_line_spans: tuple[tuple[int, int], ...] = ()
    token_fallback_used: bool = False
    omitted_prefix_lines: int = 0


@dataclass(frozen=True)
class MaterializedMemoryPlan:
    logical_plan: AgentMemoryPlan
    records: tuple[MaterializedRecord, ...]
    full_selected_tokens: int
    materialized_tokens: int

    @property
    def detail_reduction_fraction(self) -> float:
        if self.full_selected_tokens == 0:
            return 0.0
        return 1.0 - self.materialized_tokens / self.full_selected_tokens

    @property
    def materialized_retention_fraction(self) -> float:
        if self.logical_plan.full_history_tokens == 0:
            return 1.0
        return self.materialized_tokens / self.logical_plan.full_history_tokens


class ToolObservationMaterializer:
    """Compact only oversized tool observations; all other records stay whole."""

    def __init__(
        self,
        *,
        mode: MaterializationMode = MaterializationMode.WHOLE_RECORD,
        threshold_tokens: int = 512,
        head_lines: int = 20,
        tail_lines: int = 30,
        match_context_lines: int = 4,
        max_matched_lines: int = 32,
    ) -> None:
        if threshold_tokens <= 0:
            raise ValueError("threshold_tokens must be positive")
        self.mode = mode
        self.threshold_tokens = threshold_tokens
        self.head_lines = head_lines
        self.tail_lines = tail_lines
        self.match_context_lines = match_context_lines
        self.max_matched_lines = max_matched_lines

    def materialize(
        self,
        record: AgentRecord,
        *,
        query: str,
        count_tokens: TokenCounter = whitespace_tokens,
    ) -> MaterializedRecord:
        original_tokens = count_tokens(record.content)
        if (
            self.mode == MaterializationMode.WHOLE_RECORD
            or not record.has_role(AgentRecordRole.TOOL_OBSERVATION)
            or original_tokens <= self.threshold_tokens
        ):
            return MaterializedRecord(
                record.record_id,
                record.content,
                MaterializationMode.WHOLE_RECORD,
                original_tokens,
                original_tokens,
            )

        lines = record.content.splitlines()
        if self.mode == MaterializationMode.TOOL_HEAD_TAIL:
            indices = set(range(min(self.head_lines, len(lines))))
            indices.update(range(max(0, len(lines) - self.tail_lines), len(lines)))
        elif self.mode == MaterializationMode.TOOL_MATCHED_SPAN:
            terms = {
                term.lower() for term in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_./:-]*", query)
                if len(term) > 2
            }
            scored = []
            for index, line in enumerate(lines):
                line_terms = {
                    term.lower() for term in re.findall(
                        r"[A-Za-z0-9_][A-Za-z0-9_./:-]*", line
                    )
                }
                overlap = len(terms.intersection(line_terms))
                if overlap:
                    scored.append((overlap, index))
            best_overlap = max((score for score, _ in scored), default=0)
            hits = [
                index for score, index in scored if score == best_overlap
            ][: self.max_matched_lines]
            indices = set(range(min(self.head_lines, len(lines))))
            indices.update(range(max(0, len(lines) - self.tail_lines), len(lines)))
            for hit in hits:
                indices.update(range(
                    max(0, hit - self.match_context_lines),
                    min(len(lines), hit + self.match_context_lines + 1),
                ))
        else:
            raise AssertionError(f"unsupported materialization mode: {self.mode}")

        ordered = sorted(indices)
        spans = _contiguous_spans(ordered)
        selected_lines: list[str] = []
        previous_end = 0
        for start, end in spans:
            if selected_lines and start > previous_end:
                selected_lines.append(f"<elided_lines>{start - previous_end}</elided_lines>")
            selected_lines.extend(lines[start:end])
            previous_end = end
        materialized = "\n".join(selected_lines)
        return MaterializedRecord(
            record.record_id,
            materialized,
            self.mode,
            original_tokens,
            count_tokens(materialized),
            spans,
        )


def _contiguous_spans(indices: list[int]) -> tuple[tuple[int, int], ...]:
    if not indices:
        return ()
    spans: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            spans.append((start, previous + 1))
            start = index
        previous = index
    spans.append((start, previous + 1))
    return tuple(spans)


def materialize_plan(
    history: CanonicalAgentHistory,
    plan: AgentMemoryPlan,
    materializer: ToolObservationMaterializer,
    *,
    query: str,
    count_tokens: TokenCounter = whitespace_tokens,
) -> MaterializedMemoryPlan:
    records = history.record_by_id
    rows = tuple(
        materializer.materialize(records[record_id], query=query, count_tokens=count_tokens)
        for record_id in plan.selected_record_ids
    )
    return MaterializedMemoryPlan(
        logical_plan=plan,
        records=rows,
        full_selected_tokens=sum(row.original_tokens for row in rows),
        materialized_tokens=sum(row.materialized_tokens for row in rows),
    )


def materialize_tool_observation_tail_to_ceiling(
    record: AgentRecord,
    *,
    max_tokens: int,
    count_tokens: TokenCounter = whitespace_tokens,
) -> MaterializedRecord | None:
    """Keep a tokenizer-measured suffix without exceeding ``max_tokens``.

    Whole output lines are the first truncation boundary.  If even the newest
    output line does not fit, a character-boundary suffix of that line is used
    as the token-budget fallback; every candidate is re-tokenized before it is
    accepted.  The return-code/output envelope is retained when it has the
    conventional mini-swe-agent form.  ``None`` means that even the structural
    envelope and explicit elision marker do not fit.

    This helper is intentionally limited to tool observations.  It must never
    be used to trim system, task, assistant-action, mutation, or verification
    records independently of their causal turn.
    """

    if max_tokens <= 0 or not record.has_role(AgentRecordRole.TOOL_OBSERVATION):
        return None
    original_tokens = count_tokens(record.content)
    if original_tokens <= max_tokens:
        return MaterializedRecord(
            record.record_id,
            record.content,
            MaterializationMode.WHOLE_RECORD,
            original_tokens,
            original_tokens,
        )

    lines = record.content.splitlines()
    if not lines:
        return None
    head, body, tail, body_offset = _tool_observation_regions(lines)

    def candidate(kept_body_lines: list[str], *, omitted: int) -> str:
        rows = list(head)
        if omitted:
            rows.append(f'<elided_prefix_lines count="{omitted}"/>')
        rows.extend(kept_body_lines)
        rows.extend(tail)
        return "\n".join(rows)

    minimal = candidate([], omitted=len(body))
    minimal_tokens = count_tokens(minimal)
    if minimal_tokens > max_tokens:
        return None

    # Find the largest whole-line suffix that fits. Token counts are measured
    # on the final serialized candidate, so the ceiling remains strict even
    # when tokenizer merges make counts mildly non-linear.
    low, high = 0, len(body)
    best_count = 0
    best_content = minimal
    best_tokens = minimal_tokens
    while low <= high:
        keep = (low + high) // 2
        content = candidate(body[len(body) - keep :] if keep else [], omitted=len(body) - keep)
        tokens = count_tokens(content)
        if tokens <= max_tokens:
            best_count, best_content, best_tokens = keep, content, tokens
            low = keep + 1
        else:
            high = keep - 1

    token_fallback = False
    fallback_omitted_lines: int | None = None
    if best_count == 0 and body:
        # A single output line may itself exceed the budget. Keep its newest
        # suffix, using character boundaries only to locate a string whose
        # *actual tokenizer count* respects the token ceiling.
        line = body[-1]
        low, high = 0, len(line)
        while low <= high:
            keep_chars = (low + high) // 2
            fragment = line[len(line) - keep_chars :] if keep_chars else ""
            if fragment:
                rows = list(head)
                if len(body) > 1:
                    rows.append(
                        f'<elided_prefix_lines count="{len(body) - 1}"/>'
                    )
                rows.extend(("<elided_prefix_tokens/>", fragment))
                rows.extend(tail)
                content = "\n".join(rows)
            else:
                content = candidate([], omitted=len(body))
            tokens = count_tokens(content)
            if tokens <= max_tokens:
                best_content, best_tokens = content, tokens
                token_fallback = bool(fragment)
                fallback_omitted_lines = len(body) - 1 if fragment else len(body)
                low = keep_chars + 1
            else:
                high = keep_chars - 1

    selected_spans = (
        ((body_offset + len(body) - best_count, body_offset + len(body)),)
        if best_count else ()
    )
    return MaterializedRecord(
        record.record_id,
        best_content,
        MaterializationMode.MATCHED_TOKEN_TAIL,
        original_tokens,
        best_tokens,
        selected_spans,
        token_fallback,
        (
            fallback_omitted_lines
            if fallback_omitted_lines is not None else len(body) - best_count
        ),
    )


def _tool_observation_regions(
    lines: list[str],
) -> tuple[list[str], list[str], list[str], int]:
    """Separate the conventional observation envelope from its output body."""

    output_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "<output>"),
        None,
    )
    output_end = next(
        (
            index for index in range(len(lines) - 1, -1, -1)
            if lines[index].strip() == "</output>"
        ),
        None,
    )
    if output_start is None or output_end is None or output_end <= output_start:
        return [], lines, [], 0
    return (
        lines[: output_start + 1],
        lines[output_start + 1 : output_end],
        lines[output_end:],
        output_start + 1,
    )
