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


@dataclass(frozen=True)
class MaterializedRecord:
    record_id: str
    content: str
    mode: MaterializationMode
    original_tokens: int
    materialized_tokens: int
    selected_line_spans: tuple[tuple[int, int], ...] = ()


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
