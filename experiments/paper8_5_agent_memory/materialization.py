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
    TOOL_STRUCTURED_EVIDENCE = "tool_structured_evidence"
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
            evidence_query = query
            if record.content and evidence_query.endswith(record.content):
                evidence_query = evidence_query[: -len(record.content)]
            terms = _query_terms(evidence_query)
            scored = []
            for index, line in enumerate(lines):
                line_terms = _query_terms(line)
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
        elif self.mode == MaterializationMode.TOOL_STRUCTURED_EVIDENCE:
            indices = _structured_evidence_indices(
                record,
                lines,
                query=query,
                context_lines=self.match_context_lines,
                max_anchor_lines=self.max_matched_lines,
            )
            # Unknown output is not safely compressible.  A structured policy
            # must fail closed instead of quietly degenerating to arbitrary
            # head/tail sampling.
            if not indices:
                return MaterializedRecord(
                    record.record_id,
                    record.content,
                    MaterializationMode.WHOLE_RECORD,
                    original_tokens,
                    original_tokens,
                )
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


_TERM = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
_NUMBER_TERM = re.compile(r"\b\d{2,}\b")
_PATH_LINE = re.compile(
    r"(?:^|\s)(?:\.?\.?/)?(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9_+-]+(?::\d+)?"
)
_FAILURE = re.compile(
    r"(?:Traceback|AssertionError|[A-Za-z]+(?:Error|Exception)\b|"
    r"^E\s+|^FAILED\b|^ERROR\b|\bfailed\b|\bfailure\b)",
    re.IGNORECASE,
)
_DIFF_STRUCTURE = re.compile(r"^(?:diff --git|index |--- |\+\+\+ |@@ )")
_SOURCE_STRUCTURE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|struct|enum)\s+[A-Za-z_]"
)
_SEARCH_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:find\b|fd\b|rg\s+--files\b|"
    r"(?:grep|rg)\b[^\n]*(?:\s-(?:[^\s]*l[^\s]*|files-with-matches)\b))"
)
_VERIFY_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|tox|nox|python\s+-m\s+(?:pytest|unittest)|"
    r"make\s+(?:test|check|lint)|ruff|mypy|npm\s+test|cargo\s+test)\b"
)
_DIFF_COMMAND = re.compile(r"(?:^|[;&|]\s*)git\s+(?:diff|show)\b")

_QUERY_STOPWORDS = {
    "about", "after", "again", "agent", "before", "change", "code", "file",
    "from", "have", "into", "issue", "make", "only", "output", "reported",
    "should", "task", "that", "their", "then", "this", "tool", "using", "with",
}


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for raw in _TERM.findall(query):
        term = raw.lower()
        if len(term) <= 3 or term in _QUERY_STOPWORDS:
            continue
        terms.add(term)
        terms.add(re.sub(r":\d+$", "", term))
        if "/" in term:
            terms.add(term.rsplit("/", 1)[-1])
    # Multi-digit line numbers, status codes, and expected values are often the
    # only discriminator between otherwise identical source/test lines.
    terms.update(_NUMBER_TERM.findall(query))
    return terms


def _structured_evidence_indices(
    record: AgentRecord,
    lines: list[str],
    *,
    query: str,
    context_lines: int,
    max_anchor_lines: int,
) -> set[int]:
    """Select command-aware evidence lines, never arbitrary output positions.

    The scorer uses only information already visible at the current decision:
    the task/current query, the executed command stored on the observation, and
    the observation itself.  It prefers failure/traceback, diff, source-symbol,
    path, and lexical evidence.  Structural envelopes are retained separately.
    If no positive evidence exists, the caller keeps the whole record.
    """

    command = record.command or str(record.metadata.get("observation_for_command") or "")
    # The active observation is conventionally appended to the routing query.
    # It must not retrieve itself: doing so assigns lexical credit to every
    # line, including the exact noise that materialization is meant to remove.
    evidence_query = query
    if record.content and evidence_query.endswith(record.content):
        evidence_query = evidence_query[: -len(record.content)]
    terms = _query_terms(evidence_query)
    command_resources = {
        value.lower().replace("\\", "/")
        for value in record.resource_ids
        if value
    }
    verify = bool(_VERIFY_COMMAND.search(command))
    diff = bool(_DIFF_COMMAND.search(command))
    search = bool(_SEARCH_COMMAND.search(command))
    scores: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped in {"<output>", "</output>"} or stripped.startswith(
            "<returncode>"
        ):
            continue
        lower = stripped.lower().replace("\\", "/")
        line_terms = _query_terms(stripped)
        lexical = len(terms.intersection(line_terms))
        resource_match = sum(
            1 for resource in command_resources
            if resource in lower or resource.rsplit("/", 1)[-1] in lower
        )
        score = lexical * 8 + resource_match * 12
        failure = bool(_FAILURE.search(stripped))
        path = bool(_PATH_LINE.search(stripped))
        if failure:
            score += 80 if verify else 45
        # A verbose test run may contain thousands of passing-test paths. A
        # path is evidence only when the line is failing, task/resource linked,
        # or the command is not a verifier.
        if path and (not verify or failure or lexical or resource_match):
            score += 35
        if _DIFF_STRUCTURE.search(stripped):
            score += 70 if diff else 30
        if _SOURCE_STRUCTURE.search(stripped):
            score += 30
        if search and path:
            score += 20
        if score > 0:
            scores.append((score, index))

    anchors = [
        index for _, index in sorted(scores, key=lambda row: (-row[0], row[1]))[
            :max_anchor_lines
        ]
    ]
    selected: set[int] = set()
    for anchor in anchors:
        selected.update(range(
            max(0, anchor - context_lines),
            min(len(lines), anchor + context_lines + 1),
        ))

    # Keep the conventional result envelope, but do not use unrelated head or
    # tail body lines as evidence.
    head, _, tail, _ = _tool_observation_regions(lines)
    selected.update(range(len(head)))
    if tail:
        selected.update(range(len(lines) - len(tail), len(lines)))
    return selected if anchors else set()


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
