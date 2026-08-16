"""Prompt-role and query-region discovery for adaptive PRA.

PRA routing needs a retrieval objective, but that objective is not necessarily
the final prompt suffix.  This module keeps explicit caller metadata as the
strongest signal and provides a deterministic structural fallback for raw text.
All intervals are half-open token spans ``[start, end)``.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence


_TOKEN = re.compile(r"\S+")
_QUERY_LABEL = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:query|question|instruction|task|request|objective)\s*[:\-]?",
    re.IGNORECASE,
)
_CONTEXT_LABEL = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:context|document|payload|logs?|tool output|references?|urls?)\s*[:\-]?",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_LOG = re.compile(
    r"(?:^\s*\d{4}-\d{2}-\d{2}[T ]|\b(?:ERROR|WARN|INFO|DEBUG|TRACE)\b|"
    r"Traceback \(most recent call last\)|^\s*at\s+\S+\([^)]*:\d+\))"
)
_IMPERATIVE = re.compile(
    r"^\s*(?:find|explain|identify|determine|summarize|compare|why|what|which|who|when|how)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptSegment:
    """One caller-labeled prompt segment before tokenization.

    ``role`` describes function rather than serialization position.  Public
    callers should prefer this representation to manually counted token spans.
    """

    role: str
    text: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.text:
            raise ValueError("Prompt segments require a nonempty role and text.")


@dataclass(frozen=True)
class QueryRegion:
    """A selected half-open token interval and its discovery provenance."""

    start: int
    end: int
    confidence: float
    method: str
    role: str = "query"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Query regions must satisfy 0 <= start < end.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Query-region confidence must lie in [0, 1].")

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class QueryRegionSelection:
    """Resolved query regions plus confidence signals for adaptive control."""

    prompt: str
    policy: str
    regions: tuple[QueryRegion, ...]
    token_count: int
    confidence: float
    score_margin: float

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("A query-region selection cannot be empty.")
        if any(region.end > self.token_count for region in self.regions):
            raise ValueError("A query region extends beyond the tokenized prompt.")

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        return tuple((region.start, region.end) for region in self.regions)

    def selected_text(self) -> tuple[str, ...]:
        tokens = [match.group(0) for match in _TOKEN.finditer(self.prompt)]
        return tuple(" ".join(tokens[region.start : region.end]) for region in self.regions)


def token_offsets(text: str) -> tuple[tuple[int, int], ...]:
    """Return deterministic whitespace-token character offsets for SDK utilities."""

    return tuple(match.span() for match in _TOKEN.finditer(text))


def _char_to_token_span(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    starts = [left for left, _ in offsets]
    ends = [right for _, right in offsets]
    first = bisect_right(ends, start)
    last = bisect_left(starts, end)
    return (first, last) if first < last else None


def render_segments(
    segments: Sequence[PromptSegment],
) -> tuple[str, tuple[tuple[PromptSegment, int, int], ...]]:
    """Serialize structured segments while retaining exact character ranges."""

    if not segments:
        raise ValueError("At least one prompt segment is required.")
    parts: list[str] = []
    ranges = []
    cursor = 0
    for index, segment in enumerate(segments):
        if index:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(segment.text)
        cursor += len(segment.text)
        ranges.append((segment, start, cursor))
    return "".join(parts), tuple(ranges)


def _merge_regions(regions: Iterable[QueryRegion]) -> tuple[QueryRegion, ...]:
    ordered = sorted(regions, key=lambda region: (region.start, region.end))
    merged: list[QueryRegion] = []
    for region in ordered:
        if merged and region.start < merged[-1].end:
            prior = merged[-1]
            merged[-1] = QueryRegion(
                prior.start,
                max(prior.end, region.end),
                max(prior.confidence, region.confidence),
                prior.method if prior.method == region.method else "combined",
                prior.role if prior.role == region.role else "multi_role",
            )
        else:
            merged.append(region)
    return tuple(merged)


class QueryRegionSelector:
    """Resolve explicit, structured, suffix, or structural query regions.

    Structural inference is intentionally lightweight.  It recognizes labeled
    sections, questions, and instruction-like lines while discounting URLs,
    logs, code fences, and context headings.  It is a baseline and confidence
    source, not a replacement for application-provided roles.
    """

    SUPPORTED_POLICIES = {
        "head",
        "suffix",
        "explicit",
        "structured",
        "structural",
        "auto",
        "multi_region",
        "session_state",
    }

    def __init__(self, *, suffix_tokens: int = 32, max_regions: int = 2) -> None:
        if suffix_tokens <= 0 or max_regions <= 0:
            raise ValueError("Query-region limits must be positive.")
        self.suffix_tokens = suffix_tokens
        self.max_regions = max_regions

    def select(
        self,
        prompt: str | None = None,
        *,
        policy: str = "auto",
        query_spans: Sequence[tuple[int, int]] | None = None,
        segments: Sequence[PromptSegment] | None = None,
    ) -> QueryRegionSelection:
        """Select query regions, preferring spans then structured roles.

        Explicit spans are token intervals in the final serialized prompt.
        When structured segments are supplied, their serialization becomes the
        prompt and raw ``prompt`` must be omitted to prevent ambiguous mapping.
        """

        if policy not in self.SUPPORTED_POLICIES:
            raise ValueError(f"Unsupported query-region policy: {policy}")
        segment_ranges: tuple[tuple[PromptSegment, int, int], ...] = ()
        if segments is not None:
            if prompt is not None:
                raise ValueError("Pass either prompt or segments, not both.")
            prompt, segment_ranges = render_segments(segments)
        if not prompt:
            raise ValueError("Query-region selection requires a nonempty prompt.")
        offsets = token_offsets(prompt)
        if not offsets:
            raise ValueError("The prompt contains no tokens.")

        if query_spans:
            regions = tuple(
                QueryRegion(int(start), int(end), 1.0, "explicit", "query")
                for start, end in query_spans
            )
            return self._selection(prompt, "explicit", regions, len(offsets), 1.0)

        if segment_ranges:
            regions = self._from_segments(offsets, segment_ranges, policy)
            if regions:
                return self._selection(prompt, "structured", regions, len(offsets), 1.0)
            if policy in {"explicit", "structured", "session_state"}:
                raise ValueError("Structured query selection found no query-like segment.")

        if policy in {"head", "suffix"}:
            start = max(0, len(offsets) - self.suffix_tokens)
            region = QueryRegion(start, len(offsets), 0.5, policy, "recent_head")
            return self._selection(prompt, policy, (region,), len(offsets), 0.0)
        if policy in {"explicit", "structured", "session_state"}:
            raise ValueError(f"Policy {policy} requires explicit spans or structured segments.")

        candidates = self._structural_candidates(prompt, offsets)
        if not candidates:
            start = max(0, len(offsets) - self.suffix_tokens)
            fallback = QueryRegion(start, len(offsets), 0.25, "structural_fallback", "recent_head")
            return self._selection(prompt, "structural_fallback", (fallback,), len(offsets), 0.0)
        limit = self.max_regions if policy == "multi_region" else 1
        selected = _merge_regions(candidate[1] for candidate in candidates[:limit])
        margin = candidates[0][0] - candidates[1][0] if len(candidates) > 1 else candidates[0][0]
        return self._selection(prompt, "structural", selected, len(offsets), margin)

    def reinterpret(
        self,
        prompt: str,
        previous: QueryRegionSelection,
    ) -> QueryRegionSelection:
        """Escalate a failed suffix interpretation to structural/multi-region cues."""

        policy = "structural" if previous.policy in {"head", "suffix"} else "multi_region"
        return self.select(prompt, policy=policy)

    def _from_segments(
        self,
        offsets: Sequence[tuple[int, int]],
        ranges: Sequence[tuple[PromptSegment, int, int]],
        policy: str,
    ) -> tuple[QueryRegion, ...]:
        query_roles = {"query", "question", "current_state"}
        if policy in {"multi_region", "session_state"}:
            query_roles |= {"instruction", "task"}
        regions = []
        for segment, start, end in ranges:
            if segment.role.lower() not in query_roles:
                continue
            span = _char_to_token_span(offsets, start, end)
            if span is not None:
                confidence = 1.0 if segment.role.lower() in {"query", "question"} else 0.9
                regions.append(QueryRegion(*span, confidence, "structured", segment.role.lower()))
        if policy != "multi_region" and regions:
            # The latest explicit query/current-state segment is normally the
            # smallest sufficient state in a multi-turn session.
            query_only = [region for region in regions if region.role in {"query", "question", "current_state"}]
            return (query_only[-1] if query_only else regions[-1],)
        return tuple(regions[-self.max_regions :])

    def _structural_candidates(
        self, prompt: str, offsets: Sequence[tuple[int, int]]
    ) -> list[tuple[float, QueryRegion]]:
        candidates = []
        starts = [left for left, _ in offsets]
        ends = [right for _, right in offsets]
        cursor = 0
        fenced = False
        for raw_line in prompt.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            start, end = cursor, cursor + len(line)
            cursor += len(raw_line)
            if line.strip().startswith("```"):
                fenced = not fenced
                continue
            first = bisect_right(ends, start)
            last = bisect_left(starts, end)
            if first >= last:
                continue
            span = (first, last)
            score = 0.0
            if _QUERY_LABEL.search(line):
                score += 5.0
            if "?" in line:
                score += 3.0
            if _IMPERATIVE.search(line):
                score += 2.0
            if _CONTEXT_LABEL.search(line):
                score -= 4.0
            if _URL.search(line):
                score -= 4.0
            if _LOG.search(line) or fenced:
                score -= 5.0
            if score <= 0.0:
                continue
            confidence = min(0.98, 0.45 + 0.08 * score)
            candidates.append((score, QueryRegion(*span, confidence, "structural", "query")))
        return sorted(candidates, key=lambda item: (-item[0], item[1].start))

    @staticmethod
    def _selection(
        prompt: str,
        policy: str,
        regions: Sequence[QueryRegion],
        count: int,
        margin: float,
    ) -> QueryRegionSelection:
        merged = _merge_regions(regions)
        confidence = sum(region.confidence for region in merged) / len(merged)
        return QueryRegionSelection(prompt, policy, merged, count, confidence, float(margin))
