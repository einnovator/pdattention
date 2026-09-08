"""Validity-first routing over records produced by completed subagents.

The router never decides whether a record is safe to expose. Callers first ask
``AgentContextGraph`` for visible records; this module only ranks that already
authorized set. This keeps semantic selection independent from consistency and
lineage policy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .context_records import ContextRecord, RecordType
from .subagent_context import AgentContextGraph, AgentStatus


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+")
_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")
_SEARCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "from",
        "in",
        "is",
        "module",
        "of",
        "the",
        "to",
        "where",
        "which",
        "with",
    }
)
_EVIDENCE_TYPES = {
    RecordType.API_RESULT,
    RecordType.DB_RESULT,
    RecordType.FILE_READ,
    RecordType.GENERIC_DOCUMENT,
    RecordType.GENERIC_TEXT,
    RecordType.LOG_BLOCK,
    RecordType.RAG_CHUNK,
    RecordType.RAG_CHUNK_SET,
    RecordType.RAG_RESULT,
    RecordType.TERMINAL_OUTPUT,
    RecordType.TOOL_RESPONSE,
}


class DescendantRoutingMode(str, Enum):
    """Baselines and trained policy used after validity filtering."""

    ALL_VALID = "all_valid"
    LEXICAL = "lexical"
    LEARNED = "learned"
    FIELDED_BM25 = "fielded_bm25"
    ORACLE = "oracle"


@dataclass(frozen=True)
class RoutingCandidate:
    """One authorized typed record presented to a descendant selector."""

    record: ContextRecord
    text: str
    source_status: AgentStatus

    @property
    def record_uuid(self) -> str:
        return self.record.record_id

    @property
    def token_count(self) -> int:
        return len(_tokens(self.text))


@dataclass(frozen=True)
class DescendantRoutingExample:
    """Supervised query and relevant record IDs for fitting/evaluation."""

    query: str
    candidates: tuple[RoutingCandidate, ...]
    relevant_record_ids: frozenset[str]


@dataclass(frozen=True)
class DescendantRoute:
    """Ranked selection plus metrics available without answer generation."""

    mode: DescendantRoutingMode
    selected: tuple[RoutingCandidate, ...]
    scores: tuple[float, ...]
    candidate_count: int
    selected_tokens: int

    def recall(self, relevant_record_ids: Iterable[str]) -> float:
        relevant = set(relevant_record_ids)
        if not relevant:
            return 1.0
        selected = {row.record_uuid for row in self.selected}
        return len(selected & relevant) / len(relevant)

    def precision(self, relevant_record_ids: Iterable[str]) -> float:
        if not self.selected:
            return 1.0 if not set(relevant_record_ids) else 0.0
        relevant = set(relevant_record_ids)
        return sum(row.record_uuid in relevant for row in self.selected) / len(self.selected)


class DescendantRecordRouter:
    """Small interpretable ranker for completed-child evidence.

    The learned mode is a linear pairwise ranker over lexical, type, size, and
    completion features. It intentionally avoids host-model embeddings so the
    first natural benchmark can distinguish learned selection from native K/V
    transport without adding another model to the causal path.
    """

    feature_count = 12

    def __init__(self) -> None:
        self.weights = [0.0] * self.feature_count
        self.fitted = False

    def fit(
        self,
        examples: Sequence[DescendantRoutingExample],
        *,
        epochs: int = 80,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
    ) -> "DescendantRecordRouter":
        """Fit deterministic pairwise logistic preferences."""

        pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        for example in examples:
            positives = [
                row for row in example.candidates if row.record_uuid in example.relevant_record_ids
            ]
            negatives = [
                row for row in example.candidates if row.record_uuid not in example.relevant_record_ids
            ]
            for positive in positives:
                for negative in negatives:
                    pairs.append(
                        (self.features(example.query, positive), self.features(example.query, negative))
                    )
        if not pairs:
            raise ValueError("At least one positive/negative routing pair is required.")
        for _ in range(epochs):
            for positive, negative in pairs:
                difference = tuple(a - b for a, b in zip(positive, negative))
                margin = sum(weight * value for weight, value in zip(self.weights, difference))
                error = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, margin))))
                self.weights = [
                    weight + learning_rate * (error * value - l2 * weight)
                    for weight, value in zip(self.weights, difference)
                ]
        self.fitted = True
        return self

    def route(
        self,
        query: str,
        candidates: Sequence[RoutingCandidate],
        *,
        mode: DescendantRoutingMode | str = DescendantRoutingMode.LEXICAL,
        top_k: int = 1,
        oracle_record_ids: Iterable[str] = (),
    ) -> DescendantRoute:
        """Rank an already-authorized candidate set under a bounded budget."""

        mode = DescendantRoutingMode(mode)
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        oracle_ids = set(oracle_record_ids)
        if mode == DescendantRoutingMode.LEARNED and not self.fitted:
            raise ValueError("The learned descendant router must be fitted before use.")
        fielded_scores = (
            _fielded_bm25_scores(query, candidates)
            if mode == DescendantRoutingMode.FIELDED_BM25
            else {}
        )

        scored: list[tuple[float, RoutingCandidate]] = []
        for candidate in candidates:
            features = self.features(query, candidate)
            if mode == DescendantRoutingMode.ALL_VALID:
                score = 0.0
            elif mode == DescendantRoutingMode.ORACLE:
                score = 1.0 if candidate.record_uuid in oracle_ids else 0.0
            elif mode == DescendantRoutingMode.LEARNED:
                score = sum(weight * value for weight, value in zip(self.weights, features))
            elif mode == DescendantRoutingMode.FIELDED_BM25:
                score = fielded_scores[candidate.record_uuid]
            else:
                score = features[1] + features[2] + 0.5 * features[3]
            scored.append((score, candidate))
        scored.sort(key=lambda row: (-row[0], row[1].record_uuid))
        limit = len(scored) if mode == DescendantRoutingMode.ALL_VALID else min(top_k, len(scored))
        selected_rows = scored[:limit]
        return DescendantRoute(
            mode=mode,
            selected=tuple(row[1] for row in selected_rows),
            scores=tuple(row[0] for row in selected_rows),
            candidate_count=len(scored),
            selected_tokens=sum(row[1].token_count for row in selected_rows),
        )

    @staticmethod
    def features(query: str, candidate: RoutingCandidate) -> tuple[float, ...]:
        """Return bounded query-record features in a stable public order."""

        query_tokens = _tokens(query)
        text_tokens = _tokens(candidate.text)
        query_set, text_set = set(query_tokens), set(text_tokens)
        overlap = query_set & text_set
        union = query_set | text_set
        numbers = {token for token in query_set if token.isdigit()}
        identifiers = {token for token in query_set if any(char in token for char in "_./:")}
        lowered_query = " ".join(query_tokens)
        lowered_text = " ".join(text_tokens)
        record_type = candidate.record.record_type
        return (
            1.0,
            len(overlap) / max(1, len(union)),
            len(overlap) / max(1, len(query_set)),
            float(bool(lowered_query and lowered_query in lowered_text)),
            len(numbers & text_set) / max(1, len(numbers)),
            len(identifiers & text_set) / max(1, len(identifiers)),
            float(record_type == RecordType.TERMINAL_OUTPUT),
            float(record_type in {RecordType.FILE_READ, RecordType.GENERIC_DOCUMENT}),
            float(record_type == RecordType.TOOL_RESPONSE),
            float(candidate.source_status != AgentStatus.RUNNING),
            min(1.0, math.log1p(len(text_tokens)) / 10.0),
            1.0 / max(1.0, len(text_tokens)),
        )


def visible_routing_candidates(
    graph: AgentContextGraph,
    requester_agent_uuid: str,
    *,
    evidence_types: Iterable[RecordType] = _EVIDENCE_TYPES,
) -> tuple[RoutingCandidate, ...]:
    """Build candidates only from records authorized by the context graph."""

    allowed = set(evidence_types)
    rows: list[RoutingCandidate] = []
    for record in graph.visible_records(requester_agent_uuid):
        if record.agent_uuid == requester_agent_uuid or record.record_type not in allowed:
            continue
        source = graph.descriptor(record.agent_uuid or "")
        payload = record.payload
        text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, default=str)
        rows.append(RoutingCandidate(record, text, source.status))
    return tuple(rows)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(value))


def _search_stem(token: str) -> str:
    """Apply a deliberately small deterministic normalizer for source search."""

    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("ses"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 5 and token.endswith("ed"):
        token = token[:-2]
    if len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
    return token


def _search_tokens(value: str) -> tuple[str, ...]:
    tokens = (_search_stem(match.group(0)) for match in _SEARCH_TOKEN_RE.finditer(value))
    return tuple(token for token in tokens if token not in _SEARCH_STOPWORDS)


def _candidate_search_fields(candidate: RoutingCandidate) -> tuple[str, str]:
    """Return an identity field and lossless content field when structure exists."""

    payload = candidate.record.payload
    if not isinstance(payload, Mapping):
        return "", candidate.text
    arguments = payload.get("arguments")
    identity = ""
    if isinstance(arguments, Mapping):
        for name in ("path", "uri", "resource", "record_id"):
            value = arguments.get(name)
            if isinstance(value, str):
                identity = value
                break
    output = payload.get("output", candidate.text)
    content = (
        output
        if isinstance(output, str)
        else json.dumps(output, sort_keys=True, default=str)
    )
    return identity, content


def _fielded_bm25_scores(
    query: str,
    candidates: Sequence[RoutingCandidate],
    *,
    lines_per_window: int = 12,
) -> dict[str, float]:
    """Rank concentrated evidence rather than whole-record word frequency.

    Source paths and leading documentation are separate fields. The content is
    divided into bounded line windows so a long module containing many generic
    terms does not beat a short, locally relevant definition merely because of
    document length. This remains a zero-model retrieval baseline: it neither
    learns from nor inspects relevance labels.
    """

    if not candidates:
        return {}
    query_tokens = _search_tokens(query)
    fields: list[tuple[str, str, tuple[tuple[str, ...], ...]]] = []
    corpus_windows: list[tuple[str, ...]] = []
    for candidate in candidates:
        identity, content = _candidate_search_fields(candidate)
        lines = content.splitlines() or [content]
        windows = tuple(
            _search_tokens(" ".join(lines[start : start + lines_per_window]))
            for start in range(0, len(lines), lines_per_window)
        ) or ((),)
        fields.append((candidate.record_uuid, identity, windows))
        corpus_windows.extend(windows)

    document_frequency = Counter(
        token for window in corpus_windows for token in set(window)
    )
    corpus_size = len(corpus_windows)
    average_length = sum(map(len, corpus_windows)) / max(1, corpus_size)

    def window_score(window: tuple[str, ...]) -> float:
        frequencies = Counter(window)
        length_normalizer = 0.25 + 0.75 * len(window) / max(1.0, average_length)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (corpus_size - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            score += inverse_frequency * frequency * 2.2 / (
                frequency + 1.2 * length_normalizer
            )
        return score

    scores: dict[str, float] = {}
    query_set = set(query_tokens)
    for record_uuid, identity, windows in fields:
        concentrated = max(window_score(window) for window in windows)
        leading_documentation = window_score(windows[0])
        identity_overlap = len(query_set & set(_search_tokens(identity)))
        scores[record_uuid] = (
            concentrated + 0.75 * leading_documentation + 2.0 * identity_overlap
        )
    return scores
