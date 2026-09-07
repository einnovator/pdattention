"""Budgeted cross-document retrieval expansion over immutable PRA records.

The first-stage RAG/PRA selector remains frozen.  Policies in this module use
one selected span to discover additional intervals in peer selected records;
the runtime then enforces authorization, interval deduplication, and a global
token budget before handing canonical source intervals to materialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .agent_resources import hashed_semantic_vector
from .rag_evaluation import RAGChunk


_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:+-]*")
_TOKEN = re.compile(r"\S+")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_CAPITALIZED = re.compile(r"\b(?:[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+){0,3})\b")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the to was were will with".split()
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _terms(text: str, *, remove_stopwords: bool = True) -> tuple[str, ...]:
    values = tuple(token.casefold() for token in _TERM.findall(text))
    return tuple(token for token in values if token not in _STOPWORDS) if remove_stopwords else values


class CrossDocumentExpansionMode(str, Enum):
    """Built-in cross-record candidate-generation policies."""

    NONE = "none"
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"
    ENTITY = "entity"
    ORACLE = "oracle"
    CUSTOM = "custom"


class CrossDocumentDirection(str, Enum):
    """Allowed ordered source-to-target record relationships."""

    SYMMETRIC = "symmetric"
    HIGHER_TO_LOWER = "higher_to_lower"
    LOWER_TO_HIGHER = "lower_to_higher"
    TOP_RECORD_HUB = "top_record_hub"
    QUERY_SELECTED_PAIR_ONLY = "query_selected_pair_only"


class CrossDocumentGranularity(str, Enum):
    """Addressing granularity requested from the materializer."""

    CHUNK = "chunk"
    LOGICAL_INTERVAL = "logical_interval"
    HIERARCHICAL = "hierarchical"


class CrossDocumentFailure(str, Enum):
    """Stable failure labels used by experiment and SDK receipts."""

    LEXICAL_MISS = "CROSSDOC_LEXICAL_MISS"
    DENSE_MISS = "CROSSDOC_DENSE_MISS"
    HYBRID_MISS = "CROSSDOC_HYBRID_MISS"
    QUERY_DRIFT = "CROSSDOC_QUERY_DRIFT"
    DISTRACTOR_EXPANSION = "CROSSDOC_DISTRACTOR_EXPANSION"
    BUDGET_OVERFLOW = "CROSSDOC_BUDGET_OVERFLOW"
    DUPLICATE_EXPANSION = "CROSSDOC_DUPLICATE_EXPANSION"
    SECOND_HOP_DRIFT = "CROSSDOC_SECOND_HOP_DRIFT"
    SA_NO_GAIN = "CROSSDOC_SA_NO_GAIN"
    SDK_QUALIFICATION_FAIL = "CROSSDOC_SDK_QUALIFICATION_FAIL"
    UNAUTHORIZED_TARGET = "CROSSDOC_UNAUTHORIZED_TARGET"


@dataclass(frozen=True)
class CrossDocumentBudget:
    """Pair-local candidate cap and request-global materialization ceiling."""

    top_k_per_pair: int = 1
    max_extra_tokens: int = 128
    max_extra_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.top_k_per_pair <= 0 or self.max_extra_tokens < 0:
            raise ValueError("cross-document budgets require positive top-k and non-negative tokens")
        if self.max_extra_fraction is not None and not 0.0 < self.max_extra_fraction <= 1.0:
            raise ValueError("max_extra_fraction must be in (0, 1]")

    def effective_tokens(self, selected_tokens: int) -> int:
        fractional = (
            self.max_extra_tokens
            if self.max_extra_fraction is None
            else math.floor(selected_tokens * self.max_extra_fraction)
        )
        return max(0, min(self.max_extra_tokens, fractional))


@dataclass(frozen=True)
class CrossDocumentExpansionConfig:
    """Stable configuration for one bounded cross-document expansion round."""

    mode: CrossDocumentExpansionMode | str = CrossDocumentExpansionMode.NONE
    query_conditioned: bool = True
    direction: CrossDocumentDirection | str = CrossDocumentDirection.SYMMETRIC
    granularity: CrossDocumentGranularity | str = CrossDocumentGranularity.CHUNK
    budget: CrossDocumentBudget = field(default_factory=CrossDocumentBudget)
    dense_dimensions: int = 256
    dense_query_weight: float = 1.0
    dense_selected_weight: float = 1.0
    rrf_constant: float = 60.0
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    query_relevance_weight: float = 0.25
    minimum_pair_query_score: float = 0.0
    iteration_depth: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CrossDocumentExpansionMode(self.mode))
        object.__setattr__(self, "direction", CrossDocumentDirection(self.direction))
        object.__setattr__(self, "granularity", CrossDocumentGranularity(self.granularity))
        if self.dense_dimensions <= 0 or self.rrf_constant <= 0:
            raise ValueError("dense dimensions and RRF constant must be positive")
        weights = (
            self.dense_query_weight,
            self.dense_selected_weight,
            self.lexical_weight,
            self.dense_weight,
            self.query_relevance_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("cross-document weights must be finite and non-negative")
        if self.dense_query_weight + self.dense_selected_weight <= 0:
            raise ValueError("dense query and selected-span weights cannot both be zero")
        if self.iteration_depth != 1:
            raise ValueError(
                "iterative cross-document expansion is locked until the one-pass policy qualifies"
            )


@dataclass(frozen=True)
class CrossDocumentSelectedRecord:
    """One ranked record and the source intervals selected before expansion."""

    record_uri: str
    rank: int
    spans: tuple[RAGChunk, ...]
    query_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.record_uri or self.rank <= 0 or not self.spans:
            raise ValueError("selected records require a URI, positive rank, and source spans")
        if len({row.chunk_id for row in self.spans}) != len(self.spans):
            raise ValueError("selected record spans must have unique chunk IDs")
        if len({row.document_id for row in self.spans}) != 1:
            raise ValueError("selected record spans must belong to one source document")

    @property
    def selected_text(self) -> str:
        return "\n".join(row.text for row in self.spans)

    @property
    def selected_tokens(self) -> int:
        return sum(row.token_count for row in self.spans)


@dataclass(frozen=True)
class CrossDocumentExpansionRequest:
    """Immutable inputs available to built-in or custom expansion policies."""

    query: str
    selection_receipt_id: str
    selected_records: tuple[CrossDocumentSelectedRecord, ...]
    candidate_chunks_by_record: Mapping[str, tuple[RAGChunk, ...]]
    budget: CrossDocumentBudget
    authorized_record_uris: frozenset[str] | None = None
    resident_chunk_ids: frozenset[str] = frozenset()
    gold_chunk_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.selection_receipt_id or len(self.selected_records) < 2:
            raise ValueError("cross-document expansion requires a query, receipt, and two records")
        uris = tuple(row.record_uri for row in self.selected_records)
        if len(set(uris)) != len(uris):
            raise ValueError("selected record URIs must be unique")
        missing = set(uris) - set(self.candidate_chunks_by_record)
        if missing:
            raise ValueError(f"candidate chunks are missing for selected records: {sorted(missing)}")
        object.__setattr__(
            self,
            "candidate_chunks_by_record",
            {key: tuple(value) for key, value in self.candidate_chunks_by_record.items()},
        )


@dataclass(frozen=True)
class CrossDocumentCandidate:
    """One scored source-span to peer-target-chunk proposal."""

    source_record_uri: str
    source_chunk_id: str
    source_span: tuple[int, int]
    target_record_uri: str
    target_chunk_id: str
    target_span: tuple[int, int]
    target_text: str
    target_tokens: int
    source_rank: int
    target_rank: int
    score: float
    lexical_score: float | None = None
    dense_score: float | None = None
    query_score: float | None = None
    entity_overlap: float | None = None
    keyterms: tuple[str, ...] = ()
    lexical_rank: int | None = None
    dense_rank: int | None = None
    selection_reason: str = ""
    gold_support: bool = False

    @property
    def pair(self) -> tuple[str, str, str]:
        return self.source_record_uri, self.source_chunk_id, self.target_record_uri

    @property
    def candidate_id(self) -> str:
        return _digest(
            (
                self.source_record_uri,
                self.source_chunk_id,
                self.target_record_uri,
                self.target_chunk_id,
                self.target_span,
            )
        )

    def receipt_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("target_text")
        value["target_text_sha256"] = hashlib.sha256(self.target_text.encode("utf-8")).hexdigest()
        value["candidate_id"] = self.candidate_id
        return value


@dataclass(frozen=True)
class CrossDocumentExpandedSpan:
    """Canonical target interval admitted by runtime policy enforcement."""

    source_record_uri: str
    source_chunk_id: str
    target_record_uri: str
    target_chunk_id: str
    start: int
    end: int
    text: str
    token_count: int
    score: float
    reused_native_tokens: int
    new_native_tokens: int
    selection_reason: str

    @property
    def span_id(self) -> str:
        return _digest((self.target_record_uri, self.target_chunk_id, self.start, self.end))

    def receipt_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("text")
        value["text_sha256"] = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        value["span_id"] = self.span_id
        return value


@dataclass(frozen=True)
class CrossDocumentExpansionReceipt:
    """Hash-addressed explanation of proposal, filtering, and materialization."""

    policy_name: str
    policy_revision: str
    query_digest: str
    selection_receipt_id: str
    candidates: tuple[Mapping[str, object], ...]
    chosen_spans: tuple[Mapping[str, object], ...]
    requested_tokens: int
    deduplicated_tokens: int
    reused_native_tokens: int
    new_native_tokens: int
    budget: Mapping[str, object]
    search_latency_ms: float
    failures: tuple[str, ...] = ()
    schema_version: str = "paper3.3-crossdoc-expansion-v1"

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_name": self.policy_name,
            "policy_revision": self.policy_revision,
            "query_digest": self.query_digest,
            "selection_receipt_id": self.selection_receipt_id,
            "candidates": [dict(row) for row in self.candidates],
            "chosen_spans": [dict(row) for row in self.chosen_spans],
            "requested_tokens": self.requested_tokens,
            "deduplicated_tokens": self.deduplicated_tokens,
            "reused_native_tokens": self.reused_native_tokens,
            "new_native_tokens": self.new_native_tokens,
            "budget": dict(self.budget),
            "search_latency_ms": self.search_latency_ms,
            "failures": list(self.failures),
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value


@dataclass(frozen=True)
class CrossDocumentExpansionPlan:
    """Runtime-approved source intervals plus their auditable receipt."""

    spans: tuple[CrossDocumentExpandedSpan, ...]
    receipt: CrossDocumentExpansionReceipt


class CrossDocumentExpansionPolicy(Protocol):
    """Policy extension point; runtime still owns all safety enforcement."""

    name: str
    revision: str

    def propose(
        self, request: CrossDocumentExpansionRequest
    ) -> tuple[CrossDocumentCandidate, ...]: ...


CrossDocumentExpansionPlugin = CrossDocumentExpansionPolicy


class SemanticEncoder(Protocol):
    """Query/document encoder used by dense and hybrid expansion policies."""

    identity: str

    def encode_query(self, text: str) -> Sequence[float]: ...

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


SemanticEncoderLike = SemanticEncoder | Callable[[str], Sequence[float]]


@lru_cache(maxsize=32_768)
def _cached_hashed_vector(text: str, dimensions: int) -> tuple[float, ...]:
    """Reuse immutable fallback embeddings across policy/budget sweeps."""

    return tuple(hashed_semantic_vector(text, dimensions=dimensions))


class _ChunkBM25:
    """Exact BM25 index over the peer chunks available in one request."""

    def __init__(self, chunks: Sequence[RAGChunk]) -> None:
        unique = {row.chunk_id: row for row in chunks}
        self.chunks = tuple(unique[key] for key in sorted(unique))
        self.frequencies = {row.chunk_id: Counter(_terms(row.text)) for row in self.chunks}
        self.lengths = {key: sum(value.values()) for key, value in self.frequencies.items()}
        self.average_length = sum(self.lengths.values()) / max(len(self.lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for value in self.frequencies.values():
            self.document_frequency.update(value)

    def score(self, query: str, chunk: RAGChunk) -> float:
        score = 0.0
        count = max(len(self.chunks), 1)
        frequencies = self.frequencies[chunk.chunk_id]
        for term in set(_terms(query)):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = self.document_frequency.get(term, 0)
            inverse = math.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * self.lengths[chunk.chunk_id] / max(self.average_length, 1.0)
            )
            score += inverse * frequency * 2.2 / denominator
        return score


class BuiltinCrossDocumentExpansionPolicy:
    """Parameter-free lexical, dense, hybrid, entity, and oracle policies."""

    def __init__(
        self,
        config: CrossDocumentExpansionConfig,
        *,
        semantic_encoder: SemanticEncoderLike | None = None,
        policy_revision: str = "builtin-v1",
    ) -> None:
        if config.mode is CrossDocumentExpansionMode.CUSTOM:
            raise ValueError("custom mode requires a caller-provided policy")
        self.config = config
        self.name = config.mode.value
        if semantic_encoder is None:
            self.semantic_encoder_identity = (
                f"signed-hash-{config.dense_dimensions}-v1"
            )
            self._encode_query = lambda text: _cached_hashed_vector(
                text, config.dense_dimensions
            )
            self._encode_documents = lambda texts: tuple(
                _cached_hashed_vector(text, config.dense_dimensions) for text in texts
            )
        elif hasattr(semantic_encoder, "encode_query") and hasattr(
            semantic_encoder, "encode_documents"
        ):
            self.semantic_encoder_identity = str(
                getattr(semantic_encoder, "identity", type(semantic_encoder).__name__)
            )
            self._encode_query = semantic_encoder.encode_query
            self._encode_documents = semantic_encoder.encode_documents
        else:
            self.semantic_encoder_identity = str(
                getattr(semantic_encoder, "identity", type(semantic_encoder).__name__)
            )
            self._encode_query = semantic_encoder
            self._encode_documents = lambda texts: tuple(
                semantic_encoder(text) for text in texts
            )
        self.revision = f"{policy_revision}+{self.semantic_encoder_identity}"

    def propose(
        self, request: CrossDocumentExpansionRequest
    ) -> tuple[CrossDocumentCandidate, ...]:
        if self.config.mode is CrossDocumentExpansionMode.NONE:
            return ()
        all_chunks = tuple(
            chunk for chunks in request.candidate_chunks_by_record.values() for chunk in chunks
        )
        lexical = _ChunkBM25(all_chunks)
        query_vector = _normalize(self._encode_query(request.query))
        encoded_targets = self._encode_documents(tuple(row.text for row in all_chunks))
        if len(encoded_targets) != len(all_chunks):
            raise ValueError("semantic encoder returned the wrong document count")
        target_vectors = {
            row.chunk_id: _normalize(vector)
            for row, vector in zip(all_chunks, encoded_targets)
        }
        source_vectors = {
            span.chunk_id: _normalize(self._encode_query(span.text))
            for record in request.selected_records
            for span in record.spans
        }
        record_by_uri = {row.record_uri: row for row in request.selected_records}
        candidates: list[CrossDocumentCandidate] = []
        for source in request.selected_records:
            for target in request.selected_records:
                if not _direction_allowed(self.config, source, target):
                    continue
                for source_span in source.spans:
                    cross_query = _cross_query(request.query, source_span.text, self.config)
                    keyterms = _keyterms(cross_query, lexical)
                    source_vector = source_vectors[source_span.chunk_id]
                    dense_query = _normalize(
                        self.config.dense_query_weight * query_vector
                        + self.config.dense_selected_weight * source_vector
                    )
                    pair_rows: list[CrossDocumentCandidate] = []
                    for chunk in request.candidate_chunks_by_record[target.record_uri]:
                        selected_sections = {
                            row.section_id for row in target.spans if row.section_id is not None
                        }
                        if (
                            self.config.granularity
                            is CrossDocumentGranularity.HIERARCHICAL
                            and selected_sections
                            and chunk.section_id not in selected_sections
                        ):
                            continue
                        if _fully_covered(
                            (chunk.start, chunk.end),
                            tuple((row.start, row.end) for row in target.spans),
                        ):
                            continue
                        lexical_score = lexical.score(cross_query, chunk)
                        query_score = lexical.score(request.query, chunk)
                        dense_score = float(np.dot(dense_query, target_vectors[chunk.chunk_id]))
                        entity_overlap = _entity_overlap(cross_query, chunk.text, lexical)
                        target_span, target_text, target_tokens = _target_interval(
                            chunk, cross_query, self.config.granularity
                        )
                        score, reason = self._base_score(
                            lexical_score=lexical_score,
                            dense_score=dense_score,
                            query_score=query_score,
                            entity_overlap=entity_overlap,
                            gold_support=chunk.chunk_id in request.gold_chunk_ids,
                        )
                        pair_rows.append(
                            CrossDocumentCandidate(
                                source_record_uri=source.record_uri,
                                source_chunk_id=source_span.chunk_id,
                                source_span=(source_span.start, source_span.end),
                                target_record_uri=target.record_uri,
                                target_chunk_id=chunk.chunk_id,
                                target_span=target_span,
                                target_text=target_text,
                                target_tokens=target_tokens,
                                source_rank=source.rank,
                                target_rank=record_by_uri[target.record_uri].rank,
                                score=score,
                                lexical_score=lexical_score,
                                dense_score=dense_score,
                                query_score=query_score,
                                entity_overlap=entity_overlap,
                                keyterms=keyterms,
                                selection_reason=reason,
                                gold_support=chunk.chunk_id in request.gold_chunk_ids,
                            )
                        )
                    ranked = self._rank_pair(pair_rows)
                    if self.config.mode is CrossDocumentExpansionMode.ORACLE:
                        ranked = tuple(row for row in ranked if row.gold_support)
                    candidates.extend(ranked)
        return tuple(candidates)

    def _base_score(
        self,
        *,
        lexical_score: float,
        dense_score: float,
        query_score: float,
        entity_overlap: float,
        gold_support: bool,
    ) -> tuple[float, str]:
        mode = self.config.mode
        if mode is CrossDocumentExpansionMode.LEXICAL:
            return lexical_score, "bm25_cross_record"
        if mode is CrossDocumentExpansionMode.DENSE:
            return dense_score, "dense_cross_record"
        if mode is CrossDocumentExpansionMode.ENTITY:
            return entity_overlap, "entity_keyterm_cross_link"
        if mode is CrossDocumentExpansionMode.ORACLE:
            return (1.0 if gold_support else 0.0), "gold_support_oracle"
        if mode is CrossDocumentExpansionMode.HYBRID_WEIGHTED:
            return (
                self.config.lexical_weight * lexical_score
                + self.config.dense_weight * dense_score
                + self.config.query_relevance_weight * query_score,
                "weighted_lexical_dense_query",
            )
        return 0.0, "hybrid_rrf_pending_ranks"

    def _rank_pair(
        self, rows: Sequence[CrossDocumentCandidate]
    ) -> tuple[CrossDocumentCandidate, ...]:
        if self.config.mode is CrossDocumentExpansionMode.HYBRID_WEIGHTED:
            lexical = _minmax(row.lexical_score or 0.0 for row in rows)
            dense = _minmax(row.dense_score or 0.0 for row in rows)
            query = _minmax(row.query_score or 0.0 for row in rows)
            updated = []
            for index, row in enumerate(rows):
                score = (
                    self.config.lexical_weight * lexical[index]
                    + self.config.dense_weight * dense[index]
                    + self.config.query_relevance_weight * query[index]
                )
                updated.append(
                    CrossDocumentCandidate(
                        **{
                            **asdict(row),
                            "score": score,
                            "selection_reason": "normalized_weighted_lexical_dense_query",
                        }
                    )
                )
            return tuple(sorted(updated, key=lambda row: (-row.score, row.target_chunk_id)))
        if self.config.mode is not CrossDocumentExpansionMode.HYBRID_RRF:
            return tuple(sorted(rows, key=lambda row: (-row.score, row.target_chunk_id)))
        lexical_order = sorted(rows, key=lambda row: (-float(row.lexical_score or 0.0), row.target_chunk_id))
        dense_order = sorted(rows, key=lambda row: (-float(row.dense_score or 0.0), row.target_chunk_id))
        lexical_rank = {row.candidate_id: index for index, row in enumerate(lexical_order, 1)}
        dense_rank = {row.candidate_id: index for index, row in enumerate(dense_order, 1)}
        updated = []
        for row in rows:
            left = lexical_rank[row.candidate_id]
            right = dense_rank[row.candidate_id]
            score = 1.0 / (self.config.rrf_constant + left) + 1.0 / (
                self.config.rrf_constant + right
            )
            updated.append(
                CrossDocumentCandidate(
                    **{
                        **asdict(row),
                        "score": score,
                        "lexical_rank": left,
                        "dense_rank": right,
                        "selection_reason": "reciprocal_rank_fusion",
                    }
                )
            )
        return tuple(sorted(updated, key=lambda row: (-row.score, row.target_chunk_id)))


def _normalize(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _minmax(values) -> tuple[float, ...]:
    """Normalize one pair-local score channel without changing rank ties."""

    rows = tuple(float(value) for value in values)
    if not rows:
        return ()
    lower, upper = min(rows), max(rows)
    if upper == lower:
        return tuple(0.0 for _ in rows)
    return tuple((value - lower) / (upper - lower) for value in rows)


def _fully_covered(
    span: tuple[int, int], occupied: Sequence[tuple[int, int]]
) -> bool:
    """Return whether the union of occupied intervals covers the full span."""

    cursor = span[0]
    for start, end in sorted(occupied):
        if end <= cursor or start >= span[1]:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= span[1]:
            return True
    return False


def _cross_query(query: str, selected_text: str, config: CrossDocumentExpansionConfig) -> str:
    if not config.query_conditioned:
        return selected_text
    return query + "\n" + selected_text


def _direction_allowed(
    config: CrossDocumentExpansionConfig,
    source: CrossDocumentSelectedRecord,
    target: CrossDocumentSelectedRecord,
) -> bool:
    if source.record_uri == target.record_uri:
        return False
    direction = config.direction
    if direction is CrossDocumentDirection.SYMMETRIC:
        return True
    if direction is CrossDocumentDirection.HIGHER_TO_LOWER:
        return source.rank < target.rank
    if direction is CrossDocumentDirection.LOWER_TO_HIGHER:
        return source.rank > target.rank
    if direction is CrossDocumentDirection.TOP_RECORD_HUB:
        return source.rank == 1
    return (
        source.query_score >= config.minimum_pair_query_score
        and target.query_score >= config.minimum_pair_query_score
    )


def _entity_overlap(query: str, text: str, index: _ChunkBM25) -> float:
    capitalized = {value.casefold() for value in _CAPITALIZED.findall(query)}
    query_terms = Counter(_terms(query))
    rare = {
        term
        for term in query_terms
        if index.document_frequency.get(term, 0) <= max(1, len(index.chunks) // 4)
    }
    target = text.casefold()
    entity_hits = sum(value in target for value in capitalized)
    term_hits = sum(term in set(_terms(text)) for term in rare)
    return float(2 * entity_hits + term_hits)


def _keyterms(query: str, index: _ChunkBM25, *, limit: int = 12) -> tuple[str, ...]:
    """Extract deterministic entities and rare terms for receipt inspection."""

    entities = [value.casefold() for value in _CAPITALIZED.findall(query)]
    terms = sorted(
        set(_terms(query)),
        key=lambda term: (index.document_frequency.get(term, 0), term),
    )
    return tuple(dict.fromkeys((*entities, *terms)))[:limit]


def _target_interval(
    chunk: RAGChunk,
    query: str,
    granularity: CrossDocumentGranularity,
) -> tuple[tuple[int, int], str, int]:
    """Map a scored chunk to a chunk or best matching logical sentence."""

    if granularity is not CrossDocumentGranularity.LOGICAL_INTERVAL:
        return (chunk.start, chunk.end), chunk.text, chunk.token_count
    query_terms = set(_terms(query))
    sentences = tuple(_SENTENCE.finditer(chunk.text))
    if not sentences:
        return (chunk.start, chunk.end), chunk.text, chunk.token_count
    match = max(
        sentences,
        key=lambda row: (
            len(query_terms.intersection(_terms(row.group(0)))),
            -row.start(),
        ),
    )
    text = match.group(0).strip()
    leading = len(match.group(0)) - len(match.group(0).lstrip())
    start = chunk.start + match.start() + leading
    end = start + len(text)
    return (start, end), text, max(1, len(_TOKEN.findall(text)))


class CrossDocumentExpansionRuntime:
    """Enforce host authority, pair caps, interval deduplication, and budget."""

    def expand(
        self,
        request: CrossDocumentExpansionRequest,
        policy: CrossDocumentExpansionPolicy,
    ) -> CrossDocumentExpansionPlan:
        started = time.perf_counter()
        proposed = policy.propose(request)
        measured_latency_ms = (time.perf_counter() - started) * 1000.0
        # Experiment runners may freeze one candidate ranking across multiple
        # materialization budgets.  Preserve the original search cost in each
        # counterfactual receipt instead of reporting the cached tuple lookup.
        search_latency_ms = float(
            getattr(policy, "proposal_latency_ms", measured_latency_ms)
        )
        authorized = request.authorized_record_uris
        failures: set[str] = set()
        allowed: list[CrossDocumentCandidate] = []
        for row in proposed:
            if authorized is not None and row.target_record_uri not in authorized:
                failures.add(CrossDocumentFailure.UNAUTHORIZED_TARGET.value)
                continue
            if policy.name in {
                CrossDocumentExpansionMode.LEXICAL.value,
                CrossDocumentExpansionMode.ENTITY.value,
            } and row.score <= 0.0:
                continue
            allowed.append(row)

        pair_limited: list[CrossDocumentCandidate] = []
        by_pair: dict[tuple[str, str, str], list[CrossDocumentCandidate]] = defaultdict(list)
        for row in allowed:
            by_pair[row.pair].append(row)
        for pair in sorted(by_pair):
            pair_limited.extend(
                sorted(by_pair[pair], key=lambda row: (-row.score, row.target_chunk_id))[
                    : request.budget.top_k_per_pair
                ]
            )
        pair_limited.sort(key=lambda row: (-row.score, row.target_record_uri, row.target_chunk_id))
        requested_tokens = sum(row.target_tokens for row in pair_limited)
        selected_tokens = sum(row.selected_tokens for row in request.selected_records)
        token_budget = request.budget.effective_tokens(selected_tokens)
        occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for record in request.selected_records:
            occupied[record.record_uri].extend((row.start, row.end) for row in record.spans)
        chosen: list[CrossDocumentExpandedSpan] = []
        remaining = token_budget
        duplicate_tokens = 0
        for candidate in pair_limited:
            if remaining <= 0:
                failures.add(CrossDocumentFailure.BUDGET_OVERFLOW.value)
                break
            fragments, removed = _uncovered_fragments(
                candidate, occupied[candidate.target_record_uri], remaining
            )
            duplicate_tokens += removed
            if not fragments:
                continue
            for start, end, text, count in fragments:
                reused = count if candidate.target_chunk_id in request.resident_chunk_ids else 0
                chosen.append(
                    CrossDocumentExpandedSpan(
                        source_record_uri=candidate.source_record_uri,
                        source_chunk_id=candidate.source_chunk_id,
                        target_record_uri=candidate.target_record_uri,
                        target_chunk_id=candidate.target_chunk_id,
                        start=start,
                        end=end,
                        text=text,
                        token_count=count,
                        score=candidate.score,
                        reused_native_tokens=reused,
                        new_native_tokens=count - reused,
                        selection_reason=candidate.selection_reason,
                    )
                )
                occupied[candidate.target_record_uri].append((start, end))
                remaining -= count
                if remaining <= 0:
                    break
        if duplicate_tokens:
            failures.add(CrossDocumentFailure.DUPLICATE_EXPANSION.value)
        deduplicated_tokens = sum(row.token_count for row in chosen)
        if request.gold_chunk_ids and any(
            row.target_chunk_id not in request.gold_chunk_ids for row in chosen
        ):
            failures.add(CrossDocumentFailure.DISTRACTOR_EXPANSION.value)
        if not chosen and policy.name != CrossDocumentExpansionMode.NONE.value:
            miss = {
                CrossDocumentExpansionMode.LEXICAL.value: CrossDocumentFailure.LEXICAL_MISS,
                CrossDocumentExpansionMode.DENSE.value: CrossDocumentFailure.DENSE_MISS,
                CrossDocumentExpansionMode.HYBRID_RRF.value: CrossDocumentFailure.HYBRID_MISS,
                CrossDocumentExpansionMode.HYBRID_WEIGHTED.value: CrossDocumentFailure.HYBRID_MISS,
            }.get(policy.name)
            if miss is not None:
                failures.add(miss.value)
        receipt = CrossDocumentExpansionReceipt(
            policy_name=policy.name,
            policy_revision=policy.revision,
            query_digest=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
            selection_receipt_id=request.selection_receipt_id,
            candidates=tuple(row.receipt_dict() for row in pair_limited),
            chosen_spans=tuple(row.receipt_dict() for row in chosen),
            requested_tokens=requested_tokens,
            deduplicated_tokens=deduplicated_tokens,
            reused_native_tokens=sum(row.reused_native_tokens for row in chosen),
            new_native_tokens=sum(row.new_native_tokens for row in chosen),
            budget={
                "top_k_per_pair": request.budget.top_k_per_pair,
                "max_extra_tokens": request.budget.max_extra_tokens,
                "max_extra_fraction": request.budget.max_extra_fraction,
                "effective_extra_tokens": token_budget,
                "selected_native_tokens": selected_tokens,
            },
            search_latency_ms=search_latency_ms,
            failures=tuple(sorted(failures)),
        )
        return CrossDocumentExpansionPlan(tuple(chosen), receipt)


def _uncovered_fragments(
    candidate: CrossDocumentCandidate,
    occupied: Sequence[tuple[int, int]],
    remaining_tokens: int,
) -> tuple[tuple[tuple[int, int, str, int], ...], int]:
    matches = tuple(_TOKEN.finditer(candidate.target_text))
    available = []
    removed = 0
    base = candidate.target_span[0]
    for match in matches:
        start, end = base + match.start(), base + match.end()
        if any(start < previous_end and end > previous_start for previous_start, previous_end in occupied):
            removed += 1
        else:
            available.append((match, start, end))
    available = available[:remaining_tokens]
    if not available:
        return (), removed
    groups: list[list[tuple[re.Match[str], int, int]]] = []
    for value in available:
        if groups and value[0].start() <= groups[-1][-1][0].end() + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    fragments = []
    for group in groups:
        first, last = group[0], group[-1]
        text = candidate.target_text[first[0].start() : last[0].end()]
        fragments.append((first[1], last[2], text, len(group)))
    return tuple(fragments), removed


def build_cross_document_expansion_policy(
    config: CrossDocumentExpansionConfig,
    *,
    semantic_encoder: SemanticEncoderLike | None = None,
    custom_policy: CrossDocumentExpansionPolicy | None = None,
    qualification: CrossDocumentPolicyQualification | None = None,
    allow_experimental: bool = False,
) -> CrossDocumentExpansionPolicy:
    """Build a policy without silently promoting an unqualified SDK mode."""

    if config.mode is CrossDocumentExpansionMode.CUSTOM:
        if custom_policy is None:
            raise ValueError("custom cross-document mode requires custom_policy")
        return custom_policy
    if config.mode in {
        CrossDocumentExpansionMode.ORACLE,
        CrossDocumentExpansionMode.HYBRID_WEIGHTED,
    } and not allow_experimental:
        raise ValueError(f"{config.mode.value} is available only in experiment/debug mode")
    if (
        config.mode is not CrossDocumentExpansionMode.NONE
        and not allow_experimental
        and (
            qualification is None
            or not qualification.sdk_exposable
        )
    ):
        raise ValueError(
            f"{config.mode.value} has not passed the cross-document SDK qualification gate"
        )
    return BuiltinCrossDocumentExpansionPolicy(config, semantic_encoder=semantic_encoder)


@dataclass(frozen=True)
class CrossDocumentPolicyEvidence:
    """Measured evidence used to decide whether a policy can enter the SDK."""

    policy_name: str
    gap_recovered: float
    extra_native_fraction: float
    official_score_delta_ci: tuple[float, float]
    seed_count: int
    model_sizes: tuple[str, ...]
    cross_family_validated: bool
    deterministic: bool = True
    no_test_tuning: bool = True
    no_catastrophic_nll_regression: bool = True
    bounded_latency: bool = True
    graceful_fallback: bool = True
    canonical_records: bool = True
    auditable_receipts: bool = True


@dataclass(frozen=True)
class CrossDocumentPolicyQualification:
    """Fail-closed promotion decision for one measured expansion policy."""

    status: str
    sdk_exposable: bool
    reasons: tuple[str, ...]


def qualify_cross_document_policy(
    evidence: CrossDocumentPolicyEvidence,
) -> CrossDocumentPolicyQualification:
    """Apply the paper's parameter-free and strong SDK promotion gates."""

    reasons = []
    if evidence.gap_recovered < 0.50:
        reasons.append("packed-RAG gap recovery is below 50%")
    if evidence.extra_native_fraction > 0.20:
        reasons.append("extra native K/V exceeds 20%")
    if evidence.official_score_delta_ci[0] < 0:
        reasons.append("official-score interval includes regression")
    if evidence.seed_count < 5:
        reasons.append("fewer than five seeds")
    if len(set(evidence.model_sizes)) < 2:
        reasons.append("fewer than two model sizes")
    checks = {
        "policy is not deterministic": evidence.deterministic,
        "test data influenced policy selection": evidence.no_test_tuning,
        "NLL regression gate failed": evidence.no_catastrophic_nll_regression,
        "latency is unbounded": evidence.bounded_latency,
        "fallback is not graceful": evidence.graceful_fallback,
        "canonical ContextRecord integration is missing": evidence.canonical_records,
        "auditable receipts are missing": evidence.auditable_receipts,
    }
    reasons.extend(label for label, passed in checks.items() if not passed)
    sdk_exposable = not reasons
    strong = (
        sdk_exposable
        and evidence.gap_recovered >= 0.80
        and evidence.extra_native_fraction <= 0.10
        and evidence.cross_family_validated
    )
    status = "SDK_STRONG_CANDIDATE" if strong else "SDK_OPTIONAL" if sdk_exposable else "RESEARCH_ONLY"
    if not evidence.cross_family_validated and sdk_exposable:
        reasons.append("cross-family validation remains recommended")
    return CrossDocumentPolicyQualification(status, sdk_exposable, tuple(reasons))
