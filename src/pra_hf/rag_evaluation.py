"""Matched-candidate evaluation primitives for standard RAG and PRA.

The external retriever is intentionally outside the comparison.  It creates a
``CandidateReceipt`` once; standard RAG and PRA then consume the same ordered
documents.  This keeps retrieval failures separate from context selection,
materialization, and generation failures.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence

from .agent_resources import hashed_semantic_vector
from .context_records import ContextRecord, RecordType


_TERM = re.compile(r"[A-Za-z0-9_./:@+-]+")
_TOKEN = re.compile(r"\S+")


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _terms(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TERM.findall(text))


class TokenCounter(Protocol):
    """Minimal tokenizer surface needed by deterministic context packing."""

    def __call__(self, text: str) -> int: ...


def whitespace_token_count(text: str) -> int:
    """Return a dependency-free token estimate for fixture and retrieval runs."""

    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class RAGSection:
    """One named section whose character boundaries remain addressable."""

    section_id: str
    title: str
    start: int
    end: int
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.section_id:
            raise ValueError("section_id is required")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("section boundaries must describe a non-empty interval")


@dataclass(frozen=True)
class RAGDocument:
    """Versioned source document retained as one logical PRA resource."""

    document_id: str
    title: str
    text: str
    source: str = ""
    uri: str = ""
    version: str = "v1"
    mime: str = "text/plain"
    sections: tuple[RAGSection, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id or not self.text.strip():
            raise ValueError("documents require a stable ID and non-empty text")
        if len({section.section_id for section in self.sections}) != len(self.sections):
            raise ValueError("section IDs must be unique within a document")
        if any(section.end > len(self.text) for section in self.sections):
            raise ValueError("section boundaries exceed the document text")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "document_id": self.document_id,
                "title": self.title,
                "text": self.text,
                "source": self.source,
                "uri": self.uri,
                "version": self.version,
                "mime": self.mime,
                "sections": [asdict(section) for section in self.sections],
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class RAGQuestion:
    """Question, accepted answers, and independently annotated evidence."""

    example_id: str
    question: str
    answers: tuple[str, ...]
    gold_document_ids: frozenset[str]
    gold_spans: Mapping[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    question_type: str = ""

    def __post_init__(self) -> None:
        if not self.example_id or not self.question.strip() or not self.answers:
            raise ValueError("questions require an ID, text, and at least one answer")
        object.__setattr__(self, "answers", tuple(dict.fromkeys(self.answers)))
        object.__setattr__(self, "gold_document_ids", frozenset(self.gold_document_ids))
        object.__setattr__(self, "gold_spans", dict(self.gold_spans))


@dataclass(frozen=True)
class ChunkerConfig:
    """Frozen chunk construction used by both context selectors."""

    max_tokens: int = 256
    overlap_tokens: int = 32
    algorithm: str = "whitespace_offsets_v1"

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")


@dataclass(frozen=True)
class RAGChunk:
    """One independently selectable character interval in a source document."""

    chunk_id: str
    document_id: str
    ordinal: int
    start: int
    end: int
    text: str
    token_count: int
    section_id: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or self.token_count <= 0:
            raise ValueError("chunks must have a non-empty character and token extent")


def chunk_document(
    document: RAGDocument,
    config: ChunkerConfig,
    *,
    token_count: TokenCounter = whitespace_token_count,
) -> tuple[RAGChunk, ...]:
    """Create stable overlapping chunks while preserving exact source offsets."""

    matches = tuple(_TOKEN.finditer(document.text))
    if not matches:
        return ()
    stride = config.max_tokens - config.overlap_tokens
    chunks: list[RAGChunk] = []
    for ordinal, first in enumerate(range(0, len(matches), stride)):
        last = min(first + config.max_tokens, len(matches))
        start = matches[first].start()
        end = matches[last - 1].end()
        text = document.text[start:end]
        containing = next(
            (
                section.section_id
                for section in document.sections
                if section.start <= start and end <= section.end
            ),
            None,
        )
        chunks.append(
            RAGChunk(
                chunk_id=f"{document.document_id}:chunk:{ordinal}",
                document_id=document.document_id,
                ordinal=ordinal,
                start=start,
                end=end,
                text=text,
                token_count=token_count(text),
                section_id=containing,
            )
        )
        if last == len(matches):
            break
    return tuple(chunks)


@dataclass(frozen=True)
class CandidateDocument:
    """One first-stage retrieval result persisted in rank order."""

    document_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class CandidateReceipt:
    """Immutable first-stage retrieval output consumed by every condition."""

    dataset: str
    dataset_revision: str
    corpus_revision: str
    corpus_sha256: str
    example_id: str
    retriever: str
    retriever_revision: str
    index_sha256: str
    candidates: tuple[CandidateDocument, ...]
    document_fingerprints: Mapping[str, str]
    chunker: ChunkerConfig
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("candidate receipts cannot be empty")
        ids = tuple(candidate.document_id for candidate in self.candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate document IDs must be unique")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("candidate ranks must be contiguous and one-based")
        if set(ids) != set(self.document_fingerprints):
            raise ValueError("receipt fingerprints must exactly cover candidates")
        object.__setattr__(self, "document_fingerprints", dict(self.document_fingerprints))

    @property
    def candidate_document_ids(self) -> tuple[str, ...]:
        return tuple(candidate.document_id for candidate in self.candidates)

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "corpus_revision": self.corpus_revision,
            "corpus_sha256": self.corpus_sha256,
            "example_id": self.example_id,
            "retriever": self.retriever,
            "retriever_revision": self.retriever_revision,
            "index_sha256": self.index_sha256,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "document_fingerprints": dict(self.document_fingerprints),
            "chunker": asdict(self.chunker),
            "seed": self.seed,
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CandidateReceipt":
        supplied_id = value.get("receipt_id")
        receipt = cls(
            dataset=str(value["dataset"]),
            dataset_revision=str(value["dataset_revision"]),
            corpus_revision=str(value["corpus_revision"]),
            corpus_sha256=str(value["corpus_sha256"]),
            example_id=str(value["example_id"]),
            retriever=str(value["retriever"]),
            retriever_revision=str(value["retriever_revision"]),
            index_sha256=str(value["index_sha256"]),
            candidates=tuple(
                CandidateDocument(
                    str(row["document_id"]), int(row["rank"]), float(row["score"])
                )
                for row in value["candidates"]  # type: ignore[index]
            ),
            document_fingerprints={
                str(key): str(item)
                for key, item in dict(value["document_fingerprints"]).items()
            },
            chunker=ChunkerConfig(**dict(value["chunker"])),
            seed=int(value.get("seed", 0)),
        )
        if supplied_id is not None and supplied_id != receipt.receipt_id:
            raise ValueError("candidate receipt digest does not match its contents")
        return receipt

    def validate_documents(self, documents: Mapping[str, RAGDocument]) -> None:
        """Reject changed, missing, or substituted documents before evaluation."""

        if set(self.candidate_document_ids) - set(documents):
            raise ValueError("one or more frozen candidate documents are unavailable")
        for document_id, expected in self.document_fingerprints.items():
            if documents[document_id].fingerprint != expected:
                raise ValueError(f"candidate document {document_id!r} changed after retrieval")


class FirstStageBM25:
    """Small exact BM25 retriever used to create auditable candidate receipts."""

    revision = "pra_bm25_v1_k1.2_b0.75"

    def __init__(self, documents: Sequence[RAGDocument]) -> None:
        self.documents = tuple(documents)
        self.by_id = {document.document_id: document for document in self.documents}
        if len(self.by_id) != len(self.documents):
            raise ValueError("corpus document IDs must be unique")
        self.term_frequencies = {
            document.document_id: Counter(_terms(f"{document.title} {document.text}"))
            for document in self.documents
        }
        self.lengths = {
            document_id: sum(frequencies.values())
            for document_id, frequencies in self.term_frequencies.items()
        }
        self.average_length = sum(self.lengths.values()) / max(len(self.lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies.values():
            self.document_frequency.update(frequencies)
        self.index_sha256 = _digest(
            {
                "revision": self.revision,
                "documents": [
                    (document.document_id, document.fingerprint)
                    for document in self.documents
                ],
            }
        )

    def scores(self, query: str) -> Mapping[str, float]:
        query_terms = set(_terms(query))
        count = max(len(self.documents), 1)
        scores: dict[str, float] = {}
        for document_id, frequencies in self.term_frequencies.items():
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency.get(term, 0)
                inverse = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * self.lengths[document_id] / max(self.average_length, 1.0)
                )
                score += inverse * frequency * 2.2 / denominator
            scores[document_id] = score
        return scores

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.scores(query)
        ordered = sorted(
            self.documents,
            key=lambda document: (-scores[document.document_id], document.document_id),
        )[:top_k]
        return tuple(
            CandidateDocument(document.document_id, rank, scores[document.document_id])
            for rank, document in enumerate(ordered, 1)
        )


def make_candidate_receipt(
    *,
    dataset: str,
    dataset_revision: str,
    corpus_revision: str,
    corpus_sha256: str,
    question: RAGQuestion,
    retriever: FirstStageBM25,
    candidate_count: int,
    chunker: ChunkerConfig,
    ensure_gold: bool = False,
    seed: int = 0,
) -> CandidateReceipt:
    """Freeze ranked candidates, optionally injecting missing gold for L1 only."""

    candidates = list(retriever.retrieve(question.question, candidate_count))
    if ensure_gold:
        present = {candidate.document_id for candidate in candidates}
        missing = sorted(question.gold_document_ids - present)
        for document_id in missing:
            if document_id not in retriever.by_id:
                raise ValueError(f"gold document {document_id!r} is absent from the corpus")
        keep = max(candidate_count - len(missing), 0)
        candidates = candidates[:keep] + [
            CandidateDocument(document_id, 0, retriever.scores(question.question)[document_id])
            for document_id in missing[:candidate_count]
        ]
        candidates = [
            CandidateDocument(candidate.document_id, rank, candidate.score)
            for rank, candidate in enumerate(candidates, 1)
        ]
    selected_documents = {candidate.document_id: retriever.by_id[candidate.document_id] for candidate in candidates}
    return CandidateReceipt(
        dataset=dataset,
        dataset_revision=dataset_revision,
        corpus_revision=corpus_revision,
        corpus_sha256=corpus_sha256,
        example_id=question.example_id,
        retriever="bm25",
        retriever_revision=retriever.revision,
        index_sha256=retriever.index_sha256,
        candidates=tuple(candidates),
        document_fingerprints={
            document_id: document.fingerprint
            for document_id, document in selected_documents.items()
        },
        chunker=chunker,
        seed=seed,
    )


class ContextCondition(str, Enum):
    NO_PRA = "no_pra"
    PRA_NO_ADAPTOR = "pra_no_adaptor"
    PRA_ADAPTOR_BUNDLE = "pra_adaptor_bundle"
    ORACLE_GOLD_DOCUMENTS = "oracle_gold_documents"
    ORACLE_GOLD_SPANS = "oracle_gold_spans"


@dataclass(frozen=True)
class RankedChunk:
    chunk: RAGChunk
    score: float
    rank: int
    channel_ranks: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PackedContext:
    """Selected source intervals and their exact physical token accounting."""

    condition: ContextCondition
    chunks: tuple[RankedChunk, ...]
    token_budget: int
    packed_tokens: int
    candidate_tokens: int
    selector_latency_ms: float
    index_build_ms: float
    selector_name: str
    bundle_id: str | None = None
    bundle_revision: str | None = None

    @property
    def selected_document_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.chunk.document_id for row in self.chunks))

    @property
    def selected_chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.chunk.chunk_id for row in self.chunks)

    @property
    def text(self) -> str:
        return "\n\n".join(row.chunk.text for row in self.chunks)


class ChunkSelector(Protocol):
    """Context selector that ranks chunks from one frozen candidate receipt."""

    name: str

    def rank(self, query: str, chunks: Sequence[RAGChunk]) -> tuple[RankedChunk, ...]: ...


@dataclass(frozen=True)
class PreparedCandidateContext:
    """Chunks built once for all selectors and physical-budget cutoffs."""

    chunks: tuple[RAGChunk, ...]
    build_latency_ms: float

    @property
    def candidate_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.chunks)


def prepare_candidate_context(
    receipt: CandidateReceipt,
    documents: Mapping[str, RAGDocument],
    *,
    token_count: TokenCounter = whitespace_token_count,
) -> PreparedCandidateContext:
    """Validate and chunk a frozen candidate set exactly once per receipt."""

    receipt.validate_documents(documents)
    started = time.perf_counter()
    chunks = tuple(
        chunk
        for document_id in receipt.candidate_document_ids
        for chunk in chunk_document(
            documents[document_id], receipt.chunker, token_count=token_count
        )
    )
    return PreparedCandidateContext(chunks, (time.perf_counter() - started) * 1000.0)


def _bm25_chunk_scores(query: str, chunks: Sequence[RAGChunk]) -> dict[str, float]:
    frequencies = {chunk.chunk_id: Counter(_terms(chunk.text)) for chunk in chunks}
    lengths = {chunk_id: sum(value.values()) for chunk_id, value in frequencies.items()}
    average = sum(lengths.values()) / max(len(lengths), 1)
    document_frequency: Counter[str] = Counter()
    for value in frequencies.values():
        document_frequency.update(value)
    count = max(len(chunks), 1)
    scores: dict[str, float] = {}
    for chunk_id, chunk_terms in frequencies.items():
        score = 0.0
        for term in set(_terms(query)):
            frequency = chunk_terms.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(
                1.0 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * lengths[chunk_id] / max(average, 1.0)
            )
            score += inverse * frequency * 2.2 / denominator
        scores[chunk_id] = score
    return scores


class StandardRAGSelector:
    """Defensible baseline: global BM25 chunk ranking over frozen documents."""

    name = "global_bm25_chunk_packing_v1"

    def rank(self, query: str, chunks: Sequence[RAGChunk]) -> tuple[RankedChunk, ...]:
        scores = _bm25_chunk_scores(query, chunks)
        ordered = sorted(chunks, key=lambda chunk: (-scores[chunk.chunk_id], chunk.chunk_id))
        return tuple(
            RankedChunk(chunk, scores[chunk.chunk_id], rank, {"bm25": rank})
            for rank, chunk in enumerate(ordered, 1)
        )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class PRAHybridSelector:
    """Generic PRA routing via rank-fused BM25 and hashed semantic views."""

    name = "pra_bm25_hashed_semantic_rrf_v1"

    def __init__(self, *, embedding_dimensions: int = 128, rrf_constant: int = 60) -> None:
        if embedding_dimensions <= 0 or rrf_constant <= 0:
            raise ValueError("embedding dimensions and RRF constant must be positive")
        self.embedding_dimensions = embedding_dimensions
        self.rrf_constant = rrf_constant

    def rank(self, query: str, chunks: Sequence[RAGChunk]) -> tuple[RankedChunk, ...]:
        bm25 = _bm25_chunk_scores(query, chunks)
        query_vector = hashed_semantic_vector(query, dimensions=self.embedding_dimensions)
        semantic = {
            chunk.chunk_id: _cosine(
                query_vector,
                hashed_semantic_vector(chunk.text, dimensions=self.embedding_dimensions),
            )
            for chunk in chunks
        }
        bm25_order = sorted(chunks, key=lambda chunk: (-bm25[chunk.chunk_id], chunk.chunk_id))
        semantic_order = sorted(
            chunks, key=lambda chunk: (-semantic[chunk.chunk_id], chunk.chunk_id)
        )
        bm25_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(bm25_order, 1)}
        semantic_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(semantic_order, 1)}
        fused = {
            chunk.chunk_id: 1.0 / (self.rrf_constant + bm25_ranks[chunk.chunk_id])
            + 1.0 / (self.rrf_constant + semantic_ranks[chunk.chunk_id])
            for chunk in chunks
        }
        ordered = sorted(chunks, key=lambda chunk: (-fused[chunk.chunk_id], chunk.chunk_id))
        return tuple(
            RankedChunk(
                chunk,
                fused[chunk.chunk_id],
                rank,
                {
                    "bm25": bm25_ranks[chunk.chunk_id],
                    "hashed_semantic": semantic_ranks[chunk.chunk_id],
                },
            )
            for rank, chunk in enumerate(ordered, 1)
        )


def pack_ranked_chunks(
    ranked: Sequence[RankedChunk], token_budget: int
) -> tuple[RankedChunk, ...]:
    """Pack whole ranked chunks without exceeding the physical-token budget."""

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    selected: list[RankedChunk] = []
    used = 0
    for row in ranked:
        if used + row.chunk.token_count <= token_budget:
            selected.append(row)
            used += row.chunk.token_count
    return tuple(selected)


def select_context(
    *,
    condition: ContextCondition,
    selector: ChunkSelector,
    query: str,
    receipt: CandidateReceipt,
    documents: Mapping[str, RAGDocument],
    token_budget: int,
    token_count: TokenCounter = whitespace_token_count,
    bundle_id: str | None = None,
    bundle_revision: str | None = None,
) -> PackedContext:
    """Rank and pack chunks after validating the immutable candidate identity."""

    if bool(bundle_id) != bool(bundle_revision):
        raise ValueError("bundle ID and immutable revision must be supplied together")
    prepared = prepare_candidate_context(receipt, documents, token_count=token_count)
    started = time.perf_counter()
    ranked = selector.rank(query, prepared.chunks)
    selected = pack_ranked_chunks(ranked, token_budget)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return PackedContext(
        condition=condition,
        chunks=selected,
        token_budget=token_budget,
        packed_tokens=sum(row.chunk.token_count for row in selected),
        candidate_tokens=prepared.candidate_tokens,
        selector_latency_ms=latency_ms,
        index_build_ms=prepared.build_latency_ms,
        selector_name=selector.name,
        bundle_id=bundle_id,
        bundle_revision=bundle_revision,
    )


def packed_context_from_ranking(
    *,
    condition: ContextCondition,
    selector_name: str,
    ranked: Sequence[RankedChunk],
    prepared: PreparedCandidateContext,
    token_budget: int,
    selector_latency_ms: float,
    bundle_id: str | None = None,
    bundle_revision: str | None = None,
) -> PackedContext:
    """Apply a budget to a cached ranking without repeating chunk scoring."""

    selected = pack_ranked_chunks(ranked, token_budget)
    return PackedContext(
        condition=condition,
        chunks=selected,
        token_budget=token_budget,
        packed_tokens=sum(row.chunk.token_count for row in selected),
        candidate_tokens=prepared.candidate_tokens,
        selector_latency_ms=selector_latency_ms,
        index_build_ms=prepared.build_latency_ms,
        selector_name=selector_name,
        bundle_id=bundle_id,
        bundle_revision=bundle_revision,
    )


def oracle_gold_document_context(
    *,
    question: RAGQuestion,
    receipt: CandidateReceipt,
    documents: Mapping[str, RAGDocument],
    token_budget: int,
    token_count: TokenCounter = whitespace_token_count,
) -> PackedContext:
    """Research-only control that packs chunks from available gold documents."""

    receipt.validate_documents(documents)
    chunks = tuple(
        chunk
        for document_id in receipt.candidate_document_ids
        if document_id in question.gold_document_ids
        for chunk in chunk_document(documents[document_id], receipt.chunker, token_count=token_count)
    )
    ranked = tuple(RankedChunk(chunk, 1.0, rank, {"oracle": rank}) for rank, chunk in enumerate(chunks, 1))
    selected = pack_ranked_chunks(ranked, token_budget)
    return PackedContext(
        ContextCondition.ORACLE_GOLD_DOCUMENTS,
        selected,
        token_budget,
        sum(row.chunk.token_count for row in selected),
        sum(
            chunk.token_count
            for document_id in receipt.candidate_document_ids
            for chunk in chunk_document(documents[document_id], receipt.chunker, token_count=token_count)
        ),
        0.0,
        0.0,
        "oracle_gold_documents",
    )


def _dcg(relevances: Sequence[int]) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(relevances, 1))


def context_metrics(
    question: RAGQuestion,
    receipt: CandidateReceipt,
    context: PackedContext,
) -> dict[str, float | int]:
    """Compute retrieval and physical-context metrics without answer conflation."""

    candidate_ids = receipt.candidate_document_ids
    selected_ids = context.selected_document_ids
    candidate_gold = question.gold_document_ids.intersection(candidate_ids)
    selected_gold = question.gold_document_ids.intersection(selected_ids)
    document_recall = len(candidate_gold) / max(len(question.gold_document_ids), 1)
    support_coverage = len(selected_gold) / max(len(question.gold_document_ids), 1)
    first_rank = next(
        (rank for rank, document_id in enumerate(selected_ids, 1) if document_id in question.gold_document_ids),
        None,
    )
    relevances = [int(document_id in question.gold_document_ids) for document_id in selected_ids]
    ideal = sorted(relevances, reverse=True)
    false_count = len(set(selected_ids) - question.gold_document_ids)
    span_hits = 0
    span_total = sum(len(spans) for spans in question.gold_spans.values())
    for document_id, spans in question.gold_spans.items():
        for start, end in spans:
            if any(
                row.chunk.document_id == document_id
                and row.chunk.start < end
                and start < row.chunk.end
                for row in context.chunks
            ):
                span_hits += 1
    return {
        "document_recall_at_candidate_k": document_recall,
        "supporting_document_coverage": support_coverage,
        "supporting_span_coverage": span_hits / span_total if span_total else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": _dcg(relevances) / _dcg(ideal) if ideal and _dcg(ideal) else 0.0,
        "gold_document_selected_fraction": len(selected_gold) / max(len(selected_ids), 1),
        "false_selected_document_fraction": false_count / max(len(selected_ids), 1),
        "logical_candidate_tokens": context.candidate_tokens,
        "physical_context_tokens": context.packed_tokens,
        "selected_full_ratio": context.packed_tokens / max(context.candidate_tokens, 1),
        "materialization_avoidance": 1.0 - context.packed_tokens / max(context.candidate_tokens, 1),
        "packed_candidate_tokens": context.packed_tokens,
        "discarded_candidate_tokens": context.candidate_tokens - context.packed_tokens,
    }


def document_to_context_record(
    document: RAGDocument,
    chunks: Sequence[RAGChunk],
    *,
    receipt_id: str,
) -> ContextRecord:
    """Represent a candidate document as a hierarchical typed PRA resource."""

    return ContextRecord(
        record_id=document.document_id,
        record_type=RecordType.GENERIC_DOCUMENT,
        payload={
            "type": "document",
            "uri": document.uri,
            "document_id": document.document_id,
            "title": document.title,
            "source": document.source,
            "version": document.version,
            "mime": document.mime,
            "token_count": sum(chunk.token_count for chunk in chunks),
            "sections": [asdict(section) for section in document.sections],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "section_id": chunk.section_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
        },
        selection_provenance={
            "candidate_receipt_id": receipt_id,
            "document_fingerprint": document.fingerprint,
        },
        version=document.version,
        source_fingerprint=document.fingerprint,
    )


def failure_classification(
    *,
    question: RAGQuestion,
    receipt: CandidateReceipt,
    context: PackedContext,
    answer_correct: bool,
    materialization_ok: bool = True,
) -> str:
    """Assign one primary failure stage for per-example analysis."""

    if not question.gold_document_ids.issubset(receipt.candidate_document_ids):
        return "first_stage_retrieval_failure"
    if not materialization_ok:
        return "materialization_failure"
    if context.packed_tokens == 0:
        return "insufficient_token_budget"
    if not question.gold_document_ids.intersection(context.selected_document_ids):
        return (
            "standard_rag_packing_failure"
            if context.condition is ContextCondition.NO_PRA
            else "pra_document_selection_failure"
        )
    if not answer_correct:
        return "generation_failure"
    return "success"
