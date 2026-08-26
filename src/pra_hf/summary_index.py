"""Natural-language routing indices backed by unchanged native K/V chunks.

The records in this module are addresses.  They may be scored, shuffled, or
replaced without changing the source identity handed to PRA's materializer.
Keeping that boundary explicit prevents a summary experiment from silently
becoming text-RAG or summary-only answering.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np


_WORD = re.compile(r"[\w]+(?:[-'][\w]+)*", flags=re.UNICODE)


def lexical_terms(text: str) -> tuple[str, ...]:
    """Return normalization-stable lexical terms for summary routing."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_WORD.findall(normalized))


def source_sha256(text: str) -> str:
    """Hash the exact source text represented by one routing address."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SummaryFacet:
    """One independently addressable aspect of a generated chunk summary."""

    label: str
    text: str

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.text.strip():
            raise ValueError("Summary facets require non-empty labels and text.")


@dataclass(frozen=True)
class NativeKVRequest:
    """Identity-only request consumed by the unchanged native-K/V layer."""

    uri: str
    chunk_id: str
    token_start: int
    token_end: int
    source_sha256: str


@dataclass(frozen=True)
class SummaryIndexRecord:
    """Persistent routing address aligned to exactly one source K/V chunk.

    ``summary`` and ``facets`` are lossy discovery metadata.  The URI, chunk
    identifier, token span, and source hash are the immutable materialization
    identity.  Tensor payloads deliberately do not appear in this record.
    """

    uri: str
    chunk_id: str
    token_start: int
    token_end: int
    source_sha256: str
    summary: str
    facets: tuple[SummaryFacet, ...] = ()
    summary_token_count: int = 0
    generation_model: str = ""
    prompt_id: str = ""

    def __post_init__(self) -> None:
        if not self.uri or not self.chunk_id:
            raise ValueError("Summary records require stable URI and chunk identifiers.")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("Summary record token spans must be non-empty and ordered.")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a hexadecimal SHA-256 digest.")
        int(self.source_sha256, 16)
        if not self.summary.strip() and not self.facets:
            raise ValueError("A summary record must contain summary text or facets.")
        if self.summary_token_count < 0:
            raise ValueError("summary_token_count cannot be negative.")

    @property
    def identity(self) -> tuple[str, str]:
        """Return the logical native-memory identity addressed by this record."""

        return self.uri, self.chunk_id

    @property
    def address_texts(self) -> tuple[str, ...]:
        """Return independently scored summary texts, preserving facet geometry."""

        values = tuple(facet.text for facet in self.facets if facet.text.strip())
        return values or (self.summary,)

    @property
    def text_bytes(self) -> int:
        """Count UTF-8 routing text bytes without charging backing native K/V."""

        return sum(len(text.encode("utf-8")) for text in self.address_texts)

    def native_kv_request(self) -> NativeKVRequest:
        """Drop all lossy text and return only the source materialization address."""

        return NativeKVRequest(
            uri=self.uri,
            chunk_id=self.chunk_id,
            token_start=self.token_start,
            token_end=self.token_end,
            source_sha256=self.source_sha256,
        )

    def to_dict(self) -> dict:
        """Serialize the address and generation provenance for a JSONL cache."""

        return {
            "uri": self.uri,
            "chunk_id": self.chunk_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "source_sha256": self.source_sha256,
            "summary": self.summary,
            "facets": [
                {"label": facet.label, "text": facet.text} for facet in self.facets
            ],
            "summary_token_count": self.summary_token_count,
            "generation_model": self.generation_model,
            "prompt_id": self.prompt_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "SummaryIndexRecord":
        """Restore one record from the tracked JSON representation."""

        return cls(
            uri=str(value["uri"]),
            chunk_id=str(value["chunk_id"]),
            token_start=int(value["token_start"]),
            token_end=int(value["token_end"]),
            source_sha256=str(value["source_sha256"]),
            summary=str(value.get("summary", "")),
            facets=tuple(
                SummaryFacet(label=str(row["label"]), text=str(row["text"]))
                for row in value.get("facets", ())
            ),
            summary_token_count=int(value.get("summary_token_count", 0)),
            generation_model=str(value.get("generation_model", "")),
            prompt_id=str(value.get("prompt_id", "")),
        )


class SummaryIndex:
    """Validated collection of lossy addresses for one logical source."""

    def __init__(self, records: Iterable[SummaryIndexRecord]):
        self.records = tuple(records)
        if not self.records:
            raise ValueError("SummaryIndex requires at least one record.")
        identities = [record.identity for record in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("SummaryIndex identities must be unique.")

    def assert_source_alignment(
        self,
        expected: Iterable[tuple[str, str, int, int, str]],
    ) -> None:
        """Assert exact URI, span, and source-hash parity with the K/V registry."""

        observed = {
            (
                record.uri,
                record.chunk_id,
                record.token_start,
                record.token_end,
                record.source_sha256,
            )
            for record in self.records
        }
        expected_set = set(expected)
        if observed != expected_set:
            missing = expected_set - observed
            extra = observed - expected_set
            raise ValueError(
                f"Summary/source alignment failed: missing={len(missing)} extra={len(extra)}"
            )

    def shuffled_addresses(self, seed: int) -> "SummaryIndex":
        """Permute only summary content while preserving every native identity."""

        addresses = [(record.summary, record.facets) for record in self.records]
        random.Random(seed).shuffle(addresses)
        return SummaryIndex(
            replace(record, summary=summary, facets=facets, prompt_id=f"{record.prompt_id}:shuffled")
            for record, (summary, facets) in zip(self.records, addresses)
        )

    def materialization_requests(self, selected_indices: Sequence[int]) -> tuple[NativeKVRequest, ...]:
        """Translate selected addresses to unchanged native-K/V requests."""

        return tuple(self.records[index].native_kv_request() for index in selected_indices)

    @property
    def text_bytes(self) -> int:
        """Return persistent summary-text bytes for this source."""

        return sum(record.text_bytes for record in self.records)


class BM25SummaryScorer:
    """BM25 over summaries, with max pooling across independent facets."""

    def __init__(self, index: SummaryIndex, *, k1: float = 1.2, b: float = 0.75):
        self.index = index
        self.k1 = float(k1)
        self.b = float(b)
        self.documents = tuple(
            tuple(Counter(lexical_terms(text)) for text in record.address_texts)
            for record in index.records
        )
        flattened = [counter for facets in self.documents for counter in facets]
        self.average_length = sum(sum(doc.values()) for doc in flattened) / max(
            len(flattened), 1
        )
        document_frequency = Counter()
        for facets in self.documents:
            terms = set().union(*(set(facet) for facet in facets))
            document_frequency.update(terms)
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str) -> np.ndarray:
        """Score each chunk, retaining the strongest matching summary facet."""

        query_terms = lexical_terms(query)
        output = []
        for facets in self.documents:
            facet_scores = []
            for document in facets:
                length = sum(document.values())
                score = 0.0
                for term in query_terms:
                    frequency = document.get(term, 0)
                    if not frequency:
                        continue
                    denominator = frequency + self.k1 * (
                        1.0 - self.b + self.b * length / max(self.average_length, 1e-12)
                    )
                    score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
                facet_scores.append(score)
            output.append(max(facet_scores, default=0.0))
        return np.asarray(output, dtype=np.float64)


def exact_summary_scores(index: SummaryIndex, query: str) -> np.ndarray:
    """Score exact normalized query-token matches against each summary/facet."""

    query_terms = set(lexical_terms(query))
    return np.asarray(
        [
            max(
                (len(query_terms.intersection(lexical_terms(text))) for text in record.address_texts),
                default=0,
            )
            for record in index.records
        ],
        dtype=np.float64,
    )


class FrozenEmbeddingScorer:
    """Cosine scorer for externally generated, frozen summary embeddings."""

    def __init__(self, index: SummaryIndex, embeddings: Sequence[Sequence[Sequence[float]]]):
        if len(embeddings) != len(index.records):
            raise ValueError("Embedding rows must align one-to-one with summary records.")
        self.index = index
        self.embeddings = tuple(self._normalize(np.asarray(row, dtype=np.float32)) for row in embeddings)
        for record, row in zip(index.records, self.embeddings):
            if row.ndim != 2 or row.shape[0] != len(record.address_texts):
                raise ValueError("Each record needs one embedding per independently scored address.")

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    def score(self, query_embedding: Sequence[float]) -> np.ndarray:
        """Return max-facet cosine similarity for every summary record."""

        query = self._normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        return np.asarray([float((row @ query).max()) for row in self.embeddings])


def minmax(values: Sequence[float]) -> np.ndarray:
    """Map scores to [0, 1], returning zeros for a constant channel."""

    values = np.asarray(values, dtype=np.float64)
    extent = float(values.max() - values.min()) if values.size else 0.0
    return (values - values.min()) / extent if extent > 1e-12 else np.zeros_like(values)


def hybrid_scores(left: Sequence[float], right: Sequence[float], alpha: float) -> np.ndarray:
    """Fuse two independently normalized channels with validation-frozen weight."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1].")
    return alpha * minmax(left) + (1.0 - alpha) * minmax(right)


def stable_topk(scores: Sequence[float], k: int) -> tuple[int, ...]:
    """Return deterministic descending top-k indices with index tie breaking."""

    if k <= 0:
        return ()
    return tuple(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k])


def retrieval_metrics(
    scores: Sequence[float],
    positive_indices: Iterable[int],
    *,
    k: int,
) -> dict[str, float | list[int]]:
    """Compute identity-level retrieval endpoints for one query/source pair."""

    positives = set(int(index) for index in positive_indices)
    ranking = stable_topk(scores, len(scores))
    selected = ranking[:k]
    recovered = positives.intersection(selected)
    first_rank = next(
        (rank for rank, index in enumerate(ranking, start=1) if index in positives),
        None,
    )
    return {
        "evidence_recall": len(recovered) / max(len(positives), 1),
        "complete_recovery": float(bool(positives) and recovered == positives),
        "precision": len(recovered) / max(len(selected), 1),
        "reciprocal_rank": 1.0 / first_rank if first_rank is not None else 0.0,
        "selected_indices": list(selected),
        "recovered_indices": sorted(recovered),
    }


def amortized_ingestion_cost(one_time_seconds: float, query_count: int) -> float:
    """Allocate one-time summary generation cost across later queries."""

    if query_count <= 0:
        raise ValueError("query_count must be positive.")
    return float(one_time_seconds) / query_count
