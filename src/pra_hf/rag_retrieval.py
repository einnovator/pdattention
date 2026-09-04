"""Paper 3.2 retrieval backends with frozen, backend-attributed receipts.

The existing :mod:`pra_hf.rag_evaluation` receipt remains the handoff to RAG
and PRA selectors.  This module adds local dense/FAISS and hybrid retrieval plus
service descriptors without coupling retrieval to model realization.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .agent_resources import hashed_semantic_vector
from .rag_evaluation import (
    CandidateDocument,
    CandidateReceipt,
    ChunkerConfig,
    RAGDocument,
    RAGQuestion,
)


class BackendStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    BACKEND_IDENTITY_UNRESOLVED = "BACKEND_IDENTITY_UNRESOLVED"


class Retriever(Protocol):
    revision: str
    index_sha256: str

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]: ...


def _stable_digest(rows: Sequence[tuple[str, str]], revision: str) -> str:
    digest = hashlib.sha256(revision.encode("utf-8"))
    for document_id, fingerprint in rows:
        digest.update(document_id.encode("utf-8"))
        digest.update(fingerprint.encode("ascii"))
    return digest.hexdigest()


def _cosine_rows(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(
        matrix @ query,
        denominator,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=denominator != 0,
    )


class ExactDenseRetriever:
    """Exact cosine retrieval with a pinned, caller-replaceable embedder."""

    def __init__(
        self,
        documents: Sequence[RAGDocument],
        *,
        dimensions: int = 256,
        embedder: Callable[[str], Sequence[float]] | None = None,
        embedder_revision: str = "pra_hashed_semantic_v1",
    ) -> None:
        if not documents or dimensions <= 0:
            raise ValueError("dense retrieval requires documents and positive dimensions")
        self.documents = tuple(documents)
        self.by_id = {row.document_id: row for row in self.documents}
        if len({row.document_id for row in self.documents}) != len(self.documents):
            raise ValueError("corpus document IDs must be unique")
        self.dimensions = dimensions
        self.embedder_revision = embedder_revision
        self._embedder = embedder or (
            lambda text: hashed_semantic_vector(text, dimensions=dimensions)
        )
        self.revision = f"exact_cosine_v1:{embedder_revision}:d{dimensions}"
        self.matrix = np.asarray(
            [self._embedder(f"{row.title} {row.text}") for row in self.documents],
            dtype=np.float64,
        )
        if self.matrix.shape != (len(self.documents), dimensions):
            raise ValueError("embedder returned an unexpected vector shape")
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in self.documents], self.revision
        )

    def scores(self, query: str) -> Mapping[str, float]:
        vector = np.asarray(self._embedder(query), dtype=np.float64)
        if vector.shape != (self.dimensions,):
            raise ValueError("query embedder returned an unexpected vector shape")
        values = _cosine_rows(self.matrix, vector)
        return {row.document_id: float(score) for row, score in zip(self.documents, values)}

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.scores(query)
        ordered = sorted(
            self.documents, key=lambda row: (-scores[row.document_id], row.document_id)
        )[:top_k]
        return tuple(
            CandidateDocument(row.document_id, rank, scores[row.document_id])
            for rank, row in enumerate(ordered, 1)
        )


class HybridRetriever:
    """RRF over named first-stage retrievers sharing one immutable corpus."""

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        documents: Sequence[RAGDocument],
        *,
        constant: float = 60.0,
    ) -> None:
        if not retrievers or constant <= 0:
            raise ValueError("hybrid retrieval requires channels and a positive constant")
        self.retrievers = dict(retrievers)
        self.documents = tuple(documents)
        self.by_id = {row.document_id: row for row in self.documents}
        if len(self.by_id) != len(self.documents):
            raise ValueError("corpus document IDs must be unique")
        self.constant = constant
        components = ",".join(
            f"{name}={retriever.revision}" for name, retriever in sorted(self.retrievers.items())
        )
        self.revision = f"rrf_v1:k{constant:g}:{components}"
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in self.documents], self.revision
        )

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        rankings = {
            name: retriever.retrieve(query, len(self.documents))
            for name, retriever in self.retrievers.items()
        }
        return reciprocal_rank_fusion(rankings, top_k=top_k, constant=self.constant)


def make_backend_candidate_receipt(
    *,
    dataset: str,
    dataset_revision: str,
    corpus_revision: str,
    corpus_sha256: str,
    question: RAGQuestion,
    retriever: Retriever,
    retriever_id: str,
    documents: Mapping[str, RAGDocument],
    candidate_count: int,
    chunker: ChunkerConfig,
    seed: int = 0,
) -> CandidateReceipt:
    """Freeze any attributed backend's candidates for downstream matched arms."""

    candidates = retriever.retrieve(question.question, candidate_count)
    missing = {row.document_id for row in candidates} - set(documents)
    if missing:
        raise ValueError(f"retriever returned unavailable documents: {sorted(missing)!r}")
    return CandidateReceipt(
        dataset=dataset,
        dataset_revision=dataset_revision,
        corpus_revision=corpus_revision,
        corpus_sha256=corpus_sha256,
        example_id=question.example_id,
        retriever=retriever_id,
        retriever_revision=retriever.revision,
        index_sha256=retriever.index_sha256,
        candidates=candidates,
        document_fingerprints={
            row.document_id: documents[row.document_id].fingerprint for row in candidates
        },
        chunker=chunker,
        seed=seed,
    )


class FaissDenseRetriever(ExactDenseRetriever):
    """FAISS cosine index with explicit availability and pinned index settings."""

    def __init__(self, documents: Sequence[RAGDocument], **kwargs: object) -> None:
        super().__init__(documents, **kwargs)
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("FAISS is not installed; use ExactDenseRetriever") from exc
        self._faiss = faiss
        normalized = self.matrix.astype(np.float32, copy=True)
        faiss.normalize_L2(normalized)
        self._index = faiss.IndexFlatIP(self.dimensions)
        self._index.add(normalized)
        self.revision = f"faiss_indexflatip_v1:{self.embedder_revision}:d{self.dimensions}"
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in self.documents], self.revision
        )

    @staticmethod
    def status() -> BackendStatus:
        try:
            import faiss  # noqa: F401
        except ImportError:
            return BackendStatus.DEPENDENCY_MISSING
        return BackendStatus.AVAILABLE

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(self._embedder(query), dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dimensions:
            raise ValueError("query embedder returned an unexpected vector shape")
        self._faiss.normalize_L2(vector)
        scores, indices = self._index.search(vector, min(top_k, len(self.documents)))
        return tuple(
            CandidateDocument(self.documents[int(index)].document_id, rank, float(score))
            for rank, (index, score) in enumerate(zip(indices[0], scores[0]), 1)
            if index >= 0
        )


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[CandidateDocument]],
    *,
    top_k: int,
    constant: float = 60.0,
) -> tuple[CandidateDocument, ...]:
    """Fuse backend rankings without assuming comparable score scales."""

    if not rankings or top_k <= 0 or constant <= 0:
        raise ValueError("RRF requires rankings, positive top_k, and positive constant")
    scores: dict[str, float] = {}
    for rows in rankings.values():
        seen: set[str] = set()
        for row in rows:
            if row.document_id in seen:
                raise ValueError("one backend ranking contains a duplicate document")
            seen.add(row.document_id)
            scores[row.document_id] = scores.get(row.document_id, 0.0) + 1.0 / (
                constant + row.rank
            )
    ordered = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))[:top_k]
    return tuple(
        CandidateDocument(document_id, rank, scores[document_id])
        for rank, document_id in enumerate(ordered, 1)
    )


@dataclass(frozen=True)
class ServiceBackend:
    """Pinned external index identity used in candidate-receipt provenance."""

    kind: str
    endpoint: str
    index_name: str
    revision: str
    status: BackendStatus = BackendStatus.AVAILABLE

    def __post_init__(self) -> None:
        if self.status is BackendStatus.AVAILABLE and not all(
            (self.kind, self.endpoint, self.index_name, self.revision)
        ):
            raise ValueError("available service backends require complete identity")

    @property
    def backend_id(self) -> str:
        return f"{self.kind}:{self.index_name}:{self.revision}"


SUPPORTED_SERVICE_KINDS = frozenset(
    {"elasticsearch", "opensearch", "qdrant", "weaviate", "milvus", "pgvector"}
)


def service_backend(
    *, kind: str, endpoint: str, index_name: str, revision: str
) -> ServiceBackend:
    """Validate a supported service identity without performing network I/O."""

    normalized = kind.casefold()
    if normalized not in SUPPORTED_SERVICE_KINDS:
        raise ValueError(f"unsupported retrieval service kind: {kind}")
    return ServiceBackend(normalized, endpoint, index_name, revision)


def unresolved_derby_backend() -> ServiceBackend:
    """Retain the requested Derby arm without inventing an incompatible API."""

    return ServiceBackend(
        kind="derby",
        endpoint="",
        index_name="",
        revision="unresolved",
        status=BackendStatus.BACKEND_IDENTITY_UNRESOLVED,
    )
