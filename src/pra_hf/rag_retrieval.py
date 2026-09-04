"""Paper 3.2 retrieval backends with frozen, backend-attributed receipts.

The existing :mod:`pra_hf.rag_evaluation` receipt remains the handoff to RAG
and PRA selectors.  This module adds local dense/FAISS and hybrid retrieval plus
service descriptors without coupling retrieval to model realization.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from urllib.parse import quote
import uuid
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


JSONTransport = Callable[[str, str, Mapping[str, object] | None], Mapping[str, object]]


def _http_json_transport(
    method: str, url: str, payload: Mapping[str, object] | None
) -> Mapping[str, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else {}


def _http_ndjson(url: str, lines: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    body = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
    request = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-ndjson"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


class SentenceTransformerEmbedder:
    """Pinned batched dense embedder for modern local/FAISS retrieval arms."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        if not model_id or not revision or batch_size <= 0:
            raise ValueError("embedder identity, revision, and batch size are required")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model = None

    @property
    def identity(self) -> str:
        return f"sentence_transformers:{self.model_id}@{self.revision}"

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id, revision=self.revision, device=self.device
            )
        return self._model

    @property
    def dimensions(self) -> int:
        model = self._load()
        operation = getattr(model, "get_embedding_dimension", None)
        value = (
            operation()
            if operation is not None
            else model.get_sentence_embedding_dimension()
        )
        if value is None:
            raise RuntimeError("sentence transformer exposes no embedding dimensions")
        return int(value)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._load().encode(
                [self.document_prefix + text for text in texts],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        values = self._load().encode(
            [self.query_prefix + text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values[0], dtype=np.float32)

    def __call__(self, text: str) -> np.ndarray:
        return self.encode_query(text)


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
        document_texts = [f"{row.title} {row.text}" for row in self.documents]
        encode_documents = getattr(self._embedder, "encode_documents", None)
        self.matrix = np.asarray(
            encode_documents(document_texts)
            if encode_documents is not None
            else [self._embedder(text) for text in document_texts],
            dtype=np.float64,
        )
        if self.matrix.shape != (len(self.documents), dimensions):
            raise ValueError("embedder returned an unexpected vector shape")
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in self.documents], self.revision
        )

    def scores(self, query: str) -> Mapping[str, float]:
        encode_query = getattr(self._embedder, "encode_query", None)
        vector = np.asarray(
            encode_query(query) if encode_query is not None else self._embedder(query),
            dtype=np.float64,
        )
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


class CrossEncoderRerankedRetriever:
    """Rerank a frozen first-stage document cohort with a pinned cross-encoder."""

    def __init__(
        self,
        retriever: Retriever,
        documents: Sequence[RAGDocument],
        *,
        model_id: str,
        revision: str,
        candidate_count: int = 50,
        device: str = "cpu",
        batch_size: int = 16,
        score_pairs: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None,
    ) -> None:
        if candidate_count <= 0 or batch_size <= 0 or not model_id or not revision:
            raise ValueError("reranker requires valid identity, cohort, and batch size")
        self.retriever = retriever
        self.documents = {row.document_id: row for row in documents}
        if len(self.documents) != len(documents):
            raise ValueError("corpus document IDs must be unique")
        self.model_id = model_id
        self.model_revision = revision
        self.candidate_count = candidate_count
        self.device = device
        self.batch_size = batch_size
        self._score_pairs = score_pairs
        self._model = None
        self._tokenizer = None
        self.revision = (
            f"cross_encoder_rerank_v1:{model_id}@{revision}:"
            f"n{candidate_count}:first={retriever.revision}"
        )
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in documents], self.revision
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.model_revision
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.model_revision
        ).to(self.device)
        self._model.eval()

    def _scores(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        if self._score_pairs is not None:
            return tuple(float(value) for value in self._score_pairs(pairs))
        import torch

        self._load()
        assert self._model is not None and self._tokenizer is not None
        scores: list[float] = []
        with torch.inference_mode():
            for first in range(0, len(pairs), self.batch_size):
                batch = pairs[first : first + self.batch_size]
                encoded = self._tokenizer(
                    [query for query, _ in batch],
                    [text for _, text in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits.detach().float().cpu()
                values = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                scores.extend(float(value) for value in values)
        return tuple(scores)

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        first_stage = self.retriever.retrieve(
            query, max(top_k, self.candidate_count)
        )
        pairs = tuple(
            (
                query,
                f"{self.documents[row.document_id].title} "
                f"{self.documents[row.document_id].text}",
            )
            for row in first_stage
        )
        scores = self._scores(pairs)
        if len(scores) != len(first_stage):
            raise ValueError("cross-encoder returned an unexpected score count")
        ranked = sorted(
            zip(first_stage, scores),
            key=lambda value: (-value[1], value[0].document_id),
        )[:top_k]
        return tuple(
            CandidateDocument(row.document_id, rank, float(score))
            for rank, (row, score) in enumerate(ranked, 1)
        )


class ElasticsearchBM25Retriever:
    """Minimal REST-backed Elasticsearch BM25 adapter with pinned index identity."""

    def __init__(
        self,
        documents: Sequence[RAGDocument],
        *,
        endpoint: str,
        index_name: str,
        index_revision: str,
        transport: JSONTransport = _http_json_transport,
    ) -> None:
        if not endpoint or not index_name or not index_revision or not documents:
            raise ValueError("Elasticsearch adapter requires endpoint, index, and corpus")
        self.documents = {row.document_id: row for row in documents}
        if len(self.documents) != len(documents):
            raise ValueError("corpus document IDs must be unique")
        self.endpoint = endpoint.rstrip("/")
        self.index_name = index_name
        self.index_revision = index_revision
        self._transport = transport
        self.last_service_ms = 0.0
        self.revision = f"elasticsearch_bm25_v1:{index_name}@{index_revision}"
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in documents], self.revision
        )

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        response = self._transport(
            "POST",
            f"{self.endpoint}/{self.index_name}/_search",
            {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "text"],
                    }
                },
            },
        )
        self.last_service_ms = (time.perf_counter() - started) * 1000.0
        hits = response.get("hits", {})
        values = hits.get("hits", ()) if isinstance(hits, Mapping) else ()
        result = []
        for rank, hit in enumerate(values, 1):
            source = hit.get("_source", {})
            document_id = str(source.get("document_id", hit.get("_id", "")))
            if document_id not in self.documents:
                raise ValueError("Elasticsearch returned a document outside the corpus")
            result.append(CandidateDocument(document_id, rank, float(hit.get("_score", 0.0))))
        return tuple(result)


class QdrantDenseRetriever:
    """REST-backed Qdrant dense adapter preserving project document identities."""

    def __init__(
        self,
        documents: Sequence[RAGDocument],
        *,
        endpoint: str,
        collection_name: str,
        collection_revision: str,
        embedder: Callable[[str], Sequence[float]],
        dimensions: int,
        transport: JSONTransport = _http_json_transport,
    ) -> None:
        if not endpoint or not collection_name or not collection_revision or dimensions <= 0:
            raise ValueError("Qdrant adapter requires endpoint, collection, and dimensions")
        self.documents = {row.document_id: row for row in documents}
        if not self.documents or len(self.documents) != len(documents):
            raise ValueError("Qdrant adapter requires unique corpus documents")
        self.endpoint = endpoint.rstrip("/")
        self.collection_name = collection_name
        self.collection_revision = collection_revision
        self._embedder = embedder
        self.dimensions = dimensions
        self._transport = transport
        self.last_embedding_ms = 0.0
        self.last_service_ms = 0.0
        self.revision = f"qdrant_cosine_v1:{collection_name}@{collection_revision}:d{dimensions}"
        self.index_sha256 = _stable_digest(
            [(row.document_id, row.fingerprint) for row in documents], self.revision
        )

    def retrieve(self, query: str, top_k: int) -> tuple[CandidateDocument, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        encode_query = getattr(self._embedder, "encode_query", None)
        started = time.perf_counter()
        vector = np.asarray(
            encode_query(query) if encode_query is not None else self._embedder(query),
            dtype=np.float32,
        )
        if vector.shape != (self.dimensions,):
            raise ValueError("query embedder returned an unexpected vector shape")
        self.last_embedding_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        response = self._transport(
            "POST",
            f"{self.endpoint}/collections/{self.collection_name}/points/query",
            {"query": vector.tolist(), "limit": top_k, "with_payload": True},
        )
        self.last_service_ms = (time.perf_counter() - started) * 1000.0
        result = response.get("result", {})
        values = result.get("points", ()) if isinstance(result, Mapping) else ()
        rows = []
        for rank, point in enumerate(values, 1):
            payload = point.get("payload", {})
            document_id = str(payload.get("document_id", ""))
            if document_id not in self.documents:
                raise ValueError("Qdrant returned a document outside the corpus")
            rows.append(CandidateDocument(document_id, rank, float(point.get("score", 0.0))))
        return tuple(rows)


def index_elasticsearch_documents(
    documents: Sequence[RAGDocument],
    *,
    endpoint: str,
    index_name: str,
    transport: JSONTransport = _http_json_transport,
) -> None:
    """Create a fresh lexical index and bulk-load immutable RAG documents."""

    endpoint = endpoint.rstrip("/")
    try:
        transport("DELETE", f"{endpoint}/{index_name}", None)
    except Exception:
        pass
    transport(
        "PUT",
        f"{endpoint}/{index_name}",
        {
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                }
            }
        },
    )
    if transport is _http_json_transport:
        for first in range(0, len(documents), 250):
            lines: list[Mapping[str, object]] = []
            for row in documents[first : first + 250]:
                lines.extend(
                    (
                        {"index": {"_index": index_name, "_id": row.document_id}},
                        {"document_id": row.document_id, "title": row.title, "text": row.text},
                    )
                )
            response = _http_ndjson(f"{endpoint}/_bulk", lines)
            if response.get("errors"):
                raise RuntimeError("Elasticsearch bulk ingestion reported item errors")
    else:
        for row in documents:
            transport(
                "PUT",
                f"{endpoint}/{index_name}/_doc/{quote(row.document_id, safe='')}",
                {"document_id": row.document_id, "title": row.title, "text": row.text},
            )
    transport("POST", f"{endpoint}/{index_name}/_refresh", None)


def qdrant_points(
    documents: Sequence[RAGDocument], embedder: SentenceTransformerEmbedder
) -> tuple[dict[str, object], ...]:
    """Build deterministic Qdrant point payloads for batched REST ingestion."""

    vectors = embedder.encode_documents(
        [f"{row.title} {row.text}" for row in documents]
    )
    return tuple(
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, row.document_id)),
            "vector": vector.tolist(),
            "payload": {"document_id": row.document_id},
        }
        for row, vector in zip(documents, vectors)
    )


def index_qdrant_documents(
    documents: Sequence[RAGDocument],
    *,
    embedder: SentenceTransformerEmbedder,
    endpoint: str,
    collection_name: str,
    batch_size: int = 128,
    transport: JSONTransport = _http_json_transport,
) -> None:
    """Create a fresh cosine collection and upload deterministic point batches."""

    if batch_size <= 0:
        raise ValueError("Qdrant ingestion batch size must be positive")
    endpoint = endpoint.rstrip("/")
    try:
        transport("DELETE", f"{endpoint}/collections/{collection_name}", None)
    except Exception:
        pass
    transport(
        "PUT",
        f"{endpoint}/collections/{collection_name}",
        {"vectors": {"size": embedder.dimensions, "distance": "Cosine"}},
    )
    points = qdrant_points(documents, embedder)
    for first in range(0, len(points), batch_size):
        transport(
            "PUT",
            f"{endpoint}/collections/{collection_name}/points?wait=true",
            {"points": list(points[first : first + batch_size])},
        )


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
        encode_query = getattr(self._embedder, "encode_query", None)
        vector = np.asarray(
            encode_query(query) if encode_query is not None else self._embedder(query),
            dtype=np.float32,
        ).reshape(1, -1)
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
