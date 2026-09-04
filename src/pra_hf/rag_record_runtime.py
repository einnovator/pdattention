"""PRA-native document records, prompt references, and learned reranking.

This module joins the repository's existing typed-record, lightweight-reference,
URI-resolver, and RAG-selection contracts.  It deliberately stops before a
specific tensor backend: an engine consumes the ordered materialized chunk
texts in a :class:`ResolvedRecordRequest` and owns their native K/V lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol, Sequence
from urllib.parse import quote

from pra_core.references import ReferenceTable
from pra_torch.refs import parse_ref_tokens
from pra_torch.resolver import InMemoryResolver

from .context_records import ContextRecord, RecordType, rag_chunk_record
from .rag_evaluation import (
    CandidateReceipt,
    ChunkSelector,
    CrossEncoderRAGSelector,
    RAGChunk,
    RAGDocument,
    RankedChunk,
    SelectionReceipt,
    StandardRAGSelector,
    chunk_document,
    whitespace_token_count,
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def document_record_uri(dataset: str, document_id: str) -> str:
    """Return a deterministic URI for one persistent source document."""

    return f"pra://{quote(dataset, safe='')}/document/{quote(document_id, safe='')}"


def chunk_record_uri(document_uri: str, ordinal: int) -> str:
    """Return a distinct address for one source-relative document chunk."""

    if ordinal < 0:
        raise ValueError("chunk ordinal cannot be negative")
    return f"{document_uri}#chunk={ordinal}"


def root_record_uri(dataset: str, request_id: str) -> str:
    """Return a request-scoped logical root that names a candidate record set."""

    return f"pra://{quote(dataset, safe='')}/query-set/{quote(request_id, safe='')}"


@dataclass(frozen=True)
class RecordIngestionReceipt:
    """Immutable identity and source metadata for an ingested document corpus."""

    dataset: str
    records: tuple[Mapping[str, object], ...]
    resolver_revision: str = "in_memory_uri_resolver_v1"
    schema_version: str = "paper3.2-native-record-ingestion-v1"

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "resolver_revision": self.resolver_revision,
            "records": [dict(row) for row in self.records],
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value


@dataclass(frozen=True)
class RerankedCandidate:
    """One candidate's first-stage and learned-reranker ordering."""

    chunk_id: str
    document_id: str
    first_stage_rank: int
    first_stage_score: float
    reranker_rank: int
    reranker_score: float


@dataclass(frozen=True)
class RerankerReceipt:
    """Auditable learned ordering over a frozen first-stage chunk cohort."""

    query_sha256: str
    model_id: str
    model_revision: str
    candidate_count: int
    latency_ms: float
    candidates: tuple[RerankedCandidate, ...]
    schema_version: str = "paper3.2-candidate-reranker-v1"

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "query_sha256": self.query_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "candidate_count": self.candidate_count,
            "latency_ms": self.latency_ms,
            "candidates": [asdict(row) for row in self.candidates],
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value


@dataclass(frozen=True)
class RerankerResult:
    """Ranked chunks paired with the receipt that explains their ordering."""

    ranked: tuple[RankedChunk, ...]
    receipt: RerankerReceipt


class CandidateReranker(Protocol):
    """Generic second-stage ranker over frozen first-stage chunk candidates."""

    model_id: str
    revision: str

    def rank(
        self, query: str, candidates: Sequence[RankedChunk]
    ) -> RerankerResult: ...


class CrossEncoderCandidateReranker:
    """Adapt the shared cross-encoder selector to the candidate-reranker API."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        batch_size: int = 16,
        score_pairs=None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.selector = CrossEncoderRAGSelector(
            model_id=model_id,
            revision=revision,
            device=device,
            batch_size=batch_size,
            score_pairs=score_pairs,
            name_prefix="pra_reranker",
        )

    def rank(
        self, query: str, candidates: Sequence[RankedChunk]
    ) -> RerankerResult:
        started = time.perf_counter()
        reranked = self.selector.rank(query, tuple(row.chunk for row in candidates))
        latency_ms = (time.perf_counter() - started) * 1000.0
        first_by_id = {row.chunk.chunk_id: row for row in candidates}
        rows = tuple(
            RankedChunk(
                row.chunk,
                row.score,
                rank,
                {
                    **first_by_id[row.chunk.chunk_id].channel_ranks,
                    "bm25": first_by_id[row.chunk.chunk_id].rank,
                    "cross_encoder": rank,
                },
            )
            for rank, row in enumerate(reranked, 1)
        )
        receipt = RerankerReceipt(
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            model_id=self.model_id,
            model_revision=self.revision,
            candidate_count=len(candidates),
            latency_ms=latency_ms,
            candidates=tuple(
                RerankedCandidate(
                    chunk_id=row.chunk.chunk_id,
                    document_id=row.chunk.document_id,
                    first_stage_rank=first_by_id[row.chunk.chunk_id].rank,
                    first_stage_score=first_by_id[row.chunk.chunk_id].score,
                    reranker_rank=row.rank,
                    reranker_score=row.score,
                )
                for row in rows
            ),
        )
        return RerankerResult(rows, receipt)


class BM25RerankSelector:
    """BM25 candidate narrowing followed by an interchangeable learned ranker."""

    def __init__(self, reranker: CandidateReranker, *, candidate_count: int = 50) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.name = (
            f"pra_bm25_top{candidate_count}_rerank:"
            f"{reranker.model_id}@{reranker.revision}"
        )

    def rank_with_receipt(
        self, query: str, chunks: Sequence[RAGChunk]
    ) -> RerankerResult:
        first_stage = StandardRAGSelector().rank(query, chunks)[: self.candidate_count]
        return self.reranker.rank(query, first_stage)

    def rank(self, query: str, chunks: Sequence[RAGChunk]) -> tuple[RankedChunk, ...]:
        return self.rank_with_receipt(query, chunks).ranked


@dataclass(frozen=True)
class ResolvedRecordRequest:
    """Prompt-to-materialization receipt produced by the public PRA ref path."""

    request_id: str
    logical_prompt: str
    visible_model_prompt: str
    reference_table: Mapping[str, str]
    requested_record_uris: tuple[str, ...]
    resolved_document_uris: tuple[str, ...]
    materialized_chunk_uris: tuple[str, ...]
    materialized_chunk_ids: tuple[str, ...]
    materialized_texts: tuple[str, ...]
    selection_receipt_id: str
    root_uri: str | None = None
    resolution_mode: str = "explicit"
    schema_version: str = "paper3.2-native-record-request-v1"

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "logical_prompt": self.logical_prompt,
            "visible_model_prompt": self.visible_model_prompt,
            "reference_table": dict(self.reference_table),
            "requested_record_uris": list(self.requested_record_uris),
            "resolved_document_uris": list(self.resolved_document_uris),
            "materialized_chunk_uris": list(self.materialized_chunk_uris),
            "materialized_chunk_ids": list(self.materialized_chunk_ids),
            "selection_receipt_id": self.selection_receipt_id,
            "root_uri": self.root_uri,
            "resolution_mode": self.resolution_mode,
            "materialized_text_sha256": [
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in self.materialized_texts
            ],
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value


class PRADocumentRecordStore:
    """Persistent typed records exposed through lightweight prompt references."""

    def __init__(
        self,
        dataset: str,
        documents: Sequence[RAGDocument],
        *,
        chunker,
        token_count=whitespace_token_count,
    ) -> None:
        if not dataset or not documents:
            raise ValueError("record store requires a dataset identity and documents")
        self.dataset = dataset
        self.documents = {row.document_id: row for row in documents}
        if len(self.documents) != len(documents):
            raise ValueError("document IDs must be unique")
        self.document_uris: dict[str, str] = {}
        self.document_ids_by_uri: dict[str, str] = {}
        self.chunks_by_id: dict[str, RAGChunk] = {}
        self.chunk_uris: dict[str, str] = {}
        self.records: dict[str, ContextRecord] = {}
        resolver_documents: dict[str, object] = {}
        resolver_metadata: dict[str, dict] = {}
        resolver_versions: dict[str, str] = {}
        receipt_rows: list[Mapping[str, object]] = []

        for document in documents:
            uri = document_record_uri(dataset, document.document_id)
            chunks = chunk_document(document, chunker, token_count=token_count)
            child_uris = tuple(chunk_record_uri(uri, chunk.ordinal) for chunk in chunks)
            record = ContextRecord(
                record_id=uri,
                record_type=RecordType.GENERIC_DOCUMENT,
                payload={
                    "document_id": document.document_id,
                    "title": document.title,
                    "text": document.text,
                    "source": document.source,
                    "mime": document.mime,
                    "token_count": sum(chunk.token_count for chunk in chunks),
                },
                child_ids=child_uris,
                version=document.version,
                source_fingerprint=document.fingerprint,
            )
            self.records[uri] = record
            self.document_uris[document.document_id] = uri
            self.document_ids_by_uri[uri] = document.document_id
            resolver_documents[uri] = document.text
            resolver_metadata[uri] = {
                "record_type": record.record_type.value,
                "document_id": document.document_id,
                "source_sha256": document.fingerprint,
                "child_uris": list(child_uris),
            }
            resolver_versions[uri] = document.version
            for chunk, child_uri in zip(chunks, child_uris):
                child = rag_chunk_record(
                    child_uri,
                    document_uri=uri,
                    chunk_id=chunk.chunk_id,
                    source_offsets=(chunk.start, chunk.end),
                    retrieval_score=0.0,
                    text=chunk.text,
                    metadata={"ordinal": chunk.ordinal, "section_id": chunk.section_id},
                )
                self.records[child_uri] = child
                self.chunks_by_id[chunk.chunk_id] = chunk
                self.chunk_uris[chunk.chunk_id] = child_uri
                resolver_documents[child_uri] = chunk.text
                resolver_metadata[child_uri] = {
                    "record_type": child.record_type.value,
                    "document_id": document.document_id,
                    "chunk_id": chunk.chunk_id,
                    "parent_uri": uri,
                    "source_offsets": [chunk.start, chunk.end],
                    "source_sha256": child.source_fingerprint,
                }
                resolver_versions[child_uri] = document.version
            receipt_rows.append(
                {
                    "record_uri": uri,
                    "record_type": record.record_type.value,
                    "document_id": document.document_id,
                    "source_sha256": document.fingerprint,
                    "token_count": sum(chunk.token_count for chunk in chunks),
                    "chunk_uris": list(child_uris),
                    "native_cache_identity": _digest(
                        {
                            "record_uri": uri,
                            "version": document.version,
                            "source_sha256": document.fingerprint,
                        }
                    ),
                }
            )

        self.resolver = InMemoryResolver(
            resolver_documents,
            metadata=resolver_metadata,
            versions=resolver_versions,
        )
        self.ingestion_receipt = RecordIngestionReceipt(dataset, tuple(receipt_rows))

    def _table_for_uris(self, uris: Sequence[str]) -> ReferenceTable:
        table = ReferenceTable()
        for uri in uris:
            if uri not in self.records and uri not in self.resolver.documents:
                raise KeyError(f"PRA record is unavailable: {uri}")
            table.register(uri, metadata={"record_uri": uri})
        return table

    def register_root(
        self, request_id: str, candidate_document_ids: Sequence[str]
    ) -> str:
        """Register a request root whose children remain independently addressable."""

        uri = root_record_uri(self.dataset, request_id)
        child_uris = tuple(self.document_uris[value] for value in candidate_document_ids)
        local_table = {f"<REF_{index}>": child for index, child in enumerate(child_uris, 1)}
        self.resolver.documents[uri] = {
            "text": "",
            "reference_table": local_table,
            "metadata": {
                "record_type": RecordType.RAG_CHUNK_SET.value,
                "request_id": request_id,
                "child_uris": list(child_uris),
            },
            "version": "v1",
        }
        return uri

    def build_explicit_prompt(
        self, *, request_id: str, document_ids: Sequence[str], model_prompt: str
    ) -> tuple[str, ReferenceTable]:
        """Build a lightweight logical prompt without serializing record payloads."""

        uris = tuple(self.document_uris[value] for value in document_ids)
        table = self._table_for_uris(uris)
        handles = table.all()
        logical = "Use the following source records:\n" + "\n".join(
            handle.token for handle in handles
        ) + "\n\n" + model_prompt
        return logical, table

    def build_root_prompt(
        self,
        *,
        request_id: str,
        candidate_document_ids: Sequence[str],
        model_prompt: str,
    ) -> tuple[str, ReferenceTable, str]:
        """Build one root reference whose children are selected during resolution."""

        uri = self.register_root(request_id, candidate_document_ids)
        table = self._table_for_uris((uri,))
        logical = f"Use the routed source set:\n{table.all()[0].token}\n\n{model_prompt}"
        return logical, table, uri

    def resolve_request(
        self,
        *,
        request_id: str,
        logical_prompt: str,
        model_prompt: str,
        reference_table: ReferenceTable,
        selection: SelectionReceipt,
        root_uri: str | None = None,
    ) -> ResolvedRecordRequest:
        """Resolve prompt handles and bind selected intervals to native chunk records."""

        occurrences = parse_ref_tokens(logical_prompt, reference_table)
        if not occurrences or any(row.handle is None for row in occurrences):
            raise ValueError("logical prompt contains an unresolved PRA reference token")
        requested = tuple(row.handle.uri for row in occurrences if row.handle is not None)
        resolved_documents: list[str] = []
        for uri in requested:
            resolved = self.resolver.resolve(uri)
            if resolved.reference_table:
                resolved_documents.extend(resolved.reference_table.values())
            else:
                resolved_documents.append(uri)
        if root_uri is not None and root_uri not in requested:
            raise ValueError("declared root URI is absent from the logical prompt")

        selected_document_uris = tuple(
            dict.fromkeys(self.document_uris[row.document_id] for row in selection.intervals)
        )
        available = set(resolved_documents)
        if not set(selected_document_uris).issubset(available):
            raise ValueError("selection contains a document outside the resolved record set")
        materialized_chunk_uris = tuple(
            self.chunk_uris[row.chunk_id] for row in selection.intervals
        )
        materialized_texts = tuple(
            self.resolver.resolve(uri).text for uri in materialized_chunk_uris
        )
        return ResolvedRecordRequest(
            request_id=request_id,
            logical_prompt=logical_prompt,
            visible_model_prompt=model_prompt,
            reference_table={handle.token: handle.uri for handle in reference_table.all()},
            requested_record_uris=requested,
            resolved_document_uris=selected_document_uris,
            materialized_chunk_uris=materialized_chunk_uris,
            materialized_chunk_ids=tuple(row.chunk_id for row in selection.intervals),
            materialized_texts=materialized_texts,
            selection_receipt_id=selection.receipt_id,
            root_uri=root_uri,
            resolution_mode="routed_root" if root_uri else "explicit",
        )


def validate_frozen_record_selection(
    request: ResolvedRecordRequest, selection: SelectionReceipt
) -> None:
    """Fail if record resolution changed the selector's ordered chunk identity."""

    if request.selection_receipt_id != selection.receipt_id:
        raise ValueError("record request does not carry the frozen selection receipt")
    if request.materialized_chunk_ids != tuple(row.chunk_id for row in selection.intervals):
        raise ValueError("record materialization changed selected chunk order or identity")

