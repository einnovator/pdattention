from __future__ import annotations

import pytest

from pra_hf.rag_evaluation import (
    CandidateDocument,
    ChunkerConfig,
    FirstStageBM25,
    RAGDocument,
    RAGQuestion,
)
from pra_hf.rag_retrieval import (
    BackendStatus,
    CrossEncoderRerankedRetriever,
    ElasticsearchBM25Retriever,
    ExactDenseRetriever,
    FaissDenseRetriever,
    HybridRetriever,
    QdrantDenseRetriever,
    make_backend_candidate_receipt,
    reciprocal_rank_fusion,
    service_backend,
    unresolved_derby_backend,
)


def _documents() -> tuple[RAGDocument, ...]:
    return (
        RAGDocument("alpha", "Alpha", "Lisbon cobalt engine"),
        RAGDocument("beta", "Beta", "Oslo copper bridge"),
        RAGDocument("gamma", "Gamma", "Lisbon tram route"),
    )


def _embed(text: str):
    lowered = text.casefold()
    return (
        float("lisbon" in lowered),
        float("engine" in lowered),
        float("oslo" in lowered),
    )


def test_exact_dense_retrieval_is_stable_and_attributed() -> None:
    retriever = ExactDenseRetriever(
        _documents(), dimensions=3, embedder=_embed, embedder_revision="fixture-v1"
    )
    first = retriever.retrieve("Lisbon engine", 2)
    second = retriever.retrieve("Lisbon engine", 2)
    assert first == second
    assert first[0].document_id == "alpha"
    assert retriever.revision == "exact_cosine_v1:fixture-v1:d3"
    assert len(retriever.index_sha256) == 64


def test_rrf_fuses_rankings_without_using_incompatible_scores() -> None:
    lexical = (
        CandidateDocument("alpha", 1, 1000.0),
        CandidateDocument("beta", 2, 10.0),
    )
    dense = (
        CandidateDocument("beta", 1, 0.51),
        CandidateDocument("alpha", 2, 0.50),
    )
    fused = reciprocal_rank_fusion({"lexical": lexical, "dense": dense}, top_k=2)
    assert tuple(row.document_id for row in fused) == ("alpha", "beta")
    assert tuple(row.rank for row in fused) == (1, 2)

    duplicate = lexical + (CandidateDocument("alpha", 3, 0.0),)
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion({"bad": duplicate}, top_k=2)


def test_hybrid_backend_freezes_one_attributed_candidate_receipt() -> None:
    documents = _documents()
    hybrid = HybridRetriever(
        {
            "bm25": FirstStageBM25(documents),
            "dense": ExactDenseRetriever(
                documents,
                dimensions=3,
                embedder=_embed,
                embedder_revision="fixture-v1",
            ),
        },
        documents,
        constant=10,
    )
    question = RAGQuestion(
        "q1",
        "Which Lisbon document discusses an engine?",
        ("alpha",),
        frozenset({"alpha"}),
    )
    receipt = make_backend_candidate_receipt(
        dataset="fixture",
        dataset_revision="v1",
        corpus_revision="v1",
        corpus_sha256="a" * 64,
        question=question,
        retriever=hybrid,
        retriever_id="local_rrf",
        documents={row.document_id: row for row in documents},
        candidate_count=2,
        chunker=ChunkerConfig(max_tokens=8, overlap_tokens=1),
        seed=11,
    )
    assert receipt.retriever == "local_rrf"
    assert receipt.retriever_revision == hybrid.revision
    assert receipt.candidates[0].document_id == "alpha"
    assert receipt.receipt_id == receipt.from_dict(receipt.to_dict()).receipt_id


def test_cross_encoder_reranker_preserves_first_stage_identity() -> None:
    documents = _documents()
    dense = ExactDenseRetriever(
        documents, dimensions=3, embedder=_embed, embedder_revision="fixture-v1"
    )

    def score_pairs(pairs):
        return [10.0 if "tram" in document else 0.0 for _, document in pairs]

    reranker = CrossEncoderRerankedRetriever(
        dense,
        documents,
        model_id="fixture/reranker",
        revision="abc123",
        candidate_count=3,
        score_pairs=score_pairs,
    )
    rows = reranker.retrieve("Lisbon", 2)
    assert rows[0].document_id == "gamma"
    assert tuple(row.rank for row in rows) == (1, 2)
    assert "fixture/reranker@abc123" in reranker.revision


def test_elasticsearch_adapter_maps_service_hits_to_corpus_ids() -> None:
    calls = []

    def transport(method, url, payload):
        calls.append((method, url, payload))
        return {
            "hits": {
                "hits": [
                    {"_id": "service-id", "_score": 3.5, "_source": {"document_id": "beta"}}
                ]
            }
        }

    retriever = ElasticsearchBM25Retriever(
        _documents(),
        endpoint="http://search:9200",
        index_name="paper32",
        index_revision="snapshot-a",
        transport=transport,
    )
    assert retriever.retrieve("Oslo", 1)[0].document_id == "beta"
    assert calls[0][0] == "POST"
    assert calls[0][2]["size"] == 1


def test_qdrant_adapter_maps_payload_identity_and_scores() -> None:
    def transport(method, url, payload):
        return {
            "result": {
                "points": [
                    {"id": "uuid", "score": 0.91, "payload": {"document_id": "alpha"}}
                ]
            }
        }

    retriever = QdrantDenseRetriever(
        _documents(),
        endpoint="http://vector:6333",
        collection_name="paper32",
        collection_revision="snapshot-b",
        embedder=_embed,
        dimensions=3,
        transport=transport,
    )
    result = retriever.retrieve("Lisbon engine", 1)
    assert result[0].document_id == "alpha"
    assert result[0].score == pytest.approx(0.91)


def test_service_identity_is_explicit_and_derby_remains_unresolved() -> None:
    qdrant = service_backend(
        kind="QDRANT",
        endpoint="http://localhost:6333",
        index_name="paper32",
        revision="snapshot-1",
    )
    assert qdrant.backend_id == "qdrant:paper32:snapshot-1"
    assert qdrant.status is BackendStatus.AVAILABLE

    derby = unresolved_derby_backend()
    assert derby.status is BackendStatus.BACKEND_IDENTITY_UNRESOLVED
    assert derby.backend_id == "derby::unresolved"

    with pytest.raises(ValueError, match="unsupported"):
        service_backend(kind="mystery", endpoint="x", index_name="x", revision="x")


def test_faiss_reports_availability_without_hiding_missing_dependency() -> None:
    assert FaissDenseRetriever.status() in {
        BackendStatus.AVAILABLE,
        BackendStatus.DEPENDENCY_MISSING,
    }
