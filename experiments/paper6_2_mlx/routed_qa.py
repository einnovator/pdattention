"""Pure document routing helpers for the Paper 6.2 natural-QA cohort."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pra_hf.agent_resources import AgentResource, DiscoveryMode, DiscoveryRequest, PersistentResourceIndex

from experiments.paper6_2_mlx.run_answer_quality_pressure import QAExample


@dataclass(frozen=True)
class RoutedQAResult:
    """Ranked document selection and retrieval diagnostics for one question."""

    selected_document_ids: tuple[str, ...]
    ranked_document_ids: tuple[str, ...]
    selected_source: str
    evidence_recall_at_1: float
    evidence_recall_at_2: float
    evidence_recall_at_4: float
    selected_evidence_recall: float
    index_build_ms: float
    routing_ms: float
    index_bytes: int
    candidate_count: int


def _recall(ranked_ids: tuple[str, ...], evidence_ids: frozenset[str], k: int) -> float:
    if not evidence_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & evidence_ids) / len(evidence_ids)


def route_qa_documents(example: QAExample, *, top_k: int = 4) -> RoutedQAResult:
    """Rank candidate documents with the SDK hybrid index and return selected text."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not example.documents:
        raise ValueError(f"{example.example_id} has no routable documents")

    uri_to_document = {}
    resources = []
    for document in example.documents:
        uri = f"!!ref:document:{example.example_id}/{document.document_id}!!"
        uri_to_document[uri] = document
        resources.append(
            AgentResource(
                uri=uri,
                kind="document",
                namespace=example.dataset,
                name=document.title,
                version="1",
                description=document.text,
                content=document.text,
                metadata={"document_id": document.document_id},
            )
        )

    started = time.perf_counter()
    index = PersistentResourceIndex(resources)
    index_build_ms = (time.perf_counter() - started) * 1000.0
    request = DiscoveryRequest(
        query=example.question,
        namespace=example.dataset,
        top_k=min(top_k, len(resources)),
    )
    started = time.perf_counter()
    scores = index.score(request, channels=(DiscoveryMode.HYBRID,))
    ranked = tuple(sorted(scores, key=lambda row: (-row.hybrid, row.uri)))
    routing_ms = (time.perf_counter() - started) * 1000.0
    ranked_ids = tuple(
        str(uri_to_document[row.uri].document_id) for row in ranked
    )
    selected_ids = ranked_ids[:top_k]
    selected_source = "\n\n".join(
        f"Document: {uri_to_document[row.uri].title}\n{uri_to_document[row.uri].text}"
        for row in ranked[:top_k]
    )
    return RoutedQAResult(
        selected_document_ids=selected_ids,
        ranked_document_ids=ranked_ids,
        selected_source=selected_source,
        evidence_recall_at_1=_recall(ranked_ids, example.evidence_document_ids, 1),
        evidence_recall_at_2=_recall(ranked_ids, example.evidence_document_ids, 2),
        evidence_recall_at_4=_recall(ranked_ids, example.evidence_document_ids, 4),
        selected_evidence_recall=_recall(
            ranked_ids, example.evidence_document_ids, top_k
        ),
        index_build_ms=index_build_ms,
        routing_ms=routing_ms,
        index_bytes=index.estimated_bytes,
        candidate_count=len(resources),
    )
