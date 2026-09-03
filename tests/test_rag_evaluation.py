from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_evaluation import (
    CandidateReceipt,
    ChunkerConfig,
    ContextCondition,
    FirstStageBM25,
    PRAHybridSelector,
    RAGDocument,
    RAGQuestion,
    StandardRAGSelector,
    chunk_document,
    context_metrics,
    document_to_context_record,
    failure_classification,
    make_candidate_receipt,
    select_context,
)


def _fixture():
    documents = (
        RAGDocument("bridge", "Bridge", "Ada designed the cobalt engine in Lisbon."),
        RAGDocument("date", "Date", "The cobalt engine launched in 2024."),
        RAGDocument("noise", "Noise", "Lisbon has trams and many sunny streets."),
    )
    question = RAGQuestion(
        "q1",
        "When did Ada's cobalt engine launch?",
        ("2024",),
        frozenset({"bridge", "date"}),
    )
    retriever = FirstStageBM25(documents)
    receipt = make_candidate_receipt(
        dataset="fixture",
        dataset_revision="v1",
        corpus_revision="v1",
        corpus_sha256="a" * 64,
        question=question,
        retriever=retriever,
        candidate_count=3,
        chunker=ChunkerConfig(max_tokens=6, overlap_tokens=1),
        ensure_gold=True,
        seed=11,
    )
    return {document.document_id: document for document in documents}, question, receipt


def test_candidate_receipt_roundtrip_and_document_integrity() -> None:
    documents, _, receipt = _fixture()
    restored = CandidateReceipt.from_dict(receipt.to_dict())
    assert restored.receipt_id == receipt.receipt_id
    assert restored.candidate_document_ids == receipt.candidate_document_ids
    restored.validate_documents(documents)

    changed = {**documents, "date": replace(documents["date"], text="Changed source")}
    with pytest.raises(ValueError, match="changed after retrieval"):
        restored.validate_documents(changed)


def test_receipt_rejects_tampering() -> None:
    _, _, receipt = _fixture()
    value = receipt.to_dict()
    value["seed"] = 99
    with pytest.raises(ValueError, match="digest"):
        CandidateReceipt.from_dict(value)


def test_chunking_preserves_exact_offsets_and_budget() -> None:
    document = RAGDocument("d", "D", "one two three four five six seven")
    chunks = chunk_document(document, ChunkerConfig(max_tokens=3, overlap_tokens=1))
    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six seven",
    ]
    assert all(document.text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


def test_conditions_share_candidates_but_can_select_different_chunks() -> None:
    documents, question, receipt = _fixture()
    baseline = select_context(
        condition=ContextCondition.NO_PRA,
        selector=StandardRAGSelector(),
        query=question.question,
        receipt=receipt,
        documents=documents,
        token_budget=8,
    )
    pra = select_context(
        condition=ContextCondition.PRA_NO_ADAPTOR,
        selector=PRAHybridSelector(),
        query=question.question,
        receipt=receipt,
        documents=documents,
        token_budget=8,
    )
    assert baseline.packed_tokens <= 8
    assert pra.packed_tokens <= 8
    assert receipt.candidate_document_ids
    assert set(baseline.selected_document_ids).issubset(receipt.candidate_document_ids)
    assert set(pra.selected_document_ids).issubset(receipt.candidate_document_ids)


def test_metrics_and_failure_stages_stay_separate() -> None:
    documents, question, receipt = _fixture()
    baseline = select_context(
        condition=ContextCondition.NO_PRA,
        selector=StandardRAGSelector(),
        query=question.question,
        receipt=receipt,
        documents=documents,
        token_budget=8,
    )
    metrics = context_metrics(question, receipt, baseline)
    assert metrics["document_recall_at_candidate_k"] == 1.0
    assert 0.0 <= metrics["materialization_avoidance"] <= 1.0
    assert failure_classification(
        question=question,
        receipt=receipt,
        context=baseline,
        answer_correct=False,
    ) in {"generation_failure", "standard_rag_packing_failure"}


def test_document_record_preserves_hierarchy_and_receipt_provenance() -> None:
    documents, _, receipt = _fixture()
    document = documents["bridge"]
    chunks = chunk_document(document, receipt.chunker)
    record = document_to_context_record(document, chunks, receipt_id=receipt.receipt_id)
    assert record.record_type.value == "generic_document"
    assert record.payload["document_id"] == document.document_id
    assert record.payload["chunks"][0]["start"] == 0
    assert record.selection_provenance["candidate_receipt_id"] == receipt.receipt_id
