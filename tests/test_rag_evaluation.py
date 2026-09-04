from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_evaluation import (
    CandidateReceipt,
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    PRAHybridSelector,
    RAGFailureClass,
    RAGDocument,
    RAGQuestion,
    StandardRAGSelector,
    SelectionReceipt,
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
        condition=ContextCondition.NO_PRA_STANDARD_RAG,
        selector=StandardRAGSelector(),
        query=question.question,
        receipt=receipt,
        documents=documents,
        token_budget=8,
    )
    pra = select_context(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
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
        condition=ContextCondition.NO_PRA_STANDARD_RAG,
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
    ) in {
        RAGFailureClass.GENERATION_FAILURE.value,
        RAGFailureClass.STANDARD_RAG_PACKING_MISS.value,
    }


def test_selection_receipt_is_condition_independent_and_tamper_evident() -> None:
    documents, question, receipt = _fixture()
    selected = select_context(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        selector=PRAHybridSelector(),
        query=question.question,
        receipt=receipt,
        documents=documents,
        token_budget=8,
    )
    native = replace(selected, condition=ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR)
    frozen = SelectionReceipt.from_context(
        candidate_receipt_id=receipt.receipt_id,
        example_id=question.example_id,
        context=selected,
    )
    frozen.validate_context(native)
    assert SelectionReceipt.from_dict(frozen.to_dict()).receipt_id == frozen.receipt_id

    changed = replace(native, chunks=native.chunks[:-1])
    with pytest.raises(ValueError, match="differs"):
        frozen.validate_context(changed)


def test_strong_rag_cross_encoder_ranks_and_packs_same_candidates() -> None:
    documents, question, receipt = _fixture()

    def score_pairs(pairs):
        return [float("2024" in text) for _, text in pairs]

    context = select_context(
        condition=ContextCondition.NO_PRA_STANDARD_RAG,
        selector=CrossEncoderRAGSelector(
            model_id="fixture/reranker",
            revision="abc123",
            score_pairs=score_pairs,
        ),
        query=question.question,
        receipt=receipt,
        documents=documents,
        token_budget=8,
    )
    assert context.chunks[0].chunk.document_id == "date"
    assert set(context.selected_document_ids).issubset(receipt.candidate_document_ids)


def test_powered_rag_condition_ids_are_exact() -> None:
    assert ContextCondition.NO_PRA_STANDARD_RAG.value == "NO_PRA_STANDARD_RAG"
    assert ContextCondition.NO_PRA is ContextCondition.NO_PRA_STANDARD_RAG
    powered = {
        ContextCondition.NO_PRA_STANDARD_RAG,
        ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
        ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE,
        ContextCondition.PRA_NATIVE_MEMORY_BUNDLE,
    }
    assert {condition.value for condition in powered} == {
        "NO_PRA_STANDARD_RAG",
        "PRA_SELECTED_CONTEXT_NO_ADAPTOR",
        "PRA_NATIVE_MEMORY_NO_ADAPTOR",
        "PRA_SELECTED_CONTEXT_BUNDLE",
        "PRA_NATIVE_MEMORY_BUNDLE",
    }


def test_bundle_selection_receipt_requires_exact_id_and_revision() -> None:
    documents, question, candidate = _fixture()
    context = select_context(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE,
        selector=PRAHybridSelector(),
        query=question.question,
        receipt=candidate,
        documents=documents,
        token_budget=8,
        bundle_id="EInnovator/pra-fixture",
        bundle_revision="a" * 40,
    )
    receipt = SelectionReceipt.from_context(
        candidate_receipt_id=candidate.receipt_id,
        example_id=question.example_id,
        context=context,
    )
    assert receipt.bundle_revision == "a" * 40

    with pytest.raises(ValueError, match="supplied together"):
        SelectionReceipt.from_context(
            candidate_receipt_id=candidate.receipt_id,
            example_id=question.example_id,
            context=replace(context, bundle_revision=None),
        )


def test_document_record_preserves_hierarchy_and_receipt_provenance() -> None:
    documents, _, receipt = _fixture()
    document = documents["bridge"]
    chunks = chunk_document(document, receipt.chunker)
    record = document_to_context_record(document, chunks, receipt_id=receipt.receipt_id)
    assert record.record_type.value == "generic_document"
    assert record.payload["document_id"] == document.document_id
    assert record.payload["chunks"][0]["start"] == 0
    assert record.selection_provenance["candidate_receipt_id"] == receipt.receipt_id
