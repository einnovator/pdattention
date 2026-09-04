from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    FirstStageBM25,
    RAGDocument,
    RAGQuestion,
    SelectionReceipt,
    StandardRAGSelector,
    make_candidate_receipt,
    prepare_candidate_context,
    select_context,
)
from pra_hf.rag_record_runtime import (
    BM25RerankSelector,
    CrossEncoderCandidateReranker,
    PRADocumentRecordStore,
    document_record_uri,
    validate_frozen_record_selection,
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
    chunker = ChunkerConfig(max_tokens=6, overlap_tokens=1)
    receipt = make_candidate_receipt(
        dataset="fixture",
        dataset_revision="v1",
        corpus_revision="v1",
        corpus_sha256="a" * 64,
        question=question,
        retriever=retriever,
        candidate_count=3,
        chunker=chunker,
        ensure_gold=True,
        seed=11,
    )
    by_id = {row.document_id: row for row in documents}
    context = select_context(
        condition=ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
        selector=StandardRAGSelector(),
        query=question.question,
        receipt=receipt,
        documents=by_id,
        token_budget=20,
    )
    selection = SelectionReceipt.from_context(
        candidate_receipt_id=receipt.receipt_id,
        example_id=question.example_id,
        context=context,
        selector_revision="fixture",
    )
    return documents, question, receipt, chunker, context, selection


def test_record_store_ingests_stable_document_and_chunk_identities() -> None:
    documents, _, _, chunker, _, _ = _fixture()
    store = PRADocumentRecordStore("fixture", documents, chunker=chunker)

    bridge_uri = document_record_uri("fixture", "bridge")
    assert store.document_uris["bridge"] == bridge_uri
    assert store.records[bridge_uri].record_type.value == "generic_document"
    assert store.records[bridge_uri].child_ids
    assert store.ingestion_receipt.receipt_id
    assert store.resolver.resolve(bridge_uri).text == documents[0].text


def test_explicit_record_prompt_resolves_through_lightweight_reference_table() -> None:
    documents, question, _, chunker, context, selection = _fixture()
    store = PRADocumentRecordStore("fixture", documents, chunker=chunker)
    model_prompt = f"Question: {question.question}\nAnswer:"
    logical, table = store.build_explicit_prompt(
        request_id=question.example_id,
        document_ids=context.selected_document_ids,
        model_prompt=model_prompt,
    )
    request = store.resolve_request(
        request_id=question.example_id,
        logical_prompt=logical,
        model_prompt=model_prompt,
        reference_table=table,
        selection=selection,
    )

    validate_frozen_record_selection(request, selection)
    assert "<REF_1>" in request.logical_prompt
    assert "<REF_" not in request.visible_model_prompt
    assert request.materialized_chunk_ids == context.selected_chunk_ids
    assert request.materialized_texts == tuple(row.chunk.text for row in context.chunks)
    assert request.resolution_mode == "explicit"


def test_routed_root_resolves_only_selector_selected_children() -> None:
    documents, question, receipt, chunker, _, selection = _fixture()
    store = PRADocumentRecordStore("fixture", documents, chunker=chunker)
    model_prompt = f"Question: {question.question}\nAnswer:"
    logical, table, root_uri = store.build_root_prompt(
        request_id=question.example_id,
        candidate_document_ids=receipt.candidate_document_ids,
        model_prompt=model_prompt,
    )
    request = store.resolve_request(
        request_id=question.example_id,
        logical_prompt=logical,
        model_prompt=model_prompt,
        reference_table=table,
        selection=selection,
        root_uri=root_uri,
    )

    assert request.requested_record_uris == (root_uri,)
    assert request.resolution_mode == "routed_root"
    assert set(request.resolved_document_uris).issubset(
        {store.document_uris[value] for value in receipt.candidate_document_ids}
    )
    validate_frozen_record_selection(request, selection)


def test_record_resolution_rejects_selection_outside_explicit_set() -> None:
    documents, question, _, chunker, context, selection = _fixture()
    store = PRADocumentRecordStore("fixture", documents, chunker=chunker)
    only_first = (context.selected_document_ids[0],)
    logical, table = store.build_explicit_prompt(
        request_id=question.example_id,
        document_ids=only_first,
        model_prompt="Question",
    )
    with pytest.raises(ValueError, match="outside the resolved record set"):
        store.resolve_request(
            request_id=question.example_id,
            logical_prompt=logical,
            model_prompt="Question",
            reference_table=table,
            selection=selection,
        )


def test_record_selection_validation_rejects_tampered_receipt() -> None:
    documents, question, _, chunker, context, selection = _fixture()
    store = PRADocumentRecordStore("fixture", documents, chunker=chunker)
    logical, table = store.build_explicit_prompt(
        request_id=question.example_id,
        document_ids=context.selected_document_ids,
        model_prompt="Question",
    )
    request = store.resolve_request(
        request_id=question.example_id,
        logical_prompt=logical,
        model_prompt="Question",
        reference_table=table,
        selection=selection,
    )
    with pytest.raises(ValueError, match="frozen selection"):
        validate_frozen_record_selection(
            replace(request, selection_receipt_id="tampered"), selection
        )


def test_bm25_then_generic_reranker_narrows_and_persists_scores() -> None:
    documents, question, receipt, _, _, _ = _fixture()
    prepared = prepare_candidate_context(
        receipt, {row.document_id: row for row in documents}
    )

    def score_pairs(pairs):
        return [float("2024" in text) for _, text in pairs]

    reranker = CrossEncoderCandidateReranker(
        model_id="fixture-reranker",
        revision="sha-fixture",
        score_pairs=score_pairs,
    )
    selector = BM25RerankSelector(reranker, candidate_count=2)
    result = selector.rank_with_receipt(question.question, prepared.chunks)

    assert len(result.ranked) == 2
    assert result.ranked[0].chunk.document_id == "date"
    assert result.receipt.model_id == "fixture-reranker"
    assert result.receipt.model_revision == "sha-fixture"
    assert result.receipt.candidate_count == 2
    assert result.receipt.candidates[0].first_stage_rank >= 1
    assert result.receipt.receipt_id

