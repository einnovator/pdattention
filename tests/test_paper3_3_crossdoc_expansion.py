from __future__ import annotations

from experiments.paper3_3_crossdoc_expansion.run_expansion_frontier import (
    _controlled_crossdoc_fixture,
    _fixture_initial_context,
    evaluate_question,
)
from pra_hf.crossdoc_expansion import (
    CrossDocumentDirection,
    CrossDocumentExpansionMode,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    FirstStageBM25,
    SelectionReceipt,
    context_metrics,
    make_candidate_receipt,
    prepare_candidate_context,
)


def test_controlled_frontier_recovers_an_omitted_peer_span() -> None:
    documents, questions, metadata = _controlled_crossdoc_fixture()
    question = questions[0]
    by_id = {row.document_id: row for row in documents}
    retriever = FirstStageBM25(documents)
    candidate = make_candidate_receipt(
        dataset="fixture",
        dataset_revision=metadata["dataset_revision"],
        corpus_revision=metadata["corpus_revision"],
        corpus_sha256=metadata["corpus_sha256"],
        question=question,
        retriever=retriever,
        candidate_count=len(documents),
        chunker=ChunkerConfig(32, 4),
        seed=11,
    )
    prepared = prepare_candidate_context(candidate, by_id)
    context = _fixture_initial_context(question, prepared, token_budget=256)
    selection = SelectionReceipt.from_context(
        candidate_receipt_id=candidate.receipt_id,
        example_id=question.example_id,
        context=context,
        selector_revision=context.selector_name,
    )

    initial = context_metrics(question, candidate, context)
    rows, receipts = evaluate_question(
        dataset="fixture",
        question=question,
        candidate=candidate,
        prepared=prepared,
        context=context,
        selection=selection,
        modes=(CrossDocumentExpansionMode.LEXICAL,),
        directions=(CrossDocumentDirection.SYMMETRIC,),
        query_conditioning=(False,),
        top_ks=(1,),
        budgets=(16,),
        all_candidate_kv_resident=False,
    )

    assert initial["supporting_span_coverage"] == 0.5
    assert initial["gold_chunk_recall"] == 0.5
    assert rows[0]["supporting_span_coverage"] == 1.0
    assert rows[0]["gold_chunk_recall"] == 1.0
    assert rows[0]["answer_string_availability"] == 1.0
    assert rows[0]["distractor_fraction"] == 0.0
    assert receipts[0]["chosen_spans"]

