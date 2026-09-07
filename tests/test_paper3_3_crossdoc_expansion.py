from __future__ import annotations

from experiments.paper3_3_crossdoc_expansion.run_expansion_frontier import (
    _controlled_crossdoc_fixture,
    _fixture_initial_context,
    cached_record_level_context,
    evaluate_question,
    load_selection_cache,
)
from experiments.paper3_3_crossdoc_expansion.summarize_expansion import (
    build_publication_summary,
    select_validation_config,
)
from pra_hf.crossdoc_expansion import (
    CrossDocumentDirection,
    CrossDocumentExpansionMode,
    CrossDocumentGranularity,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    FirstStageBM25,
    RankedChunk,
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
        granularity=CrossDocumentGranularity.CHUNK,
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


def test_selection_cache_replays_without_reranking(tmp_path) -> None:
    documents, questions, metadata = _controlled_crossdoc_fixture()
    question = questions[0]
    candidate = make_candidate_receipt(
        dataset="fixture",
        dataset_revision=metadata["dataset_revision"],
        corpus_revision=metadata["corpus_revision"],
        corpus_sha256=metadata["corpus_sha256"],
        question=question,
        retriever=FirstStageBM25(documents),
        candidate_count=len(documents),
        chunker=ChunkerConfig(32, 4),
        seed=11,
    )
    prepared = prepare_candidate_context(candidate, {row.document_id: row for row in documents})

    class CountingSelector:
        name = "frozen-selector@test"

        def __init__(self) -> None:
            self.calls = 0

        def rank(self, query, chunks):
            del query
            self.calls += 1
            return tuple(
                RankedChunk(chunk, 1.0 / rank, rank, {"fixture": rank})
                for rank, chunk in enumerate(chunks, 1)
            )

    cache_path = tmp_path / "selection.jsonl"
    selector = CountingSelector()
    first = cached_record_level_context(
        {},
        cache_path=cache_path,
        example_id=question.example_id,
        candidate_receipt_id=candidate.receipt_id,
        prepared=prepared,
        selector=selector,
        query=question.question,
        token_budget=256,
        max_resources=2,
    )
    second = cached_record_level_context(
        load_selection_cache(cache_path),
        cache_path=cache_path,
        example_id=question.example_id,
        candidate_receipt_id=candidate.receipt_id,
        prepared=prepared,
        selector=selector,
        query=question.question,
        token_budget=256,
        max_resources=2,
    )

    assert selector.calls == 1
    assert first.selected_chunk_ids == second.selected_chunk_ids
    assert first.selector_name == second.selector_name


def test_publication_summary_freezes_best_bounded_validation_policy() -> None:
    def aggregate(mode, gain, distractors, fraction, budget):
        return {
            "mode": mode,
            "query_conditioned": True,
            "direction": "symmetric",
            "granularity": "chunk",
            "top_k_per_pair": 1,
            "max_extra_tokens": budget,
            "supporting_span_coverage_delta": gain,
            "gold_chunk_recall_delta": gain,
            "distractor_fraction_mean": distractors,
            "extra_native_fraction_mean": fraction,
            "requested_cross_tokens_mean": 20.0,
            "deduplicated_cross_tokens_mean": 10.0,
        }

    summary = (
        aggregate("lexical", 0.10, 0.2, 0.10, 64),
        aggregate("dense", 0.10, 0.4, 0.08, 32),
        aggregate("entity", 0.20, 0.0, 0.30, 128),
        aggregate("oracle", 0.40, 0.0, 0.20, 128),
    )
    best, oracle = select_validation_config(summary)
    assert best["mode"] == "lexical"
    assert oracle["mode"] == "oracle"

    rows = []
    for example_id, initial, gain in (("a", 0.0, 1.0), ("b", 1.0, 0.0)):
        rows.append(
            {
                **best,
                "example_id": example_id,
                "supporting_span_coverage": initial + gain,
                "initial_supporting_span_coverage": initial,
                "gold_chunk_recall": initial + gain,
                "initial_gold_chunk_recall": initial,
                "answer_string_availability": initial + gain,
                "initial_answer_string_availability": initial,
            }
        )
    run = {
        "summary": summary,
        "git_commit": "abc",
        "dataset": "fixture",
        "split_name": "validation",
        "split_digest": "digest",
        "selector": "selector",
        "reranker_revision": "revision",
        "question_count_evaluated": 2,
    }
    result = build_publication_summary(run, rows)
    assert result["selected_policy"]["mode"] == "lexical"
    assert result["oracle_gap_recovery"] == 0.25
    assert result["requested_to_deduplicated_token_ratio"] == 2.0
    assert result["qualification"]["sdk_exposable"] is False
