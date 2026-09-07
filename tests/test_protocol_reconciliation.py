from types import SimpleNamespace

from experiments.paper3_2_rag.run_protocol_reconciliation import _context
from pra_hf.rag_evaluation import RAGChunk, RankedChunk


def test_context_returns_budgeted_frozen_selection() -> None:
    chunks = (
        RAGChunk("d1:0", "d1", 0, 0, 5, "alpha", 3),
        RAGChunk("d2:0", "d2", 0, 0, 4, "beta", 4),
    )
    ranking = tuple(
        RankedChunk(chunk, score=1.0 / rank, rank=rank)
        for rank, chunk in enumerate(chunks, 1)
    )
    prepared = SimpleNamespace(
        chunks=chunks,
        candidate_tokens=7,
        build_latency_ms=1.25,
    )

    context = _context(
        ranked=ranking,
        prepared=prepared,
        selector_name="frozen-selector",
        selector_latency_ms=2.5,
        token_budget=7,
        max_resources=2,
        deduplicate_documents=True,
    )

    assert context.selected_chunk_ids == ("d1:0", "d2:0")
    assert context.packed_tokens == 7
    assert context.candidate_chunks == chunks


def test_context_can_reproduce_legacy_chunk_level_selection() -> None:
    chunks = (
        RAGChunk("d1:0", "d1", 0, 0, 5, "alpha", 3),
        RAGChunk("d1:1", "d1", 1, 6, 10, "beta", 3),
    )
    ranking = tuple(
        RankedChunk(chunk, score=1.0 / rank, rank=rank)
        for rank, chunk in enumerate(chunks, 1)
    )
    prepared = SimpleNamespace(
        chunks=chunks,
        candidate_tokens=6,
        build_latency_ms=1.25,
    )

    context = _context(
        ranked=ranking,
        prepared=prepared,
        selector_name="legacy-selector",
        selector_latency_ms=2.5,
        token_budget=6,
        max_resources=2,
        deduplicate_documents=False,
    )

    assert context.selected_chunk_ids == ("d1:0", "d1:1")
