from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper3_1_summary_index.ollama_sidecar import (
    JSONLGenerationCache,
    build_prompt,
    cache_key,
    generate_cached,
    parse_generated_batch,
)
from pra_hf.summary_index import (
    BM25SummaryScorer,
    FrozenEmbeddingScorer,
    SummaryFacet,
    SummaryIndex,
    SummaryIndexRecord,
    exact_summary_scores,
    hybrid_scores,
    retrieval_metrics,
    source_sha256,
)


def _record(index: int, text: str, *, facets=()) -> SummaryIndexRecord:
    source = f"source text {index}"
    return SummaryIndexRecord(
        uri="memory://paper",
        chunk_id=f"chunk-{index}",
        token_start=index * 8,
        token_end=(index + 1) * 8,
        source_sha256=source_sha256(source),
        summary=text,
        facets=tuple(facets),
        summary_token_count=len(text.split()),
        generation_model="test",
        prompt_id="retrieval-v1",
    )


def test_summary_index_preserves_native_identity_when_addresses_are_shuffled() -> None:
    index = SummaryIndex(
        [_record(0, "Ada designed the engine"), _record(1, "Babbage built machinery")]
    )

    shuffled = index.shuffled_addresses(1)

    assert [row.identity for row in shuffled.records] == [row.identity for row in index.records]
    assert [row.summary for row in shuffled.records] != [row.summary for row in index.records]
    assert shuffled.materialization_requests([1])[0].chunk_id == "chunk-1"
    assert not hasattr(shuffled.materialization_requests([1])[0], "summary")


def test_summary_index_rejects_source_alignment_drift() -> None:
    record = _record(0, "one address")
    index = SummaryIndex([record])
    expected = [(record.uri, record.chunk_id, 0, 8, record.source_sha256)]
    index.assert_source_alignment(expected)

    with pytest.raises(ValueError, match="alignment failed"):
        index.assert_source_alignment([(record.uri, record.chunk_id, 0, 7, record.source_sha256)])


def test_bm25_and_exact_pool_over_facets() -> None:
    index = SummaryIndex(
        [
            _record(0, "engine history"),
            _record(
                1,
                "unrelated summary",
                facets=(
                    SummaryFacet("entity", "Ada Lovelace"),
                    SummaryFacet("relation", "designed an analytical engine"),
                ),
            ),
        ]
    )

    bm25 = BM25SummaryScorer(index).score("Who was Ada Lovelace?")
    exact = exact_summary_scores(index, "Who was Ada Lovelace?")

    assert bm25[1] > bm25[0]
    assert exact.tolist() == [0.0, 2.0]


def test_frozen_embeddings_and_hybrid_are_alignment_checked() -> None:
    index = SummaryIndex([_record(0, "alpha"), _record(1, "beta")])
    scorer = FrozenEmbeddingScorer(index, [[[1.0, 0.0]], [[0.0, 1.0]]])

    assert scorer.score([0.9, 0.1]).tolist() == pytest.approx([0.9938837, 0.1104315])
    assert hybrid_scores([0.0, 1.0], [1.0, 0.0], 0.75).tolist() == pytest.approx(
        [0.25, 0.75]
    )
    with pytest.raises(ValueError, match="one-to-one"):
        FrozenEmbeddingScorer(index, [[[1.0, 0.0]]])


def test_retrieval_metrics_use_unique_chunk_identities() -> None:
    metrics = retrieval_metrics([0.1, 0.8, 0.7, 0.2], {1, 2}, k=2)

    assert metrics == {
        "evidence_recall": 1.0,
        "complete_recovery": 1.0,
        "precision": 1.0,
        "reciprocal_rank": 1.0,
        "selected_indices": [1, 2],
        "recovered_indices": [1, 2],
    }


def test_summary_text_bytes_exclude_native_kv_payload() -> None:
    index = SummaryIndex([_record(0, "cafe"), _record(1, "naive")])

    assert index.text_bytes == len("cafenaive".encode("utf-8"))
    assert np.isfinite(BM25SummaryScorer(index).score("cafe")).all()


def test_generation_batch_requires_exact_chunk_ids() -> None:
    prompt = build_prompt(
        [("chunk-0", "Ada wrote notes")],
        prompt_id="retrieval",
        token_budget=32,
        facet_count=2,
    )
    assert "chunk-0" in prompt
    with pytest.raises(ValueError, match="do not match"):
        parse_generated_batch(
            {"summaries": [{"id": "wrong", "summary": "notes", "facets": []}]},
            ["chunk-0"],
            {
                "prompt_eval_tokens": 10,
                "eval_tokens": 4,
                "generation_seconds": 0.5,
                "raw_response_sha256": "f" * 64,
            },
        )


def test_generation_cache_is_append_only_and_resumable(tmp_path) -> None:
    accounting = {
        "prompt_eval_tokens": 10,
        "eval_tokens": 4,
        "generation_seconds": 0.5,
        "raw_response_sha256": "f" * 64,
    }
    value = parse_generated_batch(
        {
            "summaries": [
                {
                    "id": "chunk-0",
                    "summary": "Ada wrote notes",
                    "facets": [{"label": "entity", "text": "Ada"}],
                }
            ]
        },
        ["chunk-0"],
        accounting,
    )[0]
    key = cache_key(
        model="test", prompt_id="retrieval", token_budget=32, facet_count=1, source="source"
    )
    path = tmp_path / "summaries.jsonl"
    cache = JSONLGenerationCache(path)
    cache.put(key, value)
    cache.put(key, value)

    restored = JSONLGenerationCache(path)
    assert restored.get(key) == value
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_generation_cache_rebinds_duplicate_content_to_requested_identity(tmp_path) -> None:
    class FakeClient:
        calls = 0

        def generate_text(self, model, prompt, *, seed, max_new_tokens):
            self.calls += 1
            return "shared address", {
                "prompt_eval_tokens": 5,
                "eval_tokens": 2,
                "generation_seconds": 0.1,
                "raw_response_sha256": "a" * 64,
            }

    client = FakeClient()
    outputs = generate_cached(
        client,
        JSONLGenerationCache(tmp_path / "duplicate.jsonl"),
        [("chunk-a", "identical source"), ("chunk-b", "identical source")],
        model="test",
        prompt_id="retrieval",
        token_budget=8,
        facet_count=1,
        seed=7,
        batch_size=2,
        structured_batches=False,
    )

    assert client.calls == 1
    assert [output.item_id for output in outputs] == ["chunk-a", "chunk-b"]
    assert [output.summary for output in outputs] == ["shared address", "shared address"]
