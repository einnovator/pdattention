"""Unit and scientific-contract tests for Paper 2.6 hybrid discovery."""

from __future__ import annotations

import torch

from pra_hf.hybrid_discovery import (
    HybridDiscoveryPolicy,
    TokenNativeIndex,
)
from pra_hf.iterative import GistIndex
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    ReferenceChunkMemory,
)


class WhitespaceTokenizer:
    """Small deterministic tokenizer implementing the HF methods used by the index."""

    all_special_ids = (0,)

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.pieces: dict[int, str] = {}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        values = []
        for piece in text.split():
            key = piece.casefold()
            if key not in self.vocabulary:
                index = len(self.vocabulary) + 1
                self.vocabulary[key] = index
                self.pieces[index] = piece
            values.append(self.vocabulary[key])
        return {"input_ids": values}

    def convert_ids_to_tokens(self, values):
        return [self.pieces[int(value)] for value in values]

    def decode(self, values):
        return " ".join(self.convert_ids_to_tokens(values))


def _entry(uri: str, text: str, vector: list[float]) -> PRACacheEntry:
    gist = torch.tensor([vector], dtype=torch.float32)
    kv = gist.view(1, 1, 1, -1)
    chunk = ReferenceChunkMemory(
        chunk_id=f"{uri}:0",
        source_uri=uri,
        token_start=0,
        token_end=len(text.split()),
        token_kv=LayerKV(kv.clone(), kv.clone()),
        routing_gist=ChunkRoutingGist(k=gist),
    )
    return PRACacheEntry(
        uri=uri,
        text=text,
        layer_memory={0: LayerReferenceMemory([chunk])},
    )


def _fixture():
    tokenizer = WhitespaceTokenizer()
    entries = [
        _entry("A", "BRIDGE beta", [0.95, 0.05, 0.0]),
        _entry("B", "beta final", [0.0, 1.0, 0.0]),
        _entry("X", "unrelated noise", [0.7, 0.0, 0.7]),
    ]
    index = GistIndex.from_entries(entries, 0)
    token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
    return tokenizer, index, token_index


def test_token_index_preserves_gist_identity_and_native_ids():
    tokenizer, index, token_index = _fixture()
    token_index.validate_alignment(index)
    assert [record.chunk_id for record in token_index.records] == list(index.chunk_ids)
    assert token_index.records[0].token_ids == tuple(
        tokenizer("BRIDGE beta", add_special_tokens=False)["input_ids"]
    )
    assert "bridge" in token_index.bm25_idf


def test_channel_record_retains_exact_normalized_and_provenance_scores():
    tokenizer, index, token_index = _fixture()
    query_ids = tokenizer("question bridge beta", add_special_tokens=False)["input_ids"]
    candidates = token_index.score(
        query_ids,
        torch.tensor([0.1, 0.0, -0.1]),
        tokenizer,
        HybridDiscoveryPolicy(mode="token_weighted"),
        hop=1,
        parent_id="__root__",
    )
    candidate = candidates[0]
    assert candidate.reference_uri == "A"
    assert candidate.normalized_exact_score > 0
    assert candidate.weighted_overlap_score > 0
    assert candidate.bm25_score > 0
    assert candidate.raw_exact_span is not None
    assert candidate.parent_id == "__root__"
    assert not candidate.confidence_calibrated
    assert candidate.hop == 1
