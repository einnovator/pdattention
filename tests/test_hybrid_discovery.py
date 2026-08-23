"""Unit and scientific-contract tests for Paper 2.6 hybrid discovery."""

from __future__ import annotations

import torch

from pra_hf.config import PRAConfig
from pra_hf.hybrid_discovery import (
    HybridDiscoveryPolicy,
    TokenChunkRecord,
    TokenNativeIndex,
)
from pra_hf.iterative import GistIndex, IterativeGistRouter, IterativeRoutingConfig
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


def test_iterative_token_routing_exposes_a_later_hop_anchor():
    tokenizer, index, token_index = _fixture()
    root_ids = tokenizer("question bridge", add_special_tokens=False)["input_ids"]
    result = IterativeGistRouter(index).route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(
            depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            root_anchor_alpha=0.0,
        ),
        token_index=token_index,
        root_token_ids=root_ids,
        tokenizer=tokenizer,
        discovery_policy=HybridDiscoveryPolicy(mode="token_weighted"),
    )
    assert [index.chunk_ids[row] for row in result.selected_indices] == ["A:0", "B:0"]
    assert [node.hop for node in result.graph.nodes] == [1, 2]
    assert result.graph.nodes[1].discovery_channels["associative_score"] > 0
    assert result.graph.costs["token_index_comparisons"] == 6
    assert result.graph.costs["token_index_queries"] == 2


def test_iterative_hybrid_uses_semantic_entry_then_token_association():
    tokenizer, index, token_index = _fixture()
    root_ids = tokenizer("opaque prompt", add_special_tokens=False)["input_ids"]
    result = IterativeGistRouter(index).route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(
            depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            root_anchor_alpha=0.0,
        ),
        token_index=token_index,
        root_token_ids=root_ids,
        tokenizer=tokenizer,
        discovery_policy=HybridDiscoveryPolicy(mode="iterative_hybrid"),
    )
    assert [index.chunk_ids[row] for row in result.selected_indices] == ["A:0", "B:0"]
    assert result.graph.edges[0].edge_type == "hybrid_entry"
    assert result.graph.edges[1].edge_type == "hybrid_associative"


def test_stale_token_index_is_rejected_before_routing():
    tokenizer, index, token_index = _fixture()
    stale = TokenNativeIndex(
        (
            TokenChunkRecord(
                reference_uri="wrong",
                chunk_id="wrong:0",
                layer_id=0,
                token_ids=(1,),
                normalized_tokens=("wrong",),
                bm25_terms=("wrong",),
                aliases=(),
                token_start=0,
                token_end=1,
            ),
            *token_index.records[1:],
        )
    )
    try:
        IterativeGistRouter(index).route(
            torch.tensor([1.0, 0.0, 0.0]),
            IterativeRoutingConfig(),
            token_index=stale,
            root_token_ids=tokenizer("bridge")["input_ids"],
            tokenizer=tokenizer,
        )
    except ValueError as error:
        assert "do not align" in str(error)
    else:
        raise AssertionError("Expected stale token sidecar rejection.")


def test_public_config_exposes_token_and_hybrid_iterative_modes():
    token = PRAConfig(routing_mode="token_iterative")
    hybrid = PRAConfig(routing_mode="hybrid_iterative", hybrid_token_weight=0.6)
    assert token.hybrid_discovery_policy.mode == "token_weighted"
    assert hybrid.hybrid_discovery_policy.mode == "iterative_hybrid"
    assert hybrid.hybrid_discovery_policy.token_weight == 0.6
    assert hybrid.selection_policy == "hybrid_iterative_closure"


def test_indexed_scoring_retains_exact_hit_with_bounded_candidate_work():
    tokenizer = WhitespaceTokenizer()
    entries = [
        _entry(f"R{index}", f"topic{index} unique{index}", [1.0, 0.0, 0.0])
        for index in range(24)
    ]
    index = GistIndex.from_entries(entries, 0)
    token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
    query = tokenizer("topic17 unique17")["input_ids"]
    candidates = token_index.score(
        query,
        torch.zeros(len(entries)),
        tokenizer,
        HybridDiscoveryPolicy(
            mode="token_exact", indexed=True, candidate_pool_size=6
        ),
        hop=1,
        parent_id="__root__",
    )
    assert token_index.raw_ngram_postings

    winner = min(candidates, key=lambda candidate: candidate.rank or 10_000)
    assert winner.reference_uri == "R17"
    assert token_index.last_search_stats["candidate_rows"] <= 6
    assert token_index.last_search_stats["candidate_fraction"] < 0.5


def test_indexed_sparse_scoring_does_not_materialize_full_python_result_set():
    tokenizer = WhitespaceTokenizer()
    entries = [
        _entry(f"R{index}", f"topic{index} unique{index}", [1.0, 0.0, 0.0])
        for index in range(24)
    ]
    gist_index = GistIndex.from_entries(entries, 0)
    token_index = TokenNativeIndex.from_gist_index(gist_index, tokenizer)
    candidates = token_index.score(
        tokenizer("topic17 unique17")["input_ids"],
        torch.zeros(len(entries)),
        tokenizer,
        HybridDiscoveryPolicy(
            mode="token_exact", indexed=True, candidate_pool_size=6
        ),
        hop=1,
        parent_id="__root__",
        sparse=True,
    )

    assert isinstance(candidates, dict)
    assert len(candidates) <= 6
    assert token_index.last_search_stats["returned_rows"] == len(candidates)
    winner = min(candidates.values(), key=lambda candidate: candidate.rank or 10_000)
    assert winner.reference_uri == "R17"


def test_extended_channels_populate_ngram_edit_and_embedding_scores():
    tokenizer, index, _ = _fixture()
    query = tokenizer("bridge betx")["input_ids"]
    torch.manual_seed(305)
    embedding = torch.randn(max(tokenizer.pieces) + 1, 5)
    token_index = TokenNativeIndex.from_gist_index(
        index, tokenizer, token_embedding_weight=embedding, ngram_sizes=(2,)
    )
    candidates = token_index.score(
        query,
        torch.zeros(len(index.records)),
        tokenizer,
        HybridDiscoveryPolicy(
            mode="token_edit",
            ngram_sizes=(2,),
            approximate_max_distance=1,
            enable_extended_channels=True,
        ),
        hop=1,
        parent_id="__root__",
        token_embedding_weight=embedding,
    )

    assert candidates[0].edit_score > 0
    assert candidates[0].embedding_score is not None
    assert token_index.memory_bytes() > 0


def test_automatic_alias_extraction_is_label_independent():
    tokenizer = WhitespaceTokenizer()
    index = GistIndex.from_entries(
        [_entry("memory://facts", "Ada Lovelace designed an engine", [1.0, 0.0, 0.0])],
        0,
    )
    token_index = TokenNativeIndex.from_gist_index(
        index, tokenizer, automatic_aliases=True
    )
    assert "ada lovelace" in token_index.records[0].aliases


def test_index_extension_recomputes_corpus_statistics_and_preserves_rows():
    tokenizer, index, token_index = _fixture()
    extra_index = GistIndex.from_entries(
        [_entry("D", "delta bridge", [0.0, 0.0, 1.0])], 0
    )
    extra = TokenNativeIndex.from_gist_index(extra_index, tokenizer).records
    extended = token_index.extended(extra)

    assert len(extended.records) == len(token_index.records) + 1
    assert extended.records[-1].reference_uri == "D"
    assert extended.idf["bridge"] != token_index.idf["bridge"]


def test_explicit_uri_overrides_a_stronger_semantic_distractor():
    tokenizer, index, token_index = _fixture()
    candidates = token_index.score(
        tokenizer("opaque query")["input_ids"],
        torch.tensor([-1.0, 1.0, 0.5]),
        tokenizer,
        HybridDiscoveryPolicy(mode="iterative_hybrid"),
        hop=1,
        parent_id="__root__",
        explicit_reference_uris={"A"},
    )

    winner = min(candidates, key=lambda candidate: candidate.rank or 10_000)
    assert winner.reference_uri == "A"
    assert winner.selected_channel == "explicit_reference"
    assert winner.selected_score == 1.0
