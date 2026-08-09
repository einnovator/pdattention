from __future__ import annotations

import pytest
import torch

from pra_torch.attention import PRAttention
from pra_torch.config import PRAConfig
from pra_torch.data import CharTokenizer
from pra_torch.gist import mean_pool_layer_keys
from pra_torch.gists import GistContext, compute_gists
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRABatchedMemoryCache,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
    ReferenceRoutingGists,
)
from pra_torch.model import TinyPRAModel


def _config(**overrides) -> PRAConfig:
    values = {
        "vocab_size": 32,
        "d_model": 2,
        "n_heads": 1,
        "n_layers": 1,
        "max_seq_len": 16,
        "top_k_references": 1,
        "top_k_chunks_per_reference": 1,
        "trigger_threshold": -1.0,
        "dropout": 0.0,
    }
    values.update(overrides)
    return PRAConfig(**values)


def _compute(mode: str, keys: torch.Tensor, values: torch.Tensor | None, count: int = 3):
    return compute_gists(
        keys=keys,
        values=values,
        mode=mode,
        num_gists=count,
        config=_config(),
        context=GistContext(level="chunk"),
    )


def _chunk(uri: str, chunk_id: str, keys, values) -> ReferenceChunkMemory:
    gist_k = torch.tensor(keys, dtype=torch.float32)
    gist_v = torch.tensor(values, dtype=torch.float32)
    if gist_k.ndim == 1:
        gist_k = gist_k.unsqueeze(0)
        gist_v = gist_v.unsqueeze(0)
    token_k = gist_k[:1].view(1, 1, 1, -1)
    token_v = gist_v[:1].view(1, 1, 1, -1)
    return ReferenceChunkMemory(
        chunk_id=chunk_id,
        source_uri=uri,
        token_start=0,
        token_end=1,
        token_kv=LayerKV(token_k, token_v),
        routing_gist=ChunkRoutingGist(k=gist_k, v=gist_v),
    )


def _entry(uri: str, chunk: ReferenceChunkMemory, reference_keys=None) -> PRACacheEntry:
    entry = PRACacheEntry(
        uri=uri,
        text=uri,
        layer_memory={0: LayerReferenceMemory(chunks=[chunk])},
    )
    if reference_keys is not None:
        entry.reference_gists_by_layer[0] = ReferenceRoutingGists(
            k=torch.tensor(reference_keys, dtype=torch.float32),
            mode="prototype",
        )
    return entry


@pytest.mark.parametrize("mode", ["mean", "last"])
def test_single_gist_modes_always_return_one_two_dimensional_gist(mode):
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = _compute(mode, keys, keys + 10.0, count=8)

    assert result.k.shape == (1, 2)
    assert result.v.shape == (1, 2)
    assert result.metadata["requested_gists"] == 8
    assert result.metadata["actual_gists"] == 1


@pytest.mark.parametrize("mode", ["kmeans", "som", "prototype", "hybrid"])
def test_multi_gist_strategy_shapes_edges_and_determinism(mode):
    keys = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    values = keys + 5.0

    first = _compute(mode, keys, values)
    second = _compute(mode, keys, values)
    empty = _compute(mode, keys[:0], values[:0])
    single = _compute(mode, keys[:1], values[:1])

    assert first.k.ndim == first.v.ndim == 2
    assert first.k.shape == first.v.shape
    assert 1 <= first.k.shape[0] <= 3
    assert torch.allclose(first.k, second.k)
    assert torch.allclose(first.v, second.v)
    assert empty.k.shape == empty.v.shape == (0, 2)
    assert single.k.shape == single.v.shape == (1, 2)
    assert torch.isfinite(first.k).all()
    assert torch.isfinite(first.v).all()


@pytest.mark.parametrize("mode", ["kmeans", "som", "prototype"])
def test_multi_gist_strategies_preserve_key_value_region_pairing(mode):
    keys = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [[10.0, 0.0], [12.0, 0.0], [0.0, 20.0], [0.0, 22.0]],
        dtype=torch.float32,
    )

    result = _compute(mode, keys, values, count=2)
    ordered = result.v[result.v[:, 0].argsort()]

    assert result.k.shape == result.v.shape == (2, 2)
    assert torch.allclose(ordered, torch.tensor([[0.0, 21.0], [11.0, 0.0]]))


def test_chunk_routing_reports_the_winning_gist():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", _chunk("A", "A0", [[1, 0], [0, 1]], [[10, 0], [0, 20]])))

    hit = cache.search(torch.tensor([[0.0, 1.0]]), 0, _config())[0][0]

    assert hit.winning_gist_index == 1
    assert hit.winning_gist_score == pytest.approx(1.0)
    assert hit.gist_count == 2


def test_reference_first_uses_cached_multi_gists_before_scoring_chunks(monkeypatch):
    cache = PRASimpleMemoryCache()
    entry_a = _entry(
        "A",
        _chunk("A", "A0", [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        [[1, 0], [0, 1]],
    )
    entry_b = _entry(
        "B",
        _chunk("B", "B0", [[-1, 0], [0, -1]], [[-1, 0], [0, -1]]),
        [[-1, 0], [0, -1]],
    )
    cache.put(entry_a)
    cache.put(entry_b)
    scored_uris = []
    original = cache._score_chunks

    def record_scored_entries(query, layer_id, config, *, entries=None):
        scored_uris.extend(entry.uri for entry in entries or [])
        return original(query, layer_id, config, entries=entries)

    monkeypatch.setattr(cache, "_score_chunks", record_scored_entries)
    cfg = _config(
        search_strategy="reference_first",
        reference_level_gist_mode="prototype",
        reference_gists_per_reference=2,
    )

    hit = cache.search(torch.tensor([[0.0, 1.0]]), 0, cfg)[0][0]

    assert scored_uris == ["A"]
    assert hit.reference_uri == "A"
    assert hit.winning_reference_gist_index == 1
    assert hit.reference_gist_count == 2


def test_gist_only_materializes_only_the_winning_key_value_pair():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", _chunk("A", "A0", [[1, 0], [0, 1]], [[10, 0], [0, 20]])))
    cfg = _config(detail_materialization="gist_only")
    selected = cache.search(torch.tensor([[0.0, 1.0]]), 0, cfg)[0]
    attention = PRAttention(2, 1, 8, 0, cache, config=cfg)

    keys, values, retained, _ = attention._materialize(
        selected,
        torch.zeros(1, 1, 1, 2),
    )

    assert retained[0].winning_gist_index == 1
    assert keys.shape == values.shape == (1, 1, 1, 2)
    assert torch.equal(keys.flatten(), torch.tensor([0.0, 1.0]))
    assert torch.equal(values.flatten(), torch.tensor([0.0, 20.0]))


def test_reference_gists_are_built_once_and_reused_by_search(monkeypatch):
    tokenizer = CharTokenizer(["abcdefghijklmnop"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        chunking_mode="fixed",
        fixed_chunk_tokens=4,
        max_gists_per_reference=4,
        gist_mode="prototype",
        gists_per_chunk=2,
        reference_level_gist_mode="kmeans",
        reference_gists_per_reference=2,
        search_strategy="reference_first",
        trigger_threshold=-1.0,
    )
    model = TinyPRAModel(cfg)
    entry = model.encode_reference_to_cache(
        "mem://multi",
        "abcdefghijklmnop",
        tokenizer,
        "cpu",
    )
    cache = PRASimpleMemoryCache()
    cache.put(entry)

    assert entry.reference_gists_by_layer[0].k.shape == (2, cfg.d_model)
    monkeypatch.setattr(
        cache,
        "_fallback_reference_gists",
        lambda *args, **kwargs: pytest.fail("cached reference gists were recomputed"),
    )
    selected = cache.search(torch.randn(1, cfg.d_model), 0, cfg)

    assert selected[0]


def test_default_mean_gist_matches_the_previous_single_vector_value():
    projected = torch.tensor([[[[1.0], [3.0]], [[10.0], [30.0]]]])
    old_value = mean_pool_layer_keys(projected)
    token_keys = projected.transpose(1, 2).contiguous().view(2, 2)
    result = _compute("mean", token_keys, None)

    assert result.k.shape == (1, 2)
    assert torch.equal(result.k[0], old_value)


def test_duplicate_uri_multi_gists_remain_isolated_by_batch_row():
    left = PRASimpleMemoryCache()
    right = PRASimpleMemoryCache()
    left.put(_entry("shared", _chunk("shared", "left", [[1, 0], [0, 1]], [[1, 0], [0, 1]])))
    right.put(_entry("shared", _chunk("shared", "right", [[0, 1], [1, 0]], [[0, 1], [1, 0]])))
    cache = PRABatchedMemoryCache([left, right])

    selected = cache.search(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 0, _config())

    assert selected[0][0].chunk_id == "left"
    assert selected[1][0].chunk_id == "right"
    assert selected[0][0].entry is not selected[1][0].entry


def test_cache_gists_can_be_rebuilt_without_reencoding_native_kv():
    tokenizer = CharTokenizer(["abcdefgh"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        gist_mode="mean",
    )
    model = TinyPRAModel(cfg)
    entry = model.encode_reference_to_cache("mem://x", "abcdefgh", tokenizer, "cpu")
    cache = PRASimpleMemoryCache()
    cache.put(entry)
    original_k = entry.layer_memory[0].chunks[0].token_kv.k

    model.cfg.gist_mode = "prototype"
    model.cfg.gists_per_chunk = 2
    model.rebuild_cache_routing_gists(cache, tokenizer=tokenizer)

    chunk = entry.layer_memory[0].chunks[0]
    assert chunk.token_kv.k is original_k
    assert chunk.routing_gist.method == "prototype"
    assert chunk.routing_gist.k.shape == (2, cfg.d_model)
    assert entry.metadata["gists_per_chunk"] == 2
