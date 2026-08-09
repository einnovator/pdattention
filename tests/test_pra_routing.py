from __future__ import annotations

import torch
from types import SimpleNamespace
import pytest

from pra_torch.attention import PRAttention
from pra_torch.chunking import partition_reference
from pra_torch.config import PRAConfig
from pra_torch.gist import mean_pool_layer_keys
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
)
from pra_torch.memory_batching import dynamic_memory_attention
from pra_torch.model import TinyPRAModel
from pra_torch.resolution import RecursiveReferenceCacheBuilder, ResolutionBudget
from pra_torch.resolver import InMemoryResolver
from pra_torch.data import CharTokenizer
from pra_torch.pra_train import _retrieval_metrics


def _chunk(uri, chunk_id, gist, value, *, layer_id=0, start=0, length=1):
    gist = torch.tensor(gist, dtype=torch.float32)
    value = torch.tensor(value, dtype=torch.float32)
    key = gist.view(1, 1, 1, -1).expand(1, 1, length, -1).clone()
    values = value.view(1, 1, 1, -1).expand(1, 1, length, -1).clone()
    return ReferenceChunkMemory(
        chunk_id=chunk_id,
        source_uri=uri,
        token_start=start,
        token_end=start + length,
        token_kv=LayerKV(key, values),
        routing_gist=ChunkRoutingGist(k=gist, v=value),
    )


def _entry(uri, chunks_by_layer):
    return PRACacheEntry(
        uri=uri,
        text=uri,
        layer_memory={
            layer_id: LayerReferenceMemory(chunks=list(chunks))
            for layer_id, chunks in chunks_by_layer.items()
        },
    )


def _routing_config(**overrides):
    values = dict(
        vocab_size=32,
        d_model=2,
        n_heads=1,
        n_layers=1,
        max_seq_len=8,
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        dropout=0.0,
    )
    values.update(overrides)
    return PRAConfig(**values)


def test_mean_pool_reconstructs_heads_before_averaging():
    keys = torch.tensor(
        [[[[1.0], [3.0]], [[10.0], [30.0]]]]
    )  # [1, heads=2, tokens=2, head_dim=1]
    pooled = mean_pool_layer_keys(keys)
    assert pooled.shape == (2,)
    assert torch.equal(pooled, torch.tensor([2.0, 20.0]))


def test_search_is_layer_specific_and_batch_specific():
    cache = PRASimpleMemoryCache()
    cache.put(
        _entry(
            "A",
            {
                0: [_chunk("A", "A0", [1, 0], [1, 0])],
                1: [_chunk("A", "A1", [0, 1], [1, 0], layer_id=1)],
            },
        )
    )
    cache.put(
        _entry(
            "B",
            {
                0: [_chunk("B", "B0", [0, 1], [0, 1])],
                1: [_chunk("B", "B1", [1, 0], [0, 1], layer_id=1)],
            },
        )
    )
    cfg = _routing_config()
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    layer_zero = cache.search(queries, 0, cfg)
    layer_one = cache.search(queries, 1, cfg)

    assert [hits[0].reference_uri for hits in layer_zero] == ["A", "B"]
    assert [hits[0].reference_uri for hits in layer_one] == ["B", "A"]


def test_search_retains_complete_deterministic_rankings_before_top_k():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", {0: [_chunk("A", "A0", [1, 0], [1, 0])]}))
    cache.put(_entry("B", {0: [_chunk("B", "B0", [0.8, 0.6], [0, 1])]}))

    selected = cache.search(
        torch.tensor([[1.0, 0.0]]),
        0,
        _routing_config(collect_routing_metrics=True),
    )[0]
    rankings = cache.last_rankings(0)[0]

    assert [hit.reference_uri for hit in selected] == ["A"]
    assert [row["reference_uri"] for row in rankings] == ["A", "B"]
    assert [row["reference_rank"] for row in rankings] == [1, 2]
    assert rankings[0]["chunks"][0]["gist_count"] == 1


@pytest.mark.parametrize("aggregation", ["max", "mean", "logsumexp"])
def test_tensorized_hierarchical_routing_matches_legacy_scores_and_selection(aggregation):
    cache = PRASimpleMemoryCache()
    cache.put(
        _entry(
            "A",
            {0: [
                _chunk("A", "A0", [1.0, 0.0], [1, 0]),
                _chunk("A", "A1", [0.6, 0.8], [1, 0], start=1),
            ]},
        )
    )
    cache.put(
        _entry(
            "B",
            {0: [
                _chunk("B", "B0", [0.0, 1.0], [0, 1]),
                _chunk("B", "B1", [0.8, 0.6], [0, 1], start=1),
            ]},
        )
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    common = dict(
        top_k_references=2,
        top_k_chunks_per_reference=1,
        reference_score_aggregation=aggregation,
        collect_routing_metrics=True,
    )

    legacy = cache.search(queries, 0, _routing_config(routing_backend="legacy", **common))
    legacy_rankings = cache.last_rankings(0)
    tensorized = cache.search(
        queries, 0, _routing_config(routing_backend="tensorized", **common)
    )
    tensorized_rankings = cache.last_rankings(0)

    assert [[hit.chunk_id for hit in row] for row in tensorized] == [
        [hit.chunk_id for hit in row] for row in legacy
    ]
    for expected_rows, actual_rows in zip(legacy_rankings, tensorized_rankings):
        assert [row["reference_uri"] for row in actual_rows] == [
            row["reference_uri"] for row in expected_rows
        ]
        for expected, actual in zip(expected_rows, actual_rows):
            assert actual["reference_score"] == pytest.approx(expected["reference_score"])


def test_tensorized_routing_uses_torch_topk(monkeypatch):
    cache = PRASimpleMemoryCache()
    for index, gist in enumerate(([1, 0], [0.8, 0.6], [0, 1])):
        uri = f"R{index}"
        cache.put(_entry(uri, {0: [_chunk(uri, f"{uri}0", gist, gist)]}))
    calls = []
    original = torch.topk

    def record_topk(*args, **kwargs):
        calls.append((args[0].shape, kwargs.get("dim")))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "topk", record_topk)
    cache.search(
        torch.tensor([[1.0, 0.0]]),
        0,
        _routing_config(routing_backend="tensorized", top_k_references=2),
    )

    assert len(calls) == 2  # URI top-k and per-URI chunk top-k.


def test_tensorized_routing_index_is_invalidated_when_cache_changes():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", {0: [_chunk("A", "A0", [0, 1], [0, 1])]}))
    cfg = _routing_config(routing_backend="tensorized")
    query = torch.tensor([[1.0, 0.0]])

    assert cache.search(query, 0, cfg)[0][0].reference_uri == "A"
    assert cache._tensorized_indexes
    cache.put(_entry("B", {0: [_chunk("B", "B0", [1, 0], [1, 0])]}))

    assert not cache._tensorized_indexes
    assert cache.search(query, 0, cfg)[0][0].reference_uri == "B"


def test_hierarchical_search_keeps_reference_and_chunk_budgets_distinct():
    cache = PRASimpleMemoryCache()
    cache.put(
        _entry(
            "A",
            {0: [
                _chunk("A", "A0", [0.95, 0.3122499], [1, 0], start=0),
                _chunk("A", "A1", [0.10, 0.9949874], [1, 0], start=1),
                _chunk("A", "A2", [0.05, 0.9987492], [1, 0], start=2),
            ]},
        )
    )
    cache.put(
        _entry(
            "B",
            {0: [
                _chunk("B", "B0", [0.80, 0.60], [0, 1], start=0),
                _chunk("B", "B1", [0.79, 0.613106], [0, 1], start=1),
                _chunk("B", "B2", [0.78, 0.62578], [0, 1], start=2),
            ]},
        )
    )
    cfg = _routing_config(top_k_chunks_per_reference=2)
    selected = cache.search(torch.tensor([[1.0, 0.0]]), 0, cfg)[0]
    assert [hit.reference_uri for hit in selected] == ["A", "A"]
    assert [hit.chunk_id for hit in selected] == ["A0", "A1"]


@pytest.mark.parametrize(
    ("policy", "expected_tokens"),
    (("deduplicate", 6), ("keep_duplicates", 8)),
)
def test_overlap_materialization_policy_controls_physical_kv(policy, expected_tokens):
    cache = PRASimpleMemoryCache()
    cache.put(
        _entry(
            "A",
            {
                0: [
                    _chunk("A", "A0", [1, 0], [1, 0], start=0, length=4),
                    _chunk("A", "A1", [1, 0], [1, 0], start=2, length=4),
                ]
            },
        )
    )
    cfg = _routing_config(
        top_k_chunks_per_reference=2,
        overlap_materialization=policy,
    )
    selected = cache.search(torch.tensor([[1.0, 0.0]]), 0, cfg)[0]
    attention = PRAttention(2, 1, 8, 0, cache, config=cfg)

    keys, values, _retained, duplicate_tokens = attention._materialize(
        selected, torch.zeros(1, 1, 1, 2)
    )

    assert keys.shape == values.shape == (1, 1, expected_tokens, 2)
    assert duplicate_tokens == 2


def test_logsumexp_can_prefer_multiple_moderate_chunks():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", {0: [_chunk("A", "A0", [1, 0], [1, 0])]}))
    cache.put(
        _entry(
            "B",
            {0: [
                _chunk("B", "B0", [0.8, 0.6], [0, 1]),
                _chunk("B", "B1", [0.8, -0.6], [0, 1], start=1),
            ]},
        )
    )
    cfg = _routing_config(reference_score_aggregation="logsumexp")
    selected = cache.search(torch.tensor([[1.0, 0.0]]), 0, cfg)[0]
    assert selected[0].reference_uri == "B"


def test_global_chunk_search_enforces_both_limits():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", {0: [_chunk("A", f"A{i}", [1, 0.01 * i], [1, 0], start=i) for i in range(3)]}))
    cache.put(_entry("B", {0: [_chunk("B", "B0", [0.8, 0.6], [0, 1])]}))
    cfg = _routing_config(
        search_strategy="global_chunks",
        top_k_references=2,
        top_k_chunks_per_reference=1,
    )
    selected = cache.search(torch.tensor([[1.0, 0.0]]), 0, cfg)[0]
    assert [hit.reference_uri for hit in selected] == ["A", "B"]


def test_top_k_zero_selects_none_and_reference_first_requires_representation():
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", {0: [_chunk("A", "A0", [1, 0], [1, 0])]}))
    assert cache.search(torch.tensor([[1.0, 0.0]]), 0, _routing_config(top_k_references=0)) == [[]]
    cfg = _routing_config(search_strategy="reference_first")
    try:
        cache.search(torch.tensor([[1.0, 0.0]]), 0, cfg)
    except ValueError as error:
        assert "reference_level_gist_mode" in str(error)
    else:
        raise AssertionError("reference_first accepted an undefined reference representation")


def test_attention_uses_independent_memory_per_batch_item_and_no_match_is_exact():
    cfg = _routing_config(memory_bucket_count=1, memory_transport="cross_attention")
    cache = PRASimpleMemoryCache()
    cache.put(_entry("A", {0: [_chunk("A", "A0", [1, 0], [10, 0])]}))
    cache.put(_entry("B", {0: [_chunk("B", "B0", [0, 1], [0, 20])]}))
    attention = PRAttention(2, 1, 8, 0, cache, config=cfg).eval()
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
            attention.mem_o_proj,
        ):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    x = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    output = attention(x)
    assert [hits[0].reference_uri for hits in attention.last_selected_chunks] == ["A", "B"]
    assert output[0, 0, 0] > output[1, 0, 0]
    assert output[1, 0, 1] > output[0, 0, 1]
    output.sum().backward()
    assert attention.q_proj.weight.grad is not None

    attention.trigger_threshold = 2.0
    no_match = attention(x)
    local_only = attention(x, use_pra_memory=False)
    assert torch.equal(no_match, local_only)
    assert attention.last_selected_chunks == [[], []]


def test_bucket_modes_match_outputs_and_gradients():
    base_q = torch.randn(5, 2, 3, 4)
    keys = [torch.randn(1, 2, length, 4) for length in (0, 1, 7, 3, 5)]
    values = [torch.randn(1, 2, length, 4) for length in (0, 1, 7, 3, 5)]
    outputs = []
    gradients = []
    for bucket_count in (0, 1, 2, 8):
        q = base_q.clone().requires_grad_(True)
        output, stats = dynamic_memory_attention(
            q,
            keys,
            values,
            bucket_count=bucket_count,
            bucket_strategy="optimal_contiguous",
        )
        output.sum().backward()
        outputs.append(output.detach())
        gradients.append(q.grad.detach())
        assert torch.equal(output[0], torch.zeros_like(output[0]))
        if bucket_count:
            assert stats.actual_bucket_count <= bucket_count
    for output, gradient in zip(outputs[1:], gradients[1:]):
        assert torch.allclose(output, outputs[0], atol=1e-6)
        assert torch.allclose(gradient, gradients[0], atol=1e-6)


def test_bucketing_reduces_padding_for_a_skewed_batch():
    q = torch.randn(5, 1, 1, 2)
    keys = [torch.randn(1, 1, length, 2) for length in (8, 9, 10, 11, 128)]
    values = [torch.randn(1, 1, length, 2) for length in (8, 9, 10, 11, 128)]
    _, one = dynamic_memory_attention(
        q, keys, values, bucket_count=1, bucket_strategy="optimal_contiguous"
    )
    _, two = dynamic_memory_attention(
        q, keys, values, bucket_count=2, bucket_strategy="optimal_contiguous"
    )
    assert two.allocated_positions < one.allocated_positions


def test_fixed_chunking_is_bounded_and_provenance_preserving():
    tokenizer = CharTokenizer(["abcdefghij"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        chunking_mode="fixed",
        fixed_chunk_tokens=4,
        max_gists_per_reference=2,
        gist_overflow_policy="truncate",
    )
    chunks = partition_reference("mem://x", "abcdefghij", tokenizer, cfg)
    assert [chunk.chunk_id for chunk in chunks] == ["mem://x#chunk=0", "mem://x#chunk=1"]
    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [(0, 4), (4, 8)]
    assert all(chunk.metadata["discarded_chunk_count"] == 1 for chunk in chunks)


def test_fractional_chunk_overlap_has_exact_deterministic_accounting():
    tokenizer = CharTokenizer(["abcdefghij"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        chunking_mode="fixed",
        fixed_chunk_tokens=4,
        chunk_overlap_fraction=0.25,
        max_gists_per_reference=8,
    )

    chunks = partition_reference("mem://overlap", "abcdefghij", tokenizer, cfg)

    assert cfg.resolved_chunk_overlap_tokens == 1
    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]
    assert chunks[0].metadata["encoded_tokens_including_overlap"] == 12
    assert chunks[0].metadata["covered_unique_source_tokens"] == 10
    assert chunks[0].metadata["duplication_factor"] == pytest.approx(1.2)


def test_chunk_overlap_forms_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one"):
        PRAConfig(
            fixed_chunk_tokens=8,
            fixed_chunk_overlap_tokens=1,
            chunk_overlap_fraction=0.25,
        )


def test_marker_chunking_is_deterministic_and_semantic_mode_requires_a_plugin():
    text = "first<PRA_CHUNK>second"
    tokenizer = CharTokenizer([text])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        chunking_mode="markers",
        max_gists_per_reference=4,
    )
    first = partition_reference("mem://markers", text, tokenizer, cfg)
    second = partition_reference("mem://markers", text, tokenizer, cfg)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) == 2
    with pytest.raises(NotImplementedError):
        PRAConfig(chunking_mode="semantic")


def test_summary_is_optional_and_gru_gist_has_a_gradient_path():
    tokenizer = CharTokenizer(["reference text"])
    detached_cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        use_summary=True,
    )
    detached_model = TinyPRAModel(detached_cfg)
    entry = detached_model.encode_reference_to_cache("mem://x", "reference text", tokenizer, "cpu")
    assert entry.layer_memory[0].chunks[0].routing_gist.summary_k is None
    assert not entry.layer_memory[0].chunks[0].routing_gist.k.requires_grad

    trainable_cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        gist_mode="gru",
        cache_build_mode="trainable_gist",
    )
    trainable_model = TinyPRAModel(trainable_cfg)
    trainable = trainable_model.encode_reference_to_cache("mem://x", "reference text", tokenizer, "cpu")
    trainable.layer_memory[0].chunks[0].routing_gist.k.sum().backward()
    assert any(parameter.grad is not None for parameter in trainable_model.gist_pooler.parameters())
    assert any(key.startswith("gist_pooler.") for key in trainable_model.state_dict())


def test_optional_summary_modes_fall_back_or_change_routing_explicitly():
    cache = PRASimpleMemoryCache()
    chunk_a = _chunk("A", "A0", [1, 0], [1, 0])
    chunk_a.routing_gist.summary_k = torch.tensor([0.0, 1.0])
    chunk_b = _chunk("B", "B0", [0, 1], [0, 1])
    cache.put(_entry("A", {0: [chunk_a]}))
    cache.put(_entry("B", {0: [chunk_b]}))
    cfg = _routing_config(use_summary=True, summary_mode="replace")
    selected = cache.search(torch.tensor([[0.0, 1.0]]), 0, cfg)[0]
    assert selected[0].metadata["routing_source"] == "summary"

    chunk_a.routing_gist.summary_k = None
    selected = cache.search(torch.tensor([[1.0, 0.0]]), 0, cfg)[0]
    assert selected[0].reference_uri == "A"
    assert selected[0].metadata["routing_source"] == "content"


def test_chunk_metrics_are_separate_from_cache_size_and_omitted_without_labels():
    cache = PRASimpleMemoryCache()
    cache.put(
        _entry(
            "A",
            {0: [
                _chunk("A", "A0", [1, 0], [1, 0], length=2),
                _chunk("A", "A1", [0, 1], [0, 1], start=2, length=5),
            ]},
        )
    )
    selected = cache.search(torch.tensor([[1.0, 0.0]]), 0, _routing_config())[0]
    labeled = {
        "references": [SimpleNamespace(id=1, uri="A")],
        "target_reference_ids": [1],
        "target_chunk_ids": ["A0"],
        "target_chunk_spans": [],
    }
    metrics = _retrieval_metrics(
        [cache],
        [{0: selected}],
        [labeled],
        _routing_config(),
        {0: {}},
    )
    assert metrics["retrieval_chunk_hit_at_1"] == 1.0
    assert metrics["retrieval_chunk_recall_at_k"] == 1.0
    assert metrics["cache_reference_token_count"] == 7.0
    assert metrics["memory_selected_token_count"] == 2.0
    assert metrics["chunk_gists_requested"] == 1.0
    assert metrics["chunk_gists_actual_mean"] == 1.0
    assert metrics["chunk_gists_actual_max"] == 1.0
    assert metrics["winning_chunk_gist_index"] == 0.0
    assert metrics["chunk_best_gist_score"] == pytest.approx(1.0)

    unlabeled = {**labeled, "target_chunk_ids": []}
    missing = _retrieval_metrics(
        [cache], [{0: selected}], [unlabeled], _routing_config(), {0: {}}
    )
    assert "retrieval_chunk_hit_at_1" not in missing
    assert missing["retrieval_chunk_labels_available_fraction"] == 0.0


def test_recursive_builder_is_child_first_cycle_safe_and_child_affects_parent():
    tokenizer = CharTokenizer(["parent uses <REF_1>", "child detail"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=2,
        max_seq_len=32,
        top_k_references=1,
        trigger_threshold=-1.0,
        recursive_refs_enabled=True,
        recursive_max_depth=3,
    )
    model = TinyPRAModel(cfg).eval()
    resolver = InMemoryResolver(
        {
            "mem://parent": {
                "text": "parent uses <REF_1>",
                "reference_table": {"<REF_1>": "mem://child"},
            },
            "mem://child": {"text": "child detail"},
        }
    )
    cache = PRASimpleMemoryCache()
    builder = RecursiveReferenceCacheBuilder(model, resolver, tokenizer, cache, cfg)
    parent = builder.ensure_cached("mem://parent")
    ready_order = [event.uri for event in builder.events if event.event == "ready"]
    assert ready_order == ["mem://child", "mem://parent"]
    assert parent.child_uris == ["mem://child"]

    local_parent = model.encode_reference_to_cache(
        "mem://parent-local",
        "parent uses <REF_1>",
        tokenizer,
        "cpu",
        use_pra_memory=False,
    )
    recursive_keys = parent.layer_memory[1].chunks[0].token_kv.k
    local_keys = local_parent.layer_memory[1].chunks[0].token_kv.k
    assert not torch.allclose(recursive_keys, local_keys)

    cycle_resolver = InMemoryResolver(
        {
            "A": {"text": "a <REF_1>", "reference_table": {"<REF_1>": "B"}},
            "B": {"text": "b <REF_1>", "reference_table": {"<REF_1>": "A"}},
        }
    )
    cycle_cache = PRASimpleMemoryCache()
    cycle_builder = RecursiveReferenceCacheBuilder(model, cycle_resolver, tokenizer, cycle_cache, cfg)
    cycle_builder.ensure_cached("A", budget=ResolutionBudget(4, 128))
    assert cycle_cache.has("A") and cycle_cache.has("B")
    assert any(event.event == "cycle" for event in cycle_builder.events)
