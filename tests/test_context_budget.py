from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from pra_torch.attention import PRAttention
from pra_torch.chunking import ChunkingConfig, partition_source
from pra_torch.config import PRAConfig
from pra_torch.data import CharTokenizer
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
    SelectedChunk,
)
from pra_torch.model import TinyPRAModel
from pra_torch.prompt import IMPLICIT_PROMPT_HEAD_URI, prepare_prompt_batch_for_pra


def _cfg(tokenizer, **overrides):
    values = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "max_seq_len": 32,
        "model_max_context_tokens": 16,
        "max_prompt_direct_tokens": 8,
        "prompt_overflow_mode": "implicit_reference",
        "prompt_position_mode": "historical",
        "encoding_chunking": {
            "mode": "fixed",
            "chunk_tokens": 8,
            "overlap_fraction": 0.25,
        },
        "routing_chunking": {"mode": "fixed", "chunk_tokens": 4},
        "encoding_context_mode": "overlap",
        "top_k_references": 4,
        "top_k_chunks_per_reference": 4,
        "trigger_threshold": float("-inf"),
        "dropout": 0.0,
        "device": "cpu",
    }
    values.update(overrides)
    return PRAConfig(**values)


def _selected(uri: str, score: float, length: int) -> SelectedChunk:
    key = torch.zeros(1, 2, length, 4)
    chunk = ReferenceChunkMemory(
        chunk_id=f"{uri}#chunk=0",
        source_uri=uri,
        token_start=0,
        token_end=length,
        token_kv=LayerKV(k=key, v=key.clone()),
        routing_gist=ChunkRoutingGist(k=torch.tensor([[1.0] + [0.0] * 7])),
    )
    entry = PRACacheEntry(
        uri=uri,
        text=uri,
        layer_memory={0: LayerReferenceMemory(chunks=[chunk])},
    )
    return SelectedChunk(entry, chunk, score, score, 0, 1, 1)


def test_context_config_validates_hard_limit_and_nested_chunking():
    cfg = PRAConfig(
        max_seq_len=32,
        model_max_context_tokens=16,
        context_safety_reserve_tokens=1,
        encoding_chunking={"mode": "fixed", "chunk_tokens": 8},
        routing_chunking={"mode": "fixed", "chunk_tokens": 2},
    )
    assert cfg.effective_model_max_context_tokens == 16
    assert cfg.effective_prompt_direct_tokens == 15
    assert cfg.encoding_chunking_config.chunk_tokens == 8
    assert cfg.routing_chunking_config.chunk_tokens == 2
    with pytest.raises(ValueError, match="cannot exceed"):
        PRAConfig(max_seq_len=16, model_max_context_tokens=17)
    with pytest.raises(ValueError, match="encoding_chunking"):
        PRAConfig(
            max_seq_len=32,
            model_max_context_tokens=16,
            encoding_chunking={"mode": "fixed", "chunk_tokens": 17},
        )


def test_shared_chunking_supports_fixed_overlap_and_markers():
    tokenizer = CharTokenizer(["abcdefgh", "ab<SEP>cd"])
    fixed = partition_source(
        "mem://fixed",
        tokenizer.encode("abcdefgh"),
        tokenizer,
        ChunkingConfig(mode="fixed", chunk_tokens=4, overlap_tokens=1),
    )
    assert [(chunk.token_start, chunk.token_end) for chunk in fixed] == [
        (0, 4),
        (3, 7),
        (6, 8),
    ]
    assert [(chunk.logical_start, chunk.logical_end) for chunk in fixed] == [
        (0, 4),
        (3, 7),
        (6, 8),
    ]
    marked_ids = tokenizer.encode("ab<SEP>cd")
    marked = partition_source(
        "mem://marked",
        marked_ids,
        tokenizer,
        ChunkingConfig(mode="markers", markers=("<SEP>",)),
        text="ab<SEP>cd",
    )
    assert tuple(token for chunk in marked for token in chunk.token_ids) == tuple(marked_ids)


def test_encoding_blocks_are_bounded_and_feed_multiple_routing_chunks():
    tokenizer = CharTokenizer(["abcdefghijklmnopqrstuvwxyz0123456789ABCD"])
    cfg = _cfg(tokenizer)
    model = TinyPRAModel(cfg).eval()
    ids = tokenizer.encode("abcdefghijklmnopqrstuvwxyz0123456789ABCD")
    with patch.object(
        model, "_encode_reference_tokens", wraps=model._encode_reference_tokens
    ) as encode:
        entry = model.encode_reference_tokens_to_cache(
            "mem://long",
            ids,
            tokenizer,
            "cpu",
            use_configured_max_chunks=False,
            historical_encoding=True,
        )
    lengths = [len(call.args[0]) for call in encode.call_args_list]
    chunks = entry.layer_memory[0].chunks
    assert max(lengths) <= cfg.effective_model_max_context_tokens
    assert entry.metadata["encoding_call_count"] == 5
    assert len(chunks) == 10
    assert len(chunks) > len(lengths)
    assert [chunk.token_start for chunk in chunks] == list(range(0, 40, 4))
    assert [chunk.logical_start for chunk in chunks] == list(range(0, 40, 4))
    assert [chunk.logical_end for chunk in chunks] == list(range(4, 41, 4))
    assert entry.metadata["logical_to_native_context_ratio"] == pytest.approx(2.5)


def test_prompt_head_larger_than_native_context_uses_bounded_calls():
    tokenizer = CharTokenizer(["abcdefghijklmnopqrstuvwxyz0123456789"])
    cfg = _cfg(tokenizer)
    model = TinyPRAModel(cfg).eval()
    ids = torch.tensor([tokenizer.encode("abcdefghijklmnopqrstuvwxyz0123456789")])
    prepared = prepare_prompt_batch_for_pra(model, tokenizer, ids)
    head = prepared.caches[0].get(IMPLICIT_PROMPT_HEAD_URI)
    assert len(prepared.splits[0].implicit_ids) == 28
    assert head.metadata["max_encoding_input_tokens"] <= 16
    assert head.metadata["encoding_call_count"] == 4
    assert prepared.position_offsets.tolist() == [0]
    assert sum(chunk.token_count for chunk in head.layer_memory[0].chunks) == 28


def test_budgeter_skips_oversized_hits_and_fills_with_smaller_later_hit():
    tokenizer = CharTokenizer(["x"])
    cfg = _cfg(
        tokenizer,
        model_max_context_tokens=20,
        max_materialized_memory_tokens=12,
    )
    attention = PRAttention(8, 2, 32, 0, PRASimpleMemoryCache(), config=cfg)
    candidates = [
        _selected("A", 0.97, 8),
        _selected("B", 0.93, 8),
        _selected("C", 0.90, 12),
        _selected("D", 0.81, 4),
    ]
    selected, stats = attention._budget_selection(
        candidates, direct_tokens=4, routing_candidates=4
    )
    assert [hit.reference_uri for hit in selected] == ["A", "D"]
    assert stats["memory_tokens_materialized"] == 12
    assert stats["chunks_budget_rejected"] == 2
    assert stats["highest_budget_rejected_score"] == pytest.approx(0.93)


def test_head_and_explicit_references_share_one_deterministic_budget():
    tokenizer = CharTokenizer(["x"])
    cfg = _cfg(
        tokenizer,
        model_max_context_tokens=20,
        max_materialized_memory_tokens=12,
    )
    attention = PRAttention(8, 2, 32, 0, PRASimpleMemoryCache(), config=cfg)
    candidates = [
        _selected(IMPLICIT_PROMPT_HEAD_URI, 0.9, 8),
        _selected("mem://explicit-a", 0.9, 8),
        _selected("mem://explicit-b", 0.8, 4),
    ]
    selected, stats = attention._budget_selection(
        candidates, direct_tokens=4, routing_candidates=3
    )
    assert [hit.reference_uri for hit in selected] == [
        "mem://explicit-a",
        "mem://explicit-b",
    ]
    assert stats["memory_tokens_materialized"] == 12
    assert stats["chunks_budget_rejected"] == 1


def test_native_attention_rejects_context_larger_than_hard_limit():
    tokenizer = CharTokenizer(["x"])
    cfg = _cfg(tokenizer, model_max_context_tokens=8)
    attention = PRAttention(8, 2, 32, 0, PRASimpleMemoryCache(), config=cfg)
    q = torch.zeros(1, 2, 5, 4)
    memory = torch.zeros(1, 2, 4, 4)
    with pytest.raises(ValueError, match="exceeds model_max_context_tokens"):
        attention.forward_native_kv(
            torch.zeros(1, 5, 8), [memory], [memory.clone()]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer accounting test")
def test_cpu_budget_rejection_transfers_only_final_selected_chunk():
    tokenizer = CharTokenizer(["x"])
    cfg = _cfg(
        tokenizer,
        model_max_context_tokens=12,
        max_materialized_memory_tokens=8,
        kv_cache_residency="cpu",
    )
    attention = PRAttention(8, 2, 32, 0, PRASimpleMemoryCache(), config=cfg).cuda()
    selected = [_selected("A", 0.9, 8), _selected("B", 0.8, 8)]
    (
        keys,
        _values,
        retained,
        _duplicates,
        transfer_bytes,
        _transfer_duration,
        stats,
    ) = attention._materialize(
        selected,
        torch.zeros(1, 2, 1, 4, device="cuda"),
        direct_tokens=4,
    )
    assert [hit.reference_uri for hit in retained] == ["A"]
    assert keys.shape[2] == 8
    assert transfer_bytes == 2 * 2 * 8 * 4 * 4
    assert stats["chunks_budget_rejected"] == 1


@pytest.mark.parametrize("position_encoding", ["absolute", "rope"])
def test_streaming_generation_rolls_history_beyond_native_horizon(position_encoding):
    tokenizer = CharTokenizer(["abcdefghijklmnopqrstuvwxyz0123456789"])
    cfg = _cfg(tokenizer, position_encoding=position_encoding)
    model = TinyPRAModel(cfg).eval()
    initial = torch.tensor([tokenizer.encode("abcdefgh")])
    generated = model.generate(
        initial,
        max_new_tokens=20,
        tokenizer=tokenizer,
        do_sample=False,
    )
    stats = model.last_generation_stats
    head = model.pra_cache.get(IMPLICIT_PROMPT_HEAD_URI)
    assert generated.shape[1] == 28
    assert stats["max_direct_tokens_observed"] <= 8
    assert stats["head_tokens"] == 20
    assert stats["rollover_events"] == 5
    assert stats["max_native_operation_tokens"] <= 16
    assert stats["routing_steps"] > 0
    assert stats["max_materialized_memory_tokens_observed"] <= 8
    assert head.metadata["max_encoding_input_tokens"] <= 16
    assert sum(chunk.token_count for chunk in head.layer_memory[0].chunks) == 20
    chunks = head.layer_memory[0].chunks
    assert [chunk.logical_start for chunk in chunks] == [0, 4, 8, 12, 16]
    assert [chunk.logical_end for chunk in chunks] == [4, 8, 12, 16, 20]
