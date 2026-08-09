from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from data.collators import PRACollator
from data.schemas import QuestionSample, ReferenceSample
from pra_torch.cache_services import collect_reference_metadata
from pra_torch.config import CacheServiceConfig, PRAConfig, ResolverServiceConfig
from pra_torch.data import CharTokenizer
from pra_torch.memory import PRABatchedMemoryCache, PRASimpleMemoryCache
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra
from pra_torch.pra_train import _pra_batch_step
from pra_torch.prompt import (
    IMPLICIT_PROMPT_HEAD_NAME,
    IMPLICIT_PROMPT_HEAD_URI,
    prepare_prompt_batch_for_pra,
    prepare_prompt_for_pra,
)


def _config(tokenizer, **overrides):
    values = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "max_seq_len": 8,
        "max_prompt_direct_tokens": 4,
        "prompt_overflow_mode": "implicit_reference",
        "chunking_mode": "none",
        "max_gists_per_reference": 2,
        "top_k_references": 2,
        "top_k_chunks_per_reference": 2,
        "trigger_threshold": -1.0,
        "dropout": 0.0,
        "device": "cpu",
    }
    values.update(overrides)
    return PRAConfig(**values)


def _valid_rows(tensor, mask):
    return [row[row_mask.bool()].tolist() for row, row_mask in zip(tensor, mask)]


def test_prompt_config_validation_and_effective_limit():
    cfg = PRAConfig(max_seq_len=8, max_prompt_direct_tokens=12)
    assert cfg.effective_prompt_direct_tokens == 8

    with pytest.raises(ValueError, match="max_prompt_direct_tokens"):
        PRAConfig(max_prompt_direct_tokens=0)
    with pytest.raises(ValueError, match="prompt_overflow_mode"):
        PRAConfig(prompt_overflow_mode="discard")
    with pytest.raises(ValueError, match="max_prompt_gists"):
        PRAConfig(max_prompt_gists=0)


def test_implicit_prompt_uri_is_reserved_from_explicit_references():
    reference = ReferenceSample(
        id=1,
        uri=IMPLICIT_PROMPT_HEAD_URI,
        metadata={"text": "collision"},
    )
    with pytest.raises(ValueError, match="reserved by PRA"):
        collect_reference_metadata([{"references": [reference]}])


def test_short_truncate_error_and_implicit_splits_are_token_exact():
    tokenizer = CharTokenizer(["abcdefghijkl"])
    short_cfg = _config(tokenizer)
    short = prepare_prompt_for_pra([1, 2, 3], short_cfg)
    assert short.direct_ids == (1, 2, 3)
    assert short.implicit_ids == ()

    truncate = prepare_prompt_for_pra(
        range(10),
        _config(tokenizer, prompt_overflow_mode="truncate"),
    )
    assert truncate.direct_ids == (6, 7, 8, 9)
    assert truncate.implicit_ids == ()

    with pytest.raises(ValueError, match="exceeding direct limit 4"):
        prepare_prompt_for_pra(range(10), _config(tokenizer, prompt_overflow_mode="error"))

    implicit = prepare_prompt_for_pra(range(10), short_cfg)
    assert implicit.implicit_ids == (0, 1, 2, 3, 4, 5)
    assert implicit.direct_ids == (6, 7, 8, 9)
    assert (*implicit.implicit_ids, *implicit.direct_ids) == tuple(range(10))


def test_short_prompt_feature_path_matches_existing_forward():
    tokenizer = CharTokenizer(["abc"])
    cfg = _config(tokenizer, max_prompt_direct_tokens=4)
    model = TinyPRAModel(cfg).eval()
    input_ids = torch.tensor([tokenizer.encode("abc")])

    with torch.no_grad():
        expected = model(input_ids)
        prepared = prepare_prompt_batch_for_pra(model, tokenizer, input_ids)
        actual = model(prepared.input_ids, attention_mask=prepared.attention_mask)

    assert prepared.splits[0].implicit_ids == ()
    assert torch.equal(prepared.input_ids, input_ids)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_prompt_head_is_not_capped_by_explicit_reference_limit():
    tokenizer = CharTokenizer(["abcdefghijklmnopqrstuvwxyz"])
    cfg = _config(tokenizer, max_prompt_gists=None)
    model = TinyPRAModel(cfg).eval()
    ids = torch.tensor([tokenizer.encode("abcdefghijklmnopqrstuvwxyz")])

    prepared = prepare_prompt_batch_for_pra(model, tokenizer, ids)
    entry = prepared.caches[0].get(IMPLICIT_PROMPT_HEAD_URI)
    chunks = entry.layer_memory[0].chunks

    assert len(chunks) == 3
    assert len(chunks) > cfg.max_gists_per_reference
    cached_ids = tuple(
        token for chunk in chunks for token in chunk.metadata["source_token_ids"]
    )
    assert cached_ids == tuple(ids[0, : -cfg.effective_prompt_direct_tokens].tolist())
    assert entry.metadata["implicit"] is True
    assert entry.metadata["display_name"] == IMPLICIT_PROMPT_HEAD_NAME
    assert sum(chunk.token_count for chunk in chunks) == len(ids[0]) - cfg.effective_prompt_direct_tokens


def test_prompt_specific_gist_limit_caps_only_the_implicit_head():
    tokenizer = CharTokenizer(["abcdefghijklmnopqrstuvwxyz", "external document"])
    cfg = _config(tokenizer, max_prompt_gists=2, max_gists_per_reference=1)
    model = TinyPRAModel(cfg).eval()
    cache = PRASimpleMemoryCache()
    explicit = model.encode_reference_to_cache(
        "doc://external",
        "external document",
        tokenizer,
        "cpu",
    )
    cache.put(explicit)
    ids = torch.tensor([tokenizer.encode("abcdefghijklmnopqrstuvwxyz")])

    prepared = prepare_prompt_batch_for_pra(model, tokenizer, ids, caches=[cache])
    head = cache.get(IMPLICIT_PROMPT_HEAD_URI)

    assert len(explicit.layer_memory[0].chunks) == 1
    assert len(head.layer_memory[0].chunks) == 2
    assert {entry.uri for entry in prepared.caches[0].all_entries()} == {
        "doc://external",
        IMPLICIT_PROMPT_HEAD_URI,
    }


def test_mixed_length_batch_preserves_each_tail_and_row_local_head():
    tokenizer = CharTokenizer(["abcdefghijklmnopqrstuvwxyz"])
    cfg = _config(tokenizer)
    model = TinyPRAModel(cfg).eval()
    rows = [list(range(3)), list(range(10)), list(range(18))]
    input_ids = torch.zeros((3, 18), dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = 1

    prepared = prepare_prompt_batch_for_pra(
        model,
        tokenizer,
        input_ids,
        attention_mask=mask,
    )

    assert _valid_rows(prepared.input_ids, prepared.attention_mask) == [
        [0, 1, 2],
        [6, 7, 8, 9],
        [14, 15, 16, 17],
    ]
    assert prepared.caches[0].get(IMPLICIT_PROMPT_HEAD_URI) is None
    assert prepared.caches[1].get(IMPLICIT_PROMPT_HEAD_URI) is not None
    assert prepared.caches[2].get(IMPLICIT_PROMPT_HEAD_URI) is not None
    assert prepared.caches[1].get(IMPLICIT_PROMPT_HEAD_URI) is not prepared.caches[2].get(
        IMPLICIT_PROMPT_HEAD_URI
    )
    assert [stats.implicit_tokens for stats in prepared.stats] == [0, 6, 14]
    model.set_pra_cache(PRABatchedMemoryCache(prepared.caches))
    logits = model(prepared.input_ids, attention_mask=prepared.attention_mask)
    assert logits.shape[:2] == prepared.input_ids.shape
    assert torch.isfinite(logits).all()


def test_early_prompt_chunk_is_routable_under_controlled_query():
    tokenizer = CharTokenizer(["the secret code is zebra; recent question"])
    cfg = _config(tokenizer, fixed_chunk_tokens=6, chunking_mode="fixed")
    model = TinyPRAModel(cfg).eval()
    ids = torch.tensor([tokenizer.encode("the secret code is zebra; recent question")])
    prepared = prepare_prompt_batch_for_pra(model, tokenizer, ids)
    cache = prepared.caches[0]
    head = cache.get(IMPLICIT_PROMPT_HEAD_URI)
    first_chunk = head.layer_memory[0].chunks[0]
    query = first_chunk.routing_gist.k[:1]

    selected = cache.search(query, 0, cfg)[0]

    assert selected[0].reference_uri == IMPLICIT_PROMPT_HEAD_URI
    assert selected[0].chunk_id == first_chunk.chunk_id


def test_training_step_reports_prompt_metrics_and_keeps_explicit_reference():
    reference = ReferenceSample(
        id=1,
        uri="doc://external",
        metadata={"text": "external evidence"},
    )
    sample = QuestionSample(
        id="long",
        question="old fact zebra " + "filler " * 4 + "<REF_1> question?",
        answer="zebra",
        references=[reference],
        target_reference_ids=[1],
        target_reference_uris=[reference.uri],
    )
    tokenizer = CharTokenizer([sample.question, sample.answer, "external evidence"])
    batch = PRACollator(tokenizer, max_seq_len=8)([sample])
    cfg = _config(tokenizer, fixed_chunk_tokens=6, chunking_mode="fixed")
    model = TinyPRAModel(cfg).eval()

    _loss, result = _pra_batch_step(
        model,
        batch,
        "cpu",
        tokenizer,
        ResolverServiceConfig(),
        CacheServiceConfig(),
    )

    assert result["batch"]["input_ids"].shape[1] == cfg.effective_prompt_direct_tokens
    assert result["caches"][0].has(IMPLICIT_PROMPT_HEAD_URI)
    assert result["caches"][0].has(reference.uri)
    assert result["metrics"]["prompt_implicit_tokens"] > 0
    assert result["metrics"]["prompt_implicit_chunks"] > 0
    assert "prompt_implicit_chunks_selected" in result["metrics"]


def test_generation_builds_head_before_forwarding_bounded_tail():
    tokenizer = CharTokenizer(["abcdefghijklmnop"])
    cfg = _config(tokenizer)
    model = TinyPRAModel(cfg).eval()
    input_ids = torch.tensor([tokenizer.encode("abcdefghijklmnop")])

    with patch.object(model, "forward", wraps=model.forward) as forward:
        generated = model.generate(
            input_ids,
            max_new_tokens=1,
            tokenizer=tokenizer,
            do_sample=False,
        )

    assert generated.shape[1] == input_ids.shape[1] + 1
    assert forward.call_args.args[0].shape[1] <= cfg.effective_prompt_direct_tokens
    head = model.pra_cache.get(IMPLICIT_PROMPT_HEAD_URI)
    assert head is not None
    cached_tokens = sum(chunk.token_count for chunk in head.layer_memory[0].chunks)
    assert cached_tokens == input_ids.shape[1] - cfg.effective_prompt_direct_tokens

    short_ids = torch.tensor([tokenizer.encode("abc")])
    model.generate(short_ids, max_new_tokens=0, tokenizer=tokenizer, do_sample=False)
    assert model.pra_cache.get(IMPLICIT_PROMPT_HEAD_URI) is None


def test_historical_head_encodes_once_then_slices_exact_native_kv():
    tokenizer = CharTokenizer(["abcdefghijkl"])
    cfg = _config(
        tokenizer,
        max_seq_len=16,
        max_prompt_direct_tokens=4,
        prompt_position_mode="historical",
        chunking_mode="fixed",
        fixed_chunk_tokens=2,
        max_prompt_gists=None,
    )
    model = TinyPRAModel(cfg).eval()
    ids = torch.tensor([tokenizer.encode("abcdefghijkl")])

    prepared = prepare_prompt_batch_for_pra(model, tokenizer, ids)
    head = prepared.caches[0].get(IMPLICIT_PROMPT_HEAD_URI)
    expected = model._encode_reference_tokens(
        ids[0, :-4].tolist(),
        "cpu",
        detach=True,
        use_pra_memory=False,
    )

    assert head.metadata["historical_encoding"] is True
    assert prepared.position_offsets.tolist() == [8]
    for layer_id, layer_kv in expected.items():
        actual_k = torch.cat(
            [chunk.token_kv.k for chunk in head.layer_memory[layer_id].chunks], dim=2
        )
        actual_v = torch.cat(
            [chunk.token_kv.v for chunk in head.layer_memory[layer_id].chunks], dim=2
        )
        assert torch.equal(actual_k, layer_kv.k)
        assert torch.equal(actual_v, layer_kv.v)


def test_historical_head_plus_tail_matches_dense_full_forward():
    torch.manual_seed(41)
    tokenizer = CharTokenizer(["abcdefghijkl"])
    source_cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
        model_variant="td_sa",
        dropout=0.0,
    )
    source = TinyPRAModel(source_cfg).eval()
    target_cfg = PRAConfig(
        **{
            **source_cfg.__dict__,
            "model_variant": "td_pra",
            "max_prompt_direct_tokens": 4,
            "prompt_overflow_mode": "implicit_reference",
            "prompt_position_mode": "historical",
            "chunking_mode": "fixed",
            "fixed_chunk_tokens": 2,
            "max_prompt_gists": None,
            "top_k_references": 1,
            "top_k_chunks_per_reference": 8,
            "trigger_threshold": float("-inf"),
        }
    )
    converted = convert_sa_model_to_pra(source, target_cfg).eval()
    ids = torch.tensor([tokenizer.encode("abcdefghijkl")])
    prepared = prepare_prompt_batch_for_pra(converted, tokenizer, ids)
    converted.set_pra_cache(prepared.caches[0])

    with torch.no_grad():
        expected = source(ids)[:, -4:]
        actual = converted(
            prepared.input_ids,
            attention_mask=prepared.attention_mask,
            position_offset=prepared.position_offsets,
        )

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_mixed_historical_rows_conserve_tokens_and_continue_positions():
    tokenizer = CharTokenizer(["abcdefghijklmnop"])
    cfg = _config(
        tokenizer,
        max_seq_len=20,
        prompt_position_mode="historical",
    )
    model = TinyPRAModel(cfg).eval()
    rows = [list(range(3)), list(range(10)), list(range(16))]
    input_ids = torch.zeros((3, 16), dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row)
        mask[row_index, : len(row)] = 1

    prepared = prepare_prompt_batch_for_pra(
        model, tokenizer, input_ids, attention_mask=mask
    )

    assert prepared.position_offsets.tolist() == [0, 6, 12]
    for original, split in zip(rows, prepared.splits):
        assert [*split.implicit_ids, *split.direct_ids] == original
    model.set_pra_cache(PRABatchedMemoryCache(prepared.caches))
    logits = model(
        prepared.input_ids,
        attention_mask=prepared.attention_mask,
        position_offset=prepared.position_offsets,
    )
    assert torch.isfinite(logits).all()
