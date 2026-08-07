from unittest.mock import patch

import pytest
import torch

from data.collators import PRACollator
from data.schemas import QuestionSample, ReferenceSample
from data.tokenizer import PRATokenizer
from pra_torch.cache_services import build_cache_from_metadata
from pra_torch.config import CacheServiceConfig, PRAConfig, ResolverServiceConfig
from pra_torch.memory import PRABatchedMemoryCache
from pra_torch.model import TinyPRAModel
from pra_torch.pra_train import _pra_batch_step


def _sample(row: int, references: list[dict]) -> QuestionSample:
    refs = [
        ReferenceSample(
            id=index + 1,
            uri=value["uri"],
            summary=value.get("summary"),
            metadata={"text": value.get("text", ""), **value.get("metadata", {})},
        )
        for index, value in enumerate(references)
    ]
    tokens = " ".join(f"<REF_{ref.id}>" for ref in refs)
    return QuestionSample(
        id=f"row-{row}",
        question=f"row {row} {tokens}",
        answer=f"answer-{row}",
        references=refs,
        target_reference_ids=[ref.id for ref in refs],
        target_reference_uris=[ref.uri for ref in refs],
    )


def _collate(samples: list[QuestionSample], max_seq_len: int = 160):
    corpus = [sample.question + sample.answer for sample in samples]
    for sample in samples:
        for ref in sample.references:
            corpus.extend((ref.summary or "", str(ref.metadata.get("text", ""))))
    tokenizer = PRATokenizer(corpus, extra_tokens=["<REF_0>"])
    return tokenizer, PRACollator(tokenizer, max_seq_len=max_seq_len)(samples)


def _config(tokenizer, **overrides) -> PRAConfig:
    values = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "max_seq_len": 160,
        "dropout": 0.0,
        "top_k_references": 3,
        "top_k_chunks_per_reference": 4,
        "trigger_threshold": -1.0,
        "memory_alpha": 0.5,
        "memory_bucket_count": 2,
        "device": "cpu",
    }
    values.update(overrides)
    return PRAConfig(**values)


def _legacy_singleton_logits(model, batch, tokenizer):
    logits = []
    for row_index, metadata in enumerate(batch["metadata"]):
        build_cache_from_metadata(model, tokenizer, [metadata], "cpu")
        logits.append(model(batch["input_ids"][row_index : row_index + 1]))
    return torch.cat(logits, dim=0)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batched_prompt_matches_singletons_and_calls_forward_once(batch_size):
    torch.manual_seed(17)
    samples = [
        _sample(
            row,
            [{"uri": f"doc://{row}", "text": f"private memory for row {row}"}],
        )
        for row in range(batch_size)
    ]
    tokenizer, batch = _collate(samples)
    model = TinyPRAModel(_config(tokenizer)).eval()
    resolver = ResolverServiceConfig()
    cache = CacheServiceConfig()

    with torch.no_grad(), patch.object(model, "forward", wraps=model.forward) as singleton_forward:
        expected = _legacy_singleton_logits(model, batch, tokenizer)
    assert singleton_forward.call_count == batch_size

    with torch.no_grad(), patch.object(model, "forward", wraps=model.forward) as batched_forward:
        _loss, result = _pra_batch_step(model, batch, "cpu", tokenizer, resolver, cache)

    assert batched_forward.call_count == 1
    assert result["metrics"]["prompt_forward_calls"] == 1.0
    assert result["metrics"]["logical_batch_size"] == float(batch_size)
    assert torch.allclose(result["logits"], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("strategy", ["hierarchical", "reference_first", "global_chunks"])
def test_duplicate_uri_content_is_strictly_isolated_by_batch_row(strategy):
    samples = [
        _sample(
            0,
            [{"uri": "doc://same", "text": "answer is ALPHA", "summary": "alpha summary"}],
        ),
        _sample(
            1,
            [{"uri": "doc://same", "text": "answer is BETA", "summary": "beta summary"}],
        ),
    ]
    tokenizer, batch = _collate(samples)
    cfg = _config(
        tokenizer,
        search_strategy=strategy,
        reference_level_gist_mode="mean" if strategy == "reference_first" else None,
        use_summary=True,
        summary_mode="hybrid",
        top_k_references=1,
        top_k_chunks_per_reference=1,
    )
    model = TinyPRAModel(cfg).eval()

    with torch.no_grad():
        _loss, result = _pra_batch_step(
            model,
            batch,
            "cpu",
            tokenizer,
            ResolverServiceConfig(),
            CacheServiceConfig(),
        )

    assert isinstance(result["batch_cache"], PRABatchedMemoryCache)
    with pytest.raises(RuntimeError, match="no flat entries"):
        _ = result["batch_cache"].entries
    expected_text = ["answer is ALPHA", "answer is BETA"]
    for row_index, by_layer in enumerate(result["selections"]):
        selected = by_layer[0]
        assert selected
        assert all(hit.entry.text == expected_text[row_index] for hit in selected)


@pytest.mark.parametrize("materialization", ["selected_chunks", "full_reference", "gist_only"])
def test_materialization_modes_keep_one_variable_memory_batch(materialization):
    samples = [
        _sample(0, [{"uri": "doc://short", "text": "abcdefgh"}]),
        _sample(1, [{"uri": "doc://long", "text": "x" * 31}]),
    ]
    tokenizer, batch = _collate(samples)
    cfg = _config(
        tokenizer,
        chunking_mode="fixed",
        fixed_chunk_tokens=8,
        max_gists_per_reference=8,
        top_k_references=1,
        top_k_chunks_per_reference=1,
        detail_materialization=materialization,
    )
    model = TinyPRAModel(cfg).eval()

    from pra_torch.attention import dynamic_memory_attention as real_memory_attention

    with torch.no_grad(), patch(
        "pra_torch.attention.dynamic_memory_attention",
        wraps=real_memory_attention,
    ) as memory_attention:
        _loss, result = _pra_batch_step(
            model,
            batch,
            "cpu",
            tokenizer,
            ResolverServiceConfig(),
            CacheServiceConfig(),
        )

    assert memory_attention.call_count == 1
    q, memory_k, memory_v = memory_attention.call_args.args
    assert q.shape[0] == len(samples)
    assert len(memory_k) == len(memory_v) == len(samples)
    assert all(value.shape[:2] == (1, cfg.n_heads) for value in memory_k)
    assert result["diagnostics"][0]["batching"].selected_lengths == tuple(
        value.shape[2] for value in memory_k
    )


def test_empty_and_unequal_row_memories_share_one_attention_execution():
    samples = [
        _sample(0, []),
        _sample(1, [{"uri": "doc://8", "text": "a" * 8}]),
        _sample(
            2,
            [
                {"uri": "doc://ten-a", "text": "b" * 10},
                {"uri": "doc://ten-b", "text": "c" * 10},
                {"uri": "doc://ten-c", "text": "d" * 10},
            ],
        ),
        _sample(3, [{"uri": "doc://128", "text": "e" * 128}]),
    ]
    tokenizer, batch = _collate(samples)
    cfg = _config(
        tokenizer,
        d_model=8,
        n_heads=1,
        chunking_mode="fixed",
        fixed_chunk_tokens=32,
        max_gists_per_reference=4,
    )
    model = TinyPRAModel(cfg).eval()

    with torch.no_grad():
        _loss, result = _pra_batch_step(
            model,
            batch,
            "cpu",
            tokenizer,
            ResolverServiceConfig(),
            CacheServiceConfig(),
        )

    stats = result["diagnostics"][0]["batching"]
    assert stats.selected_lengths == (0, 8, 30, 128)
    assert len(result["selections"]) == 4
    assert result["selections"][0][0] == []
    assert result["metrics"]["prompt_forward_calls"] == 1.0


def test_overlapping_chunks_are_deduplicated_independently_per_row():
    samples = [
        _sample(0, [{"uri": "doc://overlap-12", "text": "a" * 12}]),
        _sample(1, [{"uri": "doc://overlap-20", "text": "b" * 20}]),
    ]
    tokenizer, batch = _collate(samples)
    cfg = _config(
        tokenizer,
        chunking_mode="fixed",
        fixed_chunk_tokens=8,
        fixed_chunk_overlap_tokens=4,
        max_gists_per_reference=4,
        top_k_references=1,
        top_k_chunks_per_reference=4,
    )
    model = TinyPRAModel(cfg).eval()

    with torch.no_grad():
        _loss, result = _pra_batch_step(
            model,
            batch,
            "cpu",
            tokenizer,
            ResolverServiceConfig(),
            CacheServiceConfig(),
        )

    diagnostic = result["diagnostics"][0]
    assert diagnostic["batching"].selected_lengths == (12, 20)
    assert diagnostic["memory_duplicate_chunk_tokens"] == 16.0


def test_recursive_children_and_summaries_remain_row_local_with_duplicate_uris():
    samples = []
    for row, private in enumerate(("ALPHA child", "BETA child")):
        samples.append(
            _sample(
                row,
                [
                    {
                        "uri": "doc://root",
                        "text": "parent <REF_1>",
                        "summary": f"summary {private}",
                        "metadata": {
                            "reference_table": {"<REF_1>": "doc://child"},
                            "documents": {"doc://child": {"text": private}},
                        },
                    }
                ],
            )
        )
    tokenizer, batch = _collate(samples)
    cfg = _config(
        tokenizer,
        recursive_refs_enabled=True,
        recursive_max_depth=2,
        use_summary=True,
        summary_mode="replace",
        top_k_references=2,
    )
    model = TinyPRAModel(cfg).eval()

    with torch.no_grad():
        _loss, result = _pra_batch_step(
            model,
            batch,
            "cpu",
            tokenizer,
            ResolverServiceConfig(),
            CacheServiceConfig(),
        )

    assert result["caches"][0].get("doc://child").text == "ALPHA child"
    assert result["caches"][1].get("doc://child").text == "BETA child"
    assert all(cache.get("doc://root") is not None for cache in result["caches"])


@pytest.mark.parametrize("cache_build_mode", ["detached", "trainable_gist"])
def test_batched_training_step_has_finite_gradients(cache_build_mode):
    samples = [
        _sample(row, [{"uri": f"doc://{row}", "text": f"training memory {row}"}])
        for row in range(2)
    ]
    tokenizer, batch = _collate(samples)
    model = TinyPRAModel(_config(tokenizer, cache_build_mode=cache_build_mode)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with patch.object(model, "forward", wraps=model.forward) as model_forward:
        loss, result = _pra_batch_step(
            model,
            batch,
            "cpu",
            tokenizer,
            ResolverServiceConfig(),
            CacheServiceConfig(),
        )
    assert model_forward.call_count == 1
    assert torch.isfinite(loss)

    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    assert result["metrics"]["logical_batch_size"] == 2.0
