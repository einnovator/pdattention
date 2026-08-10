import warnings

import pytest
import torch

from pra_torch.attention import PRAttention
from pra_torch.config import PRAConfig
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
)
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra
from pra_torch.data import CharTokenizer
from pra_torch.pra_train import _complete_reference_rank_metrics, native_kv_gap_metrics
from pra_torch.native_metrics import recovered_context_benefit


def _config(**overrides):
    values = {
        "vocab_size": 31,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 32,
        "max_seq_len": 16,
        "dropout": 0.0,
        "trigger_threshold": -1.0,
    }
    values.update(overrides)
    return PRAConfig(**values)


def test_native_kv_is_default_and_has_no_transport_parameters():
    attention = PRAttention(
        16, 4, 16, 0, PRASimpleMemoryCache(), config=_config(n_layers=1)
    )

    assert attention.memory_transport == "native_kv"
    assert attention.mem_o_proj is None
    assert not any("mem_o_proj" in name for name, _parameter in attention.named_parameters())


def test_historical_native_kv_matches_full_causal_attention_tail():
    torch.manual_seed(11)
    attention = PRAttention(
        16, 4, 16, 0, PRASimpleMemoryCache(), config=_config(n_layers=1)
    ).eval()
    hidden = torch.randn(2, 7, 16)
    prefix_length = 4

    full = attention(hidden, use_pra_memory=False)[:, prefix_length:]
    prefix = hidden[:, :prefix_length]
    memory_k = attention.split_heads(attention.k_proj(prefix))
    memory_v = attention.split_heads(attention.v_proj(prefix))
    tail = attention.forward_native_kv(
        hidden[:, prefix_length:],
        [memory_k[row : row + 1] for row in range(hidden.shape[0])],
        [memory_v[row : row + 1] for row in range(hidden.shape[0])],
    )

    assert torch.allclose(tail, full, atol=1e-6, rtol=1e-6)


def test_sa_checkpoint_conversion_preserves_disabled_memory_logits():
    torch.manual_seed(23)
    source = TinyPRAModel(_config(model_variant="td_sa")).eval()
    target_config = _config(model_variant="td_pra", memory_transport="native_kv")

    converted = convert_sa_model_to_pra(source, target_config).eval()
    input_ids = torch.randint(0, source.cfg.vocab_size, (3, 9))

    with torch.no_grad():
        expected = source(input_ids, use_pra_memory=False)
        actual = converted(input_ids, use_pra_memory=False)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_legacy_cross_attention_flag_preserves_adapted_transport():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = _config(n_layers=1, use_cross_attention_memory=True)
    attention = PRAttention(16, 4, 16, 0, PRASimpleMemoryCache(), config=config)

    assert config.memory_transport == "cross_attention"
    assert attention.mem_o_proj is not None
    assert any("mem_o_proj" in name for name, _parameter in attention.named_parameters())


def test_invalid_transport_is_rejected():
    with pytest.raises(ValueError, match="memory_transport"):
        _config(memory_transport="new_branch")


def test_native_attention_reports_active_kv_and_transfer_volume():
    cache = PRASimpleMemoryCache()
    key = torch.randn(1, 4, 3, 4)
    value = torch.randn_like(key)
    gist = torch.randn(16)
    cache.put(
        PRACacheEntry(
            uri="mem://history",
            text="three historical positions",
            layer_memory={
                0: LayerReferenceMemory(
                    chunks=[
                        ReferenceChunkMemory(
                            chunk_id="history-0",
                            source_uri="mem://history",
                            token_start=0,
                            token_end=3,
                            token_kv=LayerKV(k=key, v=value),
                            routing_gist=ChunkRoutingGist(k=gist),
                        )
                    ]
                )
            },
        )
    )
    attention = PRAttention(16, 4, 16, 0, cache, config=_config(n_layers=1)).eval()

    output = attention(torch.randn(1, 5, 16))
    diagnostics = attention.last_diagnostics

    assert output.shape == (1, 5, 16)
    assert diagnostics["active_local_tokens"] == 5.0
    assert diagnostics["retrieved_token_kv"] == 3.0
    assert diagnostics["accessible_kv_tokens"] == 8.0
    assert diagnostics["active_memory_fraction"] == pytest.approx(3 / 8)
    assert diagnostics["retrieved_kv_storage_bytes"] == 2 * 3 * 16 * 4
    assert diagnostics["retrieved_kv_transfer_bytes"] == 0.0


def test_native_gap_metrics_keep_transport_and_retrieval_effects_separate():
    metrics = native_kv_gap_metrics(
        [
            {"condition": "sa_full", "loss": 2.0},
            {"condition": "sa_tail", "loss": 3.0},
            {"condition": "native_all", "loss": 2.1},
            {"condition": "native_oracle", "loss": 2.2},
            {"condition": "valid", "loss": 2.3},
            {"condition": "native_shuffled", "loss": 2.6},
            {"condition": "native_disabled", "loss": 3.0},
        ]
    )

    assert metrics["transport_gap"] == pytest.approx(0.1)
    assert metrics["sparse_gap"] == pytest.approx(0.1)
    assert metrics["routing_gap"] == pytest.approx(0.1)
    assert metrics["memory_benefit_oracle"] == pytest.approx(0.8)
    assert metrics["content_causality_oracle"] == pytest.approx(0.4)
    assert metrics["dependency_gain"] == pytest.approx(1.0)
    assert metrics["rcb_oracle"] == pytest.approx(0.8)


def test_recovered_context_benefit_marks_low_dependency_targets_undefined():
    assert recovered_context_benefit(
        sa_full_loss=1.0, sa_tail_loss=1.0, pra_loss=0.9
    ) is None
    assert recovered_context_benefit(
        sa_full_loss=1.0, sa_tail_loss=2.0, pra_loss=0.5
    ) == pytest.approx(1.5)


def test_complete_rank_metrics_capture_multi_target_coverage():
    rankings = {
        0: [
            {"reference_uri": "noise", "reference_score": 0.9, "chunks": []},
            {"reference_uri": "target-a", "reference_score": 0.8, "chunks": []},
            {"reference_uri": "target-b", "reference_score": 0.7, "chunks": []},
        ],
        1: [
            {"reference_uri": "target-b", "reference_score": 0.95, "chunks": []},
            {"reference_uri": "noise", "reference_score": 0.6, "chunks": []},
            {"reference_uri": "target-a", "reference_score": 0.5, "chunks": []},
        ],
    }

    metrics = _complete_reference_rank_metrics(rankings, {"target-a", "target-b"})

    assert metrics["routing_mrr"] == pytest.approx(0.75)
    assert metrics["any_target_hit_at_1"] == 1.0
    assert metrics["all_targets_hit_at_1"] == 0.0
    assert metrics["all_targets_hit_at_2"] == 1.0
    assert metrics["fraction_targets_covered_at_2"] == 1.0


def test_native_reference_slicing_matches_one_full_historical_encode():
    tokenizer = CharTokenizer(["abcdefgh"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
        model_variant="td_pra",
        reference_encoding_strategy="native_slice",
    )
    model = TinyPRAModel(cfg).eval()
    references = [
        {"uri": "mem://a", "text": "abcd", "metadata": {}},
        {"uri": "mem://b", "text": "efgh", "metadata": {}},
    ]

    entries = model.encode_reference_group_to_cache(references, tokenizer, "cpu")
    assert [
        (
            entry.layer_memory[0].chunks[0].logical_start,
            entry.layer_memory[0].chunks[0].logical_end,
        )
        for entry in entries
    ] == [(0, 4), (4, 8)]
    full = model._encode_reference_tokens(
        tokenizer.encode("abcdefgh"),
        "cpu",
        detach=True,
        use_pra_memory=False,
    )

    for layer_id, expected in full.items():
        actual_k = torch.cat(
            [entry.layer_memory[layer_id].chunks[0].token_kv.k for entry in entries],
            dim=2,
        )
        actual_v = torch.cat(
            [entry.layer_memory[layer_id].chunks[0].token_kv.v for entry in entries],
            dim=2,
        )
        assert torch.equal(actual_k, expected.k)
        assert torch.equal(actual_v, expected.v)


def test_native_historical_prompt_positions_match_full_model_tail_logits():
    torch.manual_seed(29)
    tokenizer = CharTokenizer(["abcdefgh"])
    source_cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
        model_variant="td_sa",
    )
    source = TinyPRAModel(source_cfg).eval()
    target_cfg = PRAConfig(
        **{
            **source_cfg.__dict__,
            "model_variant": "td_pra",
            "reference_encoding_strategy": "native_slice",
            "reference_position_mode": "global",
            "prompt_position_mode": "historical",
            "top_k_references": 8,
            "top_k_chunks_per_reference": 1,
            "trigger_threshold": float("-inf"),
            "detail_materialization": "full_reference",
        }
    )
    converted = convert_sa_model_to_pra(source, target_cfg).eval()
    references = [
        {"uri": "mem://a", "text": "abcd", "metadata": {}},
        {"uri": "mem://b", "text": "ef", "metadata": {}},
    ]
    cache = PRASimpleMemoryCache()
    for entry in converted.encode_reference_group_to_cache(references, tokenizer, "cpu"):
        cache.put(entry)
    converted.set_pra_cache(cache)

    full_ids = torch.tensor([tokenizer.encode("abcdefgh")])
    tail_ids = torch.tensor([tokenizer.encode("gh")])
    with torch.no_grad():
        full_tail = source(full_ids)[:, -tail_ids.shape[1] :]
        native_tail = converted(tail_ids, position_offset=6)

    assert torch.allclose(native_tail, full_tail, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA residency parity test")
def test_cpu_resident_native_kv_matches_gpu_and_transfers_only_selected_chunks():
    torch.manual_seed(43)
    tokenizer = CharTokenizer(["abcdefghijklmnop", "ijkl"])
    source_cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=32,
        model_variant="td_sa",
    )
    source = TinyPRAModel(source_cfg).to("cuda").eval()
    base = {
        **source_cfg.__dict__,
        "model_variant": "td_pra",
        "chunking_mode": "fixed",
        "fixed_chunk_tokens": 4,
        "top_k_references": 1,
        "top_k_chunks_per_reference": 1,
        "trigger_threshold": float("-inf"),
        "collect_detailed_timing": True,
    }
    gpu_model = convert_sa_model_to_pra(source, PRAConfig(**base)).to("cuda").eval()
    cpu_model = convert_sa_model_to_pra(
        source, PRAConfig(**{**base, "kv_cache_residency": "cpu", "kv_cache_pin_memory": True})
    ).to("cuda").eval()

    for model in (gpu_model, cpu_model):
        entry = model.encode_reference_to_cache(
            "mem://history", "abcdefghijklmnop", tokenizer, "cuda"
        )
        cache = PRASimpleMemoryCache()
        cache.put(entry)
        model.set_pra_cache(cache)

    query_ids = torch.tensor([tokenizer.encode("ijkl")], device="cuda")
    with torch.no_grad():
        gpu_logits = gpu_model(query_ids)
        cpu_logits = cpu_model(query_ids)

    assert torch.allclose(cpu_logits, gpu_logits, atol=2e-6, rtol=2e-6)
    gpu_selected = gpu_model.selected_chunks_by_layer()
    cpu_selected = cpu_model.selected_chunks_by_layer()
    assert {
        layer: [[hit.chunk_id for hit in row] for row in rows]
        for layer, rows in cpu_selected.items()
    } == {
        layer: [[hit.chunk_id for hit in row] for row in rows]
        for layer, rows in gpu_selected.items()
    }
    for entry in cpu_model.pra_cache.all_entries():
        for memory in entry.layer_memory.values():
            assert all(chunk.token_kv.k.device.type == "cpu" for chunk in memory.chunks)
            selected_ids = {
                hit.chunk_id for rows in cpu_selected.values() for row in rows for hit in row
            }
            assert any(chunk.chunk_id not in selected_ids for chunk in memory.chunks)
    cpu_diagnostics = cpu_model.pra_diagnostics_by_layer()
    gpu_diagnostics = gpu_model.pra_diagnostics_by_layer()
    assert all(row["retrieved_kv_transfer_bytes"] > 0 for row in cpu_diagnostics.values())
    assert all(row["retrieved_kv_transfer_bytes"] == 0 for row in gpu_diagnostics.values())


def test_block_slicing_accounts_for_encoding_overlap_without_storing_duplicates():
    tokenizer = CharTokenizer(["abcdefgh"])
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        model_variant="td_pra",
        reference_encoding_strategy="block_slice",
        encoding_block_references=2,
        encoding_overlap_fraction=0.25,
    )
    model = TinyPRAModel(cfg).eval()
    references = [
        {"uri": f"mem://{index}", "text": text, "metadata": {}}
        for index, text in enumerate(("ab", "cd", "ef", "gh"))
    ]

    entries = model.encode_reference_group_to_cache(references, tokenizer, "cpu")
    metadata = entries[0].metadata

    assert metadata["encoding_run_unique_source_tokens"] == 8
    assert metadata["encoding_run_encoded_tokens_including_overlap"] == 9
    assert metadata["encoding_run_stored_kv_tokens"] == 8
    assert metadata["encoding_run_duplication_factor"] == pytest.approx(9 / 8)
    assert metadata["max_encoding_input_tokens"] == 5
    assert sum(
        entry.layer_memory[0].chunks[0].token_count for entry in entries
    ) == 8
    assert [
        (
            entry.layer_memory[0].chunks[0].logical_start,
            entry.layer_memory[0].chunks[0].logical_end,
        )
        for entry in entries
    ] == [(0, 2), (2, 4), (4, 6), (6, 8)]
