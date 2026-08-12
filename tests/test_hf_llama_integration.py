"""Offline correctness gates for the thin Hugging Face Llama PRA adapter."""

from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.paper2_hf.llama.run_llama32_1b import validate_loaded_model
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


def _tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def _tiny_llama32_contract() -> LlamaForCausalLM:
    """Use Llama 3.2's scaled-RoPE contract at an offline-testable width."""
    config = LlamaConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=500_000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 4.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 2.0,
            "original_max_position_embeddings": 32,
        },
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def _config() -> PRAHFConfig:
    return PRAHFConfig(
        layer_ids=(-1,),
        model_max_context_tokens=64,
        max_prompt_direct_tokens=16,
        encoding_block_tokens=16,
        routing_chunk_tokens=4,
        routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
        max_materialized_memory_tokens=16,
        top_k_references=1,
        top_k_chunks_per_reference=2,
        trigger_threshold=float("-inf"),
        kv_cache_residency="cpu",
        collect_detailed_timing=False,
    )


def test_llama_disabled_adapter_has_exact_native_parity():
    torch.manual_seed(201)
    original = _tiny_llama()
    handle = inject_pra(copy.deepcopy(original), _config())
    ids = torch.tensor([[1, 7, 9, 3, 5]])

    with torch.no_grad():
        expected = original(ids, output_hidden_states=True, use_cache=True)
        actual = handle.model(ids, output_hidden_states=True, use_cache=True)
        expected_generation = original.generate(ids, max_new_tokens=3, do_sample=False)
        actual_generation = handle.model.generate(ids, max_new_tokens=3, do_sample=False)

    assert torch.equal(actual.logits, expected.logits)
    assert all(
        torch.equal(left, right)
        for left, right in zip(actual.hidden_states, expected.hidden_states)
    )
    assert torch.equal(actual_generation, expected_generation)
    assert actual.past_key_values.get_seq_length() == expected.past_key_values.get_seq_length()


def test_llama_reference_uses_native_gqa_post_rope_kv_and_memory():
    torch.manual_seed(202)
    handle = inject_pra(_tiny_llama(), _config())
    entry = handle.add_reference(
        "memory://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    chunks = entry.layer_memory[1].chunks

    assert len(chunks) == 2
    assert all(chunk.token_kv.k.shape == (1, 2, 4, 8) for chunk in chunks)
    assert all(chunk.token_kv.v.shape == (1, 2, 4, 8) for chunk in chunks)
    assert all(chunk.token_kv.k.device.type == "cpu" for chunk in chunks)
    assert all(chunk.token_kv.position_state == "post_position" for chunk in chunks)

    handle.set_memory_enabled(True)
    with torch.no_grad():
        output = handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    diagnostics = handle.diagnostics_by_layer()[1]
    assert output.logits.shape == (1, 4, 67)
    assert diagnostics["hf_query_heads"] == 4
    assert diagnostics["hf_native_kv_heads"] == 2
    assert diagnostics["retrieved_physical_kv_tokens"] == 8


def test_llama_long_prompt_rollover_stays_bounded():
    handle = inject_pra(_tiny_llama(), _config())
    prepared = handle.prepare_long_prompt(torch.arange(1, 25).unsqueeze(0))

    assert prepared.head_tokens == 8
    assert prepared.input_ids.shape == (1, 16)
    assert prepared.position_ids.tolist() == [list(range(8, 24))]
    head = next(entry for entry in handle.cache.all_entries() if entry.uri == "#__head")
    assert head.metadata["source_tokens"] == 8
    assert handle.max_native_operation_tokens <= 64


def test_llama32_checkpoint_gate_covers_parity_gqa_rope_and_head_rollover():
    torch.manual_seed(203)
    _, report = validate_loaded_model(
        _tiny_llama32_contract(),
        torch.tensor([[1, 7, 9, 3, 5]]),
        torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18] * 3]),
        torch.tensor([[1, *([21, 22, 23, 24] * 12)]]),
        native_limit=64,
        direct_limit=16,
        encoding_block_tokens=16,
    )

    assert all(report["disabled_parity"].values())
    assert report["model_contract"]["query_heads"] == 4
    assert report["model_contract"]["native_kv_heads"] == 2
    assert report["native_reference"]["position_state"] == "post_position"
    assert report["native_reference"]["positions_exact"] is True
    assert report["native_reference"]["permanently_expanded_to_query_heads"] is False
    assert report["long_prompt"]["head_reference_count"] == 1
    assert report["long_prompt"]["head_tokens"] > 0
    assert report["enabled_path"]["native_limit_violations"] == 0
    assert report["enabled_path"]["causal_prefix_exact"] is True
