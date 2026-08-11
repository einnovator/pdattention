"""Offline correctness gates for the thin Hugging Face Qwen PRA adapter."""

from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM

from pra_torch.hf import PRAHFConfig, inject_pra
from pra_torch.memory_batching import native_kv_attention


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def _hf_config(**overrides) -> PRAHFConfig:
    values = {
        "layer_ids": (-1,),
        "model_max_context_tokens": 64,
        "max_prompt_direct_tokens": 16,
        "encoding_block_tokens": 16,
        "routing_chunk_tokens": 4,
        "max_materialized_memory_tokens": 16,
        "top_k_references": 1,
        "top_k_chunks_per_reference": 2,
        "trigger_threshold": float("-inf"),
        "kv_cache_residency": "cpu",
        "collect_detailed_timing": False,
    }
    values.update(overrides)
    return PRAHFConfig(**values)


def test_qwen_disabled_adapter_has_exact_logits_hidden_states_and_generation_parity():
    torch.manual_seed(101)
    original = _tiny_qwen()
    adapted_model = copy.deepcopy(original)
    handle = inject_pra(adapted_model, _hf_config())
    ids = torch.tensor([[1, 7, 9, 3, 5]])

    with torch.no_grad():
        expected = original(ids, output_hidden_states=True, use_cache=True)
        actual = handle.model(ids, output_hidden_states=True, use_cache=True)
        expected_generation = original.generate(ids, max_new_tokens=3, do_sample=False)
        actual_generation = handle.model.generate(ids, max_new_tokens=3, do_sample=False)

    assert torch.equal(actual.logits, expected.logits)
    assert all(torch.equal(left, right) for left, right in zip(actual.hidden_states, expected.hidden_states))
    assert torch.equal(actual_generation, expected_generation)
    assert actual.past_key_values.get_seq_length() == expected.past_key_values.get_seq_length()


def test_qwen_reference_capture_keeps_native_gqa_layout_and_uses_memory():
    torch.manual_seed(103)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    entry = handle.add_reference("mem://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]]))
    chunks = entry.layer_memory[1].chunks

    assert len(chunks) == 2
    assert all(chunk.token_kv.k.shape[1] == 2 for chunk in chunks)
    assert all(chunk.token_kv.k.device.type == "cpu" for chunk in chunks)
    assert [chunk.logical_start for chunk in chunks] == [0, 4]

    handle.set_memory_enabled(True)
    with torch.no_grad():
        output = handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    diagnostics = handle.diagnostics_by_layer()[1]
    assert output.logits.shape == (1, 4, 67)
    assert diagnostics["hf_query_heads"] == 4
    assert diagnostics["hf_native_kv_heads"] == 2
    assert diagnostics["retrieved_physical_kv_tokens"] == 8


def test_native_kv_attention_replays_gqa_without_expanding_stored_memory():
    torch.manual_seed(107)
    query = torch.randn(1, 4, 3, 8)
    local_key = torch.randn(1, 2, 3, 8)
    local_value = torch.randn(1, 2, 3, 8)
    memory_key = torch.randn(1, 2, 5, 8)
    memory_value = torch.randn(1, 2, 5, 8)

    actual, _ = native_kv_attention(
        query,
        local_key,
        local_value,
        [memory_key],
        [memory_value],
    )
    full_key = torch.cat((memory_key, local_key), dim=2).repeat_interleave(2, dim=1)
    full_value = torch.cat((memory_value, local_value), dim=2).repeat_interleave(2, dim=1)
    scores = query @ full_key.transpose(-2, -1) * (8**-0.5)
    visible = torch.cat((torch.ones(3, 5, dtype=torch.bool), torch.tril(torch.ones(3, 3, dtype=torch.bool))), dim=1)
    expected = torch.softmax(scores.masked_fill(~visible[None, None], float("-inf")), dim=-1) @ full_value

    assert memory_key.shape[1] == 2
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_qwen_implicit_head_preserves_offsets_and_native_bound():
    torch.manual_seed(109)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    prompt = torch.arange(1, 41).remainder(65).unsqueeze(0)
    prepared = handle.prepare_long_prompt(prompt)

    assert prepared.head_tokens == 24
    assert prepared.input_ids.shape == (1, 16)
    assert prepared.position_ids.tolist() == [list(range(24, 40))]
    assert handle.cache.has("#__head")
    assert handle.max_native_operation_tokens <= 16
    assert handle.native_limit_violations == 0

    handle.set_memory_enabled(True)
    with torch.no_grad():
        result = handle.model(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            position_ids=prepared.position_ids,
            use_cache=False,
        )
    assert result.logits.shape == (1, 16, 67)


def test_qwen_generation_cache_does_not_duplicate_local_history_with_memory():
    torch.manual_seed(113)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    handle.add_reference("mem://generation", torch.tensor([[21, 22, 23, 24]]))
    handle.set_memory_enabled(True)
    prompt = torch.tensor([[1, 3, 5, 7]])

    with torch.no_grad():
        generated = handle.model.generate(prompt, max_new_tokens=2, do_sample=False)

    assert generated.shape == (1, 6)
    diagnostics = handle.diagnostics_by_layer()[1]
    # The last decode step sees the four prompt tokens plus one generated token.
    assert diagnostics["hf_cache_tokens"] == 5
    assert diagnostics["retrieved_physical_kv_tokens"] == 4


def test_qwen_layer_selection_rejects_non_qwen_and_out_of_range_layers():
    with pytest.raises(ValueError, match="outside"):
        inject_pra(_tiny_qwen(), _hf_config(layer_ids=(5,)))

    llama_config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    llama_config._attn_implementation = "eager"
    llama = transformers.LlamaForCausalLM(llama_config)
    with pytest.raises(TypeError, match="Only Qwen"):
        inject_pra(llama, _hf_config(layer_ids=(0,), model_max_context_tokens=64))
