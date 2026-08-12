"""Offline correctness gates for the thin Hugging Face Gemma 3 PRA adapter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

from experiments.paper2_hf.gemma.run_gemma3_1b import validate_loaded_model
from pra_hf import PRAConfig, PRAForCausalLM
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    Gemma3PRAAttentionAdapter,
    PRAHFConfig,
    gemma3_global_layer_ids,
    inject_pra,
)


class TinyTokenizer:
    """Deterministic tokenizer for the offline product API test."""

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [2 + (ord(char) % 61) for char in text]
        if add_special_tokens:
            values.insert(0, 1)
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)


def _tiny_gemma3() -> Gemma3ForCausalLM:
    config = Gemma3TextConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=16,
        sliding_window_pattern=3,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Gemma3ForCausalLM(config).eval()


def _internal_config(*, layers=(2, 5)) -> PRAHFConfig:
    return PRAHFConfig(
        layer_ids=layers,
        model_max_context_tokens=64,
        max_prompt_direct_tokens=16,
        encoding_block_tokens=16,
        routing_chunk_tokens=4,
        routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
        max_materialized_memory_tokens=16,
        context_safety_reserve_tokens=0,
        top_k_references=1,
        top_k_chunks_per_reference=2,
        trigger_threshold=float("-inf"),
        kv_cache_residency="cpu",
        collect_detailed_timing=False,
    )


def _product_config() -> PRAConfig:
    return PRAConfig(
        chunk_tokens=4,
        selected_fraction=0.5,
        max_direct_context=16,
        native_operation_limit=64,
        max_materialized_tokens=16,
        context_safety_reserve_tokens=0,
        encoding_block_tokens=16,
    )


def test_gemma3_disabled_adapter_has_exact_native_parity_and_preserves_local_layers():
    torch.manual_seed(401)
    original = _tiny_gemma3()
    wrapped_model = copy.deepcopy(original)
    local_classes = {
        layer_id: type(layer.self_attn)
        for layer_id, layer in enumerate(wrapped_model.model.layers)
        if layer.attention_type == "sliding_attention"
    }
    handle = inject_pra(wrapped_model, _internal_config())
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
    expected_cache = expected.past_key_values.to_legacy_cache()
    actual_cache = actual.past_key_values.to_legacy_cache()
    assert len(actual_cache) == len(expected_cache)
    assert all(
        torch.equal(expected_tensor, actual_tensor)
        for expected_layer, actual_layer in zip(expected_cache, actual_cache)
        for expected_tensor, actual_tensor in zip(expected_layer, actual_layer)
    )
    assert tuple(handle.adapters) == (2, 5)
    assert all(isinstance(adapter, Gemma3PRAAttentionAdapter) for adapter in handle.adapters.values())
    assert all(
        type(handle.model.model.layers[layer_id].self_attn) is expected_class
        for layer_id, expected_class in local_classes.items()
    )


def test_gemma3_rejects_sliding_layers_and_product_defaults_choose_last_global():
    model = _tiny_gemma3()
    assert gemma3_global_layer_ids(model.config) == (2, 5)
    with pytest.raises(ValueError, match="sliding-attention layers unchanged"):
        inject_pra(model, _internal_config(layers=(0,)))

    routing, consumption = _product_config().resolved_layers(_tiny_gemma3().config)
    assert routing == 5
    assert consumption == (2, 5)
    with pytest.raises(ValueError, match="sliding attention"):
        PRAConfig(routing_layer=0, consumption_layers=(2,)).resolved_layers(
            _tiny_gemma3().config
        )


def test_gemma3_adapter_uses_native_global_rope_and_qk_norms_exactly():
    torch.manual_seed(403)
    model = _tiny_gemma3()
    handle = inject_pra(model, _internal_config())
    adapter = handle.adapters[5]
    attention = adapter.original_attention
    hidden = torch.randn(1, 4, model.config.hidden_size)
    positions = torch.arange(4).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, positions)

    query, key, value = adapter.project_qkv(hidden)
    actual_query, actual_key = adapter.apply_native_position_encoding(
        query, key, position_embeddings
    )
    shape = (1, 4, -1, attention.head_dim)
    expected_query = attention.q_norm(
        attention.q_proj(hidden).view(shape).transpose(1, 2)
    )
    expected_key = attention.k_norm(
        attention.k_proj(hidden).view(shape).transpose(1, 2)
    )
    expected_value = attention.v_proj(hidden).view(shape).transpose(1, 2)
    expected_query, expected_key = apply_rotary_pos_emb(
        expected_query, expected_key, *position_embeddings
    )

    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(value, expected_value)


def test_gemma3_reference_uses_native_mqa_post_rope_kv_and_global_memory():
    torch.manual_seed(402)
    handle = inject_pra(_tiny_gemma3(), _internal_config())
    entry = handle.add_reference(
        "memory://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )

    for layer_id in (2, 5):
        chunks = entry.layer_memory[layer_id].chunks
        assert len(chunks) == 2
        assert all(chunk.token_kv.k.shape == (1, 1, 4, 8) for chunk in chunks)
        assert all(chunk.token_kv.v.shape == (1, 1, 4, 8) for chunk in chunks)
        assert all(chunk.token_kv.k.device.type == "cpu" for chunk in chunks)
        assert all(chunk.token_kv.position_state == "post_position" for chunk in chunks)
        assert [chunk.token_kv.position_ids.tolist() for chunk in chunks] == [
            [[0, 1, 2, 3]],
            [[4, 5, 6, 7]],
        ]

    handle.set_memory_enabled(True)
    with torch.no_grad():
        output = handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    diagnostics = handle.diagnostics_by_layer()
    assert output.logits.shape == (1, 4, 67)
    assert set(diagnostics) == {2, 5}
    assert all(row["hf_query_heads"] == 4 for row in diagnostics.values())
    assert all(row["hf_native_kv_heads"] == 1 for row in diagnostics.values())
    assert all(row["retrieved_physical_kv_tokens"] == 8 for row in diagnostics.values())


def test_gemma3_long_prompt_and_public_api_stay_bounded():
    pra = PRAForCausalLM.from_model(
        _tiny_gemma3(), TinyTokenizer(), pra_config=_product_config()
    )
    assert pra.routing_layer == 5
    assert pra.consumption_layers == (2, 5)
    assert pra.stats()["family"] == "gemma3"

    reference = pra.add_reference("abcdefgh")
    result = pra.generate("question" * 4, max_new_tokens=2, return_details=True)
    stats = pra.stats()

    assert reference.chunks == 2
    assert result.generated_tokens == 2
    assert result.stats["head_tokens"] > 0
    assert stats["max_native_operation_tokens"] <= 64
    assert stats["native_limit_violations"] == 0
    assert stats["routing_index_bytes"] > 0
    assert stats["resident_detail_kv_bytes"] > stats["routing_index_bytes"]


def test_gemma3_official_runner_contract_on_structurally_faithful_model():
    torch.manual_seed(404)
    model = _tiny_gemma3()
    ids = torch.tensor([[1, 4, 8, 12, 16, 20]])
    reference = torch.arange(2, 34).unsqueeze(0) % model.config.vocab_size
    long_prompt = torch.arange(2, 42).unsqueeze(0) % model.config.vocab_size

    handle, report = validate_loaded_model(
        model,
        ids,
        reference,
        long_prompt,
        native_limit=64,
        direct_limit=16,
        encoding_block_tokens=16,
    )

    assert report["architecture_audit"]["global_attention_layers"] == [2, 5]
    assert report["architecture_audit"]["local_attention_layers"] == [0, 1, 3, 4]
    assert all(report["disabled_parity"].values())
    assert report["pra_placement"]["consumption_layers"] == [2, 5]
    assert report["native_reference"]["detail_k_shape"][1:] == [1, 16, 8]
    assert report["native_reference"]["permanently_expanded_to_query_heads"] is False
    assert report["long_prompt"]["head_reference_count"] == 1
    assert report["long_prompt"]["native_limit_violations"] == 0
    assert handle.max_native_operation_tokens <= 64
