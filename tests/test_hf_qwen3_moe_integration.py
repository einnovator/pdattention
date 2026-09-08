"""Offline correctness gates for the attention-only Qwen3 MoE PRA adapter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pra_hf import PRAConfig, PRAForCausalLM
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    Qwen3MoePRAAttentionAdapter,
    describe_moe_topology,
    inject_pra,
)


class TinyTokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [2 + (ord(char) % 61) for char in text]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)


def _tiny_qwen3_moe() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(config).eval()


def _config() -> PRAHFConfig:
    return PRAHFConfig(
        layer_ids=(-1,),
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


def test_qwen3_moe_disabled_path_is_exact_and_experts_remain_host_owned():
    torch.manual_seed(601)
    original = _tiny_qwen3_moe()
    adapted = copy.deepcopy(original)
    mlps = tuple(layer.mlp for layer in adapted.model.layers)
    gates = tuple(layer.mlp.gate for layer in adapted.model.layers)
    experts = tuple(layer.mlp.experts for layer in adapted.model.layers)
    handle = inject_pra(adapted, _config())
    ids = torch.tensor([[1, 7, 9, 3, 5]])

    with torch.no_grad():
        expected = original(ids, output_hidden_states=True, use_cache=True)
        actual = handle.model(ids, output_hidden_states=True, use_cache=True)
        expected_generation = original.generate(ids, max_new_tokens=3, do_sample=False)
        actual_generation = handle.model.generate(ids, max_new_tokens=3, do_sample=False)

    assert torch.equal(actual.logits, expected.logits)
    assert all(torch.equal(left, right) for left, right in zip(actual.hidden_states, expected.hidden_states))
    assert torch.equal(actual_generation, expected_generation)
    assert tuple(layer.mlp for layer in handle.model.model.layers) == mlps
    assert tuple(layer.mlp.gate for layer in handle.model.model.layers) == gates
    assert tuple(layer.mlp.experts for layer in handle.model.model.layers) == experts
    assert isinstance(handle.adapters[1], Qwen3MoePRAAttentionAdapter)


def test_qwen3_moe_contract_describes_sparse_layers_and_host_ownership():
    topology = describe_moe_topology(_tiny_qwen3_moe())

    assert topology is not None
    assert topology.model_type == "qwen3_moe"
    assert topology.moe_layer_ids == (0, 1)
    assert topology.dense_layer_ids == ()
    assert topology.expert_count == 4
    assert topology.experts_per_token == 2
    assert topology.router_ownership == "host_model"
    assert topology.expert_execution_ownership == "host_model"
    assert topology.pra_scope == "attention_only"


def test_qwen3_moe_reference_keeps_native_gqa_and_uses_memory():
    torch.manual_seed(602)
    handle = inject_pra(_tiny_qwen3_moe(), _config())
    entry = handle.add_reference(
        "memory://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    chunks = entry.layer_memory[1].chunks

    assert len(chunks) == 2
    assert all(chunk.token_kv.k.shape == (1, 2, 4, 8) for chunk in chunks)
    assert all(chunk.token_kv.v.shape == (1, 2, 4, 8) for chunk in chunks)
    handle.set_memory_enabled(True)
    with torch.no_grad():
        output = handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    diagnostics = handle.diagnostics_by_layer()[1]
    assert output.logits.shape == (1, 4, 67)
    assert diagnostics["hf_query_heads"] == 4
    assert diagnostics["hf_native_kv_heads"] == 2
    assert diagnostics["retrieved_physical_kv_tokens"] == 8


def test_qwen3_moe_public_api_reports_topology():
    pra = PRAForCausalLM.from_model(
        _tiny_qwen3_moe(),
        TinyTokenizer(),
        pra_config=PRAConfig(
            routing_layer=-1,
            consumption_layers=(-1,),
            chunk_tokens=4,
            selected_fraction=0.5,
            max_direct_context=16,
            native_operation_limit=64,
            max_materialized_tokens=16,
            context_safety_reserve_tokens=0,
            encoding_block_tokens=16,
        ),
    )

    stats = pra.stats()
    assert stats["family"] == "qwen3_moe"
    assert stats["moe_topology"]["expert_count"] == 4
    assert stats["moe_topology"]["pra_scope"] == "attention_only"
