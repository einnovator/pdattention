"""Offline family gates for Mistral 3, GPT-OSS, and Qwen3 hybrid models."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import (
    GptOssConfig,
    GptOssForCausalLM,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
)

from pra_hf import PRAConfig, PRAForCausalLM
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    GptOssPRAAttentionAdapter,
    MistralPRAAttentionAdapter,
    PRAHFConfig,
    describe_moe_topology,
    describe_qwen3_hybrid_plan,
    inject_pra,
    resolve_decoder_anatomy,
)


class TinyTokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [2 + (ord(char) % 61) for char in text]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)


def _internal_config() -> PRAHFConfig:
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


def _tiny_mistral3() -> Mistral3ForConditionalGeneration:
    config = Mistral3Config(
        text_config={
            "model_type": "mistral",
            "vocab_size": 67,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "sliding_window": None,
            "bos_token_id": 1,
            "eos_token_id": 66,
            "pad_token_id": 0,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 8,
            "patch_size": 4,
            "image_size": 8,
            "num_channels": 3,
        },
        image_token_index=10,
        spatial_merge_size=1,
    )
    config.text_config._attn_implementation = "eager"
    return Mistral3ForConditionalGeneration(config).eval()


def _tiny_gpt_oss() -> GptOssForCausalLM:
    config = GptOssConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return GptOssForCausalLM(config).eval()


@pytest.mark.parametrize("factory", [_tiny_mistral3, _tiny_gpt_oss])
def test_extended_family_disabled_path_preserves_exact_logits(factory):
    torch.manual_seed(701)
    original = factory()
    adapted = copy.deepcopy(original)
    handle = inject_pra(adapted, _internal_config())
    input_ids = torch.tensor([[1, 7, 9, 3, 5]])

    with torch.no_grad():
        expected = original(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        actual = handle.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    assert torch.equal(actual.logits, expected.logits)
    assert all(
        torch.equal(left, right)
        for left, right in zip(actual.hidden_states, expected.hidden_states)
    )


def test_mistral3_nested_decoder_and_public_api_use_text_config():
    model = _tiny_mistral3()
    anatomy = resolve_decoder_anatomy(model)
    pra = PRAForCausalLM.from_model(
        model,
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

    assert anatomy.path == "model.language_model"
    assert anatomy.config is model.config.text_config
    assert isinstance(pra._handle.adapters[1], MistralPRAAttentionAdapter)
    assert pra.stats()["family"] == "mistral3"


def test_mistral3_reference_uses_native_gqa_and_active_memory():
    torch.manual_seed(703)
    handle = inject_pra(_tiny_mistral3(), _internal_config())
    entry = handle.add_reference(
        "memory://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    handle.set_memory_enabled(True)

    with torch.no_grad():
        output = handle.model(input_ids=torch.tensor([[1, 4, 8, 12]]), use_cache=False)

    assert entry.layer_memory[1].chunks[0].token_kv.k.shape == (1, 2, 4, 8)
    assert output.logits.shape == (1, 4, 67)
    assert handle.diagnostics_by_layer()[1]["retrieved_physical_kv_tokens"] == 8


def test_gpt_oss_keeps_sinks_router_and_experts_host_owned_with_active_memory():
    torch.manual_seed(702)
    model = _tiny_gpt_oss()
    sinks = tuple(layer.self_attn.sinks for layer in model.model.layers)
    routers = tuple(layer.mlp.router for layer in model.model.layers)
    experts = tuple(layer.mlp.experts for layer in model.model.layers)
    handle = inject_pra(model, _internal_config())
    entry = handle.add_reference(
        "memory://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    handle.set_memory_enabled(True)

    with torch.no_grad():
        output = handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)

    topology = describe_moe_topology(model)
    assert isinstance(handle.adapters[1], GptOssPRAAttentionAdapter)
    assert tuple(layer.self_attn.original_attention.sinks if hasattr(layer.self_attn, "original_attention") else layer.self_attn.sinks for layer in model.model.layers) == sinks
    assert tuple(layer.mlp.router for layer in model.model.layers) == routers
    assert tuple(layer.mlp.experts for layer in model.model.layers) == experts
    assert topology is not None
    assert topology.expert_count == 4
    assert topology.experts_per_token == 2
    assert entry.layer_memory[1].chunks[0].token_kv.k.shape[1] == 2
    assert output.logits.shape == (1, 4, 67)
    assert handle.diagnostics_by_layer()[1]["retrieved_physical_kv_tokens"] == 8


def test_gpt_oss_product_defaults_select_only_full_attention_layers():
    config = _tiny_gpt_oss().config
    routing, consumption = PRAConfig().resolved_layers(config)

    assert routing == 1
    assert consumption == (1,)


def test_qwen3_hybrid_plan_separates_qsa_from_recurrent_state():
    config = SimpleNamespace(
        model_type="qwen3_8_flash_next",
        layer_types=(
            "gated_deltanet",
            "gated_deltanet",
            "gated_deltanet",
            "qwen_sparse_attention",
        )
        * 12,
    )

    plan = describe_qwen3_hybrid_plan(config)

    assert plan.layer_count == 48
    assert len(plan.qsa_layer_ids) == 12
    assert len(plan.gdn_layer_ids) == 36
    assert plan.qsa_layer_ids[:3] == (3, 7, 11)
    assert plan.pra_scope == "qsa_attention_only"
    assert plan.recurrent_state_ownership == "host_model"
    assert plan.evaluation_status == "STRUCTURAL_ONLY"


def test_qwen3_hybrid_plan_fails_closed_for_unknown_layers():
    plan = describe_qwen3_hybrid_plan(
        SimpleNamespace(model_type="future_qwen", layer_types=("gdn", "mystery", "qsa"))
    )
    assert plan.unsupported_layer_ids == (1,)
