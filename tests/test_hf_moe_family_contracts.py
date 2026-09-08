"""Runtime SDK gates for MoE and newly qualified Hugging Face families."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
from transformers import (
    GptOssConfig,
    GptOssForCausalLM,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from pra_hf import PRAConfig, PRAForCausalLM
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    GptOssPRAAttentionAdapter,
    MistralPRAAttentionAdapter,
    PRAHFConfig,
    Qwen3MoePRAAttentionAdapter,
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


def _tiny_qwen_moe():
    config = Qwen3MoeConfig(
        vocab_size=67, hidden_size=32, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=66, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(config).eval()


def _tiny_gpt_oss():
    config = GptOssConfig(
        vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_local_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=66, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return GptOssForCausalLM(config).eval()


def _tiny_mistral3():
    config = Mistral3Config(
        text_config={
            "model_type": "mistral", "vocab_size": 67, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 64, "sliding_window": None,
            "bos_token_id": 1, "eos_token_id": 66, "pad_token_id": 0,
        },
        vision_config={
            "model_type": "pixtral", "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "head_dim": 8,
            "patch_size": 4, "image_size": 8, "num_channels": 3,
        },
        image_token_index=10,
        spatial_merge_size=1,
    )
    config.text_config._attn_implementation = "eager"
    return Mistral3ForConditionalGeneration(config).eval()


@pytest.mark.parametrize(
    ("factory", "adapter_type"),
    [
        (_tiny_qwen_moe, Qwen3MoePRAAttentionAdapter),
        (_tiny_gpt_oss, GptOssPRAAttentionAdapter),
        (_tiny_mistral3, MistralPRAAttentionAdapter),
    ],
)
def test_disabled_path_is_exact(factory, adapter_type):
    torch.manual_seed(801)
    original = factory()
    adapted = copy.deepcopy(original)
    handle = inject_pra(adapted, _internal_config())
    ids = torch.tensor([[1, 7, 9, 3, 5]])
    with torch.no_grad():
        expected = original(input_ids=ids, output_hidden_states=True, use_cache=False)
        actual = handle.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    assert torch.equal(actual.logits, expected.logits)
    assert all(
        torch.equal(left, right)
        for left, right in zip(actual.hidden_states, expected.hidden_states)
    )
    assert isinstance(handle.adapters[1], adapter_type)


@pytest.mark.parametrize("factory", [_tiny_qwen_moe, _tiny_gpt_oss, _tiny_mistral3])
def test_active_memory_preserves_native_gqa(factory):
    handle = inject_pra(factory(), _internal_config())
    entry = handle.add_reference(
        "memory://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    handle.set_memory_enabled(True)
    with torch.no_grad():
        output = handle.model(input_ids=torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    assert entry.layer_memory[1].chunks[0].token_kv.k.shape == (1, 2, 4, 8)
    assert output.logits.shape == (1, 4, 67)
    assert handle.diagnostics_by_layer()[1]["retrieved_physical_kv_tokens"] == 8


def test_moe_ownership_and_runtime_stats_are_exposed():
    model = _tiny_gpt_oss()
    routers = tuple(layer.mlp.router for layer in model.model.layers)
    experts = tuple(layer.mlp.experts for layer in model.model.layers)
    pra = PRAForCausalLM.from_model(
        model,
        TinyTokenizer(),
        pra_config=PRAConfig(
            routing_layer=-1, consumption_layers=(-1,), chunk_tokens=4,
            selected_fraction=0.5, max_direct_context=16,
            native_operation_limit=64, max_materialized_tokens=16,
            context_safety_reserve_tokens=0, encoding_block_tokens=16,
        ),
    )
    topology = describe_moe_topology(model)
    assert topology is not None
    assert topology.expert_count == 4
    assert tuple(layer.mlp.router for layer in model.model.layers) == routers
    assert tuple(layer.mlp.experts for layer in model.model.layers) == experts
    assert pra.stats()["moe_topology"]["pra_scope"] == "attention_only"


def test_nested_decoder_and_qwen_hybrid_scope():
    anatomy = resolve_decoder_anatomy(_tiny_mistral3())
    plan = describe_qwen3_hybrid_plan(
        SimpleNamespace(
            model_type="qwen4_exp_text",
            layer_types=("linear_attention",) * 3 + ("full_attention",),
        )
    )
    assert anatomy.path == "model.language_model"
    assert plan.qsa_layer_ids == (3,)
    assert plan.gdn_layer_ids == (0, 1, 2)
    assert plan.recurrent_state_ownership == "host_model"


def test_gpt_oss_defaults_select_only_full_layers():
    roles = PRAConfig().resolved_layer_roles(_tiny_gpt_oss().config)
    assert roles.routing_layers == (1,)
    assert roles.consumption_layers == (1,)
