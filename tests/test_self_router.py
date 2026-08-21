"""Correctness gates for model-native PRA query routing."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.paper3_5_adaptive_pra.self_router_study import load_factorized_oracles
from pra_hf.self_router import (
    QueryPrefillAccounting,
    ValidationProjector,
    decode_grouped_action,
    native_qk_representation,
    pool_query_tokens,
    query_span_mask,
    qwen_prefill_continue,
    qwen_prefill_prefix,
    reuse_is_semantically_valid,
)


def _tiny_qwen() -> Qwen3ForCausalLM:
    torch.manual_seed(7)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=47,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            use_cache=False,
        )
    ).eval()


def test_query_span_pooling_preserves_tensor_geometry():
    states = torch.arange(2 * 5 * 2 * 3, dtype=torch.float32).view(2, 5, 2, 3)
    mask = torch.tensor(
        [[False, True, True, False, False], [False, False, True, True, True]]
    )
    assert torch.equal(pool_query_tokens(states, mask, "last"), states[[0, 1], [2, 4]])
    assert torch.equal(pool_query_tokens(states, mask, "max"), states[[0, 1], [2, 4]])
    assert torch.allclose(pool_query_tokens(states, mask, "mean")[0], states[0, 1:3].mean(0))


def test_structured_query_span_is_half_open_and_validated():
    mask = query_span_mask(7, 2, 5)
    assert mask.shape == (1, 7)
    assert mask.nonzero(as_tuple=False)[:, 1].tolist() == [2, 3, 4]
    try:
        query_span_mask(7, 5, 5)
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("An empty query region must be rejected.")


def test_native_qk_capture_matches_manual_qwen_projection():
    model = _tiny_qwen()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        output = model(ids, output_hidden_states=True, use_cache=False)
    mask = query_span_mask(4, 1, 4)
    actual = native_qk_representation(
        model, output.hidden_states[2], 2, mask, kind="qk", pooling="mean"
    )
    layer = model.model.layers[2]
    normalized = layer.input_layernorm(output.hidden_states[2])
    q = layer.self_attn.q_norm(
        layer.self_attn.q_proj(normalized).view(1, 4, 4, 8)
    )
    k = layer.self_attn.k_norm(
        layer.self_attn.k_proj(normalized).view(1, 4, 2, 8)
    )
    expected = torch.cat(
        (q[:, 1:].mean(1).flatten(1), k[:, 1:].mean(1).flatten(1)), dim=1
    )
    assert actual.shape == (1, 48)
    assert torch.equal(actual, expected)


def test_qwen_prefix_continuation_is_exact_and_does_not_mutate_parameters():
    model = _tiny_qwen()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = model.model(ids, attention_mask=mask, use_cache=False).last_hidden_state
        state = qwen_prefill_prefix(model, ids, mask, stop_layer=2)
        actual = qwen_prefill_continue(model, state)
    assert torch.equal(actual, expected)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_query_prefill_accounting_charges_double_encoding_but_not_reuse():
    doubled = QueryPrefillAccounting(32, 12, 14, 28)
    reused = QueryPrefillAccounting(32, 12, 14, 28, reused=True)
    assert doubled.processed_token_layers == 448
    assert doubled.recomputed_query_tokens == 32
    assert doubled.normalized_cost == 0.25
    assert reused.processed_token_layers == 448
    assert reused.recomputed_query_tokens == 0
    assert reused.normalized_cost == 0.0


def test_reuse_requires_pre_memory_state_and_unchanged_query_region():
    assert reuse_is_semantically_valid(
        depth=14, first_memory_layer=24, ordinary_context_precedes_query=False
    )
    assert not reuse_is_semantically_valid(
        depth=25, first_memory_layer=24, ordinary_context_precedes_query=False
    )
    assert not reuse_is_semantically_valid(
        depth=14, first_memory_layer=24, ordinary_context_precedes_query=True
    )
    assert not reuse_is_semantically_valid(
        depth=14,
        first_memory_layer=24,
        ordinary_context_precedes_query=False,
        query_region_changed=True,
    )


def test_grouped_target_decoding_repairs_only_cross_group_budget_constraints():
    action, repaired = decode_grouped_action(2, (4, 8, 2, 2), 2)
    assert repaired
    assert action.facets == 2
    assert action.roots == action.search_budget == action.kv_budget == 4
    assert action.neighbors == 8 and action.hops == 2


def test_validation_projector_is_deterministic_and_rejects_heldout_fit():
    rows = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
    )
    left = ValidationProjector(2).fit(rows, ["validation"] * 3).transform(rows)
    right = ValidationProjector(2).fit(rows, ["validation"] * 3).transform(rows)
    assert torch.equal(left, right)
    try:
        ValidationProjector(2).fit(rows, ["validation", "test", "validation"])
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("Held-out rows must never fit the representation projector.")


def test_factorized_oracle_loader_validates_identity_and_numeric_schema(tmp_path):
    path = tmp_path / "oracles.csv"
    path.write_text(
        "dataset,example_id,seed,config_id,facets,roots,neighbors,hops,search_budget,kv_budget,chain_complete,abstract_cost\n"
        "hotpotqa,x,11,F1_R1_K2_H0_Bs2_Bkv2,1,1,2,0,2,2,1,4.5\n",
        encoding="utf-8",
    )
    key = ("hotpotqa", "x", 11)
    rows = load_factorized_oracles(path, {key})
    assert rows[key]["roots"] == 1
    assert rows[key]["chain_complete"] == 1.0
    try:
        load_factorized_oracles(path, {("qasper", "other", 11)})
    except ValueError as error:
        assert "identities" in str(error)
    else:
        raise AssertionError("Mismatched held-out identities must be rejected.")
