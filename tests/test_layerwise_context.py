from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pra_hf.layerwise_context import (
    LayerContextCollector,
    attention_token_metrics,
    branch_token_metrics,
    causal_radius_mask,
    directional_rotation,
    normalized_displacement,
)
from pra_torch.hf import PRAHFConfig, inject_pra


def _tiny_qwen():
    config = Qwen3Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
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


def _handle():
    return inject_pra(
        _tiny_qwen(),
        PRAHFConfig(
            layer_ids=(0, 2),
            model_max_context_tokens=64,
            max_prompt_direct_tokens=16,
            encoding_block_tokens=16,
            routing_chunk_tokens=4,
            max_materialized_memory_tokens=16,
            collect_detailed_timing=False,
        ),
    )


def test_collector_preserves_exact_layer_index_and_residual_identities():
    torch.manual_seed(7)
    handle = _handle()
    handle.set_attention_diagnostics(True)
    ids = torch.tensor([[1, 5, 9, 3, 4, 2]])
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    with LayerContextCollector(handle.model.model.layers, (0, 2)) as collector:
        handle.model(ids, attention_mask=torch.ones_like(ids), position_ids=positions)
        collector.validate(atol=1e-5, rtol=1e-5)
        assert tuple(collector.snapshots) == (0, 2)
        assert collector.snapshots[0].attention_input.shape == (1, 6, 32)
        assert collector.snapshots[2].attention_weights.shape == (1, 4, 6, 6)


def test_branch_metrics_match_explicit_norm_displacement_and_rotation():
    handle = _handle()
    ids = torch.tensor([[1, 2, 3, 4]])
    with LayerContextCollector(handle.model.model.layers, (0,)) as collector:
        handle.model(ids, attention_mask=torch.ones_like(ids))
        snapshot = collector.snapshots[0]
        metrics = branch_token_metrics(snapshot)
        expected = snapshot.attention_contribution.float().norm(dim=-1) / (
            snapshot.pre_attention_residual.float().norm(dim=-1) + 1e-8
        )
        assert torch.allclose(metrics["attention_contribution_ratio"], expected)
        assert torch.allclose(
            metrics["post_attention_displacement"],
            normalized_displacement(
                snapshot.pre_attention_residual, snapshot.post_attention_residual
            ),
        )
        assert torch.allclose(
            metrics["post_attention_rotation"],
            directional_rotation(
                snapshot.pre_attention_residual, snapshot.post_attention_residual
            ),
        )


def test_radius_mask_preserves_coordinates_and_restricts_only_visible_context():
    mask = causal_radius_mask(5, 2, device=torch.device("cpu"), dtype=torch.float32)
    visible = mask[0, 0] == 0
    assert visible.nonzero().tolist() == [
        [0, 0],
        [1, 0], [1, 1],
        [2, 1], [2, 2],
        [3, 2], [3, 3],
        [4, 3], [4, 4],
    ]
    full = causal_radius_mask(5, None, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(full[0, 0] == 0, torch.ones(5, 5, dtype=torch.bool).tril())


def test_full_radius_reproduces_canonical_states_and_local_is_deterministic():
    torch.manual_seed(11)
    handle = _handle()
    ids = torch.tensor([[1, 5, 9, 3, 4, 2]])
    positions = torch.tensor([[7, 8, 9, 10, 11, 12]])
    layers = handle.model.model.layers
    with LayerContextCollector(layers, (0, 2)) as collector:
        handle.model(ids, attention_mask=torch.ones_like(ids), position_ids=positions)
        canonical = {
            layer: snapshot.attention_input.clone()
            for layer, snapshot in collector.snapshots.items()
        }
        full = causal_radius_mask(6, None, device=ids.device, dtype=torch.float32)
        collector.clear()
        handle.model(ids, attention_mask={"full_attention": full}, position_ids=positions)
        for layer, snapshot in collector.snapshots.items():
            assert torch.allclose(snapshot.attention_input, canonical[layer], atol=1e-6)
        local = causal_radius_mask(6, 1, device=ids.device, dtype=torch.float32)
        collector.clear()
        handle.model(ids, attention_mask={"full_attention": local}, position_ids=positions)
        first = collector.snapshots[2].attention_input.clone()
        collector.clear()
        handle.model(ids, attention_mask={"full_attention": local}, position_ids=positions)
        second = collector.snapshots[2].attention_input
        assert torch.equal(first, second)
        assert not torch.allclose(first[:, 1:], canonical[2][:, 1:])
        assert torch.allclose(first[:, :1], canonical[2][:, :1], atol=1e-6)


def test_attention_statistics_use_query_heads_and_evidence_keys():
    weights = torch.zeros(1, 2, 3, 3)
    weights[:, :, 0, 0] = 1.0
    weights[:, :, 1, :2] = 0.5
    weights[:, :, 2, :] = 1 / 3
    evidence = torch.tensor([[False, True, False]])
    metrics = attention_token_metrics(weights, local_window=2, evidence_mask=evidence)
    assert metrics["self_attention_fraction"].shape == (1, 3)
    assert torch.allclose(metrics["effective_attention_support"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(metrics["evidence_attention_fraction"], torch.tensor([[0.0, 0.5, 1 / 3]]))
