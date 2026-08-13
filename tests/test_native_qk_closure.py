"""Tests for semantically narrowed tokenwise native-QK closure."""

from __future__ import annotations

import pytest
import torch

from pra_hf.native_closure import (
    NativeLocalQKRouter,
    NativeQKIndex,
    NativeQKRoutingConfig,
    gqa_query_to_kv_heads,
    native_local_qk_scores,
)


def _index(device: str = "cpu") -> NativeQKIndex:
    # Root selects A. Semantic narrowing from A includes X then B, while A's
    # native query resonates only with B's native key.
    semantic_memory = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.8, 0.6], [0.0, 1.0, 0.0], [0.0, 0.7, -0.7]],
        device=device,
    )
    semantic_query = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    query = torch.zeros(4, 2, 4, 2, device=device)
    key = torch.zeros(4, 2, 2, 2, device=device)
    query[0, 0, 0] = torch.tensor([3.0, 0.0], device=device)
    key[1, 0, 0] = torch.tensor([3.0, 0.0], device=device)
    key[2, 0, 0] = torch.tensor([0.0, 3.0], device=device)
    key[3, 0, 0] = torch.tensor([-3.0, 0.0], device=device)
    return NativeQKIndex(
        parent_ids=("A", "B", "X", "Y"),
        parent_spans=((0, 2), (2, 4), (4, 6), (6, 8)),
        parent_memory_gists=semantic_memory,
        local_spans=((0, 2), (2, 4), (4, 6), (6, 8)),
        local_parent_indices=torch.arange(4, device=device),
        local_memory_gists=semantic_memory,
        local_query_gists=semantic_query,
        local_pre_query=query,
        local_pre_key=key,
        token_mask=torch.ones(4, 2, dtype=torch.bool, device=device),
    )


def test_gqa_mapping_pairs_only_model_compatible_heads():
    assert gqa_query_to_kv_heads(16, 8).tolist() == [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
    ]
    with pytest.raises(ValueError, match="divisible"):
        gqa_query_to_kv_heads(15, 8)


def test_native_score_uses_compatible_head_and_exact_scaled_dot():
    query = torch.zeros(1, 2, 4, 2)
    key = torch.zeros(1, 2, 2, 2)
    query[0, 1, 3] = torch.tensor([2.0, 0.0])
    key[0, 0, 1] = torch.tensor([3.0, 0.0])
    result = native_local_qk_scores(
        query, key, torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result.scores.item() == pytest.approx(6 / (2**0.5))
    assert result.query_token.item() == 1
    assert result.key_token.item() == 0
    assert result.query_head.item() == 3
    assert result.kv_head.item() == 1
    assert result.dot_products == 2 * 2 * 4


def test_max_and_top_m_reductions_are_deterministic():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 4, 2)
    key = torch.randn(3, 3, 2, 2)
    q_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    k_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    first = native_local_qk_scores(query, key, q_mask, k_mask)
    second = native_local_qk_scores(query, key, q_mask, k_mask)
    top_m = native_local_qk_scores(
        query, key, q_mask, k_mask,
        token_reduction="top_m_mean", head_reduction="top_m_mean", top_m=2,
    )
    assert torch.equal(first.scores, second.scores)
    assert torch.isfinite(top_m.scores).all()
    assert top_m.scores.shape == (2, 3)


def test_top_m_ignores_padding_when_fewer_than_m_token_pairs_exist():
    query = torch.tensor([[[[2.0, 0.0]], [[0.0, 0.0]]]])
    key = torch.tensor([[[[3.0, 0.0]], [[0.0, 0.0]]]])
    mask = torch.tensor([[True, False]])
    result = native_local_qk_scores(
        query, key, mask, mask,
        token_reduction="top_m_mean", head_reduction="mean", top_m=4,
    )
    assert result.scores.item() == pytest.approx(6 / (2**0.5))


def test_semantic_narrowing_native_bridge_parent_dedup_and_hard_budget():
    router = NativeLocalQKRouter(_index())
    result = router.route(
        torch.tensor([1.0, 0.0, 0.0]),
        NativeQKRoutingConfig(
            max_unique_parents=2,
            candidate_pool_fraction=0.5,
            initial_parent_count=1,
            branch_top_k=1,
        ),
        evidence_parent_ids={"A", "B"},
    )
    assert result.selected_indices == (0, 1)
    assert len(result.selected_indices) == len(set(result.selected_indices)) == 2
    native = [edge for edge in result.graph.edges if edge.edge_type == "native_qk"]
    assert len(native) == 1
    assert native[0].query_head == 0 and native[0].kv_head == 0
    assert native[0].semantic_candidate_rank == 2
    assert result.graph.costs["candidate_parents"] == 2
    assert result.graph.costs["accepted_native_transitions"] <= 1
    assert result.graph.costs["proposed_native_transitions"] <= 1
    assert result.graph.costs["kv_materializations_during_closure"] == 0
    assert result.graph.costs["final_kv_tokens"] == 4


def test_semantic_narrowing_excludes_native_match_outside_pool():
    result = NativeLocalQKRouter(_index()).route(
        torch.tensor([1.0, 0.0, 0.0]),
        NativeQKRoutingConfig(
            max_unique_parents=2,
            candidate_pool_fraction=0.25,
            initial_parent_count=1,
            branch_top_k=1,
        ),
    )
    assert result.selected_indices == (0, 2)
    assert result.graph.costs["candidate_parents"] == 1


def test_threshold_has_hard_cap_and_semantic_fill_preserves_budget():
    result = NativeLocalQKRouter(_index()).route(
        torch.tensor([1.0, 0.0, 0.0]),
        NativeQKRoutingConfig(
            max_unique_parents=3,
            candidate_pool_fraction=0.75,
            initial_parent_count=1,
            branch_top_k=1,
            transition_mode="threshold",
            threshold_lambda=100.0,
        ),
    )
    assert len(result.selected_indices) == 3
    assert result.graph.costs["accepted_native_transitions"] == 0
    assert sum(node.selection_reason == "semantic_budget_fill" for node in result.graph.nodes) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_closure_cpu_cuda_identity_parity():
    config = NativeQKRoutingConfig(
        max_unique_parents=2,
        candidate_pool_fraction=0.5,
        initial_parent_count=1,
        branch_top_k=1,
    )
    cpu = NativeLocalQKRouter(_index()).route(torch.tensor([1.0, 0.0, 0.0]), config)
    gpu = NativeLocalQKRouter(_index("cuda")).route(
        torch.tensor([1.0, 0.0, 0.0], device="cuda"), config
    )
    assert gpu.selected_indices == cpu.selected_indices
    assert [edge.query_head for edge in gpu.graph.edges] == [
        edge.query_head for edge in cpu.graph.edges
    ]
