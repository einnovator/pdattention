"""Shape and deployment-boundary tests for Paper 6.5 M6."""

from __future__ import annotations

import torch

from pra_hf.native_resource_discovery import (
    NativeResolverEndpoint,
    NativeResourceSearchRequest,
    ProjectedQueryExport,
    native_mean_k_scores,
    native_token_qk_scores,
)


def test_native_scorers_accept_grouped_kv_heads_and_mask_padding():
    query = torch.ones((4, 2))
    keys = torch.zeros((2, 3, 2, 2))
    keys[0, 0] = 2.0
    keys[1, 0] = -2.0
    mask = torch.tensor([[True, False, False], [True, False, False]])
    means = native_mean_k_scores(query, keys, mask)
    tokens = native_token_qk_scores(query, keys, mask, top_r=2)
    assert means[0] > means[1]
    assert tokens[0] > tokens[1]


def test_model_server_endpoint_returns_identity_not_raw_native_state():
    endpoint = NativeResolverEndpoint(
        ("!!ref:tool:test:a:v1!!", "!!ref:tool:test:b:v1!!"),
        lambda query, mode: (0.2, 0.9),
        index_fingerprint="fixture",
    )
    result = endpoint.search(NativeResourceSearchRequest("tools", "query", top_k=1))
    assert result.hits[0].uri.endswith(":b:v1!!")
    assert not result.raw_state_exported
    assert not hasattr(result, "query")


def test_shared_memory_export_is_low_rank_not_raw_multihead_q():
    exported = ProjectedQueryExport(torch.ones(16), "revision", 27, "projection")
    assert exported.values.shape == (16,)
