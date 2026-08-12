import copy

import pytest
import torch

from experiments.paper1_5_rope.learned_routing import (
    AsymmetricLinearRouter,
    contrastive_margin_loss,
    materialize_native_payload,
    native_attention_output,
    rank_candidate_ids,
    shuffled_positive_mask,
    train_router,
    trainable_parameter_count,
)


def test_asymmetric_projection_shapes_and_parameter_count():
    router = AsymmetricLinearRouter(d_model=8, routing_dim=3)
    scores = router(torch.randn(2, 8), torch.randn(2, 5, 8))

    assert scores.shape == (2, 5)
    assert trainable_parameter_count(router) == 2 * 8 * 3


def test_margin_training_changes_router_but_not_frozen_backbone():
    torch.manual_seed(4)
    backbone = torch.nn.Linear(8, 8)
    backbone.requires_grad_(False)
    before = copy.deepcopy(backbone.state_dict())
    router = AsymmetricLinearRouter(d_model=8, routing_dim=4)
    queries = backbone(torch.randn(3, 8)).detach()
    chunks = backbone(torch.randn(3, 4, 8)).detach()
    labels = torch.tensor(
        [[True, False, False, False], [False, True, False, False], [False, False, True, False]]
    )

    history = train_router(router, queries, chunks, labels, steps=20)

    assert history[-1] < history[0]
    assert all(parameter.grad is None for parameter in backbone.parameters())
    for name, value in backbone.state_dict().items():
        assert torch.equal(value, before[name])


def test_ranking_is_deterministic_and_uses_identity_for_ties():
    scores = torch.tensor([[0.4, 0.4, 0.1]])
    ids = [["chunk-b", "chunk-a", "chunk-c"]]

    first = rank_candidate_ids(scores, ids)
    second = rank_candidate_ids(scores, ids)

    assert first == second == [["chunk-a", "chunk-b", "chunk-c"]]


def test_shuffled_labels_preserve_count_and_exclude_true_evidence():
    labels = torch.tensor(
        [[True, False, False, False], [False, True, True, False]]
    )

    shuffled = shuffled_positive_mask(labels, seed=7)

    assert torch.equal(shuffled.sum(dim=1), labels.sum(dim=1))
    assert not bool((shuffled & labels).any())
    assert torch.equal(shuffled, shuffled_positive_mask(labels, seed=7))


def test_fixed_selection_preserves_native_payload_exactly():
    payload = {
        "a": (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
        "b": (torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4)),
    }

    routed_k, routed_v = materialize_native_payload(payload, ["b", "a"])
    oracle_k, oracle_v = materialize_native_payload(payload, ["b", "a"])
    query = torch.randn(1, 2, 1, 4)
    routed_output = native_attention_output(query, routed_k, routed_v)
    oracle_output = native_attention_output(query, oracle_k, oracle_v)

    assert torch.equal(routed_k, oracle_k)
    assert torch.equal(routed_v, oracle_v)
    assert torch.equal(routed_output, oracle_output)
    assert routed_k.data_ptr() != payload["a"][0].data_ptr()


def test_margin_loss_rejects_batches_without_positive_negative_pairs():
    with pytest.raises(ValueError, match="positive-negative"):
        contrastive_margin_loss(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_router_score_parity():
    torch.manual_seed(9)
    cpu_router = AsymmetricLinearRouter(16, 8).eval()
    cuda_router = copy.deepcopy(cpu_router).cuda().eval()
    query = torch.randn(4, 16)
    chunks = torch.randn(4, 7, 16)

    cpu_scores = cpu_router(query, chunks)
    cuda_scores = cuda_router(query.cuda(), chunks.cuda()).cpu()

    assert torch.allclose(cpu_scores, cuda_scores, atol=2e-6, rtol=2e-6)
