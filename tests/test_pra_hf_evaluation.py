"""Tests for stable PRA-HF router evaluation metrics."""

import pytest
import torch

from pra_hf import PRARouter, evaluate_router_features


def test_router_evaluation_reports_first_evidence_mrr():
    router = PRARouter(2, 2)
    with torch.no_grad():
        router.query_projection.weight.copy_(torch.eye(2))
        router.memory_projection.weight.copy_(torch.eye(2))
    features = [
        {
            "dataset": "test",
            "queries": {"last": torch.tensor([1.0, 0.0])},
            "memory_gists": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "positive_mask": torch.tensor([True, False]),
            "chunk_spans": [(0, 1), (1, 2)],
        },
        {
            "dataset": "test",
            "queries": {"last": torch.tensor([1.0, 0.0])},
            "memory_gists": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "positive_mask": torch.tensor([False, True]),
            "chunk_spans": [(0, 1), (1, 2)],
        },
    ]

    report = evaluate_router_features(router, features)

    assert report["summary"]["MRR"] == pytest.approx(0.75)

