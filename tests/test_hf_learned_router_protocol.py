"""Unit tests for learned-router objectives and deterministic negative policies."""

import pytest
import torch

from experiments.paper2_hf.routing.train_learned_router import (
    _loss,
    select_training_candidates,
)
from pra_torch.hf import HFRoutingProjection


def _feature() -> dict:
    return {
        "dataset": "test",
        "example_id": "example-1",
        "queries": {"last": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        "memory_gists": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "positive_mask": torch.tensor([True, False, False, True, False, False]),
        "lexical_scores": torch.tensor([0.1, 0.9, 0.2, 0.0, 0.8, 0.0]),
        "normalized_positions": torch.linspace(0.0, 1.0, 6),
    }


def test_mixed_negative_policy_is_deterministic_and_budgeted():
    first, composition = select_training_candidates(
        _feature(), "last", policy="mixed", negatives_per_positive=1, seed=7
    )
    second, _ = select_training_candidates(
        _feature(), "last", policy="mixed", negatives_per_positive=1, seed=7
    )
    assert torch.equal(first, second)
    assert int((first & ~_feature()["positive_mask"]).sum()) == 2
    assert sum(composition.values()) == 2
    assert set(composition) == {
        "zero_shot_false_positive",
        "lexical",
        "position_matched",
        "random",
    }


def test_mined_policy_requires_and_prioritizes_learned_scores():
    with pytest.raises(ValueError, match="requires learned-router"):
        select_training_candidates(
            _feature(), "last", policy="mined", negatives_per_positive=1, seed=7
        )
    mask, composition = select_training_candidates(
        _feature(),
        "last",
        policy="mined",
        negatives_per_positive=1,
        seed=7,
        mined_scores=torch.tensor([0.0, 0.1, 0.2, 0.0, 0.99, 0.3]),
    )
    assert bool(mask[4])
    assert composition["learned_false_positive"] >= 1


@pytest.mark.parametrize("objective", ("contrastive", "margin"))
def test_objectives_are_finite_and_differentiable(objective):
    model = HFRoutingProjection(4, 3, "asymmetric_linear")
    loss, composition = _loss(
        model,
        _feature(),
        "last",
        torch.device("cpu"),
        0.1,
        False,
        7,
        objective,
        0.2,
        "mixed",
        2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert sum(composition.values()) == 4
    assert all(parameter.grad is not None for parameter in model.parameters())
