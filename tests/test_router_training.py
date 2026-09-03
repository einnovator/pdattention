from __future__ import annotations

import pytest
import torch

from pra_hf.training import train_router


def _feature(*, finite: bool) -> dict:
    query = torch.ones(4)
    memory = torch.ones(3, 4)
    if not finite:
        memory[1, 2] = torch.nan
    return {
        "example_id": "finite-contract",
        "queries": {"last": query},
        "memory_gists": memory,
        "positive_mask": torch.tensor([True, False, False]),
    }


def test_train_router_rejects_nonfinite_extracted_features() -> None:
    with pytest.raises(ValueError, match="Non-finite train routing features"):
        train_router([_feature(finite=False)], [_feature(finite=True)], steps=1)
