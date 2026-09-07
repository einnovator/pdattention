import pytest
import torch

from experiments.paper4_5_runtime.run_qwen35_qualification import (
    select_chunk_indices,
)


def test_selected_chunks_respect_budget_and_score_order() -> None:
    selected = select_chunk_indices(
        torch.tensor([0.1, 0.9, 0.8]),
        [(0, 4), (4, 8), (8, 12)],
        token_budget=8,
    )

    assert selected == [1, 2]


def test_selected_chunks_keep_one_candidate_for_tiny_budget() -> None:
    assert select_chunk_indices(
        torch.tensor([0.1, 0.9]), [(0, 8), (8, 16)], token_budget=1
    ) == [1]


def test_selected_chunks_validate_shapes_and_budget() -> None:
    with pytest.raises(ValueError, match="same candidates"):
        select_chunk_indices(torch.tensor([1.0]), [(0, 1), (1, 2)], token_budget=1)
    with pytest.raises(ValueError, match="positive"):
        select_chunk_indices(torch.tensor([1.0]), [(0, 1)], token_budget=0)
