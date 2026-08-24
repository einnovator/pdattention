import math

import pytest
import torch

from pra_hf.temporal_routing import (
    delayed_commit_index,
    interaction_contrast,
    score_diagnostics,
    selection_churn,
    stride_update_mask,
    temporal_chunk_scores,
    temporal_window,
)


def _memory():
    memory = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    return memory, torch.ones(2, 2, dtype=torch.bool)


def test_temporal_window_clips_and_preserves_anchor_provenance():
    window = temporal_window(10, anchor=2, look_behind=8, look_ahead=3)
    assert window.indices == (0, 1, 2, 3, 4, 5)
    assert window.anchor == 2
    assert window.length == 6


def test_current_reducer_uses_only_newest_query():
    memory, mask = _memory()
    queries = torch.tensor([[-10.0, 0.0], [1.0, 0.0]])
    scores, interactions = temporal_chunk_scores(
        queries, memory, mask, reducer="current", top_r=1
    )
    assert scores[0] > scores[1]
    assert interactions.shape == (2, 2, 2)


def test_late_interaction_retains_distinct_temporal_evidence():
    memory, mask = _memory()
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores, interactions = temporal_chunk_scores(
        queries, memory, mask, reducer="late_max", top_r=1
    )
    assert scores[0] > scores[1]
    diagnostics = score_diagnostics(scores, interactions)
    assert diagnostics.contributing_query_states == 2
    assert diagnostics.contributing_memory_modes == 2
    assert 0 <= diagnostics.normalized_entropy <= 1


def test_masked_modes_do_not_contribute():
    memory, _ = _memory()
    mask = torch.tensor([[True, False], [True, True]])
    queries = torch.tensor([[0.0, -1.0]])
    scores, _ = temporal_chunk_scores(
        queries, memory, mask, reducer="late_max", top_r=2
    )
    assert scores[1] > scores[0]


def test_delayed_commitment_uses_first_qualified_observed_state():
    margins = torch.tensor([0.1, 0.2, 0.8, 0.9])
    entropies = torch.tensor([1.0, 0.8, 0.3, 0.2])
    assert delayed_commit_index(
        margins,
        entropies,
        start=0,
        maximum_delay=3,
        margin_threshold=0.5,
        entropy_threshold=0.5,
    ) == 2
    assert delayed_commit_index(
        margins,
        entropies,
        start=0,
        maximum_delay=1,
        margin_threshold=0.5,
    ) == 1


def test_stride_churn_and_interaction_contracts():
    assert stride_update_mask(9, 4).tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert selection_churn(torch.tensor([1, 2]), torch.tensor([2, 3])) == pytest.approx(
        2 / 3
    )
    assert interaction_contrast(0.1, 0.2, 0.3, 0.5) == pytest.approx(0.1)


def test_invalid_temporal_arguments_fail_cleanly():
    with pytest.raises(ValueError):
        temporal_window(0, anchor=0, look_behind=1)
    with pytest.raises(ValueError):
        stride_update_mask(4, 0)
    assert math.isfinite(interaction_contrast(0, 0, 0, 0))
