"""Static contracts for the Paper 6.5 native-discovery runner."""

from __future__ import annotations

import torch

from data.agent_workflows import realistic_tool_catalog, workflow_tasks
from experiments.paper6_5_tools.run_m6_native_discovery import MODES, _metric_row, _pad_keys


def test_m6_mode_grid_contains_external_native_compressed_and_hybrid_controls():
    assert {"token", "index", "external_signed_hash"} <= set(MODES)
    assert {"native_mean_k", "native_token_qk"} <= set(MODES)
    assert {"paper2_8_rank16_ensemble", "paper2_8_rank8_centroids"} <= set(MODES)
    assert "lexical_native_hybrid" in MODES


def test_variable_resource_keys_are_padded_with_an_explicit_mask():
    rows = [
        {"k": torch.ones((2, 2, 3))},
        {"k": torch.ones((4, 2, 3))},
    ]
    keys, mask = _pad_keys(rows)
    assert keys.shape == (2, 4, 2, 3)
    assert mask.sum(dim=1).tolist() == [2, 4]


def test_multistep_metric_uses_horizon_budget_and_successor_recall():
    task = next(task for task in workflow_tasks() if len(task.steps) == 3)
    resources = realistic_tool_catalog()
    scores = torch.zeros(len(resources))
    by_name = {resource.name: index for index, resource in enumerate(resources)}
    for rank, name in enumerate(task.required_tools):
        scores[by_name[name]] = 10 - rank
    row = _metric_row(task, 11, "fixture", scores, resources, 0.0, 1)
    assert row["budget"] == 3
    assert row["required_recall_at_budget"] == 1.0
    assert row["successor_recall_at_budget"] == 1.0
    assert row["all_required_recovered"]
