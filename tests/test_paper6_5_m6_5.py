"""Static and metric contracts for Paper 6.5 M6.5."""

from __future__ import annotations

import torch

from experiments.paper6_5_tools.run_m6_5_external_semantics import (
    MODEL_SPECS,
    POLICY_MODES,
    _ranking_metrics,
)


def test_candidate_models_are_pinned_and_cover_english_and_multilingual() -> None:
    assert 2 <= len(MODEL_SPECS) <= 4
    assert {row.language_scope for row in MODEL_SPECS} == {"en", "multilingual"}
    assert all(len(row.revision) == 40 for row in MODEL_SPECS)
    assert all(row.license in {"apache-2.0", "mit"} for row in MODEL_SPECS)


def test_required_external_policy_ladder_and_oracle_are_present() -> None:
    assert tuple(f"P{index}" for index in range(10)) == tuple(
        mode.split("_", 1)[0] for mode in POLICY_MODES[:10]
    )
    assert POLICY_MODES[-1] == "P10_staged_external"


def test_ranking_metrics_use_stable_required_identity_ranks() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.3], [0.8, 0.1, 0.2]])
    metrics = _ranking_metrics(scores, (1, 2))

    assert metrics["top1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mean_rank"] == 1.5
