from __future__ import annotations

import pytest

from experiments.paper8_5_agent_memory.plot_easy14_atomic_frontier import prefix_rows


def test_prefix_curve_uses_ratio_of_sums_and_zeroes_lost_success_prefixes():
    evidence = {"runs": [{
        "sequence_id": "easy14",
        "repeat": 1,
        "strategy_id": "S08_atomic_epoch_e2",
        "strategy_config_id": "balanced_keep_two_prior_epochs_v1",
        "paired_issues": [
            {
                "candidate_materialized_tokens": 60,
                "control_full_tokens": 100,
                "candidate_calls": 10,
                "control_calls": 10,
                "candidate_official_resolved": True,
                "control_official_resolved": True,
                "lost_persistent_full_success": False,
            },
            {
                "candidate_materialized_tokens": 120,
                "control_full_tokens": 300,
                "candidate_calls": 15,
                "control_calls": 12,
                "candidate_official_resolved": False,
                "control_official_resolved": True,
                "lost_persistent_full_success": True,
            },
        ],
    }]}

    rows = prefix_rows(evidence)

    assert rows[0]["raw_saving_vs_persistent_full"] == pytest.approx(0.4)
    assert rows[0]["failure_aware_saving_vs_persistent_full"] == pytest.approx(0.4)
    assert rows[1]["raw_saving_vs_persistent_full"] == pytest.approx(0.55)
    assert rows[1]["failure_aware_saving_vs_persistent_full"] == 0.0
    assert rows[1]["resolution_delta"] == -1
    assert rows[1]["calls_delta_vs_persistent_full"] == 3
