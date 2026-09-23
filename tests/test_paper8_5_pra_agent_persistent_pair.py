from __future__ import annotations

import pytest

from experiments.paper8_5_agent_memory.reduce_pra_agent_persistent_pair import (
    reduce_pair,
)


def _manifest(policy: str, *, materialized: tuple[int, int]) -> dict:
    return {
        "agent": "pra-agent",
        "boundary_mode": "boundary_free",
        "task_order": ["one", "two"],
        "model": "model",
        "served_model": "model",
        "model_revision": "revision",
        "tokenizer": "tokenizer@revision",
        "tokenizer_revision": "revision",
        "generation": {"temperature": 0, "top_p": 1, "seed": 0},
        "history_policy": policy,
        "session_id": policy,
        "episodes": [
            {
                "instance_id": "one",
                "request_count": 4,
                "cumulative_full_history_tokens": 100,
                "cumulative_materialized_history_tokens": materialized[0],
            },
            {
                "instance_id": "two",
                "request_count": 6 if policy == "full" else 5,
                "cumulative_full_history_tokens": 300,
                "cumulative_materialized_history_tokens": materialized[1],
            },
        ],
    }


def test_reduce_pair_reports_own_and_paired_n_prefix_curves() -> None:
    result = reduce_pair(
        _manifest("full", materialized=(100, 300)),
        _manifest("instruction_epoch", materialized=(100, 180)),
        {"resolved_ids": ["one", "two"]},
        {"resolved_ids": ["one", "two"]},
    )

    first, second = result["prefix_rows"]
    assert first["n"] == 1
    assert first["own_logical_saving_fraction"] == 0
    assert second["own_logical_saving_fraction"] == pytest.approx(0.3)
    assert second["paired_workload_saving_fraction"] == pytest.approx(0.3)
    assert second["conditional_preservation_fraction"] == 1
    assert second["cumulative_request_delta"] == -1


def test_reduce_pair_rejects_nonidentical_pairing_identity() -> None:
    policy = _manifest("instruction_epoch", materialized=(100, 180))
    policy["model_revision"] = "different"

    with pytest.raises(ValueError, match="model_revision"):
        reduce_pair(
            _manifest("full", materialized=(100, 300)),
            policy,
            {"resolved_ids": []},
            {"resolved_ids": []},
        )
