from types import SimpleNamespace

from experiments.paper8_5_agent_memory.run_persistent_first_divergence_oracle import (
    _completed_epoch_addback_batches,
)


def test_completed_epoch_batches_cover_each_exclusion_once():
    composed = {
        "episodes": [
            {"episode_index": 1, "model_visible_messages": 3},
            {"episode_index": 2, "model_visible_messages": 2},
        ]
    }
    history = SimpleNamespace(records=[
        SimpleNamespace(record_id=f"record-{index:06d}") for index in range(5)
    ])
    exclusions = [
        SimpleNamespace(
            causal_group_id="turn-000000",
            record_ids=("record-000001", "record-000002"),
            excluded_tokens=11,
        ),
        SimpleNamespace(
            causal_group_id="turn-000001",
            record_ids=("record-000003", "record-000004"),
            excluded_tokens=13,
        ),
    ]

    assert _completed_epoch_addback_batches(
        composed=composed, history=history, exclusions=exclusions,
    ) == [
        {
            "epoch_index": 1,
            "causal_group_ids": ("turn-000000",),
            "excluded_tokens": 11,
        },
        {
            "epoch_index": 2,
            "causal_group_ids": ("turn-000001",),
            "excluded_tokens": 13,
        },
    ]
