import copy

import pytest

from experiments.paper8_5_agent_memory.run_oracle_subset_queue import (
    _json_digest,
    validate_queue,
)


def _inputs():
    trajectory = {"instance_id": "task", "messages": []}
    reference = {"study": "paper8_5_agent_memory_frozen_replay", "rows": []}
    queue = {
        "study": "paper8_5_oracle_headroom_subset_queue",
        "instance_id": "task",
        "reference_replay_digest": _json_digest(reference),
        "target_depth": 2,
        "trials": [
            {"trial_id": "depth2-001", "omitted_causal_group_ids": ["a", "b"]}
        ],
    }
    return queue, trajectory, reference


def test_subset_queue_validation_accepts_bound_unique_trials():
    validate_queue(*_inputs())


def test_subset_queue_validation_rejects_reference_drift():
    queue, trajectory, reference = _inputs()
    queue["reference_replay_digest"] = "wrong"
    with pytest.raises(ValueError, match="reference"):
        validate_queue(queue, trajectory, reference)


def test_subset_queue_validation_rejects_duplicate_or_wrong_depth():
    queue, trajectory, reference = _inputs()
    duplicate = copy.deepcopy(queue["trials"][0])
    queue["trials"].append(duplicate)
    with pytest.raises(ValueError, match="repeats"):
        validate_queue(queue, trajectory, reference)
    queue["trials"] = [{
        "trial_id": "bad", "omitted_causal_group_ids": ["a"]
    }]
    with pytest.raises(ValueError, match="depth"):
        validate_queue(queue, trajectory, reference)
