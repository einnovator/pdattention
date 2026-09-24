import json

import pytest

from experiments.paper8_5_agent_memory.replay_trajectory_prefix import _load_prefix


def test_load_prefix_strips_provider_fields_and_returns_expected_action(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"messages": [
        {"role": "system", "content": "s", "extra": {"ignored": True}},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "tool_calls": None},
        {"role": "user", "content": "observation", "extra": {"raw": "x"}},
        {"role": "assistant", "content": "next"},
    ]}), encoding="utf-8")

    prefix, expected = _load_prefix(path, 3)

    assert prefix == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "observation"},
    ]
    assert expected == "next"


def test_load_prefix_requires_following_assistant(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]}), encoding="utf-8")

    with pytest.raises(ValueError, match="not an assistant"):
        _load_prefix(path, 0)
