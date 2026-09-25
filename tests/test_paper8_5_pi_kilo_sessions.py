from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.run_kilo_swebench import (
    KILO_DATA_PATH,
    _native_completion_observed,
    _session_arguments as kilo_session_arguments,
)


def test_kilo_native_completion_requires_terminal_non_tool_step(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete.jsonl"
    complete.write_text(
        '{"type":"tool_use","part":{"type":"tool"}}\n'
        '{"type":"step_finish","part":{"reason":"stop"}}\n',
        encoding="utf-8",
    )
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"type":"step_finish","part":{"reason":"tool-calls"}}\n'
        '{"type":"tool_use","part":{"type":"tool"}}\n',
        encoding="utf-8",
    )
    assert _native_completion_observed(complete) is True
    assert _native_completion_observed(partial) is False
from experiments.paper8_5_agent_memory.run_pi_swebench import (
    PI_SESSION_PATH,
    _session_arguments as pi_session_arguments,
)


def test_pi_persistent_session_uses_one_dedicated_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session-store"):
        pi_session_arguments(None, True)
    docker_args, pi_args, store = pi_session_arguments(
        str(tmp_path / "pi"), True
    )
    assert store is not None and store.is_dir()
    assert docker_args[-1].endswith(f"target={PI_SESSION_PATH}")
    assert pi_args == ("--session-dir", PI_SESSION_PATH, "--continue")


def test_kilo_persistent_session_replaces_memory_database(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session-store"):
        kilo_session_arguments(None, True)
    docker_args, env_args, kilo_args, store = kilo_session_arguments(
        str(tmp_path / "kilo"), True
    )
    assert store is not None and store.is_dir()
    assert docker_args[-1].endswith(f"target={KILO_DATA_PATH}")
    assert env_args == ("--env", f"KILO_DB={KILO_DATA_PATH}/kilo.db")
    assert kilo_args == ("--continue",)
