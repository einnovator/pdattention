from __future__ import annotations

import argparse

import pytest

from experiments.paper8_5_agent_memory.run_mlx_server_bounded import (
    _bytes_from_gib,
    _server_args,
)


def test_bytes_from_gib() -> None:
    assert _bytes_from_gib(0) == 0
    assert _bytes_from_gib(1.5) == 1_610_612_736


def test_bytes_from_gib_rejects_negative_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _bytes_from_gib(-1)


def test_disable_thinking_is_forwarded_explicitly() -> None:
    assert _server_args(disable_thinking=True, forwarded=["--port", "8081"]) == [
        "--port",
        "8081",
        "--chat-template-args",
        '{"enable_thinking": false}',
    ]


def test_disable_thinking_rejects_ambiguous_template_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        _server_args(
            disable_thinking=True,
            forwarded=["--chat-template-args", "{}"],
        )
