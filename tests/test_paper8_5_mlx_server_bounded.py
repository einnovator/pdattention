from __future__ import annotations

import argparse

import pytest

from experiments.paper8_5_agent_memory.run_mlx_server_bounded import _bytes_from_gib


def test_bytes_from_gib() -> None:
    assert _bytes_from_gib(0) == 0
    assert _bytes_from_gib(1.5) == 1_610_612_736


def test_bytes_from_gib_rejects_negative_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _bytes_from_gib(-1)
