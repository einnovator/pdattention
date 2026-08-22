"""Portable launcher for cooperative local PyTorch trials."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable

import torch.multiprocessing as mp


def _rank_entry(rank: int, function: Callable, world_size: int, port: int, payload: tuple) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    function(rank, *payload)


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_local(
    function: Callable,
    *,
    world_size: int,
    args: tuple = (),
    master_port: int | None = None,
) -> None:
    """Spawn a local rank group using the same environment contract as torchrun."""

    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    port = master_port or free_local_port()

    mp.start_processes(
        _rank_entry,
        args=(function, world_size, port, args),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
