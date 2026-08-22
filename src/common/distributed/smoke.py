"""Minimal CPU/Gloo job for installation and CI smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from common.experiments.state import atomic_write_json

from .context import DistributedContext, destroy_process_group, init_process_group, wrap_model


def ddp_rank(rank: int, output_path: str) -> None:
    """Run one synchronized optimizer update and record rank-zero parameters."""

    context = init_process_group(
        DistributedContext.from_environment(strategy="ddp"), backend="gloo", device="cpu"
    )
    torch.manual_seed(7)
    model = wrap_model(torch.nn.Linear(2, 1), context, "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    value = torch.tensor([[float(rank), 1.0]])
    loss = model(value).square().mean()
    loss.backward()
    optimizer.step()
    if context.is_main:
        source = model.module if hasattr(model, "module") else model
        atomic_write_json(
            Path(output_path),
            {"world_size": context.world_size, "weight": source.weight.detach().tolist()},
        )
    destroy_process_group()
