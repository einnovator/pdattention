"""Model-independent process-group context and PyTorch strategy helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch

from .models import DistributionMode


@dataclass(frozen=True)
class DistributedContext:
    """Rank identity shared by generic training and experiment adapters."""

    strategy: str = "local"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    worker_name: str = "local"
    cluster_name: str = "local"
    backend: str | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_environment(
        cls,
        *,
        strategy: str = "ddp",
        worker_name: str = "local",
        cluster_name: str = "local",
        backend: str | None = None,
    ) -> "DistributedContext":
        return cls(
            strategy=strategy,
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            worker_name=worker_name,
            cluster_name=cluster_name,
            backend=backend,
        )


def select_backend(requested: str = "auto", device: str = "cpu") -> str:
    """Select NCCL only for CUDA; Gloo remains the portable CPU/macOS path."""

    if requested != "auto":
        if requested == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL requires an available CUDA runtime.")
        return requested
    return "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"


def init_process_group(
    context: DistributedContext | None = None,
    *,
    backend: str = "auto",
    device: str = "cpu",
    timeout_seconds: int = 300,
) -> DistributedContext:
    """Initialize a process group only when world size is greater than one."""

    context = context or DistributedContext.from_environment()
    if not context.distributed:
        return context
    if not torch.distributed.is_available():
        raise RuntimeError("This PyTorch build does not provide torch.distributed.")
    selected = select_backend(backend, device)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=selected,
            init_method="env://",
            rank=context.rank,
            world_size=context.world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
    return DistributedContext(**{**context.__dict__, "backend": selected})


def destroy_process_group() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def barrier(context: DistributedContext | None = None) -> None:
    if (
        (context is None or context.distributed)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()


def reduce_metrics(metrics: dict[str, float], context: DistributedContext) -> dict[str, float]:
    """Average scalar metrics across ranks while preserving local behavior."""

    if not context.distributed or not torch.distributed.is_initialized():
        return dict(metrics)
    keys = sorted(metrics)
    device = "cuda" if context.backend == "nccl" else "cpu"
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values /= context.world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def broadcast_object(value: Any, context: DistributedContext, source: int = 0) -> Any:
    if not context.distributed or not torch.distributed.is_initialized():
        return value
    payload = [value if context.rank == source else None]
    torch.distributed.broadcast_object_list(payload, src=source)
    return payload[0]


def wrap_model(model: torch.nn.Module, context: DistributedContext, device: str):
    """Wrap a model for DDP/FSDP while keeping local execution unchanged."""

    strategy = DistributionMode.from_value(context.strategy)
    if not context.distributed or strategy == DistributionMode.LOCAL:
        return model
    if strategy == DistributionMode.DDP:
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [context.local_rank] if device.startswith("cuda") else None
        return DistributedDataParallel(model, device_ids=device_ids)
    if strategy == DistributionMode.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel(model)
    if strategy == DistributionMode.PIPELINE:
        raise NotImplementedError("Pipeline training is reserved but not implemented.")
    raise ValueError(f"Strategy {strategy.value!r} is not a cooperative model strategy.")


def make_distributed_sampler(dataset, context: DistributedContext, *, shuffle: bool = True):
    if not context.distributed:
        return None
    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
    )


def distribute_dataloader(loader, context: DistributedContext, *, shuffle: bool = True):
    """Clone a DataLoader with a rank-aware sampler while retaining its public policy."""

    if not context.distributed:
        return loader
    sampler = make_distributed_sampler(loader.dataset, context, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor if loader.num_workers else None,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )
