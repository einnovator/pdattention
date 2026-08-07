"""Model-agnostic checkpoint serialization helpers."""

from dataclasses import asdict, is_dataclass

import torch


def serializable_config(config) -> dict:
    """Return a configuration as plain containers suitable for checkpoints."""
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return dict(config)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    """Restore a model and optional optimizer/scheduler from a checkpoint."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
