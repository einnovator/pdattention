from pathlib import Path
from dataclasses import asdict, is_dataclass

import torch


def serializable_config(config) -> dict:
    """Return config as plain Python containers safe for weights-only loading."""
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return dict(config)


def checkpoint_payload(model, optimizer, scheduler, config, epoch, global_step, best_val_loss, tokenizer) -> dict:
    """Create the serializable checkpoint dictionary."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": serializable_config(config),
        "cfg": model.cfg.__dict__,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "reference_vocabulary": {k: v for k, v in tokenizer.stoi.items() if k.startswith("<REF_")},
    }


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, global_step, best_val_loss, tokenizer) -> Path:
    """Save a checkpoint and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, scheduler, config, epoch, global_step, best_val_loss, tokenizer), path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    """Load a checkpoint into the provided model and optional optimizer state."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
