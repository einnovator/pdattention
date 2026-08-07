from pathlib import Path

import torch

from common.checkpointing import load_checkpoint, serializable_config


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


__all__ = [
    "checkpoint_payload",
    "load_checkpoint",
    "save_checkpoint",
    "serializable_config",
]
