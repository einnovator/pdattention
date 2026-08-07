"""Language-model adapters for the generic functional training engine."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from common.metrics import cuda_memory_allocated, perplexity
from common.train import create_training_state, default_batch_step, move_batch, train_model
from .config import PRAConfig, TrainConfig
from .model import TinyPRAModel


def evaluate_language_model(*, model, loader, device: str, split: str = "val") -> dict:
    was_training = model.training
    model.eval()
    loss_sum = token_count = correct = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["input_ids"], use_pra_memory=False)
            flat_labels = batch["labels"].reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_labels)
            count = int(flat_labels.numel())
            loss_sum += float(loss.detach().cpu()) * count
            token_count += count
            correct += int((logits.argmax(dim=-1).reshape(-1) == flat_labels).sum().item())
    if was_training:
        model.train()
    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = loss_sum / max(token_count, 1)
    return {
        f"{split}_loss": loss,
        "loss": loss,
        "perplexity": perplexity(loss),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
        "tokens_per_second": token_count / elapsed,
        "latency": elapsed,
        "gpu_memory_allocated": cuda_memory_allocated(device),
    }


def train_language_model(
    *,
    cfg: PRAConfig,
    train_config: TrainConfig,
    datamodule,
    model: TinyPRAModel | None = None,
):
    tokenizer = datamodule.tokenizer
    model = model or TinyPRAModel(cfg)
    checkpoint_extra = lambda: {
        "cfg": cfg.__dict__,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "tokenizer_type": type(tokenizer).__name__,
        "tokenizer_json": tokenizer.to_json() if hasattr(tokenizer, "to_json") else None,
        "dataset_stage": train_config.dataset_stage,
    }
    state = create_training_state(model, train_config, checkpoint_extra=checkpoint_extra)

    def eval_step(current_model, loader, device: str, split: str = "val"):
        return evaluate_language_model(model=current_model, loader=loader, device=device, split=split)

    def batch_step(current_model, batch, device: str):
        # PRA data uses token zero for padded/ignored labels.
        return default_batch_step(current_model, batch, device, ignore_index=0)

    return train_model(
        model=state.model,
        train_config=train_config,
        train_loader=datamodule.train_loader(),
        val_loader=datamodule.val_loader(),
        test_loader=datamodule.test_loader(),
        batch_step=batch_step,
        eval_step=eval_step,
        state=state,
    )
