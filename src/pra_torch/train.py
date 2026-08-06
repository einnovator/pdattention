"""Generic model-agnostic training runtime and loop."""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint
from .checkpointing import load_checkpoint, serializable_config
from .config import TrainConfig
from .logging import build_logger
from .metrics import RunningAverages, cuda_memory_allocated, grad_norm, perplexity


@dataclass
class TrainingState:
    """Mutable runtime state shared by functional and compatibility APIs."""

    model: torch.nn.Module  # Model being optimized; PRA behavior is injected by batch_step.
    train_config: TrainConfig  # Loop, logging, data, and checkpoint policy.
    device: str  # Resolved device on which the model and batches execute.
    optimizer: torch.optim.Optimizer  # Parameter-update rule.
    scheduler: object  # Learning-rate scheduler stepped after optimizer updates.
    run_dir: Path  # Root for this run's checkpoints, metrics, and traces.
    checkpoint: ModelCheckpoint  # Conventional latest/best checkpoint paths.
    logger: object  # Composite experiment logger.
    scaler: object | None  # CUDA gradient scaler when mixed precision is enabled.
    early_stopping: EarlyStopping  # Validation-loss stopping state.
    global_step: int = 0  # Completed optimizer updates.
    batch_step: int = 0  # Consumed batches, including gradient accumulation.
    start_epoch: int = 0  # Epoch restored from a checkpoint.
    best_val_loss: float = float("inf")  # Best value used for model selection.
    epoch_history: list[dict] = field(default_factory=list)  # Completed epoch summaries.
    checkpoint_extra: Callable[[], dict] | None = None  # Lazy model-specific payload.
    logger_closed: bool = False  # Prevents duplicate close/finalization calls.


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def seed_everything(seed: int) -> None:
    """Seed Python and Torch RNGs used by the training path."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: str) -> dict:
    """Move conventional tensor fields in a collated batch to the target device."""
    return {
        **batch,
        "input_ids": batch["input_ids"].to(device),
        "labels": batch["labels"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


def create_training_state(
    model: torch.nn.Module,
    train_config: TrainConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    checkpoint_extra: Callable[[], dict] | None = None,
) -> TrainingState:
    """Create optimizer, scheduler, logging, checkpoint, and resume state."""
    seed_everything(train_config.seed)
    device = resolve_device(train_config.device)
    model.to(device)
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: max(step, 1) / train_config.warmup_steps
            if train_config.warmup_steps and step < train_config.warmup_steps
            else 1.0,
        )

    # Artifact services are created once and shared by functional/object APIs.
    run_dir = Path(train_config.output_dir) / train_config.experiment_name
    checkpoint = ModelCheckpoint(run_dir / "checkpoints")
    logger = build_logger(train_config, run_dir)
    logger.log_config(train_config)
    state = TrainingState(
        model=model,
        train_config=train_config,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=run_dir,
        checkpoint=checkpoint,
        logger=logger,
        scaler=torch.amp.GradScaler("cuda")
        if train_config.mixed_precision and device.startswith("cuda")
        else None,
        early_stopping=EarlyStopping(train_config.early_stopping_patience),
        checkpoint_extra=checkpoint_extra,
    )
    if train_config.resume_from:
        resume_training_state(state, train_config.resume_from)
    return state


def resume_training_state(state: TrainingState, path: str | Path) -> dict:
    """Restore model and optimizer state into an existing runtime."""
    checkpoint = load_checkpoint(
        path,
        state.model,
        state.optimizer,
        state.scheduler,
        map_location=state.device,
    )
    state.start_epoch = int(checkpoint.get("epoch", 0))
    state.global_step = int(checkpoint.get("global_step", 0))
    state.batch_step = int(checkpoint.get("batch_step", state.global_step))
    state.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    state.early_stopping.best = state.best_val_loss
    return checkpoint


def save_training_state(state: TrainingState, path: str | Path, epoch: int) -> Path:
    """Save a complete functional training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict() if state.scheduler is not None else None,
        "config": serializable_config(state.train_config),
        "epoch": epoch,
        "global_step": state.global_step,
        "batch_step": state.batch_step,
        "best_val_loss": state.best_val_loss,
    }
    if state.checkpoint_extra:
        payload.update(state.checkpoint_extra())
    torch.save(payload, path)
    return path


def close_training_state(state: TrainingState) -> None:
    """Finalize experiment loggers exactly once."""
    if not state.logger_closed:
        state.logger.close()
        state.logger_closed = True


def default_batch_step(model, batch: dict, device: str) -> tuple[torch.Tensor, dict]:
    """Default language-model step used when no custom batch adapter is supplied."""
    batch = move_batch(batch, device)
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=0)
    return loss, {
        "tokens": int(batch["attention_mask"].sum().item()),
        "examples": int(batch["input_ids"].shape[0]),
    }


def validated_extra_metrics(values: dict | None, reserved: set[str]) -> dict[str, float]:
    """Detach scalar batch metrics without allowing generic metric overwrite."""
    result = {}
    for key, value in (values or {}).items():
        if key in reserved or not isinstance(key, str):
            continue
        if not key.startswith(("retrieval_", "memory_", "bucket_", "cache_", "layer_")):
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, (int, float, bool)):
            result[key] = float(value)
    return result


def _validation_checkpoint(state: TrainingState, metrics: dict, epoch: int) -> bool:
    """Persist latest/best state and return whether early stopping fired."""
    val_loss = metrics.get("val_loss", metrics.get("loss", float("inf")))
    improved = val_loss < state.best_val_loss
    if improved:
        state.best_val_loss = val_loss
    save_training_state(state, state.checkpoint.latest_path, epoch)
    if improved:
        save_training_state(state, state.checkpoint.best_path, epoch)
    return state.early_stopping.update(val_loss)


def train_model(
    *,
    model: torch.nn.Module,
    train_config: TrainConfig,
    train_loader,
    val_loader=None,
    test_loader=None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    batch_step: Callable[[torch.nn.Module, dict, str], tuple[torch.Tensor, dict]] = default_batch_step,
    eval_step: Callable[[torch.nn.Module, object, str], dict] | None = None,
    checkpoint_extra: Callable[[], dict] | None = None,
    state: TrainingState | None = None,
):
    """Train with one canonical loop and injected batch/evaluation behavior.

    PRA uses the same optimizer loop as ordinary language modeling. Its adapter
    supplies a ``batch_step`` that constructs reference caches and an ``eval_step``
    that reports retrieval/recursion metrics; this function remains model-agnostic.
    """
    run_start = time.perf_counter()
    state = state or create_training_state(
        model,
        train_config,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_extra=checkpoint_extra,
    )
    if checkpoint_extra is not None:
        state.checkpoint_extra = checkpoint_extra

    model = state.model
    optimizer = state.optimizer
    scheduler = state.scheduler
    last_val_metrics: dict = {}
    epoch_metrics: dict = {}
    stop_training = bool(
        train_config.max_steps is not None and state.global_step >= train_config.max_steps
    )
    last_epoch = state.start_epoch
    last_validation_step = -1
    training_duration_seconds = 0.0
    validation_duration_seconds = 0.0
    test_duration_seconds = 0.0
    processed_tokens = 0
    sequences_seen = 0

    def synchronize() -> None:
        if state.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed_evaluation(loader, split: str):
        synchronize()
        start = time.perf_counter()
        values = eval_step(model, loader, state.device, split=split)
        synchronize()
        duration = max(time.perf_counter() - start, 1e-9)
        values[f"{split}_duration_seconds"] = duration
        return values, duration

    # Epoch/batch control is generic; all PRA-specific work occurs inside batch_step.
    for epoch in range(state.start_epoch, train_config.epochs):
        if stop_training:
            break
        last_epoch = epoch + 1
        model.train()
        averages = RunningAverages()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{train_config.epochs}")
        accum = max(int(train_config.grad_accum_steps), 1)
        epoch_start = time.perf_counter()
        epoch_examples = 0
        epoch_tokens = 0
        epoch_train_duration = 0.0
        batch_count = 0
        last_time = time.perf_counter()
        pending_backward = False

        def optimizer_step() -> float:
            """Apply clipping/update/scheduling after accumulated backward passes."""
            nonlocal pending_backward
            if state.scaler is not None:
                state.scaler.unscale_(optimizer)
            current_grad_norm = grad_norm(model.parameters())
            if train_config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            if state.scaler is not None:
                state.scaler.step(optimizer)
                state.scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            pending_backward = False
            return current_grad_norm

        for batch_idx, batch in enumerate(pbar):
            synchronize()
            train_batch_start = time.perf_counter()
            state.batch_step += 1
            batch_count += 1
            amp_context = torch.amp.autocast("cuda") if state.scaler is not None else nullcontext()
            with amp_context:
                loss, batch_metrics = batch_step(model, batch, state.device)
                scaled_loss = loss / accum
            if state.scaler is not None:
                state.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_backward = True

            did_step = (batch_idx + 1) % accum == 0
            current_grad_norm = optimizer_step() if did_step else 0.0
            synchronize()
            train_batch_duration = max(time.perf_counter() - train_batch_start, 1e-9)
            training_duration_seconds += train_batch_duration
            epoch_train_duration += train_batch_duration
            elapsed = max(time.perf_counter() - last_time, 1e-9)
            last_time = time.perf_counter()
            examples = int(batch_metrics.get("examples", 0))
            tokens = int(batch_metrics.get("tokens", 0))
            epoch_examples += examples
            epoch_tokens += tokens
            sequences_seen += examples
            processed_tokens += tokens
            metrics = {
                "train_loss": float(loss.detach().cpu()),
                "perplexity": perplexity(float(loss.detach().cpu())),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": current_grad_norm,
                "examples_per_second": examples / elapsed,
                "tokens_per_second": tokens / elapsed,
                "gpu_memory_allocated": cuda_memory_allocated(state.device),
                "train_batch_duration_seconds": train_batch_duration,
            }
            metrics.update(validated_extra_metrics(batch_metrics.get("metrics"), set(metrics)))
            averages.update(metrics, weight=max(examples, 1))
            state.logger.log_metrics(
                {
                    **metrics,
                    "epoch": epoch + 1,
                    "batch_in_epoch": batch_idx + 1,
                    "optimizer_step": state.global_step,
                    "optimizer_updated": float(did_step),
                },
                state.batch_step,
                "train_batch",
            )

            # Validation and persistence cadence follows optimizer, not batch, steps.
            if did_step:
                if state.global_step % max(train_config.log_every_steps, 1) == 0:
                    state.logger.log_metrics(metrics, state.global_step, "train")
                if (
                    eval_step
                    and val_loader is not None
                    and state.global_step % max(train_config.eval_every_steps, 1) == 0
                ):
                    last_val_metrics, validation_duration = timed_evaluation(val_loader, "val")
                    validation_duration_seconds += validation_duration
                    last_validation_step = state.global_step
                    state.logger.log_metrics(last_val_metrics, state.global_step, "val")
                    if _validation_checkpoint(state, last_val_metrics, epoch):
                        stop_training = True
                if state.global_step % max(train_config.save_every_steps, 1) == 0:
                    save_training_state(state, state.checkpoint.latest_path, epoch)

            postfix = {
                "step": state.global_step,
                "loss": metrics["train_loss"],
                "lr": metrics["learning_rate"],
                "ex/s": metrics["examples_per_second"],
                "tok/s": metrics["tokens_per_second"],
            }
            if last_val_metrics:
                postfix["val"] = last_val_metrics.get("val_loss", last_val_metrics.get("loss", 0.0))
            pbar.set_postfix(postfix)
            if stop_training or (
                train_config.max_steps is not None and state.global_step >= train_config.max_steps
            ):
                stop_training = True
                break

        if pending_backward and not stop_training:
            optimizer_step()
            if train_config.max_steps is not None and state.global_step >= train_config.max_steps:
                stop_training = True

        # Epoch summaries preserve evolution separately from noisy batch history.
        epoch_metrics = averages.compute()
        epoch_elapsed = max(time.perf_counter() - epoch_start, 1e-9)
        if "train_loss" in epoch_metrics:
            epoch_metrics["perplexity"] = perplexity(epoch_metrics["train_loss"])
        epoch_metrics["examples_per_second"] = epoch_examples / epoch_elapsed
        epoch_metrics["tokens_per_second"] = epoch_tokens / epoch_elapsed
        epoch_metrics.update(
            {
                "epoch": epoch + 1,
                "batches": batch_count,
                "batch_step": state.batch_step,
                "optimizer_step": state.global_step,
                "examples": epoch_examples,
                "tokens": epoch_tokens,
                "duration_seconds": epoch_elapsed,
                "train_duration_seconds": epoch_train_duration,
            }
        )
        state.epoch_history.append(dict(epoch_metrics))
        state.logger.log_metrics(epoch_metrics, epoch + 1, "train_epoch")
        if eval_step and val_loader is not None:
            if last_validation_step != state.global_step:
                last_val_metrics, validation_duration = timed_evaluation(val_loader, "val")
                validation_duration_seconds += validation_duration
                last_validation_step = state.global_step
                state.logger.log_metrics(last_val_metrics, state.global_step, "val")
                if _validation_checkpoint(state, last_val_metrics, epoch + 1):
                    stop_training = True
            state.logger.log_metrics(
                {
                    **last_val_metrics,
                    "epoch": epoch + 1,
                    "optimizer_step": state.global_step,
                    "validation_duration_seconds": last_val_metrics.get("val_duration_seconds", 0.0),
                },
                epoch + 1,
                "val_epoch",
            )

    # Final test/timing records use the same injected evaluator and then close state.
    test_metrics: dict = {}
    if eval_step and test_loader is not None:
        test_metrics, test_duration_seconds = timed_evaluation(test_loader, "test")
        state.logger.log_metrics(test_metrics, state.global_step, "test")
    synchronize()
    wall_clock_seconds = max(time.perf_counter() - run_start, 1e-9)
    timing_metrics = {
        "wall_clock_seconds": wall_clock_seconds,
        "train_duration_seconds": training_duration_seconds,
        "validation_duration_seconds": validation_duration_seconds,
        "test_duration_seconds": test_duration_seconds,
        "processed_tokens": processed_tokens,
        "sequences_seen": sequences_seen,
        "optimizer_steps": state.global_step,
        "optimizer_steps_per_second": state.global_step / max(training_duration_seconds, 1e-9),
        "training_tokens_per_second": processed_tokens / max(training_duration_seconds, 1e-9),
    }
    state.logger.log_metrics(timing_metrics, state.global_step, "run")
    save_training_state(state, state.checkpoint.latest_path, last_epoch)
    close_training_state(state)
    return {
        "state": state,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": state.global_step,
        "batch_step": state.batch_step,
        "best_val_loss": state.best_val_loss,
        "train_metrics": epoch_metrics,
        "epoch_metrics": list(state.epoch_history),
        "val_metrics": last_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_dir": state.checkpoint.checkpoint_dir,
        "timing_metrics": timing_metrics,
    }
