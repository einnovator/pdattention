"""Configuration shared by model-agnostic experiment infrastructure."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge nested dictionaries into ``base`` in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping, returning an empty mapping for an empty document."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(payload).__name__}.")
    return payload


def load_yaml_config(*paths: str | Path, base: dict | None = None) -> dict:
    """Load and recursively merge YAML mappings in argument order."""
    config = copy.deepcopy(base or {})
    for path in paths:
        deep_update(config, read_yaml(path))
    return config


@dataclass
class TrainConfig:
    """Training, data-loader, logging, and artifact settings.

    Model architecture and experiment-specific services belong in a package-specific
    configuration that may subclass this base class.
    """

    experiment_name: str = "experiment"  # Run name used as the artifact-directory suffix.
    output_dir: str = "out"  # Parent directory for logs, metrics, plots, and checkpoints.
    seed: int = 0  # Seed applied to Python, NumPy, and PyTorch random generators.
    device: str = "auto"  # Execution device; auto selects CUDA when it is available.
    dtype: str = "float32"  # Requested numeric dtype for model-specific adapters to honor.

    epochs: int = 3  # Maximum complete passes over the training dataloader.
    max_steps: int | None = None  # Optional cap on optimizer updates across all epochs.
    batch_size: int = 8  # Number of examples loaded in each logical training batch.
    grad_accum_steps: int = 1  # Batches accumulated before each optimizer update.
    learning_rate: float = 3e-4  # Initial AdamW learning rate.
    weight_decay: float = 0.0  # AdamW decoupled L2 regularization coefficient.
    warmup_steps: int = 0  # Optimizer updates used to linearly ramp the learning rate.
    max_grad_norm: float = 1.0  # Global gradient-norm clipping threshold; zero disables it.

    eval_every_steps: int = 50  # Optimizer-update interval for validation.
    save_every_steps: int = 100  # Optimizer-update interval for latest checkpoints.
    log_every_steps: int = 10  # Optimizer-update interval for aggregate training logs.
    resume_from: str | None = None  # Optional checkpoint path restored before training.
    early_stopping_patience: int | None = None  # Non-improving validations allowed before stop.

    num_workers: int = 0  # Worker processes used by model-specific dataloaders.
    pin_memory: bool = False  # Ask dataloaders to stage CPU tensors in pinned memory.
    persistent_workers: bool = False  # Keep dataloader workers alive between epochs.
    dataset_stage: str = "dataset"  # Dataset split, stage, or generated-data identifier.
    data_dir: str = "data"  # Root directory from which dataset adapters load artifacts.
    max_examples: int | None = None  # Optional dataset-size limit applied by adapters.
    max_seq_len: int = 96  # Maximum token sequence length prepared by data adapters.
    shuffle: bool = True  # Randomize training-example order in supporting dataloaders.

    mixed_precision: bool = False  # Enable CUDA autocast and gradient scaling when supported.
    use_tensorboard: bool = True  # Write TensorBoard event records for the run.
    save_metric_plots: bool = True  # Render local PNG plots when metric history closes.
    use_wandb: bool = False  # Send scalar metrics to Weights & Biases.
    wandb_project: str = "transformer-experiments"  # Weights & Biases destination project.
    use_clearml: bool = False  # Send configuration and scalar metrics to ClearML.
    clearml_project: str = "Transformer Experiments"  # ClearML destination project.

    def __post_init__(self) -> None:
        """Normalize numeric values and reject invalid loop settings early."""
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.grad_accum_steps = int(self.grad_accum_steps)
        self.warmup_steps = int(self.warmup_steps)
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive.")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be non-negative when configured.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
