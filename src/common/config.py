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

    experiment_name: str = "experiment"
    output_dir: str = "out"
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    epochs: int = 3
    max_steps: int | None = None
    batch_size: int = 8
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    eval_every_steps: int = 50
    save_every_steps: int = 100
    log_every_steps: int = 10
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    resume_from: str | None = None
    use_tensorboard: bool = True
    save_metric_plots: bool = True
    use_wandb: bool = False
    wandb_project: str = "transformer-experiments"
    use_clearml: bool = False
    clearml_project: str = "Transformer Experiments"
    mixed_precision: bool = False
    early_stopping_patience: int | None = None
    dataset_stage: str = "dataset"
    data_dir: str = "data"
    max_examples: int | None = None
    max_seq_len: int = 96
    shuffle: bool = True

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
