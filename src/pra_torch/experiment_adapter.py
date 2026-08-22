"""Thin PRA adapter for the model-independent experiment runner."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from common.config import deep_update
from common.experiments.models import ExperimentContext
from common.train import resolve_device
from data.datamodules import PRADataModule

from .config import CacheServiceConfig, ResolverServiceConfig, TrainConfig
from .pra_train import train_pra_model


def _scalar_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for section in ("train_metrics", "val_metrics", "test_metrics", "timing_metrics"):
        for key, value in (result.get(section) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                metrics[key] = float(value)
    for key in ("global_step", "batch_step", "best_val_loss"):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            metrics[key] = float(value)
    return metrics


def run_pra_training_config(
    cfg: dict,
    *,
    output_path: str | Path | None = None,
    distribution_strategy: str = "local",
    distributed_backend: str = "auto",
) -> dict:
    """Build PRA data/services and call the existing generic optimizer loop."""

    train_cfg = cfg["train"]
    seed = int(train_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(train_cfg["device"])
    datamodule = PRADataModule(
        dataset_stage=train_cfg["dataset_stage"],
        data_dir=train_cfg["data_dir"],
        max_examples=train_cfg["max_examples"],
        batch_size=int(train_cfg["batch_size"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        shuffle=bool(train_cfg["shuffle"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]),
    ).load()

    # Imports stay local to keep the generic runner independent of PRA's CLI module.
    from .cli import build_pra_config

    pra_cfg = build_pra_config(
        cfg,
        vocab_size=datamodule.tokenizer.vocab_size,
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["lr"]),
        steps=int(train_cfg["steps"]),
        device=device,
    )
    destination = Path(output_path or train_cfg["out"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime_cfg = TrainConfig(
        experiment_name=destination.stem,
        output_dir=str(destination.parent if destination.parent != Path(".") else "out"),
        seed=seed,
        device=device,
        epochs=1,
        max_steps=int(train_cfg["steps"]) if int(train_cfg["steps"]) > 0 else None,
        batch_size=int(train_cfg["batch_size"]),
        grad_accum_steps=1,
        learning_rate=float(train_cfg["lr"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]),
        dataset_stage=train_cfg["dataset_stage"],
        data_dir=train_cfg["data_dir"],
        max_examples=train_cfg["max_examples"],
        shuffle=bool(train_cfg["shuffle"]),
        eval_every_steps=max(int(train_cfg["steps"]) + 1, 1),
        save_every_steps=max(int(train_cfg["steps"]) + 1, 1),
        log_every_steps=1,
        use_tensorboard=False,
        distribution_strategy=distribution_strategy,
        distributed_backend=distributed_backend,
        resolver_config=ResolverServiceConfig.from_value(cfg["resolver"]),
        cache_config=CacheServiceConfig.from_value(cfg["cache"]),
    )
    result = train_pra_model(
        cfg=pra_cfg,
        train_config=runtime_cfg,
        datamodule=datamodule,
        resolver_config=runtime_cfg.resolver_config,
        cache_config=runtime_cfg.cache_config,
    )
    if result["state"].distributed.is_main:
        model = result["model"].module if hasattr(result["model"], "module") else result["model"]
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": pra_cfg.__dict__,
                "stoi": datamodule.tokenizer.stoi,
                "itos": datamodule.tokenizer.itos,
                "dataset_stage": train_cfg["dataset_stage"],
                "resolver_config": runtime_cfg.resolver_config.__dict__,
                "cache_config": runtime_cfg.cache_config.__dict__,
            },
            destination,
        )
    return {"result": result, "checkpoint": destination, "metrics": _scalar_metrics(result)}


def run_pra_training_experiment(params: dict, context: ExperimentContext) -> dict:
    """Stable experiment callable used for seeds, sweeps, and remote workers.

    ``params._resolved_config`` may carry a fully layered CLI configuration. Other
    nested ``model``, ``pra``, ``train``, ``resolver``, and ``cache`` values are
    merged after it, and common flat training aliases remain convenient in YAML.
    """

    from .cli import load_config

    values = copy.deepcopy(params)
    config_paths = values.pop("config_paths", ())
    cfg = values.pop("_resolved_config", None) or load_config(config_paths)
    cfg = copy.deepcopy(cfg)
    for section in ("model", "pra", "train", "resolver", "cache"):
        if section in values:
            deep_update(cfg.setdefault(section, {}), values.pop(section) or {})
    aliases = {
        "seed": "seed",
        "steps": "steps",
        "batch_size": "batch_size",
        "lr": "lr",
        "device": "device",
        "dataset": "dataset_stage",
        "dataset_stage": "dataset_stage",
        "data_dir": "data_dir",
        "max_examples": "max_examples",
    }
    for source, target in aliases.items():
        if source in values:
            cfg["train"][target] = values.pop(source)
    if values:
        unknown = ", ".join(sorted(values))
        raise ValueError(f"Unknown PRA experiment parameters: {unknown}")
    checkpoint = context.output_dir / "checkpoints" / Path(cfg["train"]["out"]).name
    outcome = run_pra_training_config(
        cfg,
        output_path=checkpoint,
        distribution_strategy=context.strategy,
        distributed_backend=str(cfg.get("distributed_backend", "auto")),
    )
    checkpoint_bytes = float(checkpoint.stat().st_size) if context.rank == 0 else 0.0
    return {**outcome["metrics"], "checkpoint_bytes": checkpoint_bytes}
