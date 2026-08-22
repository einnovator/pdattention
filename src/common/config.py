"""Configuration shared by model-agnostic experiment infrastructure."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from common.distributed.models import (
    ClusterConfig,
    DistributionMode,
    WorkerConfig,
    implicit_local_cluster,
    implicit_local_worker,
    select_cluster,
)
from common.experiments.models import ExperimentDefinition
from common.experiments.sweep import set_dotted
from common.storage.registry import StorageConfig, StorageRegistry


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


def discover_yaml_files(path: str | Path) -> list[Path]:
    """Resolve a file or recursively sorted, non-hidden YAML directory."""

    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError(f"Config file must be YAML: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".yml", ".yaml"}
        and not any(part.startswith(".") for part in candidate.relative_to(path).parts)
    ]
    return sorted(files, key=lambda item: item.relative_to(path).as_posix().casefold())


def resolve_config_sources(paths: Iterable[str | Path]) -> list[Path]:
    """Expand config arguments in authoritative command-line order."""

    return [source for path in paths for source in discover_yaml_files(path)]


def load_yaml_config(*paths: str | Path, base: dict | None = None) -> dict:
    """Load files or recursively discovered directories in argument order."""
    config = copy.deepcopy(base or {})
    for path in resolve_config_sources(paths):
        deep_update(config, read_yaml(path))
    return config


def load_config_sources(*paths: str | Path, base: dict | None = None) -> tuple[dict, list[str]]:
    """Return merged YAML and normalized source provenance."""

    sources = resolve_config_sources(paths)
    config = copy.deepcopy(base or {})
    for source in sources:
        deep_update(config, read_yaml(source))
    return config, [str(source.resolve()) for source in sources]


def parse_override(value: str) -> tuple[str, Any]:
    """Parse ``PATH=YAML_VALUE`` while retaining YAML scalar/list types."""

    if "=" not in value:
        raise ValueError(f"Override must use PATH=VALUE syntax: {value!r}")
    path, raw = value.split("=", 1)
    if not path.strip():
        raise ValueError("Override path cannot be empty.")
    return path.strip(), yaml.safe_load(raw)


def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    """Apply repeatable dotted CLI overrides after all configuration files."""

    for value in overrides:
        path, parsed = parse_override(value)
        set_dotted(config, path, parsed)
    return config


@dataclass(frozen=True)
class InfrastructureConfig:
    """Validated model-independent registries synthesized from merged YAML."""

    raw: Mapping[str, Any]
    workers: Mapping[str, WorkerConfig]
    clusters: Mapping[str, ClusterConfig]
    storage: Mapping[str, StorageConfig]
    experiments: Mapping[str, ExperimentDefinition]
    sources: tuple[str, ...] = ()

    def cluster(self, requested: str | None = None) -> ClusterConfig:
        return select_cluster(self.clusters, requested)

    def storage_registry(self) -> StorageRegistry:
        return StorageRegistry(self.storage)


def resolve_storage_name(
    infrastructure: InfrastructureConfig,
    *,
    cli: str | None = None,
    experiment: ExperimentDefinition | None = None,
    cluster: ClusterConfig | None = None,
    worker: WorkerConfig | None = None,
) -> str:
    """Resolve storage from most specific CLI setting to implicit local."""

    selected = (
        cli
        or (experiment.storage if experiment else None)
        or (cluster.default_storage if cluster else None)
        or (worker.default_storage if worker else None)
        or "local"
    )
    if selected not in infrastructure.storage:
        raise ValueError(f"Unknown storage {selected!r}.")
    return selected


def resolve_infrastructure(
    config: Mapping[str, Any], *, sources: Iterable[str] = ()
) -> InfrastructureConfig:
    """Parse registries, synthesize local defaults, and validate references."""

    worker_values = dict(config.get("workers") or {})
    workers = {
        str(name): WorkerConfig.from_mapping(str(name), value)
        for name, value in worker_values.items()
    }
    workers.setdefault("local", implicit_local_worker())

    cluster_values = dict(config.get("clusters") or {})
    clusters = {
        str(name): ClusterConfig.from_mapping(str(name), value)
        for name, value in cluster_values.items()
    }
    clusters.setdefault("local", implicit_local_cluster())

    storage_values = dict(config.get("storage") or config.get("storages") or {})
    storage = {
        str(name): StorageConfig.from_mapping(str(name), value)
        for name, value in storage_values.items()
    }
    storage.setdefault(
        "local", StorageConfig.from_mapping("local", {"type": "local", "path": "out/experiments"})
    )

    # Historical PRA profile entries also live under ``experiments``. Only entries
    # declaring the generic module/file contract belong to this runner.
    experiments = {
        str(name): ExperimentDefinition.from_mapping(str(name), value)
        for name, value in (config.get("experiments") or {}).items()
        if isinstance(value, Mapping) and (value.get("module") or value.get("file"))
    }

    # Selection also detects multiple defaults.
    select_cluster(clusters)
    for cluster in clusters.values():
        resolved = cluster.resolved_workers(workers)
        if cluster.default_storage and cluster.default_storage not in storage:
            raise ValueError(
                f"Cluster {cluster.name!r} references unknown storage {cluster.default_storage!r}."
            )
        if cluster.distribution.cooperative and len(resolved) < 1:
            raise ValueError(f"Cooperative cluster {cluster.name!r} requires workers.")
    for worker in workers.values():
        if worker.default_storage and worker.default_storage not in storage:
            raise ValueError(
                f"Worker {worker.name!r} references unknown storage {worker.default_storage!r}."
            )
    for experiment in experiments.values():
        if experiment.cluster and experiment.cluster not in clusters:
            raise ValueError(
                f"Experiment {experiment.name!r} references unknown cluster {experiment.cluster!r}."
            )
        for selected_storage in (
            experiment.storage,
            experiment.results_storage,
            experiment.checkpoint_storage,
        ):
            if selected_storage and selected_storage not in storage:
                raise ValueError(
                    f"Experiment {experiment.name!r} references unknown storage {selected_storage!r}."
                )
        selected_cluster = clusters[experiment.cluster] if experiment.cluster else select_cluster(clusters)
        distribution = experiment.distribution or selected_cluster.distribution
        eligible = [
            worker
            for worker in selected_cluster.resolved_workers(workers)
            if worker.satisfies(experiment.resources)
        ]
        if len(eligible) < experiment.resources.workers:
            raise ValueError(
                f"Experiment {experiment.name!r} requests {experiment.resources.workers} workers "
                f"but cluster {selected_cluster.name!r} can satisfy only {len(eligible)}."
            )
        if distribution in {DistributionMode.DDP, DistributionMode.FSDP} and len(eligible) < 1:
            raise ValueError(f"Experiment {experiment.name!r} has no cooperative workers.")

    return InfrastructureConfig(
        raw=config,
        workers=workers,
        clusters=clusters,
        storage=storage,
        experiments=experiments,
        sources=tuple(sources),
    )


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
    distribution_strategy: str = "local"  # local, ddp, or fsdp model execution strategy.
    distributed_backend: str = "auto"  # auto selects NCCL for CUDA and Gloo otherwise.
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
        if self.distribution_strategy not in {"local", "ddp", "fsdp", "pipeline"}:
            raise ValueError(f"Unsupported distribution_strategy: {self.distribution_strategy}")
