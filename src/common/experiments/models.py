"""Typed experiment, trial, context, and lifecycle models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from common.distributed.models import DistributionMode, ResourceRequirements


class TrialState(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELLED,
            self.INTERRUPTED,
            self.SKIPPED,
        }


@dataclass(frozen=True)
class ExperimentEntrypoint:
    """Importable callable or repository-local Python file."""

    module: str | None = None
    file: str | None = None
    function: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentEntrypoint":
        module = value.get("module")
        file = value.get("file")
        function = value.get("function")
        if bool(module) == bool(file):
            raise ValueError("Experiment must define exactly one of module or file.")
        if module and not function:
            raise ValueError("Module experiments require function.")
        return cls(
            module=str(module) if module else None,
            file=str(file) if file else None,
            function=str(function) if function else None,
        )

    @property
    def script_only(self) -> bool:
        return bool(self.file and not self.function)

    def as_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 2
    backoff_seconds: float = 5.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RetryConfig":
        value = value or {}
        attempts = int(value.get("max_attempts", 2))
        backoff = float(value.get("backoff_seconds", 5.0))
        if attempts <= 0 or backoff < 0:
            raise ValueError("retry.max_attempts must be positive and backoff non-negative.")
        return cls(attempts, backoff)


@dataclass(frozen=True)
class ExperimentDefinition:
    """Reusable declaration that expands to one or more deterministic trials."""

    name: str
    entrypoint: ExperimentEntrypoint
    cluster: str | None = None
    distribution: DistributionMode | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    sweep: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    trials: tuple[Mapping[str, Any], ...] = ()
    storage: str | None = None
    results_storage: str | None = None
    checkpoint_storage: str | None = None
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    retry: RetryConfig = field(default_factory=RetryConfig)
    tags: tuple[str, ...] = ()
    description: str | None = None
    notes: str | None = None
    timeout_seconds: int | None = None

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "ExperimentDefinition":
        sweep = value.get("sweep") or {}
        explicit = tuple(value.get("trials") or ())
        if sweep and explicit:
            raise ValueError(f"Experiment {name!r} cannot define both sweep and trials.")
        normalized_sweep = {}
        for key, choices in sweep.items():
            if not isinstance(choices, (list, tuple)) or not choices:
                raise ValueError(f"Experiment {name!r} sweep {key!r} must be a non-empty list.")
            normalized_sweep[str(key)] = tuple(choices)
        timeout = value.get("timeout_seconds")
        return cls(
            name=name,
            entrypoint=ExperimentEntrypoint.from_mapping(value),
            cluster=value.get("cluster"),
            distribution=(
                DistributionMode.from_value(value["distribution"])
                if value.get("distribution") is not None
                else None
            ),
            parameters=dict(value.get("parameters") or {}),
            sweep=normalized_sweep,
            trials=explicit,
            storage=value.get("storage"),
            results_storage=value.get("results_storage"),
            checkpoint_storage=value.get("checkpoint_storage"),
            resources=ResourceRequirements.from_mapping(value.get("resources")),
            retry=RetryConfig.from_mapping(value.get("retry")),
            tags=tuple(str(item) for item in value.get("tags", ())),
            description=value.get("description"),
            notes=value.get("notes"),
            timeout_seconds=int(timeout) if timeout is not None else None,
        )


@dataclass(frozen=True)
class ExperimentContext:
    """Model-independent runtime identity passed to experiment callables."""

    experiment_name: str
    trial_id: str
    run_id: str
    worker_name: str
    cluster_name: str
    role: str
    rank: int
    local_rank: int
    world_size: int
    output_dir: Path
    storage_name: str | None
    strategy: str = "local"
    resumed: bool = False

    def as_dict(self) -> dict:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentContext":
        return cls(**{**value, "output_dir": Path(value["output_dir"])})


@dataclass
class Trial:
    """Resolved immutable trial inputs plus mutable scheduler assignment."""

    experiment_name: str
    trial_id: str
    parameters: dict[str, Any]
    entrypoint: ExperimentEntrypoint
    distribution: DistributionMode
    cluster_name: str
    storage_name: str | None
    resources: ResourceRequirements
    fingerprint: str
    state: TrialState = TrialState.PENDING
    assigned_workers: tuple[str, ...] = ()
    attempt: int = 0

    def manifest(self) -> dict:
        return {
            "experiment": self.experiment_name,
            "trial_id": self.trial_id,
            "fingerprint": self.fingerprint,
            "parameters": self.parameters,
            "cluster": self.cluster_name,
            "workers": list(self.assigned_workers),
            "distribution": self.distribution.value,
            "entrypoint": self.entrypoint.as_dict(),
            "storage": self.storage_name,
            "resources": asdict(self.resources),
        }
