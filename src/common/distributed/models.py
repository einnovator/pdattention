"""Typed worker, cluster, and resource contracts for research orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class DistributionMode(str, Enum):
    """Normalized independent-job and cooperative-training strategies."""

    LOCAL = "local"
    TRIALS = "trials"
    SEEDS = "seeds"
    SWEEP = "sweep"
    DDP = "ddp"
    FSDP = "fsdp"
    PIPELINE = "pipeline"

    @classmethod
    def from_value(cls, value: str | "DistributionMode" | None) -> "DistributionMode":
        if isinstance(value, cls):
            return value
        normalized = str(value or "local").strip().lower().replace("-", "_")
        normalized = {
            "independent": "trials",
            "multi_seed": "seeds",
            "grid": "sweep",
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"Unsupported distribution mode {value!r}; expected {choices}.") from exc

    @property
    def cooperative(self) -> bool:
        return self in {self.DDP, self.FSDP, self.PIPELINE}


@dataclass(frozen=True)
class ResourceRequirements:
    """Resources that one independent trial or cooperative trial requests."""

    device: str = "auto"
    min_memory_gb: float | None = None
    tags: tuple[str, ...] = ()
    workers: int = 1
    exclusive: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ResourceRequirements":
        value = value or {}
        workers = int(value.get("workers", 1))
        if workers <= 0:
            raise ValueError("resources.workers must be positive.")
        memory = value.get("min_memory_gb")
        return cls(
            device=str(value.get("device", "auto")),
            min_memory_gb=float(memory) if memory is not None else None,
            tags=tuple(str(item) for item in value.get("tags", ())),
            workers=workers,
            exclusive=bool(value.get("exclusive", False)),
        )


@dataclass(frozen=True)
class WorkerConfig:
    """Logical compute worker and its command transport capabilities."""

    name: str
    host: str = "localhost"
    transport: str = "local"
    device: str = "auto"
    memory_gb: float | None = None
    role: str = "worker"
    tags: tuple[str, ...] = ()
    default_storage: str | None = None
    user: str | None = None
    port: int | None = None
    python: str | None = None
    workdir: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    ssh_identity_file: str | None = None
    max_jobs: int = 1
    priority: int = 0
    enabled: bool = True
    timeout_seconds: int | None = None
    setup_command: str | None = None
    staging_dir: str = "/tmp/pra-experiments"

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any] | None) -> "WorkerConfig":
        value = value or {}
        transport = str(value.get("transport", "local")).lower()
        if transport not in {"local", "process", "ssh"}:
            raise ValueError(f"Worker {name!r} uses unsupported transport {transport!r}.")
        max_jobs = int(value.get("max_jobs", 1))
        if max_jobs <= 0:
            raise ValueError(f"Worker {name!r} max_jobs must be positive.")
        port = value.get("port")
        timeout = value.get("timeout_seconds")
        memory = value.get("memory_gb")
        host = str(value.get("host", "localhost"))
        if transport == "ssh" and not host:
            raise ValueError(f"SSH worker {name!r} requires host.")
        return cls(
            name=name,
            host=host,
            transport=transport,
            device=str(value.get("device", "auto")),
            memory_gb=float(memory) if memory is not None else None,
            role=str(value.get("role", "worker")),
            tags=tuple(str(item) for item in value.get("tags", ())),
            default_storage=value.get("default_storage"),
            user=value.get("user"),
            port=int(port) if port is not None else None,
            python=value.get("python"),
            workdir=value.get("workdir"),
            env={str(key): str(item) for key, item in (value.get("env") or {}).items()},
            ssh_identity_file=value.get("ssh_identity_file"),
            max_jobs=max_jobs,
            priority=int(value.get("priority", 0)),
            enabled=bool(value.get("enabled", True)),
            timeout_seconds=int(timeout) if timeout is not None else None,
            setup_command=value.get("setup_command"),
            staging_dir=str(value.get("staging_dir", "/tmp/pra-experiments")),
        )

    def satisfies(self, resources: ResourceRequirements) -> bool:
        if not self.enabled:
            return False
        required_device = resources.device.lower()
        worker_device = self.device.lower()
        if required_device not in {"", "auto"}:
            if worker_device == "auto":
                if required_device not in {"cpu", "cuda", "mps"}:
                    return False
            elif required_device == "cuda" and worker_device.startswith("cuda"):
                pass
            elif worker_device != required_device:
                return False
        if (
            resources.min_memory_gb is not None
            and (self.memory_gb is None or self.memory_gb < resources.min_memory_gb)
        ):
            return False
        return set(resources.tags).issubset(self.tags)

    @property
    def python_executable(self) -> str:
        return self.python or "python"

    @property
    def workdir_path(self) -> Path | None:
        return Path(self.workdir) if self.workdir else None


@dataclass(frozen=True)
class ClusterWorker:
    """Worker reference with a cluster-specific logical role."""

    name: str
    role: str | None = None

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any]) -> "ClusterWorker":
        if isinstance(value, str):
            return cls(name=value)
        return cls(name=str(value["name"]), role=value.get("role"))


@dataclass(frozen=True)
class ClusterConfig:
    """Named allocation boundary for independent or cooperative execution."""

    name: str
    workers: tuple[ClusterWorker, ...]
    default: bool = False
    distribution: DistributionMode = DistributionMode.LOCAL
    default_storage: str | None = None
    artifact_transport: str = "auto"
    checkpoint_transport: str = "auto"
    result_transport: str = "auto"
    weight_update_transport: str = "network"
    coordinator: str | None = None
    max_parallel_trials: int | None = None
    rdzv_backend: str = "c10d"
    rdzv_endpoint: str | None = None
    master_worker: str | None = None
    backend: str = "auto"
    tags: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any] | None) -> "ClusterConfig":
        value = value or {}
        transports = {}
        for field_name in (
            "artifact_transport",
            "checkpoint_transport",
            "result_transport",
            "weight_update_transport",
        ):
            default = "network" if field_name == "weight_update_transport" else "auto"
            selected = str(value.get(field_name, default)).lower()
            if selected not in {"network", "storage", "auto"}:
                raise ValueError(f"Cluster {name!r} has invalid {field_name}={selected!r}.")
            transports[field_name] = selected
        distribution = DistributionMode.from_value(value.get("distribution"))
        if distribution in {DistributionMode.DDP, DistributionMode.FSDP}:
            if transports["weight_update_transport"] == "storage":
                raise ValueError(
                    f"Cluster {name!r} cannot use storage for synchronous weight updates."
                )
        workers = tuple(ClusterWorker.from_value(item) for item in value.get("workers", ()))
        maximum = value.get("max_parallel_trials")
        if maximum is not None and int(maximum) <= 0:
            raise ValueError(f"Cluster {name!r} max_parallel_trials must be positive.")
        backend = str(value.get("backend", "auto")).lower()
        if backend not in {"auto", "gloo", "nccl"}:
            raise ValueError(f"Cluster {name!r} has unsupported backend {backend!r}.")
        return cls(
            name=name,
            workers=workers,
            default=bool(value.get("default", False)),
            distribution=distribution,
            default_storage=value.get("default_storage"),
            coordinator=value.get("coordinator"),
            max_parallel_trials=int(maximum) if maximum is not None else None,
            rdzv_backend=str(value.get("rdzv_backend", "c10d")),
            rdzv_endpoint=value.get("rdzv_endpoint"),
            master_worker=value.get("master_worker"),
            backend=backend,
            tags=tuple(str(item) for item in value.get("tags", ())),
            enabled=bool(value.get("enabled", True)),
            **transports,
        )

    def resolved_workers(self, registry: Mapping[str, WorkerConfig]) -> tuple[WorkerConfig, ...]:
        names = [member.name for member in self.workers]
        if len(names) != len(set(names)):
            raise ValueError(f"Cluster {self.name!r} contains duplicate worker references.")
        resolved = []
        for member in self.workers:
            if member.name not in registry:
                raise ValueError(
                    f"Cluster {self.name!r} references unknown worker {member.name!r}."
                )
            worker = registry[member.name]
            resolved.append(replace(worker, role=member.role or worker.role))
        return tuple(resolved)


def implicit_local_worker(overrides: Mapping[str, Any] | None = None) -> WorkerConfig:
    """Return the always-present same-process pseudo-worker."""

    base: dict[str, Any] = {
        "host": "localhost",
        "transport": "local",
        "device": "auto",
        "role": "worker",
        "tags": ["local"],
        "max_jobs": 1,
    }
    if overrides:
        base.update(overrides)
    return WorkerConfig.from_mapping("local", base)


def implicit_local_cluster(overrides: Mapping[str, Any] | None = None) -> ClusterConfig:
    """Return the always-present cluster that selects only the local worker."""

    base: dict[str, Any] = {
        "workers": [{"name": "local", "role": "worker"}],
        "distribution": "local",
    }
    if overrides:
        base.update(overrides)
    return ClusterConfig.from_mapping("local", base)


def select_cluster(
    clusters: Mapping[str, ClusterConfig], requested: str | None = None
) -> ClusterConfig:
    """Apply explicit, single-default, then implicit-local cluster selection."""

    if requested:
        if requested not in clusters:
            raise ValueError(f"Unknown cluster {requested!r}.")
        selected = clusters[requested]
        if not selected.enabled:
            raise ValueError(f"Cluster {requested!r} is disabled.")
        return selected
    defaults = [cluster for cluster in clusters.values() if cluster.default and cluster.enabled]
    if len(defaults) > 1:
        names = ", ".join(sorted(cluster.name for cluster in defaults))
        raise ValueError(f"Multiple clusters are marked default: {names}.")
    if defaults:
        return defaults[0]
    return clusters["local"]
