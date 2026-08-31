"""Engine-neutral runtime provider contract and built-in implementations."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deployment import PRAEngineCapabilities
from .engine_profiles import EngineType, PrefixCacheMode
from .product_config import pra_home
from .runtime_benchmark import run_runtime_microbenchmark, write_runtime_benchmark
from .storage_lifecycle import PRAStoragePolicy


@dataclass(frozen=True)
class RuntimeConfig:
    """Canonical model/profile/session contract translated by a provider."""

    engine: str = "hf"
    model: str | None = None
    revision: str | None = None
    pra_bundle: str | None = None
    profile: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    endpoint: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    prefix_cache_mode: str = "auto"
    session_mode: str = "request"
    cache_affinity: bool = False
    engine_options: Mapping[str, Any] = field(default_factory=dict)
    verbose: bool = False
    storage_profile: str = "balanced"
    storage_config: str | None = None

    def __post_init__(self) -> None:
        if not (0 < self.port < 65536):
            raise ValueError("Runtime port must be between 1 and 65535.")
        object.__setattr__(self, "engine_options", dict(self.engine_options))

    @property
    def resolved_endpoint(self) -> str:
        return self.endpoint or f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_storage_policy(self) -> PRAStoragePolicy:
        """Resolve a named profile, then apply an optional YAML policy."""

        if self.storage_config is not None:
            policy = PRAStoragePolicy.from_yaml(self.storage_config)
            if policy.profile != self.storage_profile and self.storage_profile != "balanced":
                raise ValueError("--storage and storage-config profile disagree.")
            return policy
        return PRAStoragePolicy.named(self.storage_profile)


@dataclass(frozen=True)
class RuntimeDoctorReport:
    engine: str
    status: str
    checks: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "status": self.status, "checks": [dict(row) for row in self.checks]}


@dataclass(frozen=True)
class RuntimeHealth:
    status: str
    endpoint: str
    detail: str = ""


@dataclass(frozen=True)
class RuntimeHandle:
    """Serializable identity of a managed or remote runtime."""

    engine: str
    endpoint: str
    model: str | None
    revision: str | None
    profile: str | None
    start_time: float
    capabilities: Mapping[str, Any]
    process_id: int | None = None
    log_path: str | None = None
    command: tuple[str, ...] = ()
    managed: bool = False
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "command": list(self.command)}


class RuntimeProvider(ABC):
    """Translate the canonical PRA runtime contract to one execution engine."""

    engine_type: str
    package_name: str | None = None
    launch_mode: str = "connect_remote"

    @abstractmethod
    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        """Return conservatively implemented capabilities."""

    def inspect(self, config: RuntimeConfig) -> dict[str, Any]:
        package = self.package_name or self.engine_type
        return {
            "engine": self.engine_type,
            "engine_version": _package_version(package),
            "launch_mode": self.launch_mode,
            "model": config.model,
            "revision": config.revision,
            "device": config.device,
            "pra_bundle": config.pra_bundle,
            "pra_profile": config.profile or "REFERENCE_CORRECTNESS",
            "endpoint": config.resolved_endpoint,
            "capabilities": self.capabilities(config).to_dict(),
            "source": "static_provider",
            "storage": config.resolved_storage_policy().to_dict(),
        }

    def doctor(self, config: RuntimeConfig) -> RuntimeDoctorReport:
        installed = self.package_name is None or importlib.util.find_spec(self.package_name) is not None
        remote = bool(config.endpoint) and self.launch_mode in {"connect_remote", "both"}
        endpoint = _probe_endpoint(config.resolved_endpoint) if remote else None
        checks = [
            {"name": "provider", "status": "READY", "detail": type(self).__name__},
            {
                "name": "dependency",
                "status": "READY" if installed else "NOT_INSTALLED",
                "detail": self.package_name or "standard library",
            },
        ]
        if remote:
            checks.append({"name": "endpoint", "status": endpoint.status, "detail": endpoint.detail})
        if not installed:
            status = "NOT_INSTALLED"
        elif endpoint is not None and endpoint.status != "READY":
            status = "MISCONFIGURED"
        else:
            status = "AVAILABLE"
        return RuntimeDoctorReport(self.engine_type, status, tuple(checks))

    def build_command(self, config: RuntimeConfig) -> list[str]:
        raise RuntimeError(f"Provider '{self.engine_type}' does not support managed launch.")

    def serve(self, config: RuntimeConfig) -> RuntimeHandle:
        if self.launch_mode == "connect_remote":
            health = _probe_endpoint(config.resolved_endpoint)
            return RuntimeHandle(
                engine=self.engine_type,
                endpoint=config.resolved_endpoint,
                model=config.model,
                revision=config.revision,
                profile=config.profile,
                start_time=time.time(),
                capabilities=self.capabilities(config).to_dict(),
                managed=False,
            )
        command = self.build_command(config)
        run_id = uuid.uuid4().hex[:12]
        log_dir = pra_home() / "runtime" / self.engine_type / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        (log_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=creationflags)
        finally:
            stdout.close()
            stderr.close()
        handle = RuntimeHandle(
            engine=self.engine_type,
            endpoint=config.resolved_endpoint,
            model=config.model,
            revision=config.revision,
            profile=config.profile,
            start_time=time.time(),
            capabilities=self.capabilities(config).to_dict(),
            process_id=process.pid,
            log_path=str(log_dir),
            command=tuple(command),
            managed=True,
            run_id=run_id,
        )
        (log_dir / "runtime.json").write_text(json.dumps(handle.to_dict(), indent=2), encoding="utf-8")
        return handle

    def health(self, handle: RuntimeHandle) -> RuntimeHealth:
        return _probe_endpoint(handle.endpoint)

    def benchmark(self, config: RuntimeConfig, output: str | Path) -> dict[str, Any]:
        result = run_runtime_microbenchmark(device=config.device)
        paths = write_runtime_benchmark(result, output)
        return {"summary": result["summary"], "artifacts": {key: str(value) for key, value in paths.items()}}

    def stop(self, handle: RuntimeHandle) -> None:
        if not handle.managed or handle.process_id is None:
            return
        _terminate_owned_process(handle)


class HFRuntimeProvider(RuntimeProvider):
    engine_type = "hf"
    package_name = "transformers"
    launch_mode = "managed_local"

    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter="hf",
            engine_type=EngineType.HUGGINGFACE,
            integration_level="E2",
            prefix_cache_mode=PrefixCacheMode.SESSION_STATE,
            session_state=True,
            logical_refs=True,
            typed_records=True,
            task_metadata=True,
            native_kv=True,
            external_kv_residency=True,
            cpu_kv=True,
            gpu_kv=True,
            streaming=True,
            selected_interval_materialization=True,
            request_lifetime=True,
            phase_selection=True,
            host_device_residency=True,
            tenant_isolation=True,
            tool_resources=True,
        )

    def build_command(self, config: RuntimeConfig) -> list[str]:
        if not config.model:
            raise ValueError("A model is required for managed HF serving.")
        command = [
            sys.executable,
            "-m",
            "pra_hf.runtime_server",
            "--model",
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
        ]
        if config.revision:
            command += ["--revision", config.revision]
        command += ["--storage", config.storage_profile]
        if config.storage_config:
            command += ["--storage-config", config.storage_config]
        return command


class GenericOpenAIRuntimeProvider(RuntimeProvider):
    engine_type = "openai"
    launch_mode = "connect_remote"

    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter="openai-compatible",
            engine_type=EngineType.OPENAI_GENERIC,
            integration_level="E0",
            prefix_cache_mode=PrefixCacheMode.UNKNOWN,
            text_fallback=True,
            streaming=True,
        )


class CommandRuntimeProvider(RuntimeProvider):
    """Built-in adapter for an upstream engine's supported launcher."""

    module: str
    base_arguments: tuple[str, ...]
    engine_enum: EngineType
    native_kv: bool = False
    integration_level: str = "E0"
    prefix_cache_mode: PrefixCacheMode = PrefixCacheMode.UNKNOWN

    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter=self.engine_type,
            engine_type=self.engine_enum,
            integration_level=self.integration_level,
            prefix_cache_mode=self.prefix_cache_mode,
            automatic_prefix_cache=self.prefix_cache_mode == PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            session_state=True,
            logical_refs=self.native_kv,
            typed_records=self.native_kv,
            native_kv=self.native_kv,
            streaming=True,
            selected_interval_materialization=self.native_kv,
            request_lifetime=self.native_kv,
            host_device_residency=self.native_kv,
            tenant_isolation=self.native_kv,
        )

    def build_command(self, config: RuntimeConfig) -> list[str]:
        if not config.model:
            raise ValueError(f"A model is required for managed {self.engine_type} serving.")
        command = [sys.executable, "-m", self.module, *self.base_arguments, "--model", config.model]
        command += ["--host", config.host, "--port", str(config.port)]
        if config.revision:
            command += ["--revision", config.revision]
        for key, value in config.engine_options.items():
            option = "--" + str(key).replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
            else:
                command += [option, str(value)]
        return command


class VLLMRuntimeProvider(CommandRuntimeProvider):
    engine_type = "vllm"
    package_name = "vllm"
    launch_mode = "both"
    module = "vllm.entrypoints.openai.api_server"
    base_arguments = ()
    engine_enum = EngineType.VLLM
    native_kv = False
    integration_level = "E0"
    prefix_cache_mode = PrefixCacheMode.AUTOMATIC_PREFIX_CACHE


class SGLangRuntimeProvider(CommandRuntimeProvider):
    engine_type = "sglang"
    package_name = "sglang"
    launch_mode = "both"
    module = "sglang.launch_server"
    base_arguments = ()
    engine_enum = EngineType.SGLANG
    native_kv = True
    integration_level = "E2"
    prefix_cache_mode = PrefixCacheMode.STATELESS


class MLXRuntimeProvider(CommandRuntimeProvider):
    engine_type = "mlx"
    package_name = "mlx"
    launch_mode = "both"
    module = "mlx_lm.server"
    base_arguments = ()
    engine_enum = EngineType.MLX
    native_kv = True
    integration_level = "E2"
    prefix_cache_mode = PrefixCacheMode.SESSION_STATE


class RuntimeProviderRegistry:
    """Mutable provider registry with optional package entry-point discovery."""

    def __init__(self) -> None:
        self._providers: dict[str, RuntimeProvider] = {}

    @classmethod
    def default(cls) -> "RuntimeProviderRegistry":
        registry = cls()
        for provider in (
            HFRuntimeProvider(),
            GenericOpenAIRuntimeProvider(),
            VLLMRuntimeProvider(),
            SGLangRuntimeProvider(),
            MLXRuntimeProvider(),
        ):
            registry.register(provider)
        registry.discover()
        return registry

    def register(self, provider: RuntimeProvider, *, replace: bool = False) -> None:
        if provider.engine_type in self._providers and not replace:
            raise ValueError(f"Runtime provider already registered: {provider.engine_type}")
        self._providers[provider.engine_type] = provider

    def resolve(self, engine: str) -> RuntimeProvider:
        try:
            return self._providers[engine]
        except KeyError as error:
            known = ", ".join(sorted(self._providers))
            raise KeyError(f"Unknown runtime engine '{engine}'. Known engines: {known}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def discover(self) -> None:
        try:
            entry_points = importlib.metadata.entry_points(group="pra.runtime_providers")
        except TypeError:
            entry_points = importlib.metadata.entry_points().get("pra.runtime_providers", ())
        for entry_point in entry_points:
            provider = entry_point.load()()
            self.register(provider, replace=True)


class RuntimeManager:
    """Service boundary shared by CLI, agent launcher, gateway, and web UI."""

    def __init__(self, registry: RuntimeProviderRegistry | None = None) -> None:
        self.registry = registry or RuntimeProviderRegistry.default()

    def inspect(self, config: RuntimeConfig) -> dict[str, Any]:
        return self.registry.resolve(config.engine).inspect(config)

    def doctor(self, config: RuntimeConfig) -> dict[str, Any]:
        return self.registry.resolve(config.engine).doctor(config).to_dict()

    def serve(self, config: RuntimeConfig) -> RuntimeHandle:
        return self.registry.resolve(config.engine).serve(config)

    def health(self, handle: RuntimeHandle) -> RuntimeHealth:
        return self.registry.resolve(handle.engine).health(handle)

    def benchmark(self, config: RuntimeConfig, output: str | Path) -> dict[str, Any]:
        return self.registry.resolve(config.engine).benchmark(config, output)

    def stop(self, handle: RuntimeHandle) -> None:
        self.registry.resolve(handle.engine).stop(handle)


def parse_engine_arguments(values: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` escape-hatch arguments with YAML scalars."""

    import yaml

    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Engine argument must use KEY=VALUE: {value}")
        key, raw = value.split("=", 1)
        if not key.strip():
            raise ValueError("Engine argument key cannot be empty.")
        result[key.strip().replace("-", "_")] = yaml.safe_load(raw)
    return result


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _probe_endpoint(endpoint: str) -> RuntimeHealth:
    for suffix in ("/health", "/health_generate", "/v1/models"):
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + suffix, timeout=1.5) as response:
                if 200 <= response.status < 300:
                    return RuntimeHealth("READY", endpoint, suffix)
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
    return RuntimeHealth("NOT_READY", endpoint, "No supported readiness endpoint responded.")


def _terminate_owned_process(handle: RuntimeHandle) -> None:
    """Terminate only a live process whose command still matches the saved handle."""

    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("Stopping managed runtimes safely requires psutil.") from error
    try:
        process = psutil.Process(handle.process_id)
    except psutil.NoSuchProcess:
        return
    current = tuple(process.cmdline())
    expected = tuple(handle.command)
    if not expected or not current or Path(current[0]).name.lower() != Path(expected[0]).name.lower():
        raise RuntimeError("Runtime PID was reused or its process identity changed; refusing to terminate it.")
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        process.kill()
