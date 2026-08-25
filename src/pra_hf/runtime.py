"""Unified product runtime for PRA model, memory, and tool resources.

The module keeps policy and mechanism boundaries explicit.  Discovery returns
stable identities, materialization moves only selected native K/V, and host
authorization remains independent of model output.  The portable PyTorch path
is the correctness baseline for later compiled or serving-engine backends.
"""

from __future__ import annotations

import json
import importlib.util
import platform
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import torch

from .agent_execution import (
    ExecutionAuthorization,
    SafeToolExecutor,
    ToolExecutionResult,
    parse_tool_call,
)
from .agent_resources import (
    AgentResource,
    DiscoveryRequest,
    DiscoveryTrace,
    ResourceDiscoveryEngine,
)
from .config import PRAConfig
from .external_memory import AuthContext, ExternalMemoryManager, PRASession
from .model import GenerationResult, PRAForCausalLM, ReferenceHandle


class RuntimeBackend(str, Enum):
    """Supported execution boundaries, not claims of measured integration."""

    HUGGINGFACE = "huggingface"
    VLLM_THIN = "vllm_thin"


class CompilationMode(str, Enum):
    """Materialization implementation selected by the runtime."""

    EAGER = "eager"
    TORCH_COMPILE = "torch_compile"


class KVLayout(str, Enum):
    """Physical organization of warm K/V before selected packing."""

    LAYER_MAJOR = "layer_major"
    CHUNK_MAJOR = "chunk_major"
    BLOCK_MAJOR = "block_major"
    REFERENCE_MAJOR = "reference_major"


@dataclass
class PRARuntimeConfig:
    """Serializable systems configuration layered over :class:`PRAConfig`.

    ``pra`` controls model semantics.  The remaining fields control physical
    execution and must not change which logical K/V tokens are selected.
    """

    pra: PRAConfig = field(default_factory=PRAConfig)
    backend: str = RuntimeBackend.HUGGINGFACE.value
    compilation: str = CompilationMode.EAGER.value
    kv_layout: str = KVLayout.LAYER_MAJOR.value
    page_tokens: int = 16
    cache_max_bytes: int = 1 << 30
    cache_max_entries: int = 1024
    prefetch_enabled: bool = False
    synchronize_timing: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.pra, dict):
            self.pra = PRAConfig.from_dict(self.pra)
        self.backend = RuntimeBackend(self.backend).value
        self.compilation = CompilationMode(self.compilation).value
        self.kv_layout = KVLayout(self.kv_layout).value
        if self.page_tokens <= 0:
            raise ValueError("page_tokens must be positive.")
        if self.cache_max_bytes <= 0 or self.cache_max_entries <= 0:
            raise ValueError("Cache limits must be positive.")
        if self.schema_version != 1:
            raise ValueError(f"Unsupported runtime schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["pra"] = self.pra.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PRARuntimeConfig":
        return cls(**dict(values))

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "pra_runtime_config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "PRARuntimeConfig":
        path = Path(directory) / "pra_runtime_config.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, order=True)
class KVInterval:
    """One logical half-open token interval in a reference and decoder layer."""

    uri: str
    layer_id: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("KV interval URI cannot be empty.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("KV intervals require 0 <= start < end.")

    @property
    def tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MaterializationPlan:
    """Deduplicated logical intervals admitted under one token budget."""

    intervals: tuple[KVInterval, ...]
    requested_tokens: int
    unique_tokens: int
    dropped_tokens: int
    budget_tokens: int

    @classmethod
    def build(
        cls,
        intervals: Sequence[KVInterval],
        *,
        max_tokens: int,
    ) -> "MaterializationPlan":
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        requested = sum(interval.tokens for interval in intervals)
        grouped: dict[tuple[str, int], list[KVInterval]] = {}
        for interval in intervals:
            grouped.setdefault((interval.uri, interval.layer_id), []).append(interval)
        merged: list[KVInterval] = []
        for (uri, layer_id), rows in sorted(grouped.items()):
            for row in sorted(rows, key=lambda item: (item.start, item.end)):
                if merged and merged[-1].uri == uri and merged[-1].layer_id == layer_id and row.start <= merged[-1].end:
                    previous = merged.pop()
                    merged.append(KVInterval(uri, layer_id, previous.start, max(previous.end, row.end)))
                else:
                    merged.append(row)
        admitted: list[KVInterval] = []
        remaining = max_tokens
        for interval in merged:
            if remaining <= 0:
                break
            stop = min(interval.end, interval.start + remaining)
            admitted.append(KVInterval(interval.uri, interval.layer_id, interval.start, stop))
            remaining -= stop - interval.start
        unique = sum(interval.tokens for interval in admitted)
        return cls(tuple(admitted), requested, unique, max(0, requested - unique), max_tokens)


@dataclass(frozen=True)
class NativeKV:
    """Layer-native K/V with shape ``[batch, kv_heads, tokens, head_dim]``."""

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape or self.key.ndim != 4:
            raise ValueError("Native K/V must have equal [B, Hkv, T, D] shapes.")

    @property
    def tokens(self) -> int:
        return int(self.key.shape[2])

    @property
    def nbytes(self) -> int:
        return (self.key.numel() + self.value.numel()) * self.key.element_size()


@dataclass(frozen=True)
class _KVPlacement:
    """Map one logical source fragment into a contiguous physical buffer."""

    uri: str
    layer_id: int
    logical_start: int
    logical_end: int
    physical_start: int
    physical_end: int


class PackedNativeKVStore:
    """Contiguous K/V backing storage with layout-specific fragment ordering.

    All sources must share batch size, physical KV heads, head width, dtype, and
    device.  Layout changes only physical order; ``slice`` restores logical
    token order before attention.
    """

    def __init__(
        self,
        sources: Mapping[tuple[str, int], NativeKV],
        *,
        layout: KVLayout | str,
        page_tokens: int = 16,
    ) -> None:
        if not sources:
            raise ValueError("PackedNativeKVStore requires at least one source.")
        if page_tokens <= 0:
            raise ValueError("page_tokens must be positive.")
        self.layout = KVLayout(layout)
        self.page_tokens = int(page_tokens)
        first = next(iter(sources.values()))
        signature = (
            first.key.shape[0],
            first.key.shape[1],
            first.key.shape[3],
            first.key.dtype,
            first.key.device,
        )
        if any(
            (
                memory.key.shape[0],
                memory.key.shape[1],
                memory.key.shape[3],
                memory.key.dtype,
                memory.key.device,
            )
            != signature
            for memory in sources.values()
        ):
            raise ValueError("Packed K/V sources must share B, Hkv, D, dtype, and device.")
        fragments: list[tuple[str, int, int, int, torch.Tensor, torch.Tensor]] = []
        for (uri, layer_id), memory in sorted(sources.items()):
            for start in range(0, memory.tokens, self.page_tokens):
                end = min(memory.tokens, start + self.page_tokens)
                fragments.append(
                    (
                        uri,
                        layer_id,
                        start,
                        end,
                        memory.key[:, :, start:end, :],
                        memory.value[:, :, start:end, :],
                    )
                )
        if self.layout == KVLayout.LAYER_MAJOR:
            fragments.sort(key=lambda row: (row[1], row[0], row[2]))
        elif self.layout == KVLayout.REFERENCE_MAJOR:
            fragments.sort(key=lambda row: (row[0], row[1], row[2]))
        elif self.layout == KVLayout.CHUNK_MAJOR:
            fragments.sort(key=lambda row: (row[2] // self.page_tokens, row[0], row[1]))
        else:
            fragments.sort(key=lambda row: (row[1], row[2] // self.page_tokens, row[0]))
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        placements: list[_KVPlacement] = []
        physical_start = 0
        for uri, layer_id, start, end, key, value in fragments:
            width = end - start
            keys.append(key)
            values.append(value)
            placements.append(
                _KVPlacement(
                    uri,
                    layer_id,
                    start,
                    end,
                    physical_start,
                    physical_start + width,
                )
            )
            physical_start += width
        self.memory = NativeKV(torch.cat(keys, dim=2), torch.cat(values, dim=2))
        self.placements = tuple(placements)
        self._tokens = {
            identity: memory.tokens for identity, memory in sources.items()
        }

    @property
    def nbytes(self) -> int:
        return self.memory.nbytes

    @property
    def index_bytes(self) -> int:
        return sum(48 + len(row.uri.encode("utf-8")) for row in self.placements)

    def token_count(self, uri: str, layer_id: int) -> int:
        try:
            return self._tokens[(uri, layer_id)]
        except KeyError as error:
            raise KeyError(f"Missing native K/V for {(uri, layer_id)!r}") from error

    def slice(self, interval: KVInterval) -> NativeKV:
        """Restore one logical interval from possibly noncontiguous pages."""

        if interval.end > self.token_count(interval.uri, interval.layer_id):
            raise ValueError(
                f"Interval {interval} exceeds {self.token_count(interval.uri, interval.layer_id)} source tokens."
            )
        parts: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for row in self.placements:
            if row.uri != interval.uri or row.layer_id != interval.layer_id:
                continue
            logical_start = max(interval.start, row.logical_start)
            logical_end = min(interval.end, row.logical_end)
            if logical_start >= logical_end:
                continue
            offset = logical_start - row.logical_start
            physical_start = row.physical_start + offset
            physical_end = physical_start + (logical_end - logical_start)
            parts.append(
                (
                    logical_start,
                    self.memory.key[:, :, physical_start:physical_end, :],
                    self.memory.value[:, :, physical_start:physical_end, :],
                )
            )
        parts.sort(key=lambda row: row[0])
        if not parts:
            raise KeyError(f"No physical placement for {interval!r}")
        keys = [row[1] for row in parts]
        values = [row[2] for row in parts]
        return NativeKV(
            torch.cat(keys, dim=2) if len(keys) > 1 else keys[0].contiguous(),
            torch.cat(values, dim=2) if len(values) > 1 else values[0].contiguous(),
        )


@dataclass(frozen=True)
class MaterializedKV:
    """Packed selected K/V grouped by layer with physical accounting."""

    layers: Mapping[int, NativeKV]
    intervals: tuple[KVInterval, ...]
    logical_tokens: int
    physical_bytes: int
    transfer_bytes: int
    temporary_bytes: int
    device: str


class KVMaterializer:
    """Portable selected-interval gather and pack correctness baseline."""

    def __init__(self, *, layout: KVLayout | str = KVLayout.LAYER_MAJOR) -> None:
        self.layout = KVLayout(layout)

    @staticmethod
    def _pack(parts: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(parts), dim=2) if len(parts) > 1 else parts[0].contiguous()

    def materialize(
        self,
        sources: Mapping[tuple[str, int], NativeKV] | PackedNativeKVStore,
        plan: MaterializationPlan,
        *,
        device: torch.device | str | None = None,
        non_blocking: bool = False,
    ) -> MaterializedKV:
        """Gather plan intervals and pack them per decoder layer.

        Source tensors and output tensors retain ``[B, Hkv, T, D]``.  GQA/MQA
        head expansion is intentionally left to the model-family attention
        adapter so the warm cache stores each physical KV head only once.
        """

        target = torch.device(device) if device is not None else None
        by_layer: dict[int, list[NativeKV]] = {}
        transfer_bytes = 0
        for interval in plan.intervals:
            if isinstance(sources, PackedNativeKVStore):
                source = sources.slice(interval)
                key, value = source.key, source.value
            else:
                try:
                    source = sources[(interval.uri, interval.layer_id)]
                except KeyError as error:
                    raise KeyError(f"Missing native K/V for {(interval.uri, interval.layer_id)!r}") from error
                if interval.end > source.tokens:
                    raise ValueError(f"Interval {interval} exceeds {source.tokens} source tokens.")
                key = source.key[:, :, interval.start : interval.end, :]
                value = source.value[:, :, interval.start : interval.end, :]
            if target is not None and key.device != target:
                transfer_bytes += (key.numel() + value.numel()) * key.element_size()
                key = key.to(target, non_blocking=non_blocking)
                value = value.to(target, non_blocking=non_blocking)
            by_layer.setdefault(interval.layer_id, []).append(NativeKV(key, value))
        packed = {
            layer_id: NativeKV(
                self._pack([part.key for part in parts]),
                self._pack([part.value for part in parts]),
            )
            for layer_id, parts in sorted(by_layer.items())
        }
        physical = sum(memory.nbytes for memory in packed.values())
        return MaterializedKV(
            layers=packed,
            intervals=plan.intervals,
            logical_tokens=plan.unique_tokens,
            physical_bytes=physical,
            transfer_bytes=transfer_bytes,
            temporary_bytes=physical,
            device=str(
                target
                or (
                    sources.memory.key.device
                    if isinstance(sources, PackedNativeKVStore)
                    else next(iter(sources.values())).key.device
                )
            ),
        )


def _indexed_gather_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V tokens without expanding physical GQA/MQA heads."""

    return (
        torch.index_select(key, 2, token_indices),
        torch.index_select(value, 2, token_indices),
    )


class SelectedKVGather:
    """Identical selected-token gather in eager or ``torch.compile`` mode."""

    def __init__(self, mode: CompilationMode | str = CompilationMode.EAGER) -> None:
        self.mode = CompilationMode(mode)
        self.compiled = False
        self.compile_error: str | None = None
        self._operation = _indexed_gather_pair
        if self.mode == CompilationMode.TORCH_COMPILE:
            if not hasattr(torch, "compile"):
                self.compile_error = "torch.compile is unavailable"
            else:
                try:
                    self._operation = torch.compile(
                        _indexed_gather_pair,
                        fullgraph=True,
                        dynamic=True,
                    )
                    self.compiled = True
                except Exception as error:  # pragma: no cover - platform dependent
                    self.compile_error = f"{type(error).__name__}: {error}"

    def __call__(self, memory: NativeKV, token_indices: torch.Tensor) -> NativeKV:
        if token_indices.ndim != 1 or token_indices.dtype != torch.long:
            raise ValueError("token_indices must be a one-dimensional torch.long tensor.")
        if token_indices.device != memory.key.device:
            token_indices = token_indices.to(memory.key.device)
        if self.mode == CompilationMode.TORCH_COMPILE and not self.compiled:
            raise RuntimeError(self.compile_error or "torch.compile initialization failed")
        key, value = self._operation(memory.key, memory.value, token_indices)
        return NativeKV(key, value)

    def inspect(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "compiled_wrapper_created": self.compiled,
            "compile_error": self.compile_error,
        }


def runtime_capabilities() -> dict[str, Any]:
    """Describe locally available runtime paths without importing optional engines."""

    cuda = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability() if cuda else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if cuda else None,
        "cuda_capability": list(capability) if capability else None,
        "torch_compile_api": hasattr(torch, "compile"),
        "triton_installed": importlib.util.find_spec("triton") is not None,
        "vllm_installed": importlib.util.find_spec("vllm") is not None,
        "sglang_installed": importlib.util.find_spec("sglang") is not None,
        "tensorrt_llm_installed": importlib.util.find_spec("tensorrt_llm") is not None,
        "mlx_installed": importlib.util.find_spec("mlx") is not None,
    }


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_loaded: int = 0
    bytes_reused: int = 0

    def snapshot(self, resident_bytes: int, entries: int) -> dict[str, int | float]:
        accesses = self.hits + self.misses
        return {
            **asdict(self),
            "entries": entries,
            "resident_bytes": resident_bytes,
            "hit_rate": self.hits / max(accesses, 1),
            "reload_amplification": self.bytes_loaded / max(resident_bytes, 1),
        }


class RuntimeKVCache:
    """Thread-safe byte-bounded LRU for opaque warm or hot runtime payloads."""

    def __init__(self, *, max_bytes: int, max_entries: int) -> None:
        if max_bytes <= 0 or max_entries <= 0:
            raise ValueError("Cache limits must be positive.")
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._entries.pop(key, None)
            if row is None:
                self.stats.misses += 1
                return None
            self._entries[key] = row
            self.stats.hits += 1
            self.stats.bytes_reused += row[1]
            return row[0]

    def put(self, key: str, value: Any, *, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes cannot be negative.")
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._resident_bytes -= old[1]
            self._entries[key] = (value, int(nbytes))
            self._resident_bytes += int(nbytes)
            self.stats.bytes_loaded += int(nbytes)
            while self._entries and (
                self._resident_bytes > self.max_bytes
                or len(self._entries) > self.max_entries
            ):
                _, (_, removed_bytes) = self._entries.popitem(last=False)
                self._resident_bytes -= removed_bytes
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return self.stats.snapshot(self._resident_bytes, len(self._entries))


@dataclass(frozen=True)
class RuntimeEvent:
    """One timed stage with explicit byte and allocation accounting."""

    name: str
    seconds: float
    input_bytes: int = 0
    output_bytes: int = 0
    peak_device_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RuntimeProfiler:
    """Request-local profiler that synchronizes CUDA only when requested."""

    def __init__(self, *, device: torch.device | str = "cpu", synchronize: bool = True) -> None:
        self.device = torch.device(device)
        self.synchronize = synchronize
        self.events: list[RuntimeEvent] = []

    def _sync(self) -> None:
        if self.synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        input_bytes: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, int]]:
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        accounting = {"output_bytes": 0}
        yield accounting
        self._sync()
        peak = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        self.events.append(
            RuntimeEvent(
                name=name,
                seconds=time.perf_counter() - started,
                input_bytes=int(input_bytes),
                output_bytes=int(accounting["output_bytes"]),
                peak_device_bytes=peak,
                metadata=dict(metadata or {}),
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "total_seconds": sum(event.seconds for event in self.events),
            "events": [asdict(event) for event in self.events],
        }


class ModelBackend(Protocol):
    """Minimal model boundary consumed by :class:`PRARuntime`."""

    name: str

    def add_reference(self, reference: str, *, text: str | None = None, uri: str | None = None) -> ReferenceHandle: ...
    def generate(self, prompt: str, **kwargs: Any) -> str | GenerationResult: ...
    def inspect(self) -> Mapping[str, Any]: ...


class HuggingFaceBackend:
    """Adapter around the Paper 2 ``PRAForCausalLM`` public interface."""

    name = RuntimeBackend.HUGGINGFACE.value

    def __init__(self, model: PRAForCausalLM) -> None:
        self.model = model

    def add_reference(self, reference: str, *, text: str | None = None, uri: str | None = None) -> ReferenceHandle:
        return self.model.add_reference(reference, text=text, uri=uri)

    def generate(self, prompt: str, **kwargs: Any) -> str | GenerationResult:
        return self.model.generate(prompt, **kwargs)

    def inspect(self) -> Mapping[str, Any]:
        return self.model.stats()


@dataclass(frozen=True)
class VLLMThinRequest:
    """Scheduler-agnostic handoff for ordinary vLLM execution.

    Only selected stable identities and packed K/V metadata cross this boundary;
    semantic scores remain in the external PRA router.
    """

    request_id: str
    prompt: str
    selected_uris: tuple[str, ...]
    materialized_tokens: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VLLMThinBackend:
    """Basic Paper 4.5 boundary; not a retrieval-aware vLLM scheduler."""

    name = RuntimeBackend.VLLM_THIN.value

    def __init__(self, executor: Callable[[VLLMThinRequest], Any] | None = None) -> None:
        self.executor = executor
        self._last_request: VLLMThinRequest | None = None

    def prepare(
        self,
        prompt: str,
        *,
        selected_uris: Sequence[str],
        materialized_tokens: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> VLLMThinRequest:
        request = VLLMThinRequest(
            request_id=uuid.uuid4().hex,
            prompt=prompt,
            selected_uris=tuple(selected_uris),
            materialized_tokens=int(materialized_tokens),
            metadata=dict(metadata or {}),
        )
        self._last_request = request
        return request

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        request = self.prepare(
            prompt,
            selected_uris=kwargs.pop("selected_uris", ()),
            materialized_tokens=kwargs.pop("materialized_tokens", 0),
            metadata=kwargs,
        )
        if self.executor is None:
            raise RuntimeError("vLLM thin executor is not installed; use prepare() for handoff.")
        return self.executor(request)

    def add_reference(self, reference: str, *, text: str | None = None, uri: str | None = None) -> ReferenceHandle:
        raise RuntimeError("References must be routed and materialized before the vLLM thin handoff.")

    def inspect(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "executor_installed": self.executor is not None,
            "scheduler_semantically_aware": False,
            "last_request": asdict(self._last_request) if self._last_request else None,
        }


class PRARuntime:
    """One SDK facade over model memory, external resources, and safe tools."""

    def __init__(
        self,
        *,
        config: PRARuntimeConfig,
        backend: ModelBackend,
        external_memory: ExternalMemoryManager | None = None,
        discovery: ResourceDiscoveryEngine | None = None,
        executor: SafeToolExecutor | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.external_memory = external_memory
        self.discovery = discovery
        self.executor = executor
        self.sessions: dict[str, PRASession] = {}
        self.hot_cache = RuntimeKVCache(
            max_bytes=config.cache_max_bytes,
            max_entries=config.cache_max_entries,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        runtime_config: PRARuntimeConfig | Mapping[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> "PRARuntime":
        config = (
            runtime_config
            if isinstance(runtime_config, PRARuntimeConfig)
            else PRARuntimeConfig.from_dict(runtime_config or {})
        )
        if config.backend != RuntimeBackend.HUGGINGFACE.value:
            raise ValueError("from_pretrained currently loads the Hugging Face backend only.")
        model = PRAForCausalLM.from_pretrained(
            model_name_or_path,
            pra_config=config.pra,
            **model_kwargs,
        )
        return cls(config=config, backend=HuggingFaceBackend(model))

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        runtime_config: PRARuntimeConfig | None = None,
        **components: Any,
    ) -> "PRARuntime":
        config = runtime_config or PRARuntimeConfig()
        wrapped = PRAForCausalLM.from_model(model, tokenizer, pra_config=config.pra)
        return cls(config=config, backend=HuggingFaceBackend(wrapped), **components)

    def open_session(
        self,
        *,
        session_id: str,
        user_id: str,
        tenant_id: str,
        auth_context: AuthContext | None = None,
        **session_kwargs: Any,
    ) -> PRASession:
        if session_id in self.sessions and not self.sessions[session_id].closed:
            raise ValueError(f"Session already open: {session_id}")
        auth = auth_context or AuthContext(tenant_id, user_id, session_id)
        session = PRASession(session_id, user_id, tenant_id, auth, **session_kwargs)
        self.sessions[session_id] = session
        return session

    def close_session(self, session: PRASession) -> None:
        if self.external_memory is not None:
            self.external_memory.teardown_session(session)
        else:
            session.resources.clear()
            session.admitted_resources.clear()
            session.warm_handles.clear()
            session.hot_handles.clear()
            session.closed = True
        self.sessions.pop(session.session_id, None)

    def add_reference(self, reference: str, *, text: str | None = None, uri: str | None = None) -> ReferenceHandle:
        return self.backend.add_reference(reference, text=text, uri=uri)

    async def add_external_reference(self, session: PRASession, **kwargs: Any):
        if self.external_memory is None:
            raise RuntimeError("No external-memory manager is installed.")
        return await self.external_memory.add_reference(session, **kwargs)

    def discover_resources(self, request: DiscoveryRequest) -> DiscoveryTrace:
        if self.discovery is None:
            raise RuntimeError("No typed resource discovery engine is installed.")
        return self.discovery.discover(request)

    def execute_tool(
        self,
        generated_text: str,
        *,
        selected_uris: Sequence[str],
        authorization: ExecutionAuthorization,
        call_id: str,
        prior_observations: Sequence[AgentResource] = (),
    ) -> ToolExecutionResult:
        if self.executor is None:
            raise RuntimeError("No safe tool executor is installed.")
        return self.executor.execute(
            parse_tool_call(generated_text),
            selected_uris=selected_uris,
            authorization=authorization,
            prior_observations=prior_observations,
            call_id=call_id,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str | GenerationResult:
        return self.backend.generate(prompt, **kwargs)

    def inspect(self) -> dict[str, Any]:
        return {
            "runtime_config": self.config.to_dict(),
            "backend": dict(self.backend.inspect()),
            "sessions": [session.safe_snapshot() for session in self.sessions.values()],
            "external_memory": (
                self.external_memory.metrics.snapshot() if self.external_memory else None
            ),
            "typed_discovery_installed": self.discovery is not None,
            "safe_executor_installed": self.executor is not None,
            "hot_cache": self.hot_cache.snapshot(),
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        """Persist non-secret runtime configuration; model weights remain external."""

        return self.config.save_pretrained(directory)
