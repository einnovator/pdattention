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
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

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
    PersistentResourceIndex,
    ResourceDiscoveryEngine,
)
from .adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorAction,
    CursorActionResult,
    CursorRecord,
    MaterializationResult,
)
from .capability_runtime import CapabilityActivation, CapabilityPaletteActivation
from .capability_sdk import AgentConfig, CapabilitySDK
from .config import PRAConfig
from .context_records import ContextRecord, RecordType, RecordViewName
from .context_store import RecordScope
from .external_memory import AuthContext, ExternalMemoryManager, PRASession
from .model import GenerationResult, PRAForCausalLM, ReferenceHandle
from .native_geometry import FrozenNativeSelection, NativeMaterializationPlan
from .progressive_context import (
    ContextExecutionResult,
    LazyNativeRegion,
    NativeIndexAudit,
    NativePRASelection,
    ProgressiveContextRuntime,
    RecordCapabilities,
)
from .typed_context import AdaptiveContextRecord
from .gateway_session import GatewaySessionRegistry, ResolvedSessionTurn
from .session_service import AgentSessionState, SessionService
from .task_context import TaskEvent, TaskGraph, TaskProvenance, attach_task_provenance
from .task_planning import (
    TaskOperation,
    apply_task_operations as validate_task_operations,
)
from .task_scope import ScopeSelection, TaskScopePolicy, TaskScopeSelector


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
    profile: str | None = None
    workload: str | None = None
    profile_registry: str | None = None
    backend: str = RuntimeBackend.HUGGINGFACE.value
    compilation: str = CompilationMode.EAGER.value
    kv_layout: str = KVLayout.LAYER_MAJOR.value
    page_tokens: int = 16
    cache_max_bytes: int = 1 << 30
    cache_max_entries: int = 1024
    prefetch_enabled: bool = False
    synchronize_timing: bool = True
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    auto_prepare_native_results: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.pra, dict):
            self.pra = PRAConfig.from_dict(self.pra)
        if isinstance(self.context_policy, dict):
            self.context_policy = ContextPolicy.from_dict(self.context_policy)
        if self.profile is not None:
            self.pra.profile = self.profile
        if self.workload is not None:
            self.pra.workload = self.workload
        if self.profile_registry is not None:
            self.pra.product_profile_registry = self.profile_registry
        # Re-run normalization after top-level convenience aliases are applied.
        self.pra.__post_init__()
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
        values["context_policy"] = self.context_policy.to_dict()
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


@dataclass(frozen=True)
class RuntimeKVCacheKey:
    """Authorization- and geometry-scoped identity for reusable native payloads.

    Native tensors are reusable only when their source revision, positional
    frame, materialization layout, and authorization scope all agree.
    """

    tenant_id: str
    user_id: str
    session_id: str
    resource_id: str
    layer_id: int | None = None
    variant: str = "native_kv"
    source_revision: str = "current"
    position_signature: str = "source_relative"
    materialization_signature: str = "full_record"
    scope_signature: str = "session"

    def __post_init__(self) -> None:
        if not all(
            (
                self.tenant_id,
                self.user_id,
                self.session_id,
                self.resource_id,
                self.variant,
                self.source_revision,
                self.position_signature,
                self.materialization_signature,
                self.scope_signature,
            )
        ):
            raise ValueError("A native cache key requires tenant, user, session, and resource IDs.")

    def reuse_compatible(self, other: object) -> bool:
        """Return whether ``other`` names the exact same reusable geometry."""

        return isinstance(other, RuntimeKVCacheKey) and self == other


class RuntimeKVCache:
    """Thread-safe byte-bounded LRU for opaque warm or hot runtime payloads."""

    def __init__(
        self,
        *,
        max_bytes: int,
        max_entries: int,
        max_bytes_per_tenant: int | None = None,
    ) -> None:
        if max_bytes <= 0 or max_entries <= 0:
            raise ValueError("Cache limits must be positive.")
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.max_bytes_per_tenant = int(max_bytes_per_tenant or max_bytes)
        if self.max_bytes_per_tenant <= 0:
            raise ValueError("max_bytes_per_tenant must be positive.")
        self._entries: OrderedDict[object, tuple[Any, int]] = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: object) -> Any | None:
        with self._lock:
            row = self._entries.pop(key, None)
            if row is None:
                self.stats.misses += 1
                return None
            self._entries[key] = row
            self.stats.hits += 1
            self.stats.bytes_reused += row[1]
            return row[0]

    @staticmethod
    def _tenant(key: object) -> str | None:
        return key.tenant_id if isinstance(key, RuntimeKVCacheKey) else None

    def _tenant_bytes(self, tenant_id: str) -> int:
        return sum(
            nbytes
            for key, (_, nbytes) in self._entries.items()
            if self._tenant(key) == tenant_id
        )

    def put(self, key: object, value: Any, *, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes cannot be negative.")
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._resident_bytes -= old[1]
            self._entries[key] = (value, int(nbytes))
            self._resident_bytes += int(nbytes)
            self.stats.bytes_loaded += int(nbytes)
            tenant_id = self._tenant(key)
            while tenant_id is not None and self._tenant_bytes(tenant_id) > self.max_bytes_per_tenant:
                victim = next(
                    candidate
                    for candidate in self._entries
                    if self._tenant(candidate) == tenant_id
                )
                _, removed_bytes = self._entries.pop(victim)
                self._resident_bytes -= removed_bytes
                self.stats.evictions += 1
            while self._entries and (
                self._resident_bytes > self.max_bytes
                or len(self._entries) > self.max_entries
            ):
                _, (_, removed_bytes) = self._entries.popitem(last=False)
                self._resident_bytes -= removed_bytes
                self.stats.evictions += 1

    def clear_scope(self, *, tenant_id: str, session_id: str | None = None) -> None:
        """Evict one tenant or session without touching another authorization scope."""

        with self._lock:
            victims = [
                key
                for key in self._entries
                if isinstance(key, RuntimeKVCacheKey)
                and key.tenant_id == tenant_id
                and (session_id is None or key.session_id == session_id)
            ]
            for key in victims:
                _, removed_bytes = self._entries.pop(key)
                self._resident_bytes -= removed_bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self.stats.snapshot(self._resident_bytes, len(self._entries))
            tenants = {
                self._tenant(key)
                for key in self._entries
                if self._tenant(key) is not None
            }
            snapshot["tenant_resident_bytes"] = {
                tenant: self._tenant_bytes(tenant)
                for tenant in sorted(tenants)
            }
            return snapshot


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


@dataclass(frozen=True)
class RuntimeToolExecution:
    """Authorized tool result plus its optional compact typed-result record."""

    execution: ToolExecutionResult
    record: AdaptiveContextRecord | None = None


class HuggingFaceBackend:
    """Adapter around the Paper 2 ``PRAForCausalLM`` public interface."""

    name = RuntimeBackend.HUGGINGFACE.value

    def __init__(self, model: PRAForCausalLM) -> None:
        self.model = model

    def add_reference(self, reference: str, *, text: str | None = None, uri: str | None = None) -> ReferenceHandle:
        return self.model.add_reference(reference, text=text, uri=uri)

    def generate(self, prompt: str, **kwargs: Any) -> str | GenerationResult:
        return self.model.generate(prompt, **kwargs)

    def route(self, prompt: str):
        """Expose frozen-routing diagnostics without coupling the facade to HF internals."""

        return self.model.route(prompt)

    def freeze_native_selection(
        self, selected: Iterable[dict[str, Any]]
    ) -> FrozenNativeSelection:
        return self.model.freeze_native_selection(selected)

    def plan_native_materialization(
        self, frozen: FrozenNativeSelection, **kwargs: Any
    ) -> NativeMaterializationPlan:
        return self.model.plan_native_materialization(frozen, **kwargs)

    def generate_with_native_plan(
        self, prompt: str, plan: NativeMaterializationPlan, **kwargs: Any
    ) -> str | GenerationResult:
        return self.model.generate_with_native_plan(prompt, plan, **kwargs)

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
    """One SDK facade over model memory, capabilities, and compact results.

    Tool and skill definitions are immutable runtime-wide capabilities. Result
    records are session-scoped because their exact backing can contain tenant
    data. The Hugging Face loader applies size-adaptive native indexing by
    default; low-level constructors must still opt in because custom backends
    may not implement model-native result references.
    """

    def __init__(
        self,
        *,
        config: PRARuntimeConfig,
        backend: ModelBackend,
        external_memory: ExternalMemoryManager | None = None,
        discovery: ResourceDiscoveryEngine | None = None,
        executor: SafeToolExecutor | None = None,
        agent_config: AgentConfig | None = None,
        capability_sdk: CapabilitySDK | None = None,
        context_policy: ContextPolicy | None = None,
        native_result_routing: bool = False,
        session_service: SessionService | None = None,
    ) -> None:
        if agent_config is not None and capability_sdk is not None:
            raise ValueError("Pass agent_config or capability_sdk, not both.")
        self.config = config
        self.backend = backend
        self.external_memory = external_memory
        self.capabilities = capability_sdk or (
            CapabilitySDK(agent_config) if agent_config is not None else None
        )
        if discovery is None and self.capabilities is not None:
            discovery = ResourceDiscoveryEngine(
                PersistentResourceIndex(self.capabilities.resources())
            )
        self.discovery = discovery
        self.executor = executor
        self.context_policy = context_policy or config.context_policy
        self.native_result_routing = bool(native_result_routing)
        self.session_service = session_service
        self.engine_sessions = GatewaySessionRegistry(session_service)
        if self.native_result_routing and not isinstance(self.backend, HuggingFaceBackend):
            raise ValueError("Native result routing requires the Hugging Face backend.")
        self.sessions: dict[str, PRASession] = {}
        self.logical_sessions: dict[str, AgentSessionState] = {}
        self.result_contexts: dict[str, ProgressiveContextRuntime] = {}
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
        external_memory: ExternalMemoryManager | None = None,
        discovery: ResourceDiscoveryEngine | None = None,
        executor: SafeToolExecutor | None = None,
        agent_config: AgentConfig | None = None,
        capability_sdk: CapabilitySDK | None = None,
        context_policy: ContextPolicy | None = None,
        native_result_routing: bool = True,
        session_service: SessionService | None = None,
        **model_kwargs: Any,
    ) -> "PRARuntime":
        config = (
            runtime_config
            if isinstance(runtime_config, PRARuntimeConfig)
            else PRARuntimeConfig.from_dict(runtime_config or {})
        )
        if config.pra.model_id is None:
            config.pra.model_id = model_name_or_path
        if config.backend != RuntimeBackend.HUGGINGFACE.value:
            raise ValueError("from_pretrained currently loads the Hugging Face backend only.")
        model = PRAForCausalLM.from_pretrained(
            model_name_or_path,
            pra_config=config.pra,
            **model_kwargs,
        )
        return cls(
            config=config,
            backend=HuggingFaceBackend(model),
            external_memory=external_memory,
            discovery=discovery,
            executor=executor,
            agent_config=agent_config,
            capability_sdk=capability_sdk,
            context_policy=context_policy,
            native_result_routing=native_result_routing,
            session_service=session_service,
        )

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
        session_id: str | None,
        user_id: str,
        tenant_id: str,
        auth_context: AuthContext | None = None,
        resume: bool = False,
        task_description: str | None = None,
        **session_kwargs: Any,
    ) -> PRASession:
        logical = None
        if self.session_service is not None:
            logical = (
                self.session_service.resolve_session(user_id, session_id)
                if resume
                else self.session_service.create_session(
                    user_id,
                    session_id,
                    tenant_id=tenant_id,
                    task_description=task_description,
                )
            )
            session_id = logical.session_id
            tenant_id = logical.tenant_id
        if session_id is None:
            raise ValueError("session_id is required without a SessionService.")
        if session_id in self.sessions and not self.sessions[session_id].closed:
            raise ValueError(f"Session already open: {session_id}")
        auth = auth_context or AuthContext(tenant_id, user_id, session_id)
        session = PRASession(session_id, user_id, tenant_id, auth, **session_kwargs)
        self.sessions[session_id] = session
        if logical is not None:
            self.logical_sessions[session_id] = logical
        scope = RecordScope(tenant_id, session_id)
        context_policy = self.context_policy
        if context_policy.local_store is not None:
            context_policy = replace(
                context_policy,
                local_store=Path(context_policy.local_store) / scope.fingerprint,
            )
        adaptive = AdaptiveContextRuntime(scope, context_policy)
        self.result_contexts[session_id] = ProgressiveContextRuntime(
            adaptive,
            chunk_tokens=self.config.pra.chunk_tokens,
            pra_model=(self.backend.model if self.native_result_routing else None),
        )
        return session

    def close_session(self, session: PRASession) -> None:
        context = self.result_contexts.pop(session.session_id, None)
        if context is not None:
            model = context.registry.pra_model
            if model is not None:
                handles = (
                    *context.registry.reference_handles.values(),
                    *context.registry.backing_reference_handles.values(),
                    *context.registry.lazy_reference_handles.values(),
                )
                for handle in {handle.uri: handle for handle in handles}.values():
                    model.remove_reference(handle)
            context.runtime.store.close()
        if self.external_memory is not None:
            self.external_memory.teardown_session(session)
        else:
            session.resources.clear()
            session.admitted_resources.clear()
            session.warm_handles.clear()
            session.hot_handles.clear()
            session.closed = True
        self.sessions.pop(session.session_id, None)

    def logical_session_for(self, session: PRASession) -> AgentSessionState:
        """Return durable logical state paired with an open physical session."""

        self.context_for(session)
        try:
            return self.logical_sessions[session.session_id]
        except KeyError as error:
            raise RuntimeError("No SessionService is installed for this runtime session.") from error

    def append_session_record(
        self,
        session: PRASession,
        record: ContextRecord,
        *,
        task_id: str | None = None,
    ) -> AgentSessionState:
        """Persist one typed record, optionally owned by the active task."""

        logical = self.logical_session_for(session)
        if task_id is None:
            task_id = logical.active_task_id
        if task_id is not None:
            record = attach_task_provenance(
                record,
                TaskProvenance(
                    task_id,
                    producing_task_id=task_id,
                    event_sequence=logical.tasks.last_sequence,
                ),
            )
        assert self.session_service is not None
        updated = self.session_service.append_record(
            logical.user_id, logical.session_id, record
        )
        self.logical_sessions[session.session_id] = updated
        return updated

    def apply_task_event(
        self, session: PRASession, event: TaskEvent
    ) -> AgentSessionState:
        """Validate and persist one idempotent task mutation."""

        logical = self.logical_session_for(session)
        assert self.session_service is not None
        updated = self.session_service.apply_task_event(
            logical.user_id, logical.session_id, event
        )
        self.logical_sessions[session.session_id] = updated
        return updated

    def apply_task_operations(
        self,
        session: PRASession,
        operations: Sequence[TaskOperation],
    ) -> AgentSessionState:
        """Validate model-proposed mutations, then persist their replayable events."""

        logical = self.logical_session_for(session)
        validation_graph = TaskGraph(logical.tasks)
        events = validate_task_operations(
            validation_graph,
            operations,
            sequence_start=logical.tasks.last_sequence,
        )
        updated = logical
        for event in events:
            updated = self.apply_task_event(session, event)
        return updated

    def select_task_context(
        self,
        session: PRASession,
        query: str,
        *,
        policy: TaskScopePolicy | str = TaskScopePolicy.TASK_ADAPTIVE,
        max_records: int = 8,
        minimum_records: int = 1,
        metadata_complete: bool = True,
    ) -> ScopeSelection:
        """Scope the durable record stream by task before ordinary retrieval."""

        logical = self.logical_session_for(session)
        if logical.active_task_id is None:
            raise RuntimeError("Task-aware context selection requires an active task.")
        return TaskScopeSelector(
            TaskGraph(logical.tasks), logical.records
        ).select(
            logical.active_task_id,
            query,
            policy=policy,
            max_records=max_records,
            minimum_records=minimum_records,
            metadata_complete=metadata_complete,
        )

    def context_for(self, session: PRASession) -> ProgressiveContextRuntime:
        """Return the session-scoped compact-result runtime after validation."""

        current = self.sessions.get(session.session_id)
        if current is not session or session.closed:
            raise ValueError("Result context requires an open runtime session.")
        return self.result_contexts[session.session_id]

    def resolve_engine_session_turn(
        self, request: Any, capabilities: Any
    ) -> ResolvedSessionTurn:
        """Apply the same history/resource-delta policy used by the gateway."""

        return self.engine_sessions.resolve_turn(
            request,
            incremental_messages=bool(
                capabilities.session_state and capabilities.incremental_messages
            ),
            resource_delta=bool(capabilities.logical_refs and capabilities.resource_delta),
        )

    def commit_engine_session_turn(
        self,
        turn: ResolvedSessionTurn,
        request: Any,
        *,
        engine_session_id: str | None,
        prefix_cache_handle: str | None = None,
    ) -> Mapping[str, Any]:
        """Commit reconstructible engine metadata after successful execution."""

        return self.engine_sessions.commit(
            turn,
            request,
            engine_session_id=engine_session_id,
            prefix_cache_handle=prefix_cache_handle,
        ).inspect()

    def capability_resources(self, *, kinds: Sequence[str] = ("tool", "skill")) -> tuple[AgentResource, ...]:
        """Return authorized compact discovery views for configured capabilities."""

        if self.capabilities is None:
            raise RuntimeError("No typed capability SDK is installed.")
        return self.capabilities.resources(kinds=kinds)

    def activate_capability_candidates(
        self, record_ids: Sequence[str]
    ) -> CapabilityPaletteActivation:
        """Activate a bounded lazy selection palette for tools and skills."""

        if self.capabilities is None:
            raise RuntimeError("No typed capability SDK is installed.")
        return self.capabilities.activate_candidates(record_ids)

    def activate_capability(self, record_id: str) -> CapabilityActivation:
        """Activate one exact full capability without semantic rediscovery."""

        if self.capabilities is None:
            raise RuntimeError("No typed capability SDK is installed.")
        return self.capabilities.activate_selected(record_id)

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

    def ingest_result(
        self,
        session: PRASession,
        payload: object,
        *,
        record_type: RecordType | str,
        capabilities: RecordCapabilities | None = None,
        provenance: Mapping[str, object] | None = None,
        ttl_seconds: float | None = None,
        expected_reuse: float = 0.0,
    ) -> AdaptiveContextRecord:
        """Store exact result backing and expose only its typed compact view."""

        context = self.context_for(session)
        record = context.ingest(
            payload,
            record_type=record_type,
            capabilities=capabilities,
            provenance=provenance,
            ttl_seconds=ttl_seconds,
            expected_reuse=expected_reuse,
        )
        if self.native_result_routing and self.config.auto_prepare_native_results:
            context.prepare_native_index(record.record_id)
        return record

    def compact_result(self, session: PRASession, record_id: str) -> object:
        """Return the bounded prompt-visible representation of one result."""

        context = self.context_for(session)
        return context.runtime.records[record_id].compact_view()

    def materialize_result(
        self,
        session: PRASession,
        record_id: str,
        *,
        level: RecordViewName | str = RecordViewName.FULL,
        selector: Mapping[str, object] | None = None,
    ) -> MaterializationResult:
        """Resolve an authorized full or selected view from exact local backing."""

        return self.context_for(session).runtime.retrieve_record(
            record_id, level=level, selector=selector
        )

    def search_results(
        self,
        session: PRASession,
        query: str,
        *,
        top_k: int = 5,
        address_kinds: Sequence[str] | None = None,
    ) -> tuple[AdaptiveContextRecord, ...]:
        """Search retrieval-only addresses without exposing full result payloads."""

        return self.context_for(session).runtime.search_records(
            query, top_k=top_k, address_kinds=address_kinds
        )

    def open_result_cursor(
        self, session: PRASession, record_id: str, **kwargs: object
    ) -> CursorRecord:
        """Open a bounded stateful cursor over an authorized result collection."""

        return self.context_for(session).runtime.open_cursor(record_id, **kwargs)

    def execute_result_cursor(
        self, session: PRASession, action: CursorAction
    ) -> CursorActionResult:
        """Execute one structured cursor operation inside the session scope."""

        return self.context_for(session).runtime.execute_cursor_action(action)

    def register_result_backing(
        self, session: PRASession, record_id: str
    ) -> ReferenceHandle:
        """Return an existing in-budget native index or request one explicitly.

        This compatibility method raises when the configured gate skips the
        full index. New code can inspect that decision through
        :meth:`prepare_result_native_index` and select a cheaper recovery path.
        """

        if not self.native_result_routing:
            raise RuntimeError(
                "Native result routing is disabled; construct PRARuntime with "
                "native_result_routing=True for an isolated model session."
            )
        context = self.context_for(session)
        if context.registry.pra_model is None:
            context.registry.pra_model = self.backend.model
        return context.register_backing_record(record_id)

    def prepare_result_native_index(
        self,
        session: PRASession,
        record_id: str,
        *,
        force: bool = False,
    ) -> NativeIndexAudit:
        """Build, size-gate, or defer one full-backing native index."""

        if not self.native_result_routing:
            raise RuntimeError("Native result routing is disabled.")
        return self.context_for(session).prepare_native_index(record_id, force=force)

    def encode_result_region_native(
        self,
        session: PRASession,
        record_id: str,
        selector: Mapping[str, object],
    ) -> LazyNativeRegion:
        """Natively encode one authorized selected region after a full-index gate."""

        if not self.native_result_routing:
            raise RuntimeError("Native result routing is disabled.")
        return self.context_for(session).encode_selected_region_native(
            record_id, selector
        )

    def result_native_index_audit(
        self, session: PRASession, record_id: str
    ) -> NativeIndexAudit:
        """Return the current lifecycle and cost record for one result index."""

        context = self.context_for(session)
        try:
            return context.registry.native_index_audits[record_id]
        except KeyError as error:
            raise KeyError(f"No native-index decision exists for {record_id!r}.") from error

    def route_result_backing(
        self, session: PRASession, query: str
    ) -> NativePRASelection:
        """Route over registered exact result backing without generating tokens."""

        if not self.native_result_routing:
            raise RuntimeError("Native result routing is disabled.")
        return self.context_for(session).native_select(query)

    def materialize_routed_result(
        self,
        session: PRASession,
        selection: NativePRASelection,
        *,
        top_k: int | None = None,
    ) -> ContextExecutionResult:
        """Decode selected original spans and register the bounded detail view."""

        if not self.native_result_routing:
            raise RuntimeError("Native result routing is disabled.")
        return self.context_for(session).materialize_native_selection(
            selection, top_k=top_k
        )

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

    def execute_tool_and_record(
        self,
        generated_text: str,
        *,
        session: PRASession,
        selected_uris: Sequence[str],
        authorization: ExecutionAuthorization,
        call_id: str,
        prior_observations: Sequence[AgentResource] = (),
        capabilities: RecordCapabilities | None = None,
        ttl_seconds: float | None = None,
        expected_reuse: float = 0.0,
    ) -> RuntimeToolExecution:
        """Execute safely, then compact successful output into a typed record."""

        execution = self.execute_tool(
            generated_text,
            selected_uris=selected_uris,
            authorization=authorization,
            call_id=call_id,
            prior_observations=prior_observations,
        )
        if not execution.executed:
            return RuntimeToolExecution(execution)
        record = self.ingest_result(
            session,
            dict(execution.output),
            record_type=RecordType.TOOL_RESPONSE,
            capabilities=capabilities,
            provenance={
                "producer_tool_uri": execution.resource_uri or "",
                "call_id": call_id,
                "observation_uri": execution.observation.uri if execution.observation else "",
            },
            ttl_seconds=ttl_seconds,
            expected_reuse=expected_reuse,
        )
        if session.session_id in self.logical_sessions:
            self.append_session_record(
                session,
                ContextRecord(
                    record_id=record.record_id,
                    record_type=RecordType.TOOL_RESPONSE,
                    payload={
                        "compact": record.compact_view(),
                        "producer_tool_uri": execution.resource_uri or "",
                        "call_id": call_id,
                        "exact_backing": record.record_id,
                    },
                    selection_provenance={
                        "exact_backing": record.record_id,
                        "compression_strategy": record.compression_strategy,
                    },
                ),
            )
        return RuntimeToolExecution(execution, record)

    def generate(self, prompt: str, **kwargs: Any) -> str | GenerationResult:
        return self.backend.generate(prompt, **kwargs)

    def route(self, prompt: str):
        """Run discovery without generation when the selected backend supports it."""

        operation = getattr(self.backend, "route", None)
        if operation is None:
            raise RuntimeError("The selected backend does not expose frozen routing.")
        return operation(prompt)

    def freeze_native_selection(
        self, selected: Iterable[dict[str, Any]]
    ) -> FrozenNativeSelection:
        """Freeze source identities so materialization geometry can vary independently."""

        operation = getattr(self.backend, "freeze_native_selection", None)
        if operation is None:
            raise RuntimeError("The selected backend cannot freeze native selections.")
        return operation(selected)

    def plan_native_materialization(
        self,
        frozen: FrozenNativeSelection,
        **kwargs: Any,
    ) -> NativeMaterializationPlan:
        """Create a record-bounded layer-native materialization plan."""

        operation = getattr(self.backend, "plan_native_materialization", None)
        if operation is None:
            raise RuntimeError("The selected backend cannot plan native materialization.")
        return operation(frozen, **kwargs)

    def generate_with_native_plan(
        self,
        prompt: str,
        plan: NativeMaterializationPlan,
        **kwargs: Any,
    ) -> str | GenerationResult:
        """Generate with a frozen plan while preserving backend capability checks."""

        operation = getattr(self.backend, "generate_with_native_plan", None)
        if operation is None:
            raise RuntimeError("The selected backend cannot consume native plans.")
        return operation(prompt, plan, **kwargs)

    def inspect(self) -> dict[str, Any]:
        capability_state = None
        if self.capabilities is not None:
            capability_state = {
                "tools": len(self.capabilities.tools),
                "skills": len(self.capabilities.skills),
                "max_candidates": self.capabilities.config.max_candidates,
                "selection_view_token_budget": (
                    self.capabilities.config.selection_view_token_budget
                ),
                "lazy_selection": self.capabilities.config.encoding.lazy_selection,
                "lazy_full": self.capabilities.config.encoding.lazy_full,
                "accounting": dict(self.capabilities.runtime.accounting()),
            }
        result_contexts = {
            session_id: {
                "scope_fingerprint": context.runtime.scope.fingerprint,
                "accounting": asdict(context.runtime.accounting()),
                "backing_store": asdict(context.runtime.store.stats()),
                "visible_pra_documents": len(context.registry.documents),
                "native_backing_references": len(
                    context.registry.backing_reference_handles
                ),
                "lazy_native_references": len(context.registry.lazy_reference_handles),
                "native_index_lifecycle": {
                    record_id: {
                        "state": audit.native_index_state.value,
                        "requested": audit.native_index_requested,
                        "built": audit.native_index_built,
                        "reason": audit.native_index_skipped_reason,
                        "tokens": audit.native_index_tokens,
                        "bytes": audit.native_index_bytes,
                        "latency_ms": audit.native_index_latency_ms,
                        "lazy_regions": audit.lazy_native_regions_encoded,
                        "lazy_tokens": audit.lazy_native_tokens,
                    }
                    for record_id, audit in context.registry.native_index_audits.items()
                },
            }
            for session_id, context in self.result_contexts.items()
        }
        return {
            "runtime_config": self.config.to_dict(),
            "profile_trace": self.config.pra.product_profile_trace(),
            "backend": dict(self.backend.inspect()),
            "sessions": [session.safe_snapshot() for session in self.sessions.values()],
            "logical_sessions": {
                session_id: {
                    "user_id": state.user_id,
                    "version": state.version,
                    "records": len(state.records),
                    "tasks": len(state.tasks.tasks),
                    "active_task_id": state.active_task_id,
                }
                for session_id, state in self.logical_sessions.items()
            },
            "session_service_installed": self.session_service is not None,
            "external_memory": (
                self.external_memory.metrics.snapshot() if self.external_memory else None
            ),
            "typed_discovery_installed": self.discovery is not None,
            "safe_executor_installed": self.executor is not None,
            "capabilities": capability_state,
            "result_contexts": result_contexts,
            "native_result_routing": self.native_result_routing,
            "engine_sessions": self.engine_sessions.inspect_all(),
            "hot_cache": self.hot_cache.snapshot(),
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        """Persist non-secret runtime configuration; model weights remain external."""

        return self.config.save_pretrained(directory)
