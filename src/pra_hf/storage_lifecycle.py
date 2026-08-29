"""Engine-neutral storage lifecycle for derived PRA native K/V.

The storage manager owns semantic retention and tier transitions. Engine
adapters only implement the attention-ready HOT representation. SOURCE data is
authoritative; every native payload managed here is a disposable derived cache.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

from .context_records import RecordType
from .product_config import pra_home, read_yaml
from .task_context import TaskStatus


class PRAStorageTier(str, Enum):
    """Semantic service levels independent of engine hardware names."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    SOURCE = "source"


class PRARetentionClass(str, Enum):
    """Expected semantic lifetime of a typed source record."""

    PERSISTENT_SHARED = "persistent_shared"
    PERSISTENT_SESSION = "persistent_session"
    RECONSTRUCTABLE = "reconstructable"
    EPHEMERAL = "ephemeral"
    TRANSIENT = "transient"


class PRAStorageEvictionPolicy(str, Enum):
    """Deterministic controls retained alongside weighted retention."""

    LRU = "lru"
    SIZE_AWARE_LRU = "size_aware_lru"
    REUSE_COUNT = "reuse_count"
    RELOAD_COST = "reload_cost"
    WEIGHTED_LRU = "weighted_lru"


_BYTE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_TIME_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _default_storage_path(tier: str) -> str:
    return str(pra_home() / tier)


def parse_byte_size(value: int | str | None) -> int | None:
    """Parse integer bytes or a compact value such as ``8GiB``."""

    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Storage byte limits cannot be negative.")
        return value
    text = str(value).strip().lower().replace(" ", "")
    for suffix in sorted(_BYTE_UNITS, key=len, reverse=True):
        if text.endswith(suffix):
            amount = float(text[: -len(suffix)])
            if amount < 0:
                raise ValueError("Storage byte limits cannot be negative.")
            return int(amount * _BYTE_UNITS[suffix])
    return int(text)


def parse_duration(value: int | float | str | None) -> float | None:
    """Parse seconds or a compact duration such as ``15m`` and ``7d``."""

    if value is None or value == "session":
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Storage durations cannot be negative.")
        return float(value)
    text = str(value).strip().lower().replace(" ", "")
    unit = text[-1]
    if unit in _TIME_UNITS:
        return float(text[:-1]) * _TIME_UNITS[unit]
    return float(text)


@dataclass(frozen=True)
class PRAStorageTierConfig:
    """Capacity, representation, and encoding policy for one tier."""

    enabled: bool = True
    path: str | None = None
    max_bytes: int | str | None = None
    representation: str = "native"
    compression: str = "none"
    kv_quantization: str = "none"
    ttl_seconds: float | str | None = None
    cold_grace_seconds: float | str = 900.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_bytes", parse_byte_size(self.max_bytes))
        object.__setattr__(self, "ttl_seconds", parse_duration(self.ttl_seconds))
        object.__setattr__(self, "cold_grace_seconds", parse_duration(self.cold_grace_seconds) or 0.0)
        if self.compression not in {"none", "gzip", "zstd"}:
            raise ValueError("compression must be none, gzip, or zstd.")
        if self.kv_quantization not in {"none", "int8"}:
            raise ValueError("kv_quantization must be none or int8.")


@dataclass(frozen=True)
class PRARecordRetentionPolicy:
    """Typed prior used by persistent eviction and lifecycle compaction."""

    retention_class: PRARetentionClass | str
    priority: float
    warm_ttl_seconds: float | str | None = None
    cold_ttl_seconds: float | str | None = None
    cold_enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "retention_class", PRARetentionClass(self.retention_class))
        object.__setattr__(self, "warm_ttl_seconds", parse_duration(self.warm_ttl_seconds))
        object.__setattr__(self, "cold_ttl_seconds", parse_duration(self.cold_ttl_seconds))
        if self.priority < 0:
            raise ValueError("Record retention priority cannot be negative.")


@dataclass(frozen=True)
class PRATaskRetentionPolicy:
    """Task-state multiplier and minimum WARM lifetime."""

    priority_multiplier: float = 1.0
    min_warm_seconds: float | str = 0.0
    compaction_delay_seconds: float | str = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_warm_seconds", parse_duration(self.min_warm_seconds) or 0.0)
        object.__setattr__(self, "compaction_delay_seconds", parse_duration(self.compaction_delay_seconds) or 0.0)
        if self.priority_multiplier < 0:
            raise ValueError("Task priority multiplier cannot be negative.")


def _default_record_policies() -> dict[str, PRARecordRetentionPolicy]:
    shared = PRARecordRetentionPolicy("persistent_shared", 1.0, "7d", "90d", True)
    session = PRARecordRetentionPolicy("persistent_session", 0.5, "1h", None, True)
    reconstructable = PRARecordRetentionPolicy("reconstructable", 0.25, "15m", None, False)
    ephemeral = PRARecordRetentionPolicy("ephemeral", 0.15, "10m", None, False)
    transient = PRARecordRetentionPolicy("transient", 0.05, "2m", None, False)
    result: dict[str, PRARecordRetentionPolicy] = {
        RecordType.GENERIC_DOCUMENT.value: shared,
        RecordType.FILE_READ.value: shared,
        RecordType.SYSTEM_INSTRUCTION.value: session,
        RecordType.SESSION_RECORD.value: session,
        RecordType.TASK_STATE.value: session,
        RecordType.GENERIC_TEXT.value: session,
        RecordType.DB_RESULT.value: reconstructable,
        RecordType.RAG_RESULT.value: reconstructable,
        RecordType.RAG_CHUNK_SET.value: reconstructable,
        RecordType.GRAPH_RESULT.value: reconstructable,
        RecordType.API_RESULT.value: ephemeral,
        RecordType.TOOL_RESPONSE.value: ephemeral,
        RecordType.LOG_BLOCK.value: transient,
        RecordType.TERMINAL_OUTPUT.value: transient,
    }
    return result


def _default_task_policies() -> dict[str, PRATaskRetentionPolicy]:
    return {
        TaskStatus.PENDING.value: PRATaskRetentionPolicy(1.25, "2h", "5m"),
        TaskStatus.ACTIVE.value: PRATaskRetentionPolicy(2.0, "2h", "5m"),
        TaskStatus.BLOCKED.value: PRATaskRetentionPolicy(1.5, "6h", "5m"),
        TaskStatus.COMPLETED.value: PRATaskRetentionPolicy(0.5, 0, "5m"),
        TaskStatus.CANCELLED.value: PRATaskRetentionPolicy(0.1, 0, "1m"),
        "waiting": PRATaskRetentionPolicy(1.5, "12h", "5m"),
        "failed": PRATaskRetentionPolicy(0.25, 0, "5m"),
    }


@dataclass(frozen=True)
class PRAStoragePolicy:
    """Resolved engine-neutral policy for all semantic storage tiers."""

    profile: str = "balanced"
    hot: PRAStorageTierConfig = field(default_factory=lambda: PRAStorageTierConfig(max_bytes="8GiB"))
    warm: PRAStorageTierConfig = field(default_factory=lambda: PRAStorageTierConfig(path=_default_storage_path("warm"), max_bytes="64GiB"))
    cold: PRAStorageTierConfig = field(default_factory=lambda: PRAStorageTierConfig(path=_default_storage_path("cold"), max_bytes="512GiB", compression="gzip"))
    eviction_policy: PRAStorageEvictionPolicy | str = PRAStorageEvictionPolicy.WEIGHTED_LRU
    record_types: Mapping[str, PRARecordRetentionPolicy] = field(default_factory=_default_record_policies)
    task_states: Mapping[str, PRATaskRetentionPolicy] = field(default_factory=_default_task_policies)
    task_aware: bool = True
    immediate_persistence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "eviction_policy", PRAStorageEvictionPolicy(self.eviction_policy))
        object.__setattr__(self, "record_types", {
            str(key): value if isinstance(value, PRARecordRetentionPolicy) else PRARecordRetentionPolicy(**value)
            for key, value in self.record_types.items()
        })
        object.__setattr__(self, "task_states", {
            str(key).lower(): value if isinstance(value, PRATaskRetentionPolicy) else PRATaskRetentionPolicy(**value)
            for key, value in self.task_states.items()
        })

    @classmethod
    def named(cls, name: str, *, home: str | Path | None = None) -> "PRAStoragePolicy":
        root = Path(home) if home is not None else pra_home()
        profiles = {
            "memory": dict(warm=PRAStorageTierConfig(path=str(root / "warm"), max_bytes="2GiB"), cold=PRAStorageTierConfig(enabled=False), immediate_persistence=False),
            "balanced": dict(warm=PRAStorageTierConfig(path=str(root / "warm"), max_bytes="64GiB"), cold=PRAStorageTierConfig(path=str(root / "cold"), max_bytes="512GiB", compression="gzip")),
            "persistent": dict(warm=PRAStorageTierConfig(path=str(root / "warm"), max_bytes="256GiB", cold_grace_seconds="1h"), cold=PRAStorageTierConfig(path=str(root / "cold"), max_bytes="2TiB", compression="gzip")),
            "minimal": dict(hot=PRAStorageTierConfig(max_bytes="1GiB"), warm=PRAStorageTierConfig(path=str(root / "warm"), max_bytes="4GiB", cold_grace_seconds="2m"), cold=PRAStorageTierConfig(path=str(root / "cold"), max_bytes="16GiB", compression="gzip")),
        }
        try:
            return cls(profile=name, **profiles[name])
        except KeyError as error:
            raise ValueError(f"Unknown storage profile {name!r}; choose {', '.join(profiles)}.") from error

    @classmethod
    def from_dict(cls, values: Mapping[str, object], *, home: str | Path | None = None) -> "PRAStoragePolicy":
        values = dict(values)
        profile = str(values.pop("profile", "balanced"))
        base = cls.named(profile, home=home)
        tier_values = {}
        for name in ("hot", "warm", "cold"):
            override = values.pop(name, None)
            current = asdict(getattr(base, name))
            if override:
                normalized_tier = dict(override)
                if "max_size" in normalized_tier:
                    normalized_tier["max_bytes"] = normalized_tier.pop("max_size")
                if "cold_grace_period" in normalized_tier:
                    normalized_tier["cold_grace_seconds"] = normalized_tier.pop("cold_grace_period")
                current.update(normalized_tier)
            tier_values[name] = PRAStorageTierConfig(**current)
        eviction = dict(values.pop("eviction", {}))
        if "eviction_policy" in values:
            eviction["policy"] = values.pop("eviction_policy")
        record_types = dict(base.record_types)
        serialized_record_types = values.pop("record_types", {})
        record_overrides = dict(eviction.pop("record_types", {})) or dict(serialized_record_types)
        for key, raw in record_overrides.items():
            normalized = dict(raw)
            if "warm_ttl" in normalized:
                normalized["warm_ttl_seconds"] = normalized.pop("warm_ttl")
            if "cold_ttl" in normalized:
                normalized["cold_ttl_seconds"] = normalized.pop("cold_ttl")
            if "cold" in normalized:
                normalized["cold_enabled"] = normalized.pop("cold")
            record_types[str(key)] = PRARecordRetentionPolicy(**normalized)
        task_states = dict(base.task_states)
        task_overrides = dict(values.pop("tasks", {})) or dict(values.pop("task_states", {}))
        for key, raw in task_overrides.items():
            normalized = dict(raw)
            if "min_warm_retention" in normalized:
                normalized["min_warm_seconds"] = normalized.pop("min_warm_retention")
            if "compaction_delay" in normalized:
                normalized["compaction_delay_seconds"] = normalized.pop("compaction_delay")
            task_states[str(key).lower()] = PRATaskRetentionPolicy(**normalized)
        policy_name = eviction.pop("policy", base.eviction_policy.value)
        task_aware = bool(values.pop("task_aware", base.task_aware))
        immediate_persistence = bool(values.pop("immediate_persistence", base.immediate_persistence))
        if eviction or values:
            unknown = sorted((*eviction, *values))
            raise ValueError(f"Unknown storage policy fields: {', '.join(unknown)}")
        return cls(profile=profile, **tier_values, eviction_policy=policy_name, record_types=record_types, task_states=task_states, task_aware=task_aware, immediate_persistence=immediate_persistence)

    @classmethod
    def from_yaml(cls, path: str | Path, *, home: str | Path | None = None) -> "PRAStoragePolicy":
        data = read_yaml(path)
        return cls.from_dict(data.get("storage", data), home=home)

    def record_policy(self, record_type: str) -> PRARecordRetentionPolicy:
        return self.record_types.get(record_type, PRARecordRetentionPolicy("reconstructable", 0.25, "15m", None, False))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["eviction_policy"] = self.eviction_policy.value
        return json.loads(json.dumps(value))


@dataclass(frozen=True)
class PRAStorageFingerprint:
    """Compatibility identity for persisted native K/V payloads."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology: str
    dtype: str
    layout: str
    position_policy: str
    consumer_profile: str
    resource_version: str
    compression: str = "none"
    quantization: str = "none"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PRAStorageEntry:
    """Semantic metadata used for retention; never contains engine payloads."""

    logical_key: str
    record_type: str
    retention_class: PRARetentionClass | str
    tenant_id: str
    session_id: str | None
    task_id: str | None
    task_status: str | None
    resource_version: str
    detail_bytes: int
    security_scope: str | None = None
    source_reconstructable: bool = True
    reconstruction_cost_ms: float = 0.0
    created_ns: int = field(default_factory=time.time_ns)
    last_access_ns: int = field(default_factory=time.time_ns)
    last_selected_ns: int | None = None
    selection_count: int = 0
    last_consumed_ns: int | None = None
    consumption_count: int = 0
    reuse_count: int = 0
    dependent_record_count: int = 0
    current_tier: PRAStorageTier | str = PRAStorageTier.SOURCE
    hot_bytes: int = 0
    warm_bytes: int = 0
    cold_bytes: int = 0
    compression: str = "none"
    quantization: str = "none"
    request_pin_count: int = 0
    shared_reference_count: int = 0
    persistence_eligible_ns: int | None = None
    compaction_due_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "retention_class", PRARetentionClass(self.retention_class))
        object.__setattr__(self, "current_tier", PRAStorageTier(self.current_tier))
        if not self.logical_key or not self.tenant_id or self.detail_bytes < 0:
            raise ValueError("Storage entries require identity, tenant, and nonnegative bytes.")


class PRAStorageBackend(ABC):
    """Byte-oriented backend; semantic policy never depends on its class."""

    @abstractmethod
    def contains(self, key: str) -> bool: ...

    @abstractmethod
    def put(self, key: str, payload: bytes, metadata: Mapping[str, object]) -> int: ...

    @abstractmethod
    def get(self, key: str, metadata: Mapping[str, object]) -> bytes: ...

    @abstractmethod
    def remove(self, key: str) -> None: ...

    @abstractmethod
    def bytes_used(self) -> int: ...

    @abstractmethod
    def keys(self) -> tuple[str, ...]: ...


class MemoryKVStore(PRAStorageBackend):
    """Lossless in-process WARM backend used by tests and memory profiles."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[bytes, dict[str, object]]] = {}

    def contains(self, key: str) -> bool:
        return key in self._values

    def put(self, key: str, payload: bytes, metadata: Mapping[str, object]) -> int:
        self._values[key] = (bytes(payload), dict(metadata))
        return len(payload)

    def get(self, key: str, metadata: Mapping[str, object]) -> bytes:
        payload, stored = self._values[key]
        expected = metadata.get("fingerprint")
        if expected is not None and stored.get("fingerprint") != expected:
            raise ValueError("Persisted PRA fingerprint is incompatible.")
        return payload

    def remove(self, key: str) -> None:
        self._values.pop(key, None)

    def bytes_used(self) -> int:
        return sum(len(value[0]) for value in self._values.values())

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))


class FileKVStore(PRAStorageBackend):
    """Hashed atomic file store with strict fingerprint verification."""

    def __init__(self, path: str | Path, *, compression: str = "none") -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        if compression not in {"none", "gzip", "zstd"}:
            raise ValueError("Unsupported storage compression.")

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        directory = self.path / digest[:2] / digest[2:4]
        return directory / f"{digest}.bin", directory / f"{digest}.json"

    def _encode(self, payload: bytes) -> bytes:
        if self.compression == "none":
            return payload
        if self.compression == "gzip":
            return gzip.compress(payload)
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError("zstd storage requires the optional zstandard package.") from error
        return zstandard.ZstdCompressor().compress(payload)

    def _decode(self, payload: bytes) -> bytes:
        if self.compression == "none":
            return payload
        if self.compression == "gzip":
            return gzip.decompress(payload)
        import zstandard
        return zstandard.ZstdDecompressor().decompress(payload)

    def contains(self, key: str) -> bool:
        payload, manifest = self._paths(key)
        return payload.exists() and manifest.exists()

    def put(self, key: str, payload: bytes, metadata: Mapping[str, object]) -> int:
        payload_path, manifest_path = self._paths(key)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = self._encode(bytes(payload))
        manifest = {**dict(metadata), "logical_key": key, "stored_bytes": len(encoded), "compression": self.compression, "payload_sha256": hashlib.sha256(payload).hexdigest()}
        temp_payload = payload_path.with_suffix(".bin.tmp")
        temp_manifest = manifest_path.with_suffix(".json.tmp")
        temp_payload.write_bytes(encoded)
        temp_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(temp_payload, payload_path)
        os.replace(temp_manifest, manifest_path)
        return len(encoded)

    def get(self, key: str, metadata: Mapping[str, object]) -> bytes:
        payload_path, manifest_path = self._paths(key)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("logical_key") != key:
            raise ValueError("Persisted PRA logical identity is incompatible.")
        expected = metadata.get("fingerprint")
        if expected is not None and manifest.get("fingerprint") != expected:
            raise ValueError("Persisted PRA fingerprint is incompatible.")
        payload = self._decode(payload_path.read_bytes())
        if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
            raise ValueError("Persisted PRA payload checksum failed.")
        return payload

    def remove(self, key: str) -> None:
        for path in self._paths(key):
            path.unlink(missing_ok=True)

    def bytes_used(self) -> int:
        return sum(path.stat().st_size for path in self.path.rglob("*.bin"))

    def keys(self) -> tuple[str, ...]:
        keys = []
        for path in self.path.rglob("*.json"):
            try:
                keys.append(str(json.loads(path.read_text(encoding="utf-8"))["logical_key"]))
            except (OSError, ValueError, KeyError):
                continue
        return tuple(sorted(keys))


class PRAHotBridge(Protocol):
    """Only engine-specific contract needed by semantic storage policy."""

    def load_hot(self, logical_key: str, payload: bytes) -> int: ...
    def release_hot(self, logical_key: str) -> None: ...
    def pin_hot(self, logical_key: str, request_id: str) -> None: ...
    def unpin_hot(self, logical_key: str, request_id: str) -> None: ...
    def hot_bytes(self, logical_key: str) -> int: ...


class InMemoryHotBridge:
    """Portable HOT baseline shared by HF and engine policy tests."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.pins: dict[str, set[str]] = {}

    def load_hot(self, logical_key: str, payload: bytes) -> int:
        self.payloads.setdefault(logical_key, bytes(payload))
        return len(self.payloads[logical_key])

    def release_hot(self, logical_key: str) -> None:
        if self.pins.get(logical_key):
            raise RuntimeError("Cannot release request-pinned PRA HOT state.")
        self.payloads.pop(logical_key, None)

    def pin_hot(self, logical_key: str, request_id: str) -> None:
        if logical_key not in self.payloads:
            raise KeyError(logical_key)
        self.pins.setdefault(logical_key, set()).add(request_id)

    def unpin_hot(self, logical_key: str, request_id: str) -> None:
        self.pins.get(logical_key, set()).discard(request_id)

    def hot_bytes(self, logical_key: str) -> int:
        return len(self.payloads.get(logical_key, b""))


class ResidencyManagerHotBridge:
    """Adapt :class:`EnginePRAResidencyManager` to semantic HOT operations.

    ``decode`` maps persisted bytes to the engine-native object. ``encode`` is
    intentionally absent: demotion receives canonical persistence bytes from
    the engine integration, keeping engine tensor serialization explicit.
    """

    def __init__(self, manager: object, decode: Callable[[bytes], object]) -> None:
        self.manager = manager
        self.decode = decode

    def load_hot(self, logical_key: str, payload: bytes) -> int:
        self.manager.resolve(logical_key, lambda: (self.decode(payload), len(payload)))
        return self.manager.hot_bytes(logical_key)

    def release_hot(self, logical_key: str) -> None:
        self.manager.release(logical_key)

    def pin_hot(self, logical_key: str, request_id: str) -> None:
        self.manager.pin(logical_key, request_id)

    def unpin_hot(self, logical_key: str, request_id: str) -> None:
        self.manager.unpin(logical_key, request_id)

    def hot_bytes(self, logical_key: str) -> int:
        return self.manager.hot_bytes(logical_key)


def quantize_int8_array(values: object) -> tuple[bytes, dict[str, object]]:
    """Encode one floating NumPy array with per-array symmetric int8 scaling."""

    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    scale = maximum / 127.0 if maximum else 1.0
    quantized = np.rint(array / scale).clip(-127, 127).astype(np.int8)
    metadata = {"shape": list(array.shape), "source_dtype": str(array.dtype), "scale": scale, "quantization": "int8"}
    return quantized.tobytes(order="C"), metadata


def dequantize_int8_array(payload: bytes, metadata: Mapping[str, object]) -> object:
    """Restore float32 values from :func:`quantize_int8_array` output."""

    import numpy as np

    shape = tuple(int(value) for value in metadata["shape"])
    return np.frombuffer(payload, dtype=np.int8).reshape(shape).astype(np.float32) * float(metadata["scale"])


@dataclass
class PRAStorageMetrics:
    """Disjoint counters for tier, I/O, lifecycle, and policy behavior."""

    hits: dict[str, int] = field(default_factory=lambda: {tier.value: 0 for tier in PRAStorageTier})
    misses: dict[str, int] = field(default_factory=lambda: {tier.value: 0 for tier in PRAStorageTier})
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    reloads: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    persistence_writes: int = 0
    wasted_persistence_writes: int = 0
    task_aware_retention_hits: int = 0
    task_close_bytes_freed: int = 0
    session_close_bytes_freed: int = 0
    promotion_latency_ns: int = 0
    persistence_latency_ns: int = 0
    decompression_latency_ns: int = 0
    dequantization_latency_ns: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PRAStorageManager:
    """Apply record/task-aware lifecycle policy to opaque native K/V bytes."""

    def __init__(self, policy: PRAStoragePolicy, *, hot: PRAHotBridge | None = None, warm: PRAStorageBackend | None = None, cold: PRAStorageBackend | None = None) -> None:
        self.policy = policy
        self.hot = hot or InMemoryHotBridge()
        self.warm = warm or self._backend(policy.warm)
        self.cold = cold or self._backend(policy.cold)
        self.entries: dict[str, PRAStorageEntry] = {}
        self._source_loaders: dict[str, Callable[[], bytes]] = {}
        self._fingerprints: dict[str, str] = {}
        self.metrics = PRAStorageMetrics()

    @staticmethod
    def _backend(config: PRAStorageTierConfig) -> PRAStorageBackend | None:
        if not config.enabled:
            return None
        if config.path is None:
            return MemoryKVStore()
        return FileKVStore(config.path, compression=config.compression)

    def register(self, entry: PRAStorageEntry, payload: bytes, *, source_loader: Callable[[], bytes] | None = None, fingerprint: str | None = None, now_ns: int | None = None) -> PRAStorageEntry:
        """Register derived detail HOT without immediately writing persistence."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        record_policy = self.policy.record_policy(entry.record_type)
        entry = replace(entry, retention_class=record_policy.retention_class, current_tier=PRAStorageTier.HOT, hot_bytes=self.hot.load_hot(entry.logical_key, payload), detail_bytes=len(payload), persistence_eligible_ns=now_ns + int(self.policy.warm.cold_grace_seconds * 1e9), compression="none", quantization="none")
        self.entries[entry.logical_key] = entry
        self._source_loaders[entry.logical_key] = source_loader or (lambda value=bytes(payload): value)
        if fingerprint is not None:
            self._fingerprints[entry.logical_key] = fingerprint
        self._enforce_hot_quota(entry.logical_key, now_ns)
        if self.policy.immediate_persistence:
            self.demote_hot(entry.logical_key, now_ns=now_ns)
        return self.entries[entry.logical_key]

    def _metadata(self, key: str) -> dict[str, object]:
        entry = self.entries[key]
        return {"fingerprint": self._fingerprints.get(key), "record_type": entry.record_type, "retention_class": entry.retention_class.value, "tenant_id": entry.tenant_id, "session_id": entry.session_id, "task_id": entry.task_id}

    def _enforce_hot_quota(self, protected_key: str, now_ns: int) -> None:
        limit = self.policy.hot.max_bytes
        if limit is None:
            return
        if self.entries[protected_key].hot_bytes > limit:
            self.hot.release_hot(protected_key)
            self.entries[protected_key] = replace(
                self.entries[protected_key], current_tier=PRAStorageTier.SOURCE, hot_bytes=0
            )
            raise MemoryError("One PRA object exceeds the complete HOT budget.")
        while sum(entry.hot_bytes for entry in self.entries.values()) > limit:
            candidates = [
                entry for entry in self.entries.values()
                if entry.current_tier == PRAStorageTier.HOT
                and entry.logical_key != protected_key
                and not entry.request_pin_count
            ]
            if not candidates:
                raise MemoryError("Pinned PRA objects exhaust the HOT budget.")
            victim = min(
                candidates,
                key=lambda entry: (
                    self.retention_score(entry, now_ns=now_ns, persistent=False),
                    entry.logical_key,
                ),
            )
            self.demote_hot(victim.logical_key, now_ns=now_ns)

    def record_access(self, key: str, *, selected: bool = False, consumed: bool = False, now_ns: int | None = None) -> None:
        now_ns = time.time_ns() if now_ns is None else now_ns
        entry = self.entries[key]
        self.entries[key] = replace(entry, last_access_ns=now_ns, last_selected_ns=now_ns if selected else entry.last_selected_ns, selection_count=entry.selection_count + int(selected), last_consumed_ns=now_ns if consumed else entry.last_consumed_ns, consumption_count=entry.consumption_count + int(consumed), reuse_count=entry.reuse_count + int(entry.selection_count > 0 and selected))

    def demote_hot(self, key: str, *, payload: bytes | None = None, now_ns: int | None = None) -> PRAStorageEntry:
        """Release HOT and retain lossless WARM when policy permits."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        entry = self.entries[key]
        if entry.request_pin_count:
            raise RuntimeError("Cannot demote request-pinned PRA storage.")
        if payload is None:
            payload = self.hot.payloads[key] if isinstance(self.hot, InMemoryHotBridge) else self._source_loaders[key]()
        stored = 0
        target = PRAStorageTier.SOURCE
        if self.warm is not None and self.policy.warm.enabled:
            started = time.monotonic_ns()
            stored = self.warm.put(key, payload, self._metadata(key))
            self.metrics.persistence_latency_ns += time.monotonic_ns() - started
            self.metrics.bytes_written += stored
            self.metrics.persistence_writes += 1
            target = PRAStorageTier.WARM
        self.hot.release_hot(key)
        self.metrics.demotions += 1
        self.entries[key] = replace(entry, current_tier=target, hot_bytes=0, warm_bytes=stored, last_access_ns=now_ns)
        return self.entries[key]

    def promote(self, key: str, *, request_id: str | None = None, tenant_id: str | None = None, authorization_scopes: Iterable[str] = (), now_ns: int | None = None) -> bytes:
        """Promote exact WARM/COLD bytes, or reconstruct from SOURCE."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        entry = self.entries[key]
        if tenant_id is not None and tenant_id != entry.tenant_id:
            raise PermissionError("Cross-tenant PRA storage promotion is forbidden.")
        if entry.security_scope and entry.security_scope not in set(authorization_scopes):
            raise PermissionError("The request is not authorized for this PRA storage object.")
        started = time.monotonic_ns()
        payload: bytes
        if entry.current_tier == PRAStorageTier.HOT:
            self.metrics.hits["hot"] += 1
            payload = self.hot.payloads[key] if isinstance(self.hot, InMemoryHotBridge) else self._source_loaders[key]()
        elif self.warm is not None and self.warm.contains(key):
            self.metrics.hits["warm"] += 1
            payload = self.warm.get(key, self._metadata(key))
            self.metrics.bytes_read += entry.warm_bytes
            self.metrics.promotions += 1
            self.metrics.reloads += 1
        elif self.cold is not None and self.cold.contains(key):
            self.metrics.hits["cold"] += 1
            payload = self.cold.get(key, self._metadata(key))
            self.metrics.bytes_read += entry.cold_bytes
            self.metrics.promotions += 1
            self.metrics.reloads += 1
        else:
            self.metrics.hits["source"] += 1
            payload = self._source_loaders[key]()
            self.metrics.reloads += 1
        hot_bytes = self.hot.load_hot(key, payload)
        if request_id is not None:
            self.hot.pin_hot(key, request_id)
        self.metrics.promotion_latency_ns += time.monotonic_ns() - started
        self.entries[key] = replace(entry, current_tier=PRAStorageTier.HOT, hot_bytes=hot_bytes, request_pin_count=entry.request_pin_count + int(request_id is not None), last_access_ns=now_ns)
        self._enforce_hot_quota(key, now_ns)
        return payload

    def unpin(self, key: str, request_id: str) -> None:
        entry = self.entries[key]
        self.hot.unpin_hot(key, request_id)
        self.entries[key] = replace(entry, request_pin_count=max(0, entry.request_pin_count - 1))

    def update_task_status(self, task_id: str, status: TaskStatus | str, *, now_ns: int | None = None) -> None:
        now_ns = time.time_ns() if now_ns is None else now_ns
        status_value = status.value if isinstance(status, TaskStatus) else str(status).lower()
        task_policy = self.policy.task_states.get(status_value, PRATaskRetentionPolicy())
        closed = status_value in {"completed", "failed", "cancelled"}
        for key, entry in tuple(self.entries.items()):
            if entry.task_id == task_id:
                due = now_ns + int(task_policy.compaction_delay_seconds * 1e9) if closed else None
                self.entries[key] = replace(entry, task_status=status_value, compaction_due_ns=due)

    def close_session(self, session_id: str, *, now_ns: int | None = None) -> int:
        """Compact session-only native detail while preserving shared resources."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        before = self.usage()["total_native_bytes"]
        for key, entry in tuple(self.entries.items()):
            if entry.session_id != session_id or entry.retention_class == PRARetentionClass.PERSISTENT_SHARED:
                continue
            self._drop_to_source(key, remove_warm=True, remove_cold=entry.retention_class in {PRARetentionClass.EPHEMERAL, PRARetentionClass.TRANSIENT})
        freed = max(0, before - self.usage()["total_native_bytes"])
        self.metrics.session_close_bytes_freed += freed
        return freed

    def _drop_to_source(self, key: str, *, remove_warm: bool, remove_cold: bool) -> None:
        entry = self.entries[key]
        if entry.current_tier == PRAStorageTier.HOT and not entry.request_pin_count:
            self.hot.release_hot(key)
        if remove_warm and self.warm is not None:
            self.warm.remove(key)
        if remove_cold and self.cold is not None:
            self.cold.remove(key)
        self.entries[key] = replace(entry, current_tier=PRAStorageTier.SOURCE, hot_bytes=0, warm_bytes=0 if remove_warm else entry.warm_bytes, cold_bytes=0 if remove_cold else entry.cold_bytes)
        self.metrics.evictions += 1

    def retention_score(self, entry: PRAStorageEntry, *, now_ns: int | None = None, persistent: bool = True) -> float:
        """Return deterministic value density; lower entries leave first."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        age_seconds = max(0.0, (now_ns - entry.last_access_ns) / 1e9)
        if self.policy.eviction_policy == PRAStorageEvictionPolicy.LRU:
            return -age_seconds
        if self.policy.eviction_policy == PRAStorageEvictionPolicy.SIZE_AWARE_LRU:
            return -age_seconds / max(entry.detail_bytes, 1)
        if self.policy.eviction_policy == PRAStorageEvictionPolicy.REUSE_COUNT:
            return float(entry.reuse_count)
        if self.policy.eviction_policy == PRAStorageEvictionPolicy.RELOAD_COST:
            return (entry.reuse_count + 1) * max(entry.reconstruction_cost_ms, 0.001)
        record = self.policy.record_policy(entry.record_type)
        task = self.policy.task_states.get((entry.task_status or "").lower(), PRATaskRetentionPolicy())
        task_multiplier = task.priority_multiplier if self.policy.task_aware else 1.0
        recency = math.exp(-age_seconds / 3600.0)
        frequency = math.log1p(entry.selection_count + entry.consumption_count + entry.reuse_count)
        reload_cost = math.log1p(max(entry.reconstruction_cost_ms, 0.0))
        shared = math.log1p(entry.shared_reference_count)
        dependency = math.log1p(entry.dependent_record_count) * 2.0
        reconstructable_discount = 1.0 if entry.source_reconstructable else 0.0
        byte_cost = math.log1p(max(entry.detail_bytes, 1)) / 20.0 if persistent else entry.detail_bytes / max(self.policy.hot.max_bytes or 1, 1)
        return (recency + frequency + reload_cost + record.priority + shared + dependency - reconstructable_discount - byte_cost) * task_multiplier

    def run_maintenance(self, *, now_ns: int | None = None) -> None:
        """Apply delayed task compaction, WARM-to-COLD aging, and quotas."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        before = self.usage()["total_native_bytes"]
        for key, entry in tuple(self.entries.items()):
            record = self.policy.record_policy(entry.record_type)
            task = self.policy.task_states.get((entry.task_status or "").lower(), PRATaskRetentionPolicy())
            age = (now_ns - entry.last_access_ns) / 1e9
            if self.policy.task_aware and entry.task_status in {"pending", "active", "blocked", "waiting"} and age < task.min_warm_seconds:
                self.metrics.task_aware_retention_hits += 1
                continue
            if entry.compaction_due_ns is not None and now_ns >= entry.compaction_due_ns and entry.dependent_record_count == 0:
                if entry.retention_class in {PRARetentionClass.TRANSIENT, PRARetentionClass.EPHEMERAL} or (entry.consumption_count <= 1 and entry.reuse_count == 0 and entry.record_type == RecordType.TOOL_RESPONSE.value):
                    self._drop_to_source(key, remove_warm=True, remove_cold=True)
                    continue
            if entry.current_tier == PRAStorageTier.HOT and not entry.request_pin_count and now_ns >= (entry.persistence_eligible_ns or now_ns):
                self.demote_hot(key, now_ns=now_ns)
                entry = self.entries[key]
            warm_age_limit = max(record.warm_ttl_seconds or 0.0, task.min_warm_seconds if self.policy.task_aware else 0.0, self.policy.warm.cold_grace_seconds)
            if entry.current_tier == PRAStorageTier.WARM and age >= warm_age_limit:
                payload = self.warm.get(key, self._metadata(key)) if self.warm is not None and self.warm.contains(key) else None
                if payload is not None and self.cold is not None and record.cold_enabled:
                    started = time.monotonic_ns()
                    stored = self.cold.put(key, payload, self._metadata(key))
                    self.metrics.persistence_latency_ns += time.monotonic_ns() - started
                    self.metrics.bytes_written += stored
                    self.metrics.persistence_writes += 1
                    self.warm.remove(key)
                    self.entries[key] = replace(entry, current_tier=PRAStorageTier.COLD, warm_bytes=0, cold_bytes=stored, compression=self.policy.cold.compression, quantization=self.policy.cold.kv_quantization)
                elif entry.source_reconstructable:
                    self._drop_to_source(key, remove_warm=True, remove_cold=False)
            elif entry.current_tier == PRAStorageTier.COLD and record.cold_ttl_seconds is not None and age >= record.cold_ttl_seconds and entry.source_reconstructable:
                self._drop_to_source(key, remove_warm=False, remove_cold=True)
        self._enforce_quota(PRAStorageTier.WARM, self.warm, self.policy.warm.max_bytes, now_ns)
        self._enforce_quota(PRAStorageTier.COLD, self.cold, self.policy.cold.max_bytes, now_ns)
        after = self.usage()["total_native_bytes"]
        if after < before and any(entry.compaction_due_ns is not None and now_ns >= entry.compaction_due_ns for entry in self.entries.values()):
            self.metrics.task_close_bytes_freed += before - after

    def _enforce_quota(self, tier: PRAStorageTier, backend: PRAStorageBackend | None, limit: int | None, now_ns: int) -> None:
        if backend is None or limit is None:
            return
        while backend.bytes_used() > limit:
            candidates = [entry for entry in self.entries.values() if entry.current_tier == tier and not entry.request_pin_count]
            if not candidates:
                raise MemoryError(f"Pinned PRA entries exceed the {tier.value} quota.")
            victim = min(candidates, key=lambda entry: (self.retention_score(entry, now_ns=now_ns), entry.logical_key))
            backend.remove(victim.logical_key)
            self.entries[victim.logical_key] = replace(victim, current_tier=PRAStorageTier.SOURCE, warm_bytes=0 if tier == PRAStorageTier.WARM else victim.warm_bytes, cold_bytes=0 if tier == PRAStorageTier.COLD else victim.cold_bytes)
            self.metrics.evictions += 1

    def usage(self) -> dict[str, int]:
        hot = sum(entry.hot_bytes for entry in self.entries.values())
        warm = self.warm.bytes_used() if self.warm is not None else 0
        cold = self.cold.bytes_used() if self.cold is not None else 0
        return {"hot_bytes": hot, "warm_bytes": warm, "cold_bytes": cold, "total_native_bytes": hot + warm + cold, "source_only_objects": sum(entry.current_tier == PRAStorageTier.SOURCE for entry in self.entries.values())}

    def inspect(self) -> dict[str, object]:
        by_tier = {tier.value: sum(entry.current_tier == tier for entry in self.entries.values()) for tier in PRAStorageTier}
        return {"profile": self.policy.profile, "policy": self.policy.to_dict(), "usage": self.usage(), "objects": by_tier, "metrics": self.metrics.to_dict()}
