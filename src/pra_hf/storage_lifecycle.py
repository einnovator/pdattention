"""Engine-neutral storage lifecycle for derived PRA native K/V.

The storage manager owns semantic retention and tier transitions. Engine
adapters only implement the attention-ready HOT representation. SOURCE data is
authoritative; every native payload managed here is a disposable derived cache.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import mmap
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .context_records import RecordType
from .product_config import pra_home, read_yaml
from .task_context import TaskStatus
from .observability import DISABLED_OBSERVABILITY, Observability


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


class PRAPositionBindingMode(str, Enum):
    """Whether persistent keys already contain a fixed RoPE phase."""

    POST_ROPE = "post_rope"
    PRE_ROPE = "pre_rope"


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
    per_tenant_max_bytes: int | str | None = None
    representation: str = "native"
    compression: str = "none"
    kv_quantization: str = "none"
    ttl_seconds: float | str | None = None
    cold_grace_seconds: float | str = 900.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_bytes", parse_byte_size(self.max_bytes))
        object.__setattr__(
            self, "per_tenant_max_bytes", parse_byte_size(self.per_tenant_max_bytes)
        )
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
    maintenance_interval_seconds: float | str = 30.0

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
        object.__setattr__(
            self,
            "maintenance_interval_seconds",
            parse_duration(self.maintenance_interval_seconds) or 0.0,
        )

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
        maintenance_interval_seconds = values.pop(
            "maintenance_interval_seconds", base.maintenance_interval_seconds
        )
        if eviction or values:
            unknown = sorted((*eviction, *values))
            raise ValueError(f"Unknown storage policy fields: {', '.join(unknown)}")
        return cls(profile=profile, **tier_values, eviction_policy=policy_name, record_types=record_types, task_states=task_states, task_aware=task_aware, immediate_persistence=immediate_persistence, maintenance_interval_seconds=maintenance_interval_seconds)

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
class PRARopeContract:
    """Host geometry required to turn pre-RoPE keys into request K/V."""

    model_revision: str
    layer_frequency_digest: str
    scaling_policy: tuple[str, ...]
    rope_dims: tuple[int, ...]
    layout: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scaling_policy", tuple(str(value) for value in self.scaling_policy)
        )
        object.__setattr__(self, "rope_dims", tuple(int(value) for value in self.rope_dims))
        object.__setattr__(self, "layout", tuple(str(value) for value in self.layout))
        layers = len(self.rope_dims)
        if not self.model_revision or not self.layer_frequency_digest:
            raise ValueError("RoPE contracts require model and frequency identity.")
        if layers == 0 or len(self.scaling_policy) != layers or len(self.layout) != layers:
            raise ValueError("RoPE contract fields must describe the same nonzero layer count.")
        if any(dimension <= 0 or dimension % 2 for dimension in self.rope_dims):
            raise ValueError("RoPE dimensions must be positive and even.")

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()


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
    position_binding_mode: PRAPositionBindingMode | str = PRAPositionBindingMode.POST_ROPE
    rope_contract: PRARopeContract | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_binding_mode",
            PRAPositionBindingMode(self.position_binding_mode),
        )
        if isinstance(self.rope_contract, Mapping):
            object.__setattr__(self, "rope_contract", PRARopeContract(**self.rope_contract))
        if (
            self.position_binding_mode is PRAPositionBindingMode.PRE_ROPE
            and self.rope_contract is None
        ):
            raise ValueError("Pre-RoPE storage requires an exact host RoPE contract.")

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
    source_uri: str | None = None
    source_sha256: str | None = None
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

    def metadata(self, key: str) -> Mapping[str, object]:
        """Return durable metadata when the backend supports discovery."""

        raise NotImplementedError

    def get_range(
        self, key: str, start: int, end: int, metadata: Mapping[str, object]
    ) -> bytes:
        """Read one byte interval without requiring callers to load the blob."""

        return self.get(key, metadata)[start:end]


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

    def metadata(self, key: str) -> Mapping[str, object]:
        return dict(self._values[key][1])


class FileKVStore(PRAStorageBackend):
    """Hashed atomic file store with strict fingerprint verification."""

    def __init__(self, path: str | Path, *, compression: str = "none") -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        if compression not in {"none", "gzip", "zstd"}:
            raise ValueError("Unsupported storage compression.")
        for temporary in self.path.rglob("*.tmp"):
            temporary.unlink(missing_ok=True)

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
        self._write_synced(temp_payload, encoded)
        self._write_synced(
            temp_manifest, json.dumps(manifest, sort_keys=True).encode("utf-8")
        )
        os.replace(temp_payload, payload_path)
        os.replace(temp_manifest, manifest_path)
        return len(encoded)

    @staticmethod
    def _write_synced(path: Path, payload: bytes) -> None:
        with path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def put_segments(
        self,
        key: str,
        segments: Mapping[str, bytes],
        metadata: Mapping[str, object],
    ) -> int:
        """Atomically persist named contiguous regions for layer-aware WARM reads."""

        if self.compression != "none":
            raise ValueError("Named segment persistence requires an uncompressed store.")
        ordered = tuple((str(name), bytes(value)) for name, value in segments.items())
        offsets: dict[str, dict[str, object]] = {}
        cursor = 0
        buffer = io.BytesIO()
        for name, value in ordered:
            buffer.write(value)
            offsets[name] = {
                "start": cursor,
                "end": cursor + len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            cursor += len(value)
        return self.put(key, buffer.getvalue(), {**dict(metadata), "segments": offsets})

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

    def metadata(self, key: str) -> Mapping[str, object]:
        _payload_path, manifest_path = self._paths(key)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_range(
        self, key: str, start: int, end: int, metadata: Mapping[str, object]
    ) -> bytes:
        if start < 0 or end < start:
            raise ValueError("Invalid PRA storage byte interval.")
        if self.compression != "none":
            return super().get_range(key, start, end, metadata)
        payload_path, _manifest_path = self._paths(key)
        with payload_path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                return bytes(mapped[start:end])

    def get_segments(
        self,
        key: str,
        names: Sequence[str],
        metadata: Mapping[str, object],
    ) -> dict[str, bytes]:
        """Load only requested named regions and verify each region checksum."""

        manifest = self.metadata(key)
        expected = metadata.get("fingerprint")
        if expected is not None and manifest.get("fingerprint") != expected:
            raise ValueError("Persisted PRA fingerprint is incompatible.")
        segments = dict(manifest.get("segments", {}))
        result: dict[str, bytes] = {}
        for name in names:
            if name not in segments:
                raise KeyError(f"Unknown persisted PRA segment: {name}")
            descriptor = dict(segments[name])
            payload = self.get_range(
                key, int(descriptor["start"]), int(descriptor["end"]), metadata
            )
            if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Persisted PRA segment checksum failed: {name}")
            result[name] = payload
        return result


class MemoryMappedKVStore(FileKVStore):
    """Uncompressed file backend optimized for direct and partial WARM reads."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, compression="none")


class PRAHotBridge(Protocol):
    """Only engine-specific contract needed by semantic storage policy."""

    def load_hot(self, logical_key: str, payload: bytes) -> int: ...
    def load_hot_value(self, logical_key: str, value: object, byte_count: int) -> int: ...
    def get_hot(self, logical_key: str) -> object: ...
    def release_hot(self, logical_key: str) -> None: ...
    def pin_hot(self, logical_key: str, request_id: str) -> None: ...
    def unpin_hot(self, logical_key: str, request_id: str) -> None: ...
    def hot_bytes(self, logical_key: str) -> int: ...


class InMemoryHotBridge:
    """Portable HOT baseline shared by HF and engine policy tests."""

    def __init__(self) -> None:
        self.payloads: dict[str, object] = {}
        self.sizes: dict[str, int] = {}
        self.pins: dict[str, set[str]] = {}

    def load_hot(self, logical_key: str, payload: bytes) -> int:
        self.payloads.setdefault(logical_key, bytes(payload))
        self.sizes.setdefault(logical_key, len(payload))
        return self.sizes[logical_key]

    def load_hot_value(self, logical_key: str, value: object, byte_count: int) -> int:
        self.payloads.setdefault(logical_key, value)
        self.sizes.setdefault(logical_key, int(byte_count))
        return self.sizes[logical_key]

    def get_hot(self, logical_key: str) -> object:
        return self.payloads[logical_key]

    def release_hot(self, logical_key: str) -> None:
        if self.pins.get(logical_key):
            raise RuntimeError("Cannot release request-pinned PRA HOT state.")
        self.payloads.pop(logical_key, None)
        self.sizes.pop(logical_key, None)

    def pin_hot(self, logical_key: str, request_id: str) -> None:
        if logical_key not in self.payloads:
            raise KeyError(logical_key)
        self.pins.setdefault(logical_key, set()).add(request_id)

    def unpin_hot(self, logical_key: str, request_id: str) -> None:
        self.pins.get(logical_key, set()).discard(request_id)

    def hot_bytes(self, logical_key: str) -> int:
        return self.sizes.get(logical_key, 0)


class DecodingHotBridge(InMemoryHotBridge):
    """Restore opaque persisted bytes to an engine-native in-process object."""

    def __init__(self, decode: Callable[[bytes], object]) -> None:
        super().__init__()
        self.decode = decode

    def load_hot(self, logical_key: str, payload: bytes) -> int:
        value = self.decode(payload)
        byte_count = int(getattr(value, "nbytes", len(payload)))
        return self.load_hot_value(logical_key, value, byte_count)


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

    def load_hot_value(self, logical_key: str, value: object, byte_count: int) -> int:
        self.manager.resolve(logical_key, lambda: (value, int(byte_count)))
        return self.manager.hot_bytes(logical_key)

    def get_hot(self, logical_key: str) -> object:
        return self.manager.get(logical_key)

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


class PRAColdCodec(Protocol):
    """Engine codec for optional lossy COLD representation."""

    def encode(
        self, payload: bytes, quantization: str
    ) -> tuple[bytes, Mapping[str, object]]: ...

    def decode(self, payload: bytes, metadata: Mapping[str, object]) -> bytes: ...


class Float32Int8ColdCodec:
    """Reference codec for contiguous float32 payloads used by controls."""

    def encode(
        self, payload: bytes, quantization: str
    ) -> tuple[bytes, Mapping[str, object]]:
        if quantization == "none":
            return bytes(payload), {"quantization": "none"}
        if quantization != "int8":
            raise ValueError(f"Unsupported COLD quantization: {quantization}")
        import numpy as np

        if len(payload) % np.dtype(np.float32).itemsize:
            raise ValueError("Float32 int8 COLD codec requires aligned float32 bytes.")
        values = np.frombuffer(payload, dtype=np.float32)
        encoded, metadata = quantize_int8_array(values)
        return encoded, {**metadata, "original_bytes": len(payload)}

    def decode(self, payload: bytes, metadata: Mapping[str, object]) -> bytes:
        if metadata.get("quantization", "none") == "none":
            return bytes(payload)
        values = dequantize_int8_array(payload, metadata)
        return values.astype("float32").tobytes(order="C")


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


def _storage_locked(method):
    """Serialize lifecycle mutations while allowing reentrant tier changes."""

    def synchronized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    synchronized.__name__ = method.__name__
    synchronized.__doc__ = method.__doc__
    return synchronized


def _storage_observed(span_name: str):
    """Wrap coarse lifecycle operations without affecting disabled behavior."""

    def decorate(method):
        def observed(self, *args, **kwargs):
            key = str(args[0]) if args else "unknown"
            before = self.entries.get(key)
            source_tier = before.current_tier.value if before is not None else "unknown"
            started = time.perf_counter()
            with self.observability.span(
                span_name,
                lambda: {
                    "pra.engine": self.engine,
                    "pra.storage.source_tier": source_tier,
                },
            ):
                result = method(self, *args, **kwargs)
            elapsed = time.perf_counter() - started
            self._observe_usage()
            if span_name == "pra.storage.promote" and source_tier != "hot":
                self.observability.increment(
                    "pra_storage_promotions_total",
                    labels={"engine": self.engine, "storage_tier": source_tier},
                )
                self.observability.observe(
                    "pra_storage_promotion_duration_seconds",
                    elapsed,
                    labels={"engine": self.engine, "storage_tier": source_tier},
                )
                if source_tier in {"warm", "cold"}:
                    self.observability.increment(
                        "pra_storage_reloads_total",
                        labels={"engine": self.engine, "storage_tier": source_tier},
                    )
                elif source_tier == "source":
                    self.observability.increment(
                        "pra_storage_source_reads_total", labels={"engine": self.engine}
                    )
                    self.observability.increment(
                        "pra_storage_reconstructions_total", labels={"engine": self.engine}
                    )
            elif span_name == "pra.storage.demote":
                target = self.entries[key].current_tier.value
                self.observability.increment(
                    "pra_storage_demotions_total",
                    labels={"engine": self.engine, "storage_tier": target},
                )
            return result

        observed.__name__ = method.__name__
        observed.__doc__ = method.__doc__
        return observed

    return decorate


class PRAStorageManager:
    """Apply record/task-aware lifecycle policy to opaque native K/V bytes."""

    state_schema = "pra-storage-state-v2"

    def __init__(
        self,
        policy: PRAStoragePolicy,
        *,
        hot: PRAHotBridge | None = None,
        warm: PRAStorageBackend | None = None,
        cold: PRAStorageBackend | None = None,
        cold_codec: PRAColdCodec | None = None,
        source_resolver: Callable[[PRAStorageEntry], bytes] | None = None,
        state_path: str | Path | None = None,
        recover: bool = True,
        observability: Observability | None = None,
        engine: str = "unknown",
    ) -> None:
        self.policy = policy
        self.hot = hot or InMemoryHotBridge()
        self.warm = warm or self._backend(policy.warm)
        self.cold = cold or self._backend(policy.cold)
        self.cold_codec = cold_codec
        if policy.cold.kv_quantization != "none" and cold_codec is None:
            raise ValueError(
                "Quantized COLD storage requires an engine-compatible cold_codec."
            )
        self.source_resolver = source_resolver
        self.entries: dict[str, PRAStorageEntry] = {}
        self._source_loaders: dict[str, Callable[[], bytes]] = {}
        self._fingerprints: dict[str, str] = {}
        self._cold_metadata: dict[str, dict[str, object]] = {}
        self.metrics = PRAStorageMetrics()
        self.observability = observability or DISABLED_OBSERVABILITY
        self.engine = str(engine)
        self._lock = threading.RLock()
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self.state_path = self._resolve_state_path(state_path)
        if recover:
            self.recover()
        # Publish an explicit zero for every configured tier. Grafana should
        # distinguish an empty store from a missing or unscraped runtime.
        self._observe_usage()

    @staticmethod
    def _backend(config: PRAStorageTierConfig) -> PRAStorageBackend | None:
        if not config.enabled:
            return None
        if config.path is None:
            return MemoryKVStore()
        if config.compression == "none" and config.representation in {
            "native",
            "mmap",
            "contiguous",
        }:
            return MemoryMappedKVStore(config.path)
        return FileKVStore(config.path, compression=config.compression)

    def _resolve_state_path(self, state_path: str | Path | None) -> Path | None:
        if state_path is not None:
            return Path(state_path).expanduser().resolve()
        for config in (self.policy.warm, self.policy.cold):
            if config.enabled and config.path:
                return Path(config.path).expanduser().resolve().parent / "lifecycle.json"
        return None

    def _observe_usage(self) -> None:
        if not self.observability.metrics_enabled:
            return
        usage = self.usage()
        self.observability.set_gauge(
            "pra_storage_hot_bytes", usage["hot_bytes"], labels={"engine": self.engine}
        )
        self.observability.set_gauge(
            "pra_storage_warm_bytes", usage["warm_bytes"], labels={"engine": self.engine}
        )
        self.observability.set_gauge(
            "pra_storage_cold_bytes", usage["cold_bytes"], labels={"engine": self.engine}
        )

    @staticmethod
    def _entry_dict(entry: PRAStorageEntry) -> dict[str, object]:
        result = asdict(entry)
        result["retention_class"] = entry.retention_class.value
        result["current_tier"] = entry.current_tier.value
        return result

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "schema": self.state_schema,
            "profile": self.policy.profile,
            "metrics": self.metrics.to_dict(),
            "entries": [
                {
                    "entry": self._entry_dict(entry),
                    "fingerprint": self._fingerprints.get(key),
                    "cold_metadata": self._cold_metadata.get(key, {}),
                }
                for key, entry in sorted(self.entries.items())
            ],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            FileKVStore._write_synced(
                temporary, json.dumps(payload, sort_keys=True).encode("utf-8")
            )
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    @_storage_locked
    def recover(self) -> int:
        """Rehydrate durable semantic metadata after a runtime restart."""

        if self.state_path is None or not self.state_path.exists():
            return 0
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema") != self.state_schema:
            raise ValueError("Unsupported PRA storage lifecycle state schema.")
        stored_metrics = payload.get("metrics")
        if stored_metrics is not None:
            self.metrics = PRAStorageMetrics(**dict(stored_metrics))
        recovered = 0
        for row in payload.get("entries", ()):
            entry = PRAStorageEntry(**dict(row["entry"]))
            key = entry.logical_key
            if self.warm is not None and self.warm.contains(key):
                manifest = self.warm.metadata(key)
                entry = replace(
                    entry,
                    current_tier=PRAStorageTier.WARM,
                    hot_bytes=0,
                    warm_bytes=int(manifest.get("stored_bytes", entry.warm_bytes)),
                )
            elif self.cold is not None and self.cold.contains(key):
                manifest = self.cold.metadata(key)
                entry = replace(
                    entry,
                    current_tier=PRAStorageTier.COLD,
                    hot_bytes=0,
                    warm_bytes=0,
                    cold_bytes=int(manifest.get("stored_bytes", entry.cold_bytes)),
                )
            else:
                entry = replace(
                    entry,
                    current_tier=PRAStorageTier.SOURCE,
                    hot_bytes=0,
                    warm_bytes=0,
                    cold_bytes=0,
                )
            entry = replace(entry, request_pin_count=0)
            self.entries[key] = entry
            if row.get("fingerprint") is not None:
                self._fingerprints[key] = str(row["fingerprint"])
            self._cold_metadata[key] = dict(row.get("cold_metadata", {}))
            recovered += 1
        return recovered

    @_storage_locked
    def attach_source_loader(self, key: str, loader: Callable[[], bytes]) -> None:
        """Reconnect an authoritative SOURCE provider after restart."""

        if key not in self.entries:
            raise KeyError(key)
        self._source_loaders[key] = loader

    def _load_source(self, key: str) -> bytes:
        loader = self._source_loaders.get(key)
        if loader is not None:
            payload = bytes(loader())
        elif self.source_resolver is not None:
            payload = bytes(self.source_resolver(self.entries[key]))
        else:
            entry = self.entries[key]
            source_uri = entry.source_uri
            if source_uri is None:
                raise FileNotFoundError(
                    f"PRA SOURCE for {key!r} is unavailable after durable-cache eviction."
                )
            if source_uri.startswith("file://"):
                parsed = urlparse(source_uri)
                source_uri = url2pathname(unquote(parsed.path))
                if parsed.netloc:
                    source_uri = f"//{parsed.netloc}{source_uri}"
                elif os.name == "nt" and source_uri[:1] in {"/", "\\"} and source_uri[2:3] == ":":
                    source_uri = source_uri[1:]
            payload = Path(source_uri).expanduser().read_bytes()
        expected = self.entries[key].source_sha256
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"PRA SOURCE checksum failed for {key!r}.")
        return payload

    def start_maintenance(self) -> None:
        """Start one idempotent background policy-maintenance worker."""

        interval = float(self.policy.maintenance_interval_seconds)
        if interval <= 0 or self._maintenance_thread is not None:
            return
        self._maintenance_stop.clear()

        def worker() -> None:
            while not self._maintenance_stop.wait(interval):
                self.run_maintenance()

        self._maintenance_thread = threading.Thread(
            target=worker, name="pra-storage-maintenance", daemon=True
        )
        self._maintenance_thread.start()

    def close(self) -> None:
        """Stop maintenance and durably checkpoint semantic lifecycle state."""

        self._maintenance_stop.set()
        thread = self._maintenance_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.policy.maintenance_interval_seconds * 2))
        self._maintenance_thread = None
        with self._lock:
            self._persist_state()

    @_storage_locked
    def register(
        self,
        entry: PRAStorageEntry,
        payload: bytes,
        *,
        hot_value: object | None = None,
        source_loader: Callable[[], bytes] | None = None,
        fingerprint: str | None = None,
        now_ns: int | None = None,
    ) -> PRAStorageEntry:
        """Register derived detail HOT without immediately writing persistence."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        record_policy = self.policy.record_policy(entry.record_type)
        hot_bytes = (
            self.hot.load_hot(entry.logical_key, payload)
            if hot_value is None
            else self.hot.load_hot_value(entry.logical_key, hot_value, entry.detail_bytes)
        )
        entry = replace(entry, retention_class=record_policy.retention_class, current_tier=PRAStorageTier.HOT, hot_bytes=hot_bytes, detail_bytes=max(len(payload), entry.detail_bytes), persistence_eligible_ns=now_ns + int(self.policy.warm.cold_grace_seconds * 1e9), compression="none", quantization="none")
        self.entries[entry.logical_key] = entry
        self._source_loaders[entry.logical_key] = source_loader or (lambda value=bytes(payload): value)
        if fingerprint is not None:
            self._fingerprints[entry.logical_key] = fingerprint
        self._enforce_hot_quota(entry.logical_key, now_ns)
        if self.policy.immediate_persistence:
            self.demote_hot(entry.logical_key, now_ns=now_ns)
        self._observe_usage()
        self._persist_state()
        return self.entries[entry.logical_key]

    @_storage_locked
    def register_source(
        self,
        entry: PRAStorageEntry,
        source_loader: Callable[[], bytes],
        *,
        fingerprint: str | None = None,
    ) -> PRAStorageEntry:
        """Register canonical SOURCE before native detail is materialized."""

        record_policy = self.policy.record_policy(entry.record_type)
        entry = replace(
            entry,
            retention_class=record_policy.retention_class,
            current_tier=PRAStorageTier.SOURCE,
            hot_bytes=0,
            warm_bytes=0,
            cold_bytes=0,
        )
        self.entries[entry.logical_key] = entry
        self._source_loaders[entry.logical_key] = source_loader
        if fingerprint is not None:
            self._fingerprints[entry.logical_key] = fingerprint
        self._persist_state()
        return entry

    def _metadata(self, key: str) -> dict[str, object]:
        entry = self.entries[key]
        return {"fingerprint": self._fingerprints.get(key), "record_type": entry.record_type, "retention_class": entry.retention_class.value, "tenant_id": entry.tenant_id, "session_id": entry.session_id, "task_id": entry.task_id, "storage_entry": self._entry_dict(entry)}

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

    @_storage_locked
    def record_access(self, key: str, *, selected: bool = False, consumed: bool = False, now_ns: int | None = None) -> None:
        now_ns = time.time_ns() if now_ns is None else now_ns
        entry = self.entries[key]
        self.entries[key] = replace(entry, last_access_ns=now_ns, last_selected_ns=now_ns if selected else entry.last_selected_ns, selection_count=entry.selection_count + int(selected), last_consumed_ns=now_ns if consumed else entry.last_consumed_ns, consumption_count=entry.consumption_count + int(consumed), reuse_count=entry.reuse_count + int(entry.selection_count > 0 and selected))
        self._persist_state()

    @_storage_locked
    def update_dependencies(self, task_id: str, dependent_count: int) -> None:
        """Synchronize open downstream task references with retention metadata."""

        if dependent_count < 0:
            raise ValueError("dependent_count cannot be negative.")
        for key, entry in tuple(self.entries.items()):
            if entry.task_id == task_id:
                self.entries[key] = replace(
                    entry, dependent_record_count=int(dependent_count)
                )
        self._persist_state()

    @_storage_observed("pra.storage.demote")
    @_storage_locked
    def demote_hot(self, key: str, *, payload: bytes | None = None, now_ns: int | None = None) -> PRAStorageEntry:
        """Release HOT and retain lossless WARM when policy permits."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        entry = self.entries[key]
        if entry.request_pin_count:
            raise RuntimeError("Cannot demote request-pinned PRA storage.")
        if payload is None:
            if isinstance(self.hot, InMemoryHotBridge):
                value = self.hot.get_hot(key)
                payload = value if isinstance(value, bytes) else self._load_source(key)
            else:
                payload = self._load_source(key)
        stored = 0
        target = PRAStorageTier.SOURCE
        if self.warm is not None and self.policy.warm.enabled:
            if self.warm.contains(key):
                stored = int(
                    self.warm.metadata(key).get("stored_bytes", entry.warm_bytes)
                )
            else:
                started = time.monotonic_ns()
                stored = self.warm.put(key, payload, self._metadata(key))
                self.metrics.persistence_latency_ns += time.monotonic_ns() - started
                self.metrics.bytes_written += stored
                self.metrics.persistence_writes += 1
            target = PRAStorageTier.WARM
            if self.cold is not None and self.cold.contains(key):
                self.cold.remove(key)
        self.hot.release_hot(key)
        self.metrics.demotions += 1
        self.entries[key] = replace(
            entry,
            current_tier=target,
            hot_bytes=0,
            warm_bytes=stored,
            cold_bytes=0 if target == PRAStorageTier.WARM else entry.cold_bytes,
            last_access_ns=now_ns,
        )
        self._persist_state()
        return self.entries[key]

    @_storage_observed("pra.storage.promote")
    @_storage_locked
    def promote(self, key: str, *, request_id: str | None = None, tenant_id: str | None = None, authorization_scopes: Iterable[str] = (), now_ns: int | None = None) -> object:
        """Promote exact WARM/COLD bytes, or reconstruct from SOURCE."""

        now_ns = time.time_ns() if now_ns is None else now_ns
        entry = self.entries[key]
        if tenant_id is not None and tenant_id != entry.tenant_id:
            raise PermissionError("Cross-tenant PRA storage promotion is forbidden.")
        if entry.security_scope and entry.security_scope not in set(authorization_scopes):
            raise PermissionError("The request is not authorized for this PRA storage object.")
        started = time.monotonic_ns()
        payload: bytes
        promoted_from: PRAStorageTier | None = None
        if entry.current_tier == PRAStorageTier.HOT:
            self.metrics.hits["hot"] += 1
            value = self.hot.get_hot(key)
            if request_id is not None:
                self.hot.pin_hot(key, request_id)
                self.entries[key] = replace(
                    entry,
                    request_pin_count=entry.request_pin_count + 1,
                    last_access_ns=now_ns,
                )
            return value
        elif self.warm is not None and self.warm.contains(key):
            self.metrics.hits["warm"] += 1
            payload = self.warm.get(key, self._metadata(key))
            self.metrics.bytes_read += entry.warm_bytes
            self.metrics.promotions += 1
            self.metrics.reloads += 1
            promoted_from = PRAStorageTier.WARM
        elif self.cold is not None and self.cold.contains(key):
            self.metrics.hits["cold"] += 1
            payload = self.cold.get(key, self._metadata(key))
            if self.cold_codec is not None:
                decode_started = time.monotonic_ns()
                payload = self.cold_codec.decode(
                    payload, self._cold_metadata.get(key, {})
                )
                self.metrics.dequantization_latency_ns += (
                    time.monotonic_ns() - decode_started
                )
            self.metrics.bytes_read += entry.cold_bytes
            self.metrics.promotions += 1
            self.metrics.reloads += 1
            promoted_from = PRAStorageTier.COLD
        else:
            self.metrics.hits["source"] += 1
            payload = self._load_source(key)
            self.metrics.reloads += 1
        hot_bytes = self.hot.load_hot(key, payload)
        # HOT is an attention-ready copy above the durable tier, not a move
        # out of it. Keeping WARM/COLD intact makes promotion crash-recoverable
        # and avoids rewriting the same immutable payload after every request.
        warm_bytes = entry.warm_bytes
        cold_bytes = entry.cold_bytes
        if request_id is not None:
            self.hot.pin_hot(key, request_id)
        self.metrics.promotion_latency_ns += time.monotonic_ns() - started
        self.entries[key] = replace(entry, current_tier=PRAStorageTier.HOT, hot_bytes=hot_bytes, warm_bytes=warm_bytes, cold_bytes=cold_bytes, request_pin_count=entry.request_pin_count + int(request_id is not None), last_access_ns=now_ns)
        self._enforce_hot_quota(key, now_ns)
        self._persist_state()
        return self.hot.get_hot(key)

    @_storage_locked
    def unpin(self, key: str, request_id: str) -> None:
        entry = self.entries[key]
        self.hot.unpin_hot(key, request_id)
        self.entries[key] = replace(entry, request_pin_count=max(0, entry.request_pin_count - 1))
        self._persist_state()

    @contextmanager
    def pin_request(
        self,
        request_id: str,
        keys: Iterable[str],
        *,
        tenant_id: str | None = None,
        authorization_scopes: Iterable[str] = (),
    ):
        """Pin one authorized object set across complete request lifetime.

        ``tenant_id=None`` is reserved for trusted internal maintenance callers.
        Serving adapters must supply tenant and scopes so promotion followed by
        pinning cannot become a confused-deputy path across sessions.
        """

        unique = tuple(dict.fromkeys(map(str, keys)))
        scopes = set(authorization_scopes)
        with self._lock:
            for key in unique:
                entry = self.entries[key]
                if tenant_id is not None and entry.tenant_id != tenant_id:
                    raise PermissionError(
                        "Cross-tenant PRA request pinning is forbidden."
                    )
                if entry.security_scope and entry.security_scope not in scopes:
                    raise PermissionError(
                        "The request is not authorized to pin this PRA storage object."
                    )
                if entry.current_tier != PRAStorageTier.HOT:
                    self.promote(
                        key,
                        tenant_id=tenant_id,
                        authorization_scopes=scopes,
                    )
                    entry = self.entries[key]
                self.hot.pin_hot(key, request_id)
                self.entries[key] = replace(
                    entry, request_pin_count=entry.request_pin_count + 1
                )
            self._persist_state()
        try:
            yield
        finally:
            for key in unique:
                if key in self.entries and self.entries[key].request_pin_count:
                    self.unpin(key, request_id)

    @_storage_locked
    def update_task_status(self, task_id: str, status: TaskStatus | str, *, now_ns: int | None = None) -> None:
        now_ns = time.time_ns() if now_ns is None else now_ns
        status_value = status.value if isinstance(status, TaskStatus) else str(status).lower()
        task_policy = self.policy.task_states.get(status_value, PRATaskRetentionPolicy())
        closed = status_value in {"completed", "failed", "cancelled"}
        for key, entry in tuple(self.entries.items()):
            if entry.task_id == task_id:
                due = now_ns + int(task_policy.compaction_delay_seconds * 1e9) if closed else None
                self.entries[key] = replace(entry, task_status=status_value, compaction_due_ns=due)
        self._persist_state()

    @_storage_locked
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
        self._persist_state()
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

    @_storage_locked
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
                    cold_payload = payload
                    cold_metadata: Mapping[str, object] = {
                        "quantization": "none"
                    }
                    if self.cold_codec is not None:
                        cold_payload, cold_metadata = self.cold_codec.encode(
                            payload, self.policy.cold.kv_quantization
                        )
                    started = time.monotonic_ns()
                    stored = self.cold.put(
                        key,
                        cold_payload,
                        {**self._metadata(key), "cold_codec": dict(cold_metadata)},
                    )
                    self._cold_metadata[key] = dict(cold_metadata)
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
        self._enforce_tenant_quotas(PRAStorageTier.WARM, self.warm, now_ns)
        self._enforce_tenant_quotas(PRAStorageTier.COLD, self.cold, now_ns)
        after = self.usage()["total_native_bytes"]
        if after < before and any(entry.compaction_due_ns is not None and now_ns >= entry.compaction_due_ns for entry in self.entries.values()):
            self.metrics.task_close_bytes_freed += before - after
        self._persist_state()

    def _enforce_quota(self, tier: PRAStorageTier, backend: PRAStorageBackend | None, limit: int | None, now_ns: int) -> None:
        if backend is None or limit is None:
            return
        while backend.bytes_used() > limit:
            candidates = [
                entry
                for entry in self.entries.values()
                if (
                    entry.warm_bytes > 0
                    if tier == PRAStorageTier.WARM
                    else entry.cold_bytes > 0
                )
            ]
            if not candidates:
                raise MemoryError(f"PRA entries exceed the {tier.value} quota.")
            victim = min(candidates, key=lambda entry: (self.retention_score(entry, now_ns=now_ns), entry.logical_key))
            backend.remove(victim.logical_key)
            warm_bytes = 0 if tier == PRAStorageTier.WARM else victim.warm_bytes
            cold_bytes = 0 if tier == PRAStorageTier.COLD else victim.cold_bytes
            current = (
                PRAStorageTier.HOT
                if victim.hot_bytes
                else PRAStorageTier.WARM
                if warm_bytes
                else PRAStorageTier.COLD
                if cold_bytes
                else PRAStorageTier.SOURCE
            )
            self.entries[victim.logical_key] = replace(
                victim,
                current_tier=current,
                warm_bytes=warm_bytes,
                cold_bytes=cold_bytes,
            )
            self.metrics.evictions += 1

    def _enforce_tenant_quotas(
        self,
        tier: PRAStorageTier,
        backend: PRAStorageBackend | None,
        now_ns: int,
    ) -> None:
        if backend is None:
            return
        config = self.policy.warm if tier == PRAStorageTier.WARM else self.policy.cold
        limit = config.per_tenant_max_bytes
        if limit is None:
            return
        tenants = sorted({entry.tenant_id for entry in self.entries.values()})
        for tenant_id in tenants:
            def tenant_bytes() -> int:
                return sum(
                    entry.warm_bytes if tier == PRAStorageTier.WARM else entry.cold_bytes
                    for entry in self.entries.values()
                    if entry.tenant_id == tenant_id
                )

            while tenant_bytes() > limit:
                candidates = [
                    entry
                    for entry in self.entries.values()
                    if entry.tenant_id == tenant_id
                    and (
                        entry.warm_bytes > 0
                        if tier == PRAStorageTier.WARM
                        else entry.cold_bytes > 0
                    )
                ]
                if not candidates:
                    raise MemoryError(
                        f"PRA entries exceed tenant {tenant_id!r} {tier.value} quota."
                    )
                victim = min(
                    candidates,
                    key=lambda entry: (
                        self.retention_score(entry, now_ns=now_ns),
                        entry.logical_key,
                    ),
                )
                backend.remove(victim.logical_key)
                warm_bytes = 0 if tier == PRAStorageTier.WARM else victim.warm_bytes
                cold_bytes = 0 if tier == PRAStorageTier.COLD else victim.cold_bytes
                current = (
                    PRAStorageTier.HOT
                    if victim.hot_bytes
                    else PRAStorageTier.WARM
                    if warm_bytes
                    else PRAStorageTier.COLD
                    if cold_bytes
                    else PRAStorageTier.SOURCE
                )
                self.entries[victim.logical_key] = replace(
                    victim,
                    current_tier=current,
                    warm_bytes=warm_bytes,
                    cold_bytes=cold_bytes,
                )
                self.metrics.evictions += 1

    def usage(self) -> dict[str, int]:
        hot = sum(entry.hot_bytes for entry in self.entries.values())
        warm = self.warm.bytes_used() if self.warm is not None else 0
        cold = self.cold.bytes_used() if self.cold is not None else 0
        return {"hot_bytes": hot, "warm_bytes": warm, "cold_bytes": cold, "total_native_bytes": hot + warm + cold, "source_only_objects": sum(entry.current_tier == PRAStorageTier.SOURCE for entry in self.entries.values())}

    def inspect(self) -> dict[str, object]:
        by_tier = {tier.value: sum(entry.current_tier == tier for entry in self.entries.values()) for tier in PRAStorageTier}
        return {"profile": self.policy.profile, "policy": self.policy.to_dict(), "usage": self.usage(), "objects": by_tier, "metrics": self.metrics.to_dict()}
