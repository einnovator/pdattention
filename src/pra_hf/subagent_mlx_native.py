"""Live MLX native-K/V port for Paper 9 subagent reuse experiments.

This is a deliberately narrow engine adapter. It encodes an immutable tool
result once, retains the host model's post-RoPE K/V arrays, and gives each
child a fresh local cache backed by those shared arrays. The current public
MLX seam transiently concatenates shared and local K/V for host attention; it
avoids persistent per-child copies but is not segmented zero-copy attention.
Persistence and tiering remain the Paper 4.5 runtime's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic_ns
from typing import Callable
from uuid import uuid4

from .context_records import ContextRecord
from .subagent_context import NativeKVHandle


@dataclass(frozen=True)
class MLXLayerKV:
    """One immutable layer in host ``[batch, kv_head, token, dim]`` layout."""

    keys: object
    values: object
    key_dtype: str | None = None
    value_dtype: str | None = None

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass(frozen=True)
class MLXSubagentMemory:
    """Layer-aligned native state for one reusable typed record."""

    layers: tuple[MLXLayerKV, ...]
    token_count: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


@dataclass(frozen=True)
class MLXHostSubagentMemory:
    """Host-tier copy of one reusable record's layer K/V arrays."""

    layers: tuple[MLXLayerKV, ...]
    token_count: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


@dataclass
class _MLXPoolEntry:
    session_uuid: str
    record_uuid: str
    device: MLXSubagentMemory | None
    host: MLXHostSubagentMemory | None
    active_leases: int = 0
    last_access: int = 0


@dataclass(frozen=True)
class MLXStateLease:
    lease_uuid: str
    session_uuid: str
    target_agent_uuid: str
    record_uuid: str


class MLXStatePoolError(RuntimeError):
    """Raised when a native-state lifecycle or isolation contract is violated."""


class MLXSubagentStatePool:
    """Session-isolated lifecycle manager for reusable direct-MLX state.

    Device capacity triggers inactive LRU demotion to a host representation;
    record-count pressure triggers inactive LRU eviction. Active leases are
    never demoted or evicted. Cancellation releases one target lease, while
    session termination invalidates that session's bindings and records only.
    """

    def __init__(
        self,
        *,
        device_capacity_bytes: int,
        max_records: int,
        export_to_host: Callable[[MLXSubagentMemory], MLXHostSubagentMemory] | None = None,
        import_to_device: Callable[[MLXHostSubagentMemory], MLXSubagentMemory] | None = None,
    ) -> None:
        if device_capacity_bytes <= 0 or max_records <= 0:
            raise ValueError("Device capacity and max_records must be positive.")
        self.device_capacity_bytes = int(device_capacity_bytes)
        self.max_records = int(max_records)
        self._export_to_host = export_to_host or _mlx_memory_to_host
        self._import_to_device = import_to_device or _mlx_memory_to_device
        self._entries: dict[tuple[str, str], _MLXPoolEntry] = {}
        self._leases: dict[str, MLXStateLease] = {}
        self._target_leases: dict[tuple[str, str], str] = {}
        self._lock = RLock()
        self.events: list[dict[str, object]] = []

    @property
    def device_bytes(self) -> int:
        with self._lock:
            return sum(entry.device.nbytes for entry in self._entries.values() if entry.device)

    def put(self, session_uuid: str, record_uuid: str, memory: MLXSubagentMemory) -> None:
        if not session_uuid or not record_uuid:
            raise ValueError("Session and record identities are required.")
        key = (session_uuid, record_uuid)
        with self._lock:
            if key in self._entries:
                raise MLXStatePoolError(f"Native record already exists: {key}")
            while len(self._entries) >= self.max_records:
                self._evict_lru_inactive()
            self._make_device_room(memory.nbytes)
            self._entries[key] = _MLXPoolEntry(
                session_uuid,
                record_uuid,
                memory,
                None,
                last_access=monotonic_ns(),
            )
            self._event("put", key, tier="device", bytes=memory.nbytes)

    def acquire(
        self, session_uuid: str, record_uuid: str, target_agent_uuid: str
    ) -> tuple[MLXStateLease, MLXSubagentMemory]:
        key = (session_uuid, record_uuid)
        target = (session_uuid, target_agent_uuid)
        with self._lock:
            if target in self._target_leases:
                raise MLXStatePoolError(f"Target already owns a native-state lease: {target}")
            entry = self._entries.get(key)
            if entry is None:
                raise MLXStatePoolError(f"No native record in requested session: {key}")
            if entry.device is None:
                assert entry.host is not None
                self._make_device_room(entry.host.nbytes, exclude=key)
                entry.device = self._import_to_device(entry.host)
                self._event("promote", key, tier="device", bytes=entry.device.nbytes)
            lease = MLXStateLease(uuid4().hex, session_uuid, target_agent_uuid, record_uuid)
            self._leases[lease.lease_uuid] = lease
            self._target_leases[target] = lease.lease_uuid
            entry.active_leases += 1
            entry.last_access = monotonic_ns()
            self._event("acquire", key, target_agent_uuid=target_agent_uuid)
            return lease, entry.device

    def release(self, lease_uuid: str, *, reason: str = "complete") -> bool:
        with self._lock:
            lease = self._leases.pop(lease_uuid, None)
            if lease is None:
                return False
            self._target_leases.pop((lease.session_uuid, lease.target_agent_uuid), None)
            entry = self._entries.get((lease.session_uuid, lease.record_uuid))
            if entry is not None:
                entry.active_leases -= 1
                entry.last_access = monotonic_ns()
            self._event(
                "release",
                (lease.session_uuid, lease.record_uuid),
                target_agent_uuid=lease.target_agent_uuid,
                reason=reason,
            )
            return True

    def cancel_target(self, session_uuid: str, target_agent_uuid: str) -> bool:
        with self._lock:
            lease_uuid = self._target_leases.get((session_uuid, target_agent_uuid))
            return bool(lease_uuid and self.release(lease_uuid, reason="cancel"))

    def terminate_session(self, session_uuid: str) -> int:
        with self._lock:
            for target, lease_uuid in list(self._target_leases.items()):
                if target[0] == session_uuid:
                    self.release(lease_uuid, reason="session_terminate")
            keys = [key for key in self._entries if key[0] == session_uuid]
            for key in keys:
                self._entries.pop(key)
                self._event("evict", key, reason="session_terminate")
            return len(keys)

    def demote(self, session_uuid: str, record_uuid: str) -> None:
        key = (session_uuid, record_uuid)
        with self._lock:
            entry = self._inactive_entry(key)
            if entry.device is None:
                return
            entry.host = self._export_to_host(entry.device)
            entry.device = None
            entry.last_access = monotonic_ns()
            self._event("demote", key, tier="host", bytes=entry.host.nbytes)

    def evict(self, session_uuid: str, record_uuid: str) -> None:
        key = (session_uuid, record_uuid)
        with self._lock:
            self._inactive_entry(key)
            self._entries.pop(key)
            self._event("evict", key, reason="explicit")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "device_capacity_bytes": self.device_capacity_bytes,
                "device_bytes": self.device_bytes,
                "records": [
                    {
                        "session_uuid": entry.session_uuid,
                        "record_uuid": entry.record_uuid,
                        "tier": "device" if entry.device is not None else "host",
                        "active_leases": entry.active_leases,
                    }
                    for entry in sorted(
                        self._entries.values(), key=lambda row: (row.session_uuid, row.record_uuid)
                    )
                ],
                "active_leases": len(self._leases),
                "events": list(self.events),
            }

    def _inactive_entry(self, key: tuple[str, str]) -> _MLXPoolEntry:
        entry = self._entries.get(key)
        if entry is None:
            raise MLXStatePoolError(f"Unknown native record: {key}")
        if entry.active_leases:
            raise MLXStatePoolError(f"Cannot move or evict active native record: {key}")
        return entry

    def _make_device_room(
        self, required_bytes: int, *, exclude: tuple[str, str] | None = None
    ) -> None:
        if required_bytes > self.device_capacity_bytes:
            raise MLXStatePoolError("Native record exceeds device-tier capacity.")
        while self.device_bytes + required_bytes > self.device_capacity_bytes:
            candidates = [
                (key, entry)
                for key, entry in self._entries.items()
                if key != exclude and entry.device is not None and not entry.active_leases
            ]
            if not candidates:
                raise MLXStatePoolError("Device capacity is exhausted by active native records.")
            key, _ = min(candidates, key=lambda row: row[1].last_access)
            self.demote(*key)

    def _evict_lru_inactive(self) -> None:
        candidates = [(key, entry) for key, entry in self._entries.items() if not entry.active_leases]
        if not candidates:
            raise MLXStatePoolError("Record capacity is exhausted by active native records.")
        key, _ = min(candidates, key=lambda row: row[1].last_access)
        self._entries.pop(key)
        self._event("evict", key, reason="record_capacity")

    def _event(self, action: str, key: tuple[str, str], **details: object) -> None:
        self.events.append(
            {
                "action": action,
                "session_uuid": key[0],
                "record_uuid": key[1],
                **details,
            }
        )


def _mlx_memory_to_host(memory: MLXSubagentMemory) -> MLXHostSubagentMemory:
    import mlx.core as mx
    import numpy as np

    states = [(layer.keys, layer.values) for layer in memory.layers]
    mx.eval(states)

    def copy(array: object) -> tuple[object, str]:
        dtype = str(array.dtype).rsplit(".", 1)[-1]
        # NumPy cannot consume MLX's bfloat16 PEP 3118 buffer directly. Keep
        # the exact 16-bit payload on the host and restore its dtype on import.
        if dtype == "bfloat16":
            return np.array(mx.view(array, mx.uint16)), dtype
        return np.array(array), dtype

    layers = []
    for keys, values in states:
        host_keys, key_dtype = copy(keys)
        host_values, value_dtype = copy(values)
        layers.append(
            MLXLayerKV(host_keys, host_values, key_dtype, value_dtype)
        )
    return MLXHostSubagentMemory(
        tuple(layers),
        memory.token_count,
    )


def _mlx_memory_to_device(memory: MLXHostSubagentMemory) -> MLXSubagentMemory:
    import mlx.core as mx

    def restore(array: object, dtype: str | None) -> object:
        value = mx.array(array)
        if dtype is None:
            return value
        target = getattr(mx, dtype)
        return mx.view(value, target) if dtype == "bfloat16" else value.astype(target)

    layers = tuple(
        MLXLayerKV(
            restore(layer.keys, layer.key_dtype),
            restore(layer.values, layer.value_dtype),
        )
        for layer in memory.layers
    )
    mx.eval([(layer.keys, layer.values) for layer in layers])
    return MLXSubagentMemory(layers, memory.token_count)


class MLXSubagentKVCache:
    """Expose immutable shared K/V before a request-local MLX prompt cache.

    ``update_and_fetch`` uses the public host-cache contract, which requires a
    transient concatenated attention view. The immutable stored arrays remain
    shared, but this transient cost matters under memory pressure and is
    reported explicitly by the Paper 9 live benchmark.
    """

    def __init__(self, local_cache: object, memory: MLXLayerKV, position_base: int):
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        return (self.memory.keys, self.memory.values, *local)

    @state.setter
    def state(self, value) -> None:
        raise RuntimeError("Shared subagent memory is immutable.")

    @property
    def nbytes(self) -> int:
        return self.memory.nbytes + int(getattr(self.local_cache, "nbytes", 0))

    def empty(self) -> bool:
        return False

    def is_trimmable(self) -> bool:
        operation = getattr(self.local_cache, "is_trimmable", None)
        return bool(operation and operation())

    def trim(self, n: int) -> int:
        return int(self.local_cache.trim(n))

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        local_keys, local_values = self.local_cache.update_and_fetch(keys, values)
        return (
            mx.concatenate((self.memory.keys, local_keys), axis=2),
            mx.concatenate((self.memory.values, local_values), axis=2),
        )

    def make_mask(self, n: int, return_array: bool = False, window_size=None, **kwargs):
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        local_offset = int(self.local_cache.offset)
        if window_size is None:
            local = create_causal_mask(n, local_offset)
        else:
            local = create_causal_mask(
                n, local_offset, window_size=min(int(window_size), local_offset + n)
            )
        memory = mx.ones((n, int(self.memory.keys.shape[2])), dtype=mx.bool_)
        return mx.concatenate((memory, local), axis=1)


class MLXSubagentNativePort:
    """Concrete ``NativeKVReusePort`` backed by public ``mlx-lm`` caches."""

    encoding_revision = "paper9-mlx-post-rope-v1"
    position_contract = "contiguous_source_local"

    def __init__(self, model: object, tokenizer: object, *, model_id: str, revision: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.memories: dict[str, MLXSubagentMemory] = {}
        self.target_records: dict[str, str] = {}

    def encode(self, record: ContextRecord) -> NativeKVHandle:
        """Encode a tool result with the unchanged host model exactly once."""

        payload = record.payload
        if isinstance(payload, dict):
            value = payload.get("output", payload)
        else:
            value = payload
        token_ids = tuple(self.tokenizer.encode(str(value), add_special_tokens=False))
        if not token_ids:
            raise ValueError("Cannot encode an empty reusable record.")
        memory = encode_mlx_subagent_memory(self.model, token_ids)
        self.memories[record.record_id] = memory
        return NativeKVHandle(
            record_uuid=record.record_id,
            model_id=self.model_id,
            model_revision=self.revision,
            encoding_revision=self.encoding_revision,
            position_contract=self.position_contract,
            token_count=memory.token_count,
            byte_count=memory.nbytes,
        )

    def materialize(self, handle: NativeKVHandle, *, target_agent_uuid: str) -> bool:
        """Bind a compatible immutable record to one child request."""

        if handle.record_uuid not in self.memories:
            return False
        self.target_records[target_agent_uuid] = handle.record_uuid
        return True

    def requested_handle(self) -> NativeKVHandle:
        """Return the compatibility template supplied during runtime lookup."""

        return NativeKVHandle(
            record_uuid="requested",
            model_id=self.model_id,
            model_revision=self.revision,
            encoding_revision=self.encoding_revision,
            position_contract=self.position_contract,
            token_count=0,
            byte_count=0,
        )

    def prompt_cache(self, target_agent_uuid: str):
        """Create fresh local caches around the target's selected shared state."""

        try:
            memory = self.memories[self.target_records[target_agent_uuid]]
        except KeyError as error:
            raise KeyError(f"No native memory is attached to {target_agent_uuid!r}.") from error
        return make_mlx_subagent_cache(self.model, memory)


def encode_mlx_subagent_memory(model: object, token_ids: tuple[int, ...]) -> MLXSubagentMemory:
    """Capture post-RoPE K/V produced by a normal contiguous host prefill."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model)
    model(mx.array(token_ids, dtype=mx.int32)[None], cache=caches)
    states = [cache.state for cache in caches]
    mx.eval(states)
    layers = []
    for state in states:
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError("The MLX model exposed a non-attention cache layer.")
        layers.append(MLXLayerKV(state[0], state[1]))
    return MLXSubagentMemory(tuple(layers), len(token_ids))


def make_mlx_subagent_cache(model: object, memory: MLXSubagentMemory):
    """Wrap one immutable memory with fresh request-local cache objects."""

    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model)
    if len(local) != len(memory.layers):
        raise ValueError("Native memory does not match the model layer count.")
    return [
        MLXSubagentKVCache(cache, layer, memory.token_count)
        for cache, layer in zip(local, memory.layers)
    ]
