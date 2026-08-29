"""HiCache-style placement for SGLang's external PRA K/V namespace.

The tiers are deliberately separate from SGLang's Radix/prefix cache.  L1
contains active MLX arrays, L2 contains host-array representations, and L3 is
an on-disk NPZ backing store.  On Apple unified memory, L1/L2 describe runtime
ownership and readiness rather than physically distinct GPU and CPU DRAM.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable
from typing import TYPE_CHECKING

from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory

if TYPE_CHECKING:
    from pra_sglang.hicache_backend import PRAHiCacheL3Backend


class PRAHiCacheTier(str, Enum):
    """Storage/readiness tier for immutable selected native K/V."""

    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


@dataclass
class PRAHiCacheMetrics:
    """Placement, promotion, and capacity counters for one cache instance."""

    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    l2_to_l1_promotions: int = 0
    l3_to_l2_promotions: int = 0
    l1_to_l2_demotions: int = 0
    l2_to_l3_demotions: int = 0
    promotion_ms: float = 0.0
    l1_bytes: int = 0
    l2_bytes: int = 0
    l3_bytes: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class _HostArray:
    """NumPy-compatible payload plus the MLX dtype it must reconstruct."""

    data: object
    logical_dtype: str

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def shape(self):
        return self.data.shape

    def __array__(self, dtype=None):
        import numpy as np

        return np.asarray(self.data, dtype=dtype)


def _host_array(array: object) -> _HostArray:
    import numpy as np

    if isinstance(array, _HostArray):
        return array
    logical_dtype = str(array.dtype)
    if "bfloat16" in logical_dtype:
        import mlx.core as mx

        data = np.asarray(array.view(mx.uint16)).copy()
    else:
        data = np.asarray(array).copy()
    return _HostArray(data, logical_dtype)


def _default_to_host(memory: MLXNativeMemory) -> MLXNativeMemory:
    return MLXNativeMemory(
        tuple(
            MLXNativeLayerKV(_host_array(layer.keys), _host_array(layer.values))
            for layer in memory.layers
        ),
        source_tokens=memory.source_tokens,
    )


def _default_to_device(memory: MLXNativeMemory) -> MLXNativeMemory:
    import mlx.core as mx

    def restore(array: object):
        if not isinstance(array, _HostArray):
            return mx.array(array)
        value = mx.array(array.data)
        if "bfloat16" in array.logical_dtype:
            return value.view(mx.bfloat16)
        dtype = getattr(mx, array.logical_dtype, None)
        return value if dtype is None else value.astype(dtype)

    result = MLXNativeMemory(
        tuple(
            MLXNativeLayerKV(restore(layer.keys), restore(layer.values))
            for layer in memory.layers
        ),
        source_tokens=memory.source_tokens,
    )
    mx.eval(
        *(array for layer in result.layers for array in (layer.keys, layer.values))
    )
    return result


class SGLangPRAHiCache:
    """Promote immutable PRA K/V through L3, L2, and attention-ready L1.

    Capacity eviction is LRU within each memory tier. L1 eviction demotes to
    L2, and L2 eviction persists to L3, preserving the external-memory object
    while removing its more expensive representation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_l1_bytes: int,
        max_l2_bytes: int,
        to_host: Callable[[MLXNativeMemory], MLXNativeMemory] = _default_to_host,
        to_device: Callable[[MLXNativeMemory], MLXNativeMemory] = _default_to_device,
        l3_backend: PRAHiCacheL3Backend | None = None,
    ) -> None:
        if max_l1_bytes <= 0 or max_l2_bytes <= 0:
            raise ValueError("PRA HiCache L1 and L2 byte budgets must be positive.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_l1_bytes = int(max_l1_bytes)
        self.max_l2_bytes = int(max_l2_bytes)
        self._to_host = to_host
        self._to_device = to_device
        self._l3_backend = l3_backend
        self._l1: OrderedDict[str, MLXNativeMemory] = OrderedDict()
        self._l2: OrderedDict[str, MLXNativeMemory] = OrderedDict()
        self._l3: dict[str, Path | None] = {}
        self._metrics = PRAHiCacheMetrics()

    def _stem(self, logical_key: str) -> str:
        return hashlib.sha256(logical_key.encode("utf-8")).hexdigest()

    def _paths(self, logical_key: str) -> tuple[Path, Path]:
        stem = self._stem(logical_key)
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    def _refresh_bytes(self) -> None:
        self._metrics.l1_bytes = sum(memory.nbytes for memory in self._l1.values())
        self._metrics.l2_bytes = sum(memory.nbytes for memory in self._l2.values())
        if self._l3_backend is None:
            self._metrics.l3_bytes = sum(
                path.stat().st_size
                for path in self._l3.values()
                if path is not None and path.exists()
            )
        else:
            self._metrics.l3_bytes = sum(
                self._l3_backend.size(key) for key in self._l3
            )

    def _write_l3(self, logical_key: str, memory: MLXNativeMemory) -> None:
        import numpy as np

        if self._l3_backend is not None:
            self._l3_backend.put(logical_key, memory)
            self._l3[logical_key] = None
            return

        host = self._to_host(memory)
        arrays_path, manifest_path = self._paths(logical_key)
        arrays = {}
        dtypes = {}
        for index, layer in enumerate(host.layers):
            for suffix, array in (("k", layer.keys), ("v", layer.values)):
                name = f"layer_{index:04d}_{suffix}"
                arrays[name] = array.data if isinstance(array, _HostArray) else array
                dtypes[name] = (
                    array.logical_dtype
                    if isinstance(array, _HostArray)
                    else str(array.dtype)
                )
        np.savez_compressed(arrays_path, **arrays)
        manifest_path.write_text(
            json.dumps(
                {
                    "logical_key": logical_key,
                    "source_tokens": host.source_tokens,
                    "layer_count": len(host.layers),
                    "logical_dtypes": dtypes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._l3[logical_key] = arrays_path

    def _read_l3(self, logical_key: str) -> MLXNativeMemory:
        import numpy as np

        if self._l3_backend is not None:
            memory = self._l3_backend.get(logical_key)
            self._l3[logical_key] = None
            return memory

        arrays_path = self._l3.get(logical_key)
        if arrays_path is None:
            arrays_path, manifest_path = self._paths(logical_key)
            if not arrays_path.exists() or not manifest_path.exists():
                raise KeyError(logical_key)
            self._l3[logical_key] = arrays_path
        else:
            manifest_path = arrays_path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("logical_key") != logical_key:
            raise RuntimeError("PRA HiCache manifest identity does not match its key.")
        with np.load(arrays_path) as arrays:
            layers = tuple(
                MLXNativeLayerKV(
                    _HostArray(
                        arrays[f"layer_{index:04d}_k"].copy(),
                        manifest.get("logical_dtypes", {}).get(
                            f"layer_{index:04d}_k",
                            str(arrays[f"layer_{index:04d}_k"].dtype),
                        ),
                    ),
                    _HostArray(
                        arrays[f"layer_{index:04d}_v"].copy(),
                        manifest.get("logical_dtypes", {}).get(
                            f"layer_{index:04d}_v",
                            str(arrays[f"layer_{index:04d}_v"].dtype),
                        ),
                    ),
                )
                for index in range(int(manifest["layer_count"]))
            )
        return MLXNativeMemory(layers, int(manifest["source_tokens"]))

    def _fit_l2(self, required_bytes: int, *, protected_key: str) -> None:
        if required_bytes > self.max_l2_bytes:
            raise MemoryError("One PRA K/V object exceeds the configured L2 budget.")
        while self._metrics.l2_bytes + required_bytes > self.max_l2_bytes:
            key, memory = self._l2.popitem(last=False)
            if key == protected_key:
                self._l2[key] = memory
                raise MemoryError("PRA HiCache cannot free sufficient L2 capacity.")
            self._write_l3(key, memory)
            self._metrics.l2_to_l3_demotions += 1
            self._refresh_bytes()

    def _place_l2(self, logical_key: str, memory: MLXNativeMemory) -> None:
        existing = self._l2.pop(logical_key, None)
        self._refresh_bytes()
        required = 0 if existing is memory else memory.nbytes
        self._fit_l2(required, protected_key=logical_key)
        self._l2[logical_key] = memory
        self._refresh_bytes()

    def _fit_l1(self, required_bytes: int, *, protected_key: str) -> None:
        if required_bytes > self.max_l1_bytes:
            raise MemoryError("One PRA K/V object exceeds the configured L1 budget.")
        while self._metrics.l1_bytes + required_bytes > self.max_l1_bytes:
            key, memory = self._l1.popitem(last=False)
            if key == protected_key:
                self._l1[key] = memory
                raise MemoryError("PRA HiCache cannot free sufficient L1 capacity.")
            self._place_l2(key, self._to_host(memory))
            self._metrics.l1_to_l2_demotions += 1
            self._refresh_bytes()

    def put(
        self,
        logical_key: str,
        memory: MLXNativeMemory,
        *,
        tier: PRAHiCacheTier | str = PRAHiCacheTier.L1,
    ) -> None:
        """Insert or replace one immutable memory at the requested tier."""

        key = str(logical_key)
        target = PRAHiCacheTier(tier)
        self.remove(key)
        if target is PRAHiCacheTier.L1:
            self._fit_l1(memory.nbytes, protected_key=key)
            self._l1[key] = memory
        elif target is PRAHiCacheTier.L2:
            host = self._to_host(memory)
            self._place_l2(key, host)
        else:
            self._write_l3(key, memory)
        self._refresh_bytes()

    def get(
        self,
        logical_key: str,
        *,
        target: PRAHiCacheTier | str = PRAHiCacheTier.L1,
    ) -> MLXNativeMemory:
        """Resolve a memory and promote it to the requested readiness tier."""

        key = str(logical_key)
        destination = PRAHiCacheTier(target)
        started = time.perf_counter()
        if key in self._l1:
            self._metrics.l1_hits += 1
            memory = self._l1.pop(key)
            self._l1[key] = memory
            return memory
        if key in self._l2:
            self._metrics.l2_hits += 1
            host = self._l2.pop(key)
            self._l2[key] = host
        else:
            try:
                host = self._read_l3(key)
            except KeyError:
                self._metrics.misses += 1
                raise
            self._metrics.l3_hits += 1
            if destination is not PRAHiCacheTier.L3:
                self._place_l2(key, host)
                self._metrics.l3_to_l2_promotions += 1
        if destination is PRAHiCacheTier.L3:
            return host
        if destination is PRAHiCacheTier.L2:
            return host
        memory = self._to_device(host)
        self._fit_l1(memory.nbytes, protected_key=key)
        self._l1[key] = memory
        self._metrics.l2_to_l1_promotions += 1
        self._metrics.promotion_ms += (time.perf_counter() - started) * 1000.0
        self._refresh_bytes()
        return memory

    def placement(self, logical_key: str) -> PRAHiCacheTier | None:
        """Return the highest ready tier currently containing the key."""

        key = str(logical_key)
        if key in self._l1:
            return PRAHiCacheTier.L1
        if key in self._l2:
            return PRAHiCacheTier.L2
        arrays_path, manifest_path = self._paths(key)
        backend_exists = (
            self._l3_backend is not None and self._l3_backend.exists(key)
        )
        if key in self._l3 or backend_exists or (
            arrays_path.exists() and manifest_path.exists()
        ):
            return PRAHiCacheTier.L3
        return None

    def remove(self, logical_key: str) -> None:
        """Remove all tier representations for one logical PRA identity."""

        key = str(logical_key)
        self._l1.pop(key, None)
        self._l2.pop(key, None)
        arrays_path, manifest_path = self._paths(key)
        arrays_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        if self._l3_backend is not None:
            self._l3_backend.remove(key)
        self._l3.pop(key, None)
        self._refresh_bytes()

    def metrics(self) -> PRAHiCacheMetrics:
        """Return a detached metrics snapshot."""

        self._refresh_bytes()
        return PRAHiCacheMetrics(**self._metrics.to_dict())
