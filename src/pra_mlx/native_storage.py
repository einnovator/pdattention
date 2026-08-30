"""Layer-segmented mmap storage for MLX-format native PRA memories."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping, Sequence

from pra_hf.storage_lifecycle import MemoryMappedKVStore, PRAStorageBackend


class MLXNativeSegmentStore(PRAStorageBackend):
    """Persist every native K/V array as an independently checked segment.

    The generic lifecycle manager continues to exchange complete serialized
    memories. Engine code that needs only selected consumer layers can call
    :meth:`get_layer_arrays` without reading neighboring layers.
    """

    schema = "pra-mlx-segment-store-v1"

    def __init__(self, path: str | Path) -> None:
        self.store = MemoryMappedKVStore(path)

    @staticmethod
    def _array_bytes(value: object) -> bytes:
        import numpy as np

        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return stream.getvalue()

    def contains(self, key: str) -> bool:
        return self.store.contains(key)

    def put(self, key: str, payload: bytes, metadata: Mapping[str, object]) -> int:
        import numpy as np

        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            names = tuple(sorted(archive.files))
            segments = {
                name: self._array_bytes(archive[name])
                for name in names
            }
        return self.store.put_segments(
            key,
            segments,
            {
                **dict(metadata),
                "native_segment_schema": self.schema,
                "native_segment_names": names,
            },
        )

    def _segment_names(self, key: str) -> tuple[str, ...]:
        metadata = self.store.metadata(key)
        if metadata.get("native_segment_schema") != self.schema:
            raise ValueError("Unsupported MLX native segment-store schema.")
        return tuple(map(str, metadata["native_segment_names"]))

    def get(self, key: str, metadata: Mapping[str, object]) -> bytes:
        import numpy as np

        names = self._segment_names(key)
        segments = self.store.get_segments(key, names, metadata)
        arrays = {
            name: np.load(io.BytesIO(segments[name]), allow_pickle=False)
            for name in names
        }
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        return stream.getvalue()

    def get_layer_arrays(
        self,
        key: str,
        layers: Sequence[int],
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        """Load only K/V arrays for the requested consumer-layer indices."""

        import numpy as np

        names = tuple(
            name
            for layer in dict.fromkeys(map(int, layers))
            for name in (f"layer_{layer:04d}_k", f"layer_{layer:04d}_v")
        )
        available = set(self._segment_names(key))
        missing = [name for name in names if name not in available]
        if missing:
            raise KeyError(f"Unknown native K/V layer segments: {missing}")
        segments = self.store.get_segments(key, names, metadata)
        return {
            name: np.load(io.BytesIO(payload), allow_pickle=False)
            for name, payload in segments.items()
        }

    def remove(self, key: str) -> None:
        self.store.remove(key)

    def bytes_used(self) -> int:
        return self.store.bytes_used()

    def keys(self) -> tuple[str, ...]:
        return self.store.keys()

    def metadata(self, key: str) -> Mapping[str, object]:
        return self.store.metadata(key)

    def get_range(
        self, key: str, start: int, end: int, metadata: Mapping[str, object]
    ) -> bytes:
        return self.store.get_range(key, start, end, metadata)
