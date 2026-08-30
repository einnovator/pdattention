"""Hugging Face native-reference realization of shared PRA storage tiers."""

from __future__ import annotations

import io
from dataclasses import dataclass

from .model import ReferenceHandle


def serialize_reference(model: object, uri: str) -> bytes:
    """Serialize one complete PRA cache entry without retaining model globals."""

    import torch

    entry = model._handle.cache.get(uri)
    if entry is None:
        raise KeyError(uri)
    handle = model._references[uri]
    stream = io.BytesIO()
    torch.save({"entry": entry, "handle": handle}, stream)
    return stream.getvalue()


@dataclass
class HFReferenceHotBridge:
    """Publish persisted PRA entries into the model cache only while HOT."""

    model: object

    def __post_init__(self) -> None:
        self._handles: dict[str, ReferenceHandle] = {}
        self._sizes: dict[str, int] = {}
        self._pins: dict[str, set[str]] = {}

    def _publish(self, logical_key: str, payload: bytes) -> ReferenceHandle:
        import torch

        restored = torch.load(io.BytesIO(payload), map_location=self.model.device, weights_only=False)
        entry = restored["entry"]
        handle = restored["handle"]
        self.model._handle.cache.put(entry)
        self.model._references[handle.uri] = handle
        self._handles[logical_key] = handle
        self._sizes[logical_key] = len(payload)
        return handle

    def load_hot(self, logical_key: str, payload: bytes) -> int:
        if logical_key not in self._handles:
            self._publish(logical_key, payload)
        return self._sizes[logical_key]

    def load_hot_value(self, logical_key: str, value: object, byte_count: int) -> int:
        if not isinstance(value, ReferenceHandle):
            raise TypeError("HF HOT storage requires a ReferenceHandle.")
        self._handles.setdefault(logical_key, value)
        self._sizes.setdefault(logical_key, int(byte_count))
        return self._sizes[logical_key]

    def get_hot(self, logical_key: str) -> object:
        return self._handles[logical_key]

    def release_hot(self, logical_key: str) -> None:
        if self._pins.get(logical_key):
            raise RuntimeError("Cannot release request-pinned HF PRA memory.")
        handle = self._handles.pop(logical_key, None)
        self._sizes.pop(logical_key, None)
        if handle is not None:
            self.model.remove_reference(handle)

    def pin_hot(self, logical_key: str, request_id: str) -> None:
        if logical_key not in self._handles:
            raise KeyError(logical_key)
        self._pins.setdefault(logical_key, set()).add(request_id)

    def unpin_hot(self, logical_key: str, request_id: str) -> None:
        self._pins.get(logical_key, set()).discard(request_id)

    def hot_bytes(self, logical_key: str) -> int:
        return self._sizes.get(logical_key, 0)
