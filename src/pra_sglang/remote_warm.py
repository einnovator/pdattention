"""Off-node byte storage for SGLang PRA HiCache objects.

The transport implements the small ``set/get/exists`` surface used by
SGLang's storage adapters. Remote object identity remains separate from
Radix/prefix cache keys, and physical network traffic is counted explicitly.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class RemoteWarmClientMetrics:
    """Cold/warm transport counts observed by one model-host client."""

    reads: int = 0
    writes: int = 0
    exists_checks: int = 0
    deletes: int = 0
    read_bytes: int = 0
    written_bytes: int = 0
    read_ns: int = 0
    write_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class HTTPHiCacheStorageClient:
    """Adapt an HTTP object store to SGLang's tensor-oriented HiCache API."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._metrics = RemoteWarmClientMetrics()
        self._lock = threading.Lock()

    def _object_url(self, key: str) -> str:
        encoded = urllib.parse.quote(str(key), safe="")
        return f"{self.base_url}/v1/objects/{encoded}"

    def _record(self, **increments: int) -> None:
        with self._lock:
            for name, value in increments.items():
                setattr(self._metrics, name, getattr(self._metrics, name) + value)

    def health(self) -> dict[str, object]:
        with urllib.request.urlopen(
            f"{self.base_url}/health", timeout=self.timeout_seconds
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def set(self, key: str, value: object) -> bool:
        payload = bytes(value.detach().cpu().numpy())
        request = urllib.request.Request(
            self._object_url(key),
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            method="PUT",
        )
        started = time.monotonic_ns()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            accepted = 200 <= response.status < 300
            response.read()
        self._record(
            writes=1,
            written_bytes=len(payload),
            write_ns=time.monotonic_ns() - started,
        )
        return accepted

    def get(self, key: str, target: object):
        started = time.monotonic_ns()
        try:
            with urllib.request.urlopen(
                self._object_url(key), timeout=self.timeout_seconds
            ) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        if len(payload) != int(target.numel()):
            raise IOError(
                f"Remote WARM object size mismatch for {key}: "
                f"{len(payload)} != {int(target.numel())}"
            )
        import torch

        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        target.copy_(source)
        self._record(
            reads=1,
            read_bytes=len(payload),
            read_ns=time.monotonic_ns() - started,
        )
        return target

    def exists(self, key: str) -> bool:
        request = urllib.request.Request(self._object_url(key), method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                present = response.status == 200
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            present = False
        self._record(exists_checks=1)
        return present

    def remove(self, key: str) -> None:
        request = urllib.request.Request(self._object_url(key), method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        self._record(deletes=1)

    def metrics(self) -> RemoteWarmClientMetrics:
        with self._lock:
            return RemoteWarmClientMetrics(**self._metrics.to_dict())

    def reset_metrics(self) -> None:
        with self._lock:
            self._metrics = RemoteWarmClientMetrics()
