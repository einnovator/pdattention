"""Storage-neutral artifact persistence contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class StorageBackend(ABC):
    """Named persistent location used independently from worker transport."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def put(self, local_path: str | Path, key: str) -> str:
        """Publish one local file and return its backend URI/path."""

    @abstractmethod
    def get(self, key: str, local_path: str | Path) -> Path:
        """Retrieve one key to a local path."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether a key exists."""

    @abstractmethod
    def list(self, prefix: str = "") -> Iterable[str]:
        """List keys below a logical prefix."""

    @abstractmethod
    def uri(self, key: str = "") -> str:
        """Return a safe, non-credential URI for provenance."""
