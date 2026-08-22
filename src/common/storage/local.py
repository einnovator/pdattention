"""Atomic local/shared-filesystem artifact storage."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .base import StorageBackend


class LocalStorage(StorageBackend):
    """Filesystem backend suitable for local and mounted shared storage."""

    def __init__(self, name: str, root: str | Path):
        super().__init__(name)
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key.replace("\\", "/")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Storage key escapes root: {key!r}.")
        return candidate

    def put(self, local_path: str | Path, key: str) -> str:
        source = Path(local_path)
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            shutil.copy2(source, temporary_path)
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        return str(target)

    def get(self, key: str, local_path: str | Path) -> Path:
        source = self._path(key)
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str = ""):
        root = self._path(prefix)
        if not root.exists():
            return []
        if root.is_file():
            return [root.relative_to(self.root).as_posix()]
        return sorted(path.relative_to(self.root).as_posix() for path in root.rglob("*") if path.is_file())

    def uri(self, key: str = "") -> str:
        return str(self._path(key))
