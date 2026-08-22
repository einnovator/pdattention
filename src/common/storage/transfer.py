"""Small storage-neutral tree transfer helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import StorageBackend


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def put_tree(storage: StorageBackend, root: str | Path, prefix: str = "") -> dict[str, str]:
    root = Path(root)
    uploaded = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        key = "/".join(part for part in (prefix.strip("/"), path.relative_to(root).as_posix()) if part)
        uploaded[key] = storage.put(path, key)
    return uploaded


def get_tree(storage: StorageBackend, prefix: str, root: str | Path) -> list[Path]:
    root = Path(root)
    downloaded = []
    normalized = prefix.strip("/")
    for key in storage.list(normalized):
        relative = key[len(normalized) :].lstrip("/") if normalized else key
        downloaded.append(storage.get(key, root / relative))
    return downloaded
