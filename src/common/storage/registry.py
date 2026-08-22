"""Typed storage configuration and lazy backend construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import StorageBackend
from .local import LocalStorage


@dataclass(frozen=True)
class StorageConfig:
    """Named backend definition; credential references remain private metadata."""

    name: str
    type: str = "local"
    path: str = "out/experiments"
    bucket: str | None = None
    credentials_file: str | None = None
    profile: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    project: str | None = None

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any] | None) -> "StorageConfig":
        value = value or {}
        kind = str(value.get("type", "local")).lower()
        if kind not in {"local", "s3", "gcs"}:
            raise ValueError(f"Storage {name!r} has unsupported type {kind!r}.")
        if kind == "local" and not value.get("path"):
            raise ValueError(f"Local storage {name!r} requires path.")
        if kind in {"s3", "gcs"} and not value.get("bucket"):
            raise ValueError(f"{kind.upper()} storage {name!r} requires bucket.")
        return cls(
            name=name,
            type=kind,
            path=str(value.get("path", "out/experiments")),
            bucket=value.get("bucket"),
            credentials_file=value.get("credentials_file"),
            profile=value.get("profile"),
            region=value.get("region"),
            endpoint_url=value.get("endpoint_url"),
            project=value.get("project"),
        )

    def safe_manifest(self) -> dict:
        """Return storage provenance without credential paths or profiles."""

        return {
            "name": self.name,
            "type": self.type,
            "uri": self.safe_uri(),
        }

    def safe_uri(self) -> str:
        if self.type == "local":
            return str(Path(self.path).expanduser())
        scheme = "s3" if self.type == "s3" else "gs"
        suffix = self.path.strip("/")
        return f"{scheme}://{self.bucket}/{suffix}".rstrip("/")


class StorageRegistry:
    """Construct configured backends only when selected."""

    def __init__(self, configs: Mapping[str, StorageConfig]):
        self.configs = dict(configs)
        self._instances: dict[str, StorageBackend] = {}

    def get(self, name: str) -> StorageBackend:
        if name not in self.configs:
            raise ValueError(f"Unknown storage {name!r}.")
        if name in self._instances:
            return self._instances[name]
        config = self.configs[name]
        if config.type == "local":
            backend: StorageBackend = LocalStorage(name, config.path)
        elif config.type == "s3":
            from .s3 import S3Storage

            backend = S3Storage(
                name,
                bucket=str(config.bucket),
                prefix=config.path,
                profile=config.profile,
                region=config.region,
                endpoint_url=config.endpoint_url,
            )
        else:
            from .gcs import GCSStorage

            backend = GCSStorage(
                name,
                bucket=str(config.bucket),
                prefix=config.path,
                project=config.project,
                credentials_file=config.credentials_file,
            )
        self._instances[name] = backend
        return backend
