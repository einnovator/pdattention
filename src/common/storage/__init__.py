"""Named local and optional cloud artifact stores."""

from .base import StorageBackend
from .local import LocalStorage
from .registry import StorageConfig, StorageRegistry
from .transfer import get_tree, put_tree, sha256_file

__all__ = [
    "LocalStorage",
    "StorageBackend",
    "StorageConfig",
    "StorageRegistry",
    "get_tree",
    "put_tree",
    "sha256_file",
]
