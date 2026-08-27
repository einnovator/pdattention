"""Scoped lossless backing storage for typed adaptive-context records.

The store owns exact serialized tool results. Prompt-visible records retain only
an opaque identity, compact views, and retrieval addresses. Authorization is
checked before every read, selector operation, cursor open, or deletion.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from .context_records import RecordType


@dataclass(frozen=True)
class RecordScope:
    """Tenant/session boundary used for every backing-state operation."""

    tenant_id: str
    session_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.session_id:
            raise ValueError("tenant_id and session_id are required.")

    @property
    def fingerprint(self) -> str:
        value = f"{self.tenant_id}\0{self.session_id}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()[:24]


@dataclass(frozen=True)
class BackingRecord:
    """Persisted metadata for one exact payload, excluding the payload itself."""

    record_id: str
    record_type: RecordType | str
    scope: RecordScope
    content_hash: str
    encoding: str
    size_bytes: int
    created_at: float
    expires_at: float | None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", RecordType(self.record_type))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


@dataclass(frozen=True)
class StoreStats:
    """Current local-store occupancy after expired entries are removed."""

    records: int
    payload_bytes: int
    max_bytes: int | None


class BackingStoreError(RuntimeError):
    """Base error for backing-state lifecycle failures."""


class RecordNotFound(BackingStoreError):
    """Raised when no live backing record resolves within the supplied scope."""


class RecordAccessDenied(BackingStoreError):
    """Raised when a known record is addressed from a different scope."""


class LocalBackingStore:
    """Content-addressed local store with scope, TTL, and bounded occupancy.

    Payload files live below a hash of the tenant/session scope, preventing
    accidental cross-session file reuse. JSON-compatible objects preserve exact
    JSON values; strings and bytes are retained byte-for-byte.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_bytes: int | None = None,
        persistent: bool = False,
    ) -> None:
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive when provided.")
        if root is None:
            root = Path(tempfile.mkdtemp(prefix="pra-adaptive-context-"))
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.persistent = persistent
        self._records: dict[str, BackingRecord] = {}
        self._load_manifests()

    @staticmethod
    def _encode(payload: object) -> tuple[str, bytes]:
        if isinstance(payload, bytes):
            return "bytes", payload
        if isinstance(payload, str):
            return "text", payload.encode("utf-8")
        try:
            value = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TypeError("Backing payload must be bytes, text, or JSON-compatible.") from exc
        return "json", value

    @staticmethod
    def _decode(encoding: str, payload: bytes) -> object:
        if encoding == "bytes":
            return payload
        if encoding == "text":
            return payload.decode("utf-8")
        if encoding == "json":
            return json.loads(payload.decode("utf-8"))
        raise BackingStoreError(f"Unsupported backing encoding: {encoding}")

    def _scope_dir(self, scope: RecordScope) -> Path:
        return self.root / scope.fingerprint

    def _paths(self, record: BackingRecord) -> tuple[Path, Path]:
        identity_hash = hashlib.sha256(record.record_id.encode("utf-8")).hexdigest()
        base = self._scope_dir(record.scope) / identity_hash
        return base.with_suffix(".payload"), base.with_suffix(".json")

    def _load_manifests(self) -> None:
        for path in self.root.glob("*/*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["scope"] = RecordScope(**raw["scope"])
                record = BackingRecord(**raw)
                payload_path, _ = self._paths(record)
                if payload_path.is_file() and not record.expired:
                    self._records[record.record_id] = record
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self.cleanup()

    def put(
        self,
        payload: object,
        *,
        record_type: RecordType | str,
        scope: RecordScope,
        provenance: Mapping[str, object] | None = None,
        ttl_seconds: float | None = None,
    ) -> BackingRecord:
        """Store exact payload bytes and return a stable scoped descriptor."""

        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when provided.")
        encoding, encoded = self._encode(payload)
        record_type = RecordType(record_type)
        digest = hashlib.sha256(encoded).hexdigest()
        normalized_provenance = json.loads(json.dumps(
            dict(provenance or {}), sort_keys=True, ensure_ascii=True, default=str
        ))
        logical_hash = hashlib.sha256(
            record_type.value.encode("utf-8")
            + b"\0"
            + digest.encode("ascii")
            + b"\0"
            + json.dumps(normalized_provenance, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record_id = (
            f"pra-record://{scope.fingerprint}/{record_type.value}/sha256:{logical_hash}"
        )
        now = time.time()
        existing = self._records.get(record_id)
        if existing is not None and not existing.expired:
            return existing
        record = BackingRecord(
            record_id=record_id,
            record_type=record_type,
            scope=scope,
            content_hash=digest,
            encoding=encoding,
            size_bytes=len(encoded),
            created_at=now,
            expires_at=now + ttl_seconds if ttl_seconds is not None else None,
            provenance=normalized_provenance,
        )
        payload_path, manifest_path = self._paths(record)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(encoded)
        manifest = asdict(record)
        manifest["record_type"] = record.record_type.value
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=True), encoding="utf-8"
        )
        self._records[record_id] = record
        self._enforce_limit(exclude=record_id)
        return record

    def _resolve(self, record_id: str, scope: RecordScope) -> BackingRecord:
        record = self._records.get(record_id)
        if record is None:
            raise RecordNotFound(record_id)
        if record.scope != scope:
            raise RecordAccessDenied(record_id)
        if record.expired:
            for path in self._paths(record):
                path.unlink(missing_ok=True)
            self._records.pop(record_id, None)
            raise RecordNotFound(record_id)
        return record

    def descriptor(self, record_id: str, *, scope: RecordScope) -> BackingRecord:
        """Return authorized metadata without reading payload bytes."""

        return self._resolve(record_id, scope)

    def get(self, record_id: str, *, scope: RecordScope) -> object:
        """Read and hash-verify one exact payload after scope authorization."""

        record = self._resolve(record_id, scope)
        payload_path, _ = self._paths(record)
        try:
            encoded = payload_path.read_bytes()
        except OSError as exc:
            raise RecordNotFound(record_id) from exc
        if hashlib.sha256(encoded).hexdigest() != record.content_hash:
            raise BackingStoreError(f"Backing payload failed hash verification: {record_id}")
        return self._decode(record.encoding, encoded)

    def delete(self, record_id: str, *, scope: RecordScope) -> None:
        """Revoke and remove one record in the authorized scope."""

        record = self._resolve(record_id, scope)
        for path in self._paths(record):
            path.unlink(missing_ok=True)
        self._records.pop(record_id, None)

    def cleanup(self) -> int:
        """Remove expired entries and, for ephemeral stores, orphan manifests."""

        removed = 0
        for record_id, record in tuple(self._records.items()):
            if not record.expired:
                continue
            for path in self._paths(record):
                path.unlink(missing_ok=True)
            self._records.pop(record_id, None)
            removed += 1
        return removed

    def _enforce_limit(self, *, exclude: str) -> None:
        if self.max_bytes is None:
            return
        while self.stats().payload_bytes > self.max_bytes:
            candidates = [row for key, row in self._records.items() if key != exclude]
            if not candidates:
                record = self._records[exclude]
                for path in self._paths(record):
                    path.unlink(missing_ok=True)
                self._records.pop(exclude, None)
                raise BackingStoreError("Payload exceeds the backing-store size limit.")
            oldest = min(candidates, key=lambda row: row.created_at)
            self.delete(oldest.record_id, scope=oldest.scope)

    def stats(self) -> StoreStats:
        self.cleanup()
        return StoreStats(
            records=len(self._records),
            payload_bytes=sum(record.size_bytes for record in self._records.values()),
            max_bytes=self.max_bytes,
        )

    def close(self) -> None:
        """Delete live payloads when configured as an ephemeral local cache."""

        if self.persistent:
            return
        for record in tuple(self._records.values()):
            for path in self._paths(record):
                path.unlink(missing_ok=True)
        self._records.clear()

    def __enter__(self) -> LocalBackingStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
