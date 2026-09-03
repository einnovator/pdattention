"""Session resources, persistent input history, and transcript export."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .context_records import ContextRecord, RecordType


@dataclass(frozen=True)
class AttachmentInfo:
    attachment_id: str
    uri: str
    name: str
    mime_type: str
    size_bytes: int
    sha256: str
    source: str
    active: bool = True
    kind: str = "attachment"
    line_count: int | None = None


class AttachmentManager:
    """Append typed attachment and paste records to the active logical session."""

    def __init__(self, append: Callable[[ContextRecord], Any], records: Callable[[], tuple[ContextRecord, ...]]) -> None:
        self._append, self._records = append, records

    def add(self, path: str | Path) -> AttachmentInfo:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        body = source.read_bytes()
        mime = mimetypes.guess_type(source.name)[0] or ({".md": "text/markdown"}.get(source.suffix.casefold())) or "application/octet-stream"
        text = body.decode("utf-8") if mime.startswith("text/") or source.suffix.casefold() in {".py", ".md", ".json", ".yaml", ".yml", ".toml"} else None
        return self._store(source.name, mime, body, str(source), text=text)

    def add_paste(self, text: str) -> AttachmentInfo:
        number = 1 + sum(1 for row in self.list(include_inactive=True) if row.kind == "paste")
        body = text.encode("utf-8")
        return self._store(f"paste-{number}", "text/plain", body, f"session://paste/{number}", text=text, kind="paste")

    def add_mcp_resource(
        self, server: str, uri: str, content: str, *, mime_type: str = "text/plain"
    ) -> AttachmentInfo:
        """Persist one explicitly read MCP resource without losing its remote identity."""
        body = content.encode("utf-8")
        digest = hashlib.sha256(body).hexdigest()
        attachment_id = str(1 + len(self.list(include_inactive=True)))
        payload = {
            "attachment_id": attachment_id, "uri": uri, "name": uri,
            "mime_type": mime_type, "size_bytes": len(body), "sha256": digest,
            "source": f"mcp:{server}", "source_server": server, "active": True,
            "kind": "mcp-resource", "text": content,
            "line_count": len(content.splitlines()), "created_at": time.time(),
        }
        self._append(ContextRecord(
            f"mcp-resource:{server}:{digest[:12]}", RecordType.MCP_RESOURCE, payload
        ))
        return _info(payload)

    def _store(self, name: str, mime: str, body: bytes, source: str, *, text: str | None, kind: str = "attachment") -> AttachmentInfo:
        digest = hashlib.sha256(body).hexdigest()
        number = 1 + len(self.list(include_inactive=True))
        attachment_id = str(number)
        uri = source if kind == "paste" else Path(source).as_uri()
        payload: dict[str, Any] = {"attachment_id": attachment_id, "uri": uri, "name": name,
            "mime_type": mime, "size_bytes": len(body), "sha256": digest, "source": source,
            "active": True, "kind": kind, "created_at": time.time()}
        if text is not None:
            payload["text"] = text
            payload["line_count"] = len(text.splitlines())
        self._append(ContextRecord(f"session-resource:{attachment_id}:{digest[:12]}",
                                  RecordType.USER_PASTE if kind == "paste" else RecordType.ATTACHMENT, payload))
        return _info(payload)

    def detach(self, attachment_id: str) -> None:
        if not any(row.attachment_id == attachment_id and row.active for row in self.list()):
            raise KeyError(attachment_id)
        self._append(ContextRecord(f"session-resource-detach:{attachment_id}:{time.time_ns()}",
                                  RecordType.ATTACHMENT_EVENT,
                                  {"attachment_id": attachment_id, "active": False, "timestamp": time.time()}))

    def list(self, *, include_inactive: bool = False) -> tuple[AttachmentInfo, ...]:
        values: dict[str, AttachmentInfo] = {}
        for record in self._records():
            if record.record_type in {RecordType.ATTACHMENT, RecordType.USER_PASTE, RecordType.MCP_RESOURCE} and isinstance(record.payload, Mapping):
                values[str(record.payload["attachment_id"])] = _info(record.payload)
            elif record.record_type == RecordType.ATTACHMENT_EVENT and isinstance(record.payload, Mapping):
                key = str(record.payload["attachment_id"])
                if key in values:
                    values[key] = AttachmentInfo(**{**values[key].__dict__, "active": bool(record.payload["active"])})
        return tuple(row for row in values.values() if include_inactive or row.active)


class HistoryManager:
    """Bounded, duplicate-aware input history with optional disk persistence."""

    def __init__(self, path: str | Path | None = None, *, limit: int = 1000, suppress_duplicates: bool = True) -> None:
        self.path = Path(path).expanduser() if path else None
        self.limit, self.suppress_duplicates = limit, suppress_duplicates
        self.entries = self.path.read_text(encoding="utf-8").splitlines()[-limit:] if self.path and self.path.is_file() else []

    def add(self, value: str, *, sensitive: bool = False) -> None:
        value = value.rstrip()
        if not value or sensitive or (self.suppress_duplicates and self.entries and self.entries[-1] == value):
            return
        self.entries = (*self.entries, value)[-self.limit:] if self.limit else ()
        self.entries = list(self.entries)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\n".join(self.entries) + "\n", encoding="utf-8")

    def search(self, text: str) -> tuple[str, ...]:
        return tuple(row for row in self.entries if text.casefold() in row.casefold())

    def clear(self) -> None:
        self.entries.clear()
        if self.path and self.path.exists():
            self.path.unlink()


def export_session(state: Any, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.casefold() == ".json":
        payload = {"session_id": state.session_id, "user_id": state.user_id,
                   "metadata": dict(state.metadata), "records": [
                       {"id": row.record_id, "type": row.record_type.value, "payload": row.payload}
                       for row in state.records]}
        target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        lines = [f"# PRA Session {state.session_id}", ""]
        for row in state.records:
            payload = row.payload
            if isinstance(payload, Mapping) and payload.get("role") and payload.get("text"):
                lines.extend((f"## {str(payload['role']).title()}", "", str(payload["text"]), ""))
            elif row.record_type in {RecordType.ATTACHMENT, RecordType.USER_PASTE, RecordType.MCP_RESOURCE}:
                lines.append(f"- Resource `{row.record_id}` ({row.record_type.value})")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _info(payload: Mapping[str, Any]) -> AttachmentInfo:
    return AttachmentInfo(str(payload["attachment_id"]), str(payload["uri"]), str(payload["name"]),
                          str(payload["mime_type"]), int(payload["size_bytes"]), str(payload["sha256"]),
                          str(payload["source"]), bool(payload.get("active", True)), str(payload.get("kind", "attachment")),
                          int(payload["line_count"]) if payload.get("line_count") is not None else None)
