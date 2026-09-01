"""Serve immutable PRA WARM objects from an off-node file-backed host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@dataclass
class StoreMetrics:
    puts: int = 0
    gets: int = 0
    heads: int = 0
    deletes: int = 0
    bytes_written: int = 0
    bytes_read: int = 0


class RemoteWarmStore:
    """Atomic file-backed immutable-object store with aggregate counters."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics = StoreMetrics()
        self.lock = threading.Lock()

    def path(self, key: str) -> Path:
        return self.root / hashlib.sha256(key.encode("utf-8")).hexdigest()

    def put(self, key: str, payload: bytes) -> None:
        destination = self.path(key)
        temporary = destination.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_bytes(payload)
        temporary.replace(destination)
        with self.lock:
            self.metrics.puts += 1
            self.metrics.bytes_written += len(payload)

    def get(self, key: str) -> bytes | None:
        path = self.path(key)
        if not path.exists():
            return None
        payload = path.read_bytes()
        with self.lock:
            self.metrics.gets += 1
            self.metrics.bytes_read += len(payload)
        return payload

    def exists(self, key: str) -> bool:
        with self.lock:
            self.metrics.heads += 1
        return self.path(key).exists()

    def remove(self, key: str) -> bool:
        path = self.path(key)
        existed = path.exists()
        path.unlink(missing_ok=True)
        with self.lock:
            self.metrics.deletes += 1
        return existed

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return asdict(self.metrics)


def build_server(host: str, port: int, store: RemoteWarmStore) -> ThreadingHTTPServer:
    """Create the HTTP server; callers own its thread and shutdown."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _key(self) -> str | None:
            prefix = "/v1/objects/"
            if not self.path.startswith(prefix):
                return None
            return urllib.parse.unquote(self.path[len(prefix) :])

        def _reply(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                payload = json.dumps(
                    {"status": "ok", "service": "pra-sglang-remote-warm"}
                ).encode("utf-8")
                self._reply(200, payload, "application/json")
                return
            if self.path == "/metrics":
                payload = json.dumps(store.snapshot()).encode("utf-8")
                self._reply(200, payload, "application/json")
                return
            key = self._key()
            payload = None if key is None else store.get(key)
            if payload is None:
                self._reply(404, b"", "application/octet-stream")
                return
            self._reply(200, payload, "application/octet-stream")

        def do_HEAD(self) -> None:  # noqa: N802
            key = self._key()
            if key is None or not store.exists(key):
                self._reply(404, b"", "application/octet-stream")
                return
            self._reply(200, b"", "application/octet-stream")

        def do_PUT(self) -> None:  # noqa: N802
            key = self._key()
            if key is None:
                self._reply(404, b"", "application/json")
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            if len(payload) != length:
                self._reply(400, b"", "application/json")
                return
            store.put(key, payload)
            self._reply(201, b"{}", "application/json")

        def do_DELETE(self) -> None:  # noqa: N802
            key = self._key()
            if key is None:
                self._reply(404, b"", "application/json")
                return
            existed = store.remove(key)
            self._reply(200 if existed else 404, b"{}", "application/json")

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18161)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    server = build_server(args.host, args.port, RemoteWarmStore(args.root))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(
        json.dumps(
            {"status": "ready", "address": server.server_address, "started": started}
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
