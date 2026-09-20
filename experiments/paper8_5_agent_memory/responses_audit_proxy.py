"""Byte-preserving Responses API proxy with explicit generation controls.

The cross-agent campaign uses this proxy only for FULL admission.  It records
the exact request envelope, fills generation parameters that Codex does not
currently expose on its command line, and streams the upstream response
without translating the Responses protocol into Chat Completions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib import error, request


@dataclass(frozen=True)
class ResponsesAuditConfig:
    upstream_base_url: str
    expected_model: str
    trace_path: Path
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_output_tokens: int = 1024
    timeout_seconds: int = 3600


class ResponsesAuditProxy:
    """Forward Responses requests while preserving their native wire format."""

    def __init__(self, config: ResponsesAuditConfig) -> None:
        self.config = config
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._request_index = 0

    def _append_trace(self, row: Mapping[str, Any]) -> None:
        self.config.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.config.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")

    def _next_index(self) -> int:
        with self._lock:
            self._request_index += 1
            return self._request_index

    def start(self) -> str:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                return

            def _send_json(self, status: int, value: Mapping[str, Any]) -> None:
                body = json.dumps(dict(value)).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                # Codex probes the model catalog. Forward it, but do not treat a
                # catalog schema mismatch as an experiment request.
                self._forward_unmodified("GET", None)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if self.path.rstrip("/") != "/v1/responses":
                    self._forward_unmodified("POST", raw)
                    return
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._send_json(400, {"error": {"message": str(exc)}})
                    return
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": {"message": "request must be an object"}})
                    return
                model = str(payload.get("model") or "")
                if model != owner.config.expected_model:
                    self._send_json(
                        400,
                        {"error": {"message": f"unexpected model {model!r}"}},
                    )
                    return
                payload.setdefault("temperature", owner.config.temperature)
                payload.setdefault("top_p", owner.config.top_p)
                payload.setdefault("seed", owner.config.seed)
                payload.setdefault("max_output_tokens", owner.config.max_output_tokens)
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                request_index = owner._next_index()
                started = time.monotonic()
                trace: dict[str, Any] = {
                    "schema_version": 1,
                    "request_index": request_index,
                    "path": self.path,
                    "request": payload,
                    "request_sha256": hashlib.sha256(body).hexdigest(),
                    "uses_previous_response_id": bool(payload.get("previous_response_id")),
                    "input_item_count": len(payload.get("input", []))
                    if isinstance(payload.get("input"), list)
                    else None,
                }
                try:
                    status, headers, response_body = self._fetch("POST", body)
                    trace.update(
                        status=status,
                        elapsed_seconds=time.monotonic() - started,
                        response_bytes=len(response_body),
                        response_sha256=hashlib.sha256(response_body).hexdigest(),
                    )
                    owner._append_trace(trace)
                    self._relay(status, headers, response_body)
                except Exception as exc:  # fail closed and leave an audit row
                    trace.update(
                        status=502,
                        elapsed_seconds=time.monotonic() - started,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    owner._append_trace(trace)
                    self._send_json(502, {"error": {"message": str(exc)}})

            def _target(self) -> str:
                root = owner.config.upstream_base_url.rstrip("/")
                suffix = self.path
                if root.endswith("/v1") and suffix.startswith("/v1/"):
                    suffix = suffix[3:]
                return root + suffix

            def _fetch(self, method: str, body: bytes | None):
                headers = {
                    "Accept": self.headers.get("Accept", "application/json"),
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                }
                authorization = self.headers.get("Authorization")
                if authorization:
                    headers["Authorization"] = authorization
                upstream = request.Request(
                    self._target(), data=body, headers=headers, method=method
                )
                try:
                    response = request.urlopen(
                        upstream, timeout=owner.config.timeout_seconds
                    )
                except error.HTTPError as exc:
                    return exc.code, exc.headers, exc.read()
                with response:
                    return response.status, response.headers, response.read()

            def _relay(self, status: int, headers: Any, body: bytes) -> None:
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _forward_unmodified(self, method: str, body: bytes | None) -> None:
                try:
                    status, headers, response_body = self._fetch(method, body)
                    self._relay(status, headers, response_body)
                except Exception as exc:
                    self._send_json(502, {"error": {"message": str(exc)}})

        self._server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://host.docker.internal:{self._server.server_port}/v1"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
