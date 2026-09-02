from __future__ import annotations

import socket
from http.server import BaseHTTPRequestHandler

from pra_hf.http_server import PRAThreadingHTTPServer


def test_server_bind_does_not_require_reverse_dns(monkeypatch) -> None:
    def reject_reverse_dns(host: str) -> str:
        raise AssertionError(f"unexpected reverse DNS lookup for {host}")

    monkeypatch.setattr(socket, "getfqdn", reject_reverse_dns)
    server = PRAThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        assert server.server_name == "127.0.0.1"
        assert server.server_port > 0
    finally:
        server.server_close()
