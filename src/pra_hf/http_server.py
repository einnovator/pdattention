"""HTTP server primitives that avoid name-service work during local binding."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from socketserver import TCPServer


class PRAThreadingHTTPServer(ThreadingHTTPServer):
    """Bind immediately without reverse-resolving the listening address."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)
