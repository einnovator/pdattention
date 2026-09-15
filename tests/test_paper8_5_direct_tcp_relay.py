from __future__ import annotations

import socket
import socketserver
import threading

from experiments.paper8_5_agent_memory.direct_tcp_relay import build_server


class _Echo(socketserver.BaseRequestHandler):
    def handle(self):
        while data := self.request.recv(65536):
            self.request.sendall(data)


def test_relay_preserves_bytes_bidirectionally():
    upstream = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Echo)
    upstream_thread = threading.Thread(
        target=upstream.serve_forever, daemon=True
    )
    upstream_thread.start()
    relay = build_server(
        listen_host="127.0.0.1", listen_port=0,
        upstream_host="127.0.0.1", upstream_port=upstream.server_address[1],
    )
    relay_thread = threading.Thread(target=relay.serve_forever, daemon=True)
    relay_thread.start()
    try:
        payload = b"ordinary OpenAI request bytes\x00\xff"
        with socket.create_connection(relay.server_address, timeout=2) as client:
            client.sendall(payload)
            assert client.recv(len(payload)) == payload
    finally:
        relay.shutdown()
        relay.server_close()
        upstream.shutdown()
        upstream.server_close()
        relay_thread.join(timeout=2)
        upstream_thread.join(timeout=2)
