"""Transparent TCP relay for hosts whose Python runtime lacks LAN permission."""

from __future__ import annotations

import sys

# Direct execution places this directory first and would shadow the standard
# library's ``selectors`` module with the Paper 8.5 policy module.
if sys.path and sys.path[0].replace("\\", "/").endswith(
    "/experiments/paper8_5_agent_memory"
):
    del sys.path[0]

import argparse
import json
import select
import socket
import socketserver
from typing import Sequence


class _RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, listen: tuple[str, int], upstream: tuple[str, int]):
        self.upstream = upstream
        super().__init__(listen, _RelayHandler)


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        upstream = socket.create_connection(self.server.upstream, timeout=30)
        with upstream:
            peers = (self.request, upstream)
            while True:
                readable, _, _ = select.select(peers, (), (), 60)
                for source in readable:
                    data = source.recv(1024 * 1024)
                    if not data:
                        return
                    target = upstream if source is self.request else self.request
                    target.sendall(data)


def build_server(
    *, listen_host: str, listen_port: int,
    upstream_host: str, upstream_port: int,
) -> _RelayServer:
    return _RelayServer(
        (listen_host, listen_port), (upstream_host, upstream_port)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with build_server(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
    ) as server:
        print(json.dumps({
            "status": "listening",
            "listen": f"{args.listen_host}:{server.server_address[1]}",
            "upstream": f"{args.upstream_host}:{args.upstream_port}",
            "payload_transformation": False,
        }), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
