from __future__ import annotations

import threading

import pytest

from experiments.paper6_1_sglang.serve_remote_warm import RemoteWarmStore, build_server
from pra_sglang.remote_warm import HTTPHiCacheStorageClient


def test_remote_warm_tensor_round_trip_and_metrics(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    store = RemoteWarmStore(tmp_path / "remote")
    server = build_server("127.0.0.1", 0, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = HTTPHiCacheStorageClient(f"http://{host}:{port}")
        source = torch.arange(64, dtype=torch.uint8)
        assert client.health()["status"] == "ok"
        assert client.set("tenant/session/resource", source)
        assert client.exists("tenant/session/resource")
        target = torch.empty_like(source)
        assert client.get("tenant/session/resource", target) is target
        assert torch.equal(source, target)
        metrics = client.metrics().to_dict()
        assert metrics["writes"] == 1
        assert metrics["reads"] == 1
        assert metrics["written_bytes"] == source.numel()
        assert metrics["read_bytes"] == source.numel()
        client.remove("tenant/session/resource")
        assert not client.exists("tenant/session/resource")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
