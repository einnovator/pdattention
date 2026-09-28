from __future__ import annotations

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from experiments.paper4_5_agent import run_endpoint_preflight
from experiments.paper4_5_agent.serve_hf_agent_pra import _direct_handler


def test_endpoint_preflight_forwards_frozen_run_lease(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_preflight(options):
        captured["endpoint_run_token"] = options.endpoint_run_token
        return {"status": "qualified"}

    output = tmp_path / "preflight.json"
    monkeypatch.setattr(run_endpoint_preflight, "gateway_preflight", fake_preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_endpoint_preflight",
            "--base-url",
            "http://127.0.0.1:18126/v1",
            "--mode",
            "no-pra",
            "--served-model",
            "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
            "--endpoint-run-token",
            "frozen-lease",
            "--output",
            str(output),
        ],
    )

    run_endpoint_preflight.main()

    assert captured["endpoint_run_token"] == "frozen-lease"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "gateway_preflight": {"status": "qualified"}
    }


def test_hf_health_discloses_memory_placement_controls() -> None:
    class Executor:
        max_model_len = 8192
        dense_attention_implementation = "sdpa"
        load_in_4bit = True
        cpu_offload_enabled = True
        max_gpu_memory_mib = 6200
        max_cpu_memory_gib = 64
        hf_device_map = {"model.layers.0": "0", "model.layers.1": "cpu"}
        chat_template_profile = "native"
        chat_template_digest = "digest"

        @staticmethod
        def capabilities():
            return {"agent_history_kv_qualified": True}

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _direct_handler(Executor(), "model"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/health"
        ) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["load_in_4bit"] is True
    assert payload["cpu_offload_enabled"] is True
    assert payload["max_gpu_memory_mib"] == 6200
    assert payload["max_cpu_memory_gib"] == 64
    assert payload["hf_device_map"] == {
        "model.layers.0": "0",
        "model.layers.1": "cpu",
    }
