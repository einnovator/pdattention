from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib import request

import pytest

from experiments.paper8_5_agent_memory.responses_audit_proxy import (
    ResponsesAuditConfig,
    ResponsesAuditProxy,
)
from experiments.paper8_5_agent_memory.run_codex_swebench import (
    build_parser,
    _event_summary,
    _ollama_forward_target,
    _observed_model_identity,
    _trace_summary,
)


def test_codex_runner_accepts_native_ollama_provider_mode() -> None:
    args = build_parser().parse_args([
        "--instance-id", "django__django-15277",
        "--output", "out",
        "--reference-trajectory", "trajectory.json",
        "--upstream-base-url", "http://model.example:11434",
        "--provider-mode", "ollama_oss",
        "--ollama-tags-url", "http://model.example:11434/api/tags",
        "--model-revision", "revision",
    ])
    assert args.provider_mode == "ollama_oss"
    assert args.model_context_window == 65536


def test_native_codex_parses_remote_ollama_forward_target() -> None:
    assert _ollama_forward_target("http://192.168.1.6:11435") == (
        "192.168.1.6",
        11435,
    )
    assert _ollama_forward_target("http://model.example") == (
        "model.example",
        11434,
    )
    with pytest.raises(ValueError, match="must not contain a path"):
        _ollama_forward_target("http://model.example:11434/v1")


def test_native_codex_fails_closed_without_observed_model_identity() -> None:
    args = build_parser().parse_args([
        "--instance-id", "django__django-15277",
        "--output", "out",
        "--reference-trajectory", "trajectory.json",
        "--upstream-base-url", "http://model.example:11434",
        "--provider-mode", "ollama_oss",
    ])

    with pytest.raises(ValueError, match="observed model revision"):
        _observed_model_identity(args)


def test_codex_event_summary_preserves_native_action_types(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    rows = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "file_change"}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
    ]
    events.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = _event_summary(events)
    assert summary["event_rows"] == 4
    assert summary["item_types"] == {"command_execution": 1, "file_change": 1}
    assert summary["final_turn_usage"] == {"input_tokens": 100, "output_tokens": 20}


def test_codex_trace_summary_detects_provider_managed_continuation(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join([
            json.dumps({"status": 200, "input_item_count": 3, "uses_previous_response_id": False}),
            json.dumps({"status": 200, "input_item_count": 1, "uses_previous_response_id": True}),
        ]) + "\n",
        encoding="utf-8",
    )
    summary = _trace_summary(trace)
    assert summary["request_count"] == 2
    assert summary["all_success"] is True
    assert summary["uses_previous_response_id"] is True
    assert summary["input_item_counts"] == [3, 1]


def test_responses_proxy_fills_controls_without_protocol_translation(
    tmp_path: Path,
) -> None:
    captured = {}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["payload"] = json.loads(self.rfile.read(length))
            body = json.dumps({"id": "response-1", "object": "response"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace = tmp_path / "trace.jsonl"
    proxy = ResponsesAuditProxy(ResponsesAuditConfig(
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        expected_model="qwen",
        trace_path=trace,
        max_output_tokens=99,
    ))
    # Unit tests run outside Docker, so replace the advertised Docker hostname.
    url = proxy.start().replace("host.docker.internal", "127.0.0.1")
    try:
        payload = json.dumps({"model": "qwen", "input": [{"role": "user"}]}).encode()
        response = request.urlopen(request.Request(
            url + "/responses",
            data=payload,
            headers={"Content-Type": "application/json"},
        ))
        assert response.status == 200
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
    assert captured["path"] == "/v1/responses"
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["top_p"] == 1.0
    assert captured["payload"]["seed"] == 0
    assert captured["payload"]["max_output_tokens"] == 99
    row = json.loads(trace.read_text(encoding="utf-8"))
    assert row["input_item_count"] == 1
    assert row["status"] == 200
