"""Exercise the PRA Gateway and verify its Prometheus and Tempo dashboards."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pra_hf.http_server import PRAThreadingHTTPServer


PROMQL_COUNTER = (
    'pra_gateway_requests_total{engine="openai_generic",'
    'execution_mode="G10",status="success"}'
)
PROMQL_RATE = (
    "sum(rate(pra_gateway_requests_total[5m])) "
    "by (engine,execution_mode,status)"
)
TRACEQL = '{ name = "pra.gateway.request" }'


class ControlledBackend(BaseHTTPRequestHandler):
    """Return deterministic completions while preserving normal HTTP transport."""

    def do_GET(self) -> None:  # noqa: N802
        self._json(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not_found"})
            return
        self._json(
            200,
            {
                "id": "gateway-observability-e2e",
                "object": "chat.completion",
                "model": request.get("model", "pra-e2e-stub"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "PRA gateway telemetry OK",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            },
        )

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


def _get_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, value: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return _get_json(url)
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(0.25)
    raise RuntimeError(f"Endpoint did not become ready: {url}: {last_error}")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_text(url: str, marker: str, *, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                value = response.read().decode("utf-8")
            if marker in value:
                return value
        except (OSError, UnicodeError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"Endpoint did not expose {marker!r}: {url}: {last_error}")


def _prometheus_query(base_url: str, expression: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"query": expression})
    value = _get_json(f"{base_url.rstrip('/')}/api/v1/query?{query}")
    if value.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {value}")
    return list(value["data"]["result"])


def _wait_prometheus(
    base_url: str, expression: str, *, timeout: float = 45.0
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = _prometheus_query(base_url, expression)
        if rows and any(float(row["value"][1]) > 0 for row in rows):
            return rows
        time.sleep(1.0)
    raise RuntimeError(f"Prometheus returned no positive data for {expression}")


def _wait_tempo(base_url: str, expression: str, *, timeout: float = 45.0) -> list[dict[str, Any]]:
    deadline = time.time() + timeout
    query = urllib.parse.urlencode({"q": expression, "limit": 100})
    while time.time() < deadline:
        value = _get_json(f"{base_url.rstrip('/')}/api/search?{query}")
        traces = list(value.get("traces", ()))
        if traces:
            return traces
        time.sleep(1.0)
    raise RuntimeError(f"Tempo returned no traces for {expression}")


def _dashboard(base_url: str, uid: str) -> dict[str, Any]:
    return _get_json(f"{base_url.rstrip('/')}/api/dashboards/uid/{uid}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--metrics-port", type=int, default=9466)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--otlp-endpoint", default="http://127.0.0.1:4317")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--tempo-url", default="http://127.0.0.1:3200")
    parser.add_argument("--grafana-url", default="http://127.0.0.1:3000")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.requests < 2:
        raise SystemExit("--requests must be at least 2 to establish a rate.")
    gateway_port = args.gateway_port or _free_port()
    backend = PRAThreadingHTTPServer(("127.0.0.1", 0), ControlledBackend)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    started = time.time()
    process: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="pra-gateway-observability-") as temporary:
        root = Path(temporary)
        config = root / "observability.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "observability": {
                        "enabled": True,
                        "service": {
                            "name": "pra-gateway",
                            "environment": "lab",
                            "attributes": {
                                "pra.engine": "gateway",
                                "pra.model_family": "e2e-stub",
                                "machine.role": socket.gethostname(),
                            },
                        },
                        "prometheus": {
                            "enabled": True,
                            "host": "0.0.0.0",
                            "port": args.metrics_port,
                            "path": "/metrics",
                        },
                        "otel": {
                            "enabled": True,
                            "endpoint": args.otlp_endpoint,
                            "protocol": "grpc",
                            "traces": True,
                            "metrics": False,
                            "sampler": "always_on",
                            "sample_rate": 1.0,
                        },
                        "content": {"capture": "none"},
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        log_path = root / "gateway.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pra_hf.cli",
                    "gateway",
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(gateway_port),
                    "--mode",
                    "selected-context",
                    "--backend",
                    "openai",
                    "--backend-url",
                    f"http://127.0.0.1:{backend.server_port}",
                    "--observability",
                    str(config),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        try:
            health = _wait_json(f"http://127.0.0.1:{gateway_port}/health")
            if health.get("protocol") != "pra" or health.get("endpoint_type") != "gateway":
                raise RuntimeError(f"Port {gateway_port} is not a PRA Gateway: {health}")
            _wait_text(
                f"http://127.0.0.1:{args.metrics_port}/metrics",
                "pra_gateway_requests_total",
            )
            completions = []
            for index in range(args.requests):
                completions.append(
                    _post_json(
                        f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                        {
                            "model": "pra-e2e-stub",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"Gateway telemetry request {index}",
                                }
                            ],
                        },
                    )
                )
                if index == 0:
                    _wait_prometheus(args.prometheus_url, PROMQL_COUNTER)
            counter_rows = _wait_prometheus(args.prometheus_url, PROMQL_COUNTER)
            rate_rows = _wait_prometheus(args.prometheus_url, PROMQL_RATE)
            traces = _wait_tempo(args.tempo_url, TRACEQL)
            metrics_dashboard = _dashboard(args.grafana_url, "pra-gateway")
            trace_dashboard = _dashboard(args.grafana_url, "pra-gateway-otel")
            report = {
                "status": "PASS",
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "timestamp": time.time(),
                "duration_seconds": time.time() - started,
                "host": socket.gethostname(),
                "request_count": len(completions),
                "response_marker": completions[-1]["choices"][0]["message"]["content"],
                "gateway_health": health,
                "prometheus": {
                    "counter_query": PROMQL_COUNTER,
                    "counter_rows": counter_rows,
                    "dashboard_query": PROMQL_RATE,
                    "dashboard_rows": rate_rows,
                },
                "tempo": {
                    "dashboard_query": TRACEQL,
                    "trace_count": len(traces),
                    "traces": traces[:10],
                },
                "grafana": {
                    "metrics_dashboard": metrics_dashboard["meta"]["url"],
                    "trace_dashboard": trace_dashboard["meta"]["url"],
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            backend.shutdown()
            backend.server_close()


if __name__ == "__main__":
    main()
