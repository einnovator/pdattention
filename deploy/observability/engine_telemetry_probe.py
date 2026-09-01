"""Expose PRA metrics and OTLP spans for one running engine endpoint.

The probe is deliberately engine-neutral. It performs a bounded health request,
then records that request through PRA's normal telemetry API. This makes an
engine visible in the shared lab dashboards even when the engine itself has no
native OpenTelemetry exporter; native engine metrics remain separate scrape
targets when available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from pra_hf.observability import (
        OTelConfig,
        Observability,
        ObservabilityConfig,
        PrometheusConfig,
        ServiceConfig,
    )
except ImportError:
    # A probe can be copied beside observability.py without installing PRA.
    module_path = Path(__file__).with_name("observability.py")
    spec = importlib.util.spec_from_file_location("pra_probe_observability", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    OTelConfig = module.OTelConfig
    Observability = module.Observability
    ObservabilityConfig = module.ObservabilityConfig
    PrometheusConfig = module.PrometheusConfig
    ServiceConfig = module.ServiceConfig


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--engine", required=True)
    result.add_argument("--model", default="unknown")
    result.add_argument("--machine", default=socket.gethostname())
    result.add_argument("--engine-url")
    result.add_argument(
        "--queries",
        type=Path,
        help="JSON array of named dataset prompts; POST each to ENGINE_URL.",
    )
    result.add_argument("--max-tokens", type=int, default=32)
    result.add_argument("--otlp-endpoint", required=True)
    result.add_argument("--metrics-port", type=int, default=9464)
    result.add_argument("--interval", type=float, default=5.0)
    result.add_argument("--timeout", type=float, default=3.0)
    result.add_argument("--iterations", type=int, default=0)
    return result


def check_endpoint(url: str | None, timeout: float) -> tuple[str, float]:
    if not url:
        return "telemetry_only", 0.0
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = "success" if response.status < 500 else "error"
            response.read(1024)
    except (OSError, urllib.error.URLError):
        status = "unavailable"
    return status, time.perf_counter() - started


def run_query(
    url: str,
    model: str,
    query: dict,
    timeout: float,
    max_tokens: int,
) -> tuple[str, float, int]:
    """Execute one bounded OpenAI-compatible dataset query."""

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": str(query["prompt"])}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    generated_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        generated_tokens = int(result.get("usage", {}).get("completion_tokens", 0))
        status = "success"
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        status = "error"
    return status, time.perf_counter() - started, generated_tokens


def main() -> None:
    args = parser().parse_args()
    if args.queries and not args.engine_url:
        raise SystemExit("--queries requires --engine-url pointing at chat completions.")
    queries = []
    if args.queries:
        queries = json.loads(args.queries.read_text(encoding="utf-8"))
        if not queries or any("dataset" not in row or "prompt" not in row for row in queries):
            raise SystemExit("--queries must contain dataset and prompt fields.")
    engine = args.engine.replace("-", "_").lower()
    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            service=ServiceConfig(
                name=f"pra-{engine}",
                environment="lab",
                attributes={
                    "pra.engine": engine,
                    "pra.model_family": args.model,
                    "pra.profile": "observability",
                    "pra.execution_mode": "probe",
                    "machine.role": args.machine,
                },
            ),
            prometheus=PrometheusConfig(
                enabled=True, host="0.0.0.0", port=args.metrics_port
            ),
            otel=OTelConfig(
                enabled=True,
                endpoint=args.otlp_endpoint,
                sampler="always_on",
                sample_rate=1.0,
            ),
        ),
        start_server=True,
    )
    count = 0
    try:
        while args.iterations <= 0 or count < args.iterations:
            query = queries[count % len(queries)] if queries else None
            if query is None:
                status, elapsed = check_endpoint(args.engine_url, args.timeout)
                generated_tokens = 0
                source_tokens = 0
                selected_tokens = 0
            else:
                status, elapsed, generated_tokens = run_query(
                    args.engine_url,
                    args.model,
                    query,
                    args.timeout,
                    args.max_tokens,
                )
                source_tokens = int(
                    query.get("source_tokens", len(str(query["prompt"]).split()))
                )
                selected_tokens = int(query.get("selected_tokens", source_tokens))
            labels = {
                "engine": engine,
                "model_family": args.model,
                "profile": "observability",
                "execution_mode": "probe",
                "status": status,
            }
            span_name = "pra.engine.request" if query is not None else "pra.engine.health"
            with telemetry.span(
                span_name,
                {
                    "pra.engine": engine,
                    "pra.model_family": args.model,
                    "pra.machine": args.machine,
                    "pra.engine.status": status,
                    "pra.engine.request.duration_ms": elapsed * 1000.0,
                    "pra.dataset": query.get("dataset") if query else "health",
                    "pra.context.source_tokens": source_tokens,
                    "pra.context.selected_tokens": selected_tokens,
                    "pra.generated_tokens": generated_tokens,
                },
            ):
                telemetry.increment("pra_engine_requests_total", labels=labels)
                telemetry.observe(
                    "pra_engine_request_duration_seconds", elapsed, labels=labels
                )
                if query is not None:
                    context_labels = {
                        "engine": engine,
                        "profile": "observability",
                        "execution_mode": "E0",
                    }
                    telemetry.increment(
                        "pra_context_source_tokens_total",
                        source_tokens,
                        labels=context_labels,
                    )
                    telemetry.increment(
                        "pra_context_selected_tokens_total",
                        selected_tokens,
                        labels=context_labels,
                    )
                    telemetry.increment(
                        "pra_context_new_materialized_tokens_total",
                        selected_tokens,
                        labels=context_labels,
                    )
                if status in {"success", "telemetry_only"}:
                    telemetry.increment(
                        "pra_engine_successful_requests_total",
                        labels={key: value for key, value in labels.items() if key != "status"},
                    )
                else:
                    telemetry.increment(
                        "pra_engine_errors_total",
                        labels={key: value for key, value in labels.items() if key != "status"},
                    )
            count += 1
            if args.iterations <= 0 or count < args.iterations:
                time.sleep(args.interval)
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
