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


def main() -> None:
    args = parser().parse_args()
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
            status, elapsed = check_endpoint(args.engine_url, args.timeout)
            labels = {
                "engine": engine,
                "model_family": args.model,
                "profile": "observability",
                "execution_mode": "probe",
                "status": status,
            }
            with telemetry.span(
                "pra.engine.health",
                {
                    "pra.engine": engine,
                    "pra.model_family": args.model,
                    "pra.machine": args.machine,
                    "pra.engine.status": status,
                    "pra.engine.request.duration_ms": elapsed * 1000.0,
                },
            ):
                telemetry.increment("pra_engine_requests_total", labels=labels)
                telemetry.observe(
                    "pra_engine_request_duration_seconds", elapsed, labels=labels
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
