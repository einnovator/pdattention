"""Serve one synthetic PRA metric for deployment smoke tests."""

from __future__ import annotations

import argparse
import time

from pra_hf.observability import Observability, ObservabilityConfig, PrometheusConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            prometheus=PrometheusConfig(
                enabled=True, host="0.0.0.0", port=9464
            ),
        ),
        start_server=True,
    )
    telemetry.increment(
        "pra_gateway_requests_total",
        labels={"engine": "smoke", "execution_mode": "E0", "status": "success"},
    )
    try:
        time.sleep(args.seconds)
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
