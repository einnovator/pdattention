"""Measure transport-neutral router compilation at realistic fleet sizes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from pra_router.adapters import (
    AgentGatewayAdapter,
    BifrostRouterAdapter,
    KubernetesGAIEAdapter,
    LiteLLMRouterAdapter,
    MemoryRouterTransport,
    ReferenceRouterAdapter,
)
from pra_router.controller import RouterDesiredState


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper4_5_runtime/router_control_scale.json"
ADAPTERS = (
    LiteLLMRouterAdapter,
    AgentGatewayAdapter,
    KubernetesGAIEAdapter,
    ReferenceRouterAdapter,
    BifrostRouterAdapter,
)


def desired_state(kind: str, count: int) -> RouterDesiredState:
    backends = [
        {
            "id": f"engine-{index}:model", "engine_instance_id": f"engine-{index}",
            "runtime_model_id": "model", "inference_url": f"http://engine-{index}:8000",
            "engine": "vllm", "engine_version": "0.11", "model_id": "org/model",
            "model_revision": "base-sha", "bundle_id": "bundle", "bundle_revision": "bundle-sha",
            "profile": "BALANCED", "modes": ["selected-context", "native-memory"],
            "qualification_tier": "ENGINE_QUALIFIED", "approval_state": "APPROVED",
            "region": "eu", "cluster": "gpu", "health": "READY", "maintenance": False,
            "weight": float(index % 4 + 1), "labels": {"rack": str(index % 8)}, "metadata": {},
        }
        for index in range(count)
    ]
    return RouterDesiredState.model_validate({
        "router": {
            "id": f"{kind}-scale", "kind": kind, "management_url": "http://router-admin",
            "metadata": {"namespace": "pra", "gateway": "pra-gateway"},
        },
        "desired_revision": 1,
        "routes": [{
            "id": "chat", "public_model": "pra/chat", "route_kind": "llm",
            "policy": {
                "id": "balanced", "strategy": "weighted", "constraints": {},
                "preferences": {"qualified_first": True}, "fallback": [],
            },
            "pools": [{
                "id": "production", "model_id": "org/model", "selectors": {},
                "metadata": {"port": 8000}, "fallback": False,
                "backends": backends, "excluded": [],
            }],
        }],
    })


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def run(*, repetitions: int = 7) -> dict[str, Any]:
    rows = []
    for count in (10, 100, 1000):
        for adapter_class in ADAPTERS:
            adapter = adapter_class(MemoryRouterTransport())
            state = desired_state(adapter_class.kind, count)
            durations = []
            payload = {}
            for _ in range(repetitions):
                started = time.perf_counter()
                payload = adapter.compile(state)
                durations.append((time.perf_counter() - started) * 1000)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            rows.append({
                "adapter": adapter_class.kind,
                "backend_count": count,
                "repetitions": repetitions,
                "compile_ms_mean": statistics.fmean(durations),
                "compile_ms_p95": percentile(durations, 0.95),
                "config_bytes": len(encoded),
            })
    return {
        "schema": "pra-router-control-scale/1",
        "scope": "offline desired-state compilation; excludes external router apply and request throughput",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    result = run(repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
