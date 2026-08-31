"""Controlled AirLLM/PRA lifecycle and memory-frontier benchmark.

This benchmark measures the hook ordering and I/O overlap independently of a
language model. It is deliberately labelled CONTROLLED_MODEL in its output;
live quality and generation claims come from separate engine runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pra_hf.airllm_adapter import AirLLMPRAAdapter, AirLLMTransfer


MIB = 1024 * 1024


class TimedPRAStore:
    """Layer-local store with explicit bandwidth and reconstruction costs."""

    def __init__(self, layer_bytes: dict[int, int], bandwidth_mib_s: float, rebuild_ms: float = 0.0):
        self.layer_bytes = layer_bytes
        self.bandwidth = bandwidth_mib_s * MIB
        self.rebuild_seconds = rebuild_ms / 1000.0
        self.loaded: set[int] = set()
        self.active: set[int] = set()

    def load_layer(self, layer_id: int, device: object) -> AirLLMTransfer:
        hit = layer_id in self.loaded
        size = self.layer_bytes.get(layer_id, 0)
        started = time.perf_counter()
        if not hit:
            time.sleep(size / self.bandwidth + self.rebuild_seconds)
            self.loaded.add(layer_id)
        elapsed = time.perf_counter() - started
        return AirLLMTransfer(
            layer_id=layer_id,
            bytes_read=0 if hit else size,
            bytes_transferred=0 if hit else size,
            disk_seconds=elapsed,
            payload=layer_id,
            cache_hit=hit,
        )

    def activate_layer(self, layer_id: int, payload: Any) -> None:
        self.active.add(layer_id)

    def release_layer(self, layer_id: int) -> None:
        self.active.discard(layer_id)
        self.loaded.discard(layer_id)


class ControlledAirLLM:
    """Small hook-compatible model that emulates AirLLM's weight executor."""

    def __init__(self, layers: int, weight_layer_bytes: int, bandwidth_mib_s: float, compute_ms: float):
        self.layers = [torch.nn.Identity() for _ in range(layers)]
        self._streamed_indices = list(range(layers))
        self.device = torch.device("cpu")
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.weight_layer_bytes = weight_layer_bytes
        self.bandwidth = bandwidth_mib_s * MIB
        self.compute_seconds = compute_ms / 1000.0
        self._weight_future: Future[None] | None = None
        self._weight_future_layer: int | None = None
        self.weight_stall_seconds = 0.0
        self.weight_bytes_read = 0
        for layer_id, module in enumerate(self.layers):
            module._airllm_weight_layer = layer_id
            module.register_forward_pre_hook(self._weight_pre_hook)

    def _read_weight(self) -> None:
        time.sleep(self.weight_layer_bytes / self.bandwidth)

    def _weight_pre_hook(self, module: Any, args: tuple[Any, ...]) -> None:
        layer_id = int(module._airllm_weight_layer)
        started = time.perf_counter()
        if self._weight_future_layer == layer_id and self._weight_future is not None:
            self._weight_future.result()
        else:
            self._read_weight()
        self.weight_stall_seconds += time.perf_counter() - started
        self.weight_bytes_read += self.weight_layer_bytes
        next_layer = layer_id + 1
        if next_layer < len(self.layers):
            self._weight_future = self._executor.submit(self._read_weight)
            self._weight_future_layer = next_layer

    def run(self) -> None:
        value = torch.ones(1)
        for layer in self.layers:
            value = layer(value)
            time.sleep(self.compute_seconds)

    def close(self) -> None:
        self._executor.shutdown(wait=True)


@dataclass(frozen=True)
class BenchmarkConfig:
    layers: int = 8
    kv_heads: int = 4
    head_dim: int = 64
    dtype_bytes: int = 2
    weight_layer_mib: int = 8
    disk_bandwidth_mib_s: float = 800.0
    compute_ms: float = 3.0
    selected_tokens: int = 128
    query_tokens: int = 128


def _kv_bytes(tokens: int, layers: int, cfg: BenchmarkConfig) -> int:
    return 2 * tokens * layers * cfg.kv_heads * cfg.head_dim * cfg.dtype_bytes


def run_condition(
    *,
    source_tokens: int,
    profile: str,
    residency: str,
    prefetch: str,
    tier: str,
    repeats: int,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    profile_layers = {
        "reference_correctness": tuple(range(cfg.layers)),
        "balanced": tuple(range(cfg.layers // 2, cfg.layers)),
        "economy": tuple(range(max(0, cfg.layers - 2), cfg.layers)),
    }[profile]
    pra_per_layer = _kv_bytes(cfg.selected_tokens, 1, cfg)
    tier_config = {
        "warm": (800.0, 0.0, 1.0),
        "cold": (1200.0, 0.35, 0.5),
        "source": (2400.0, 0.9, 0.0),
    }[tier]
    bandwidth, rebuild_ms, persisted_scale = tier_config
    store = TimedPRAStore(
        {layer: int(pra_per_layer * max(persisted_scale, 0.05)) for layer in profile_layers},
        bandwidth_mib_s=bandwidth,
        rebuild_ms=rebuild_ms,
    )
    model = ControlledAirLLM(
        cfg.layers,
        cfg.weight_layer_mib * MIB,
        cfg.disk_bandwidth_mib_s,
        cfg.compute_ms,
    )
    adapter = AirLLMPRAAdapter(
        store,
        consumer_layers=profile_layers,
        residency_mode=residency,
        hot_layers=profile_layers[-2:],
        prefetch_mode=prefetch,
    ).bind(model)
    durations: list[float] = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            model.run()
            durations.append(time.perf_counter() - started)
        summary = adapter.summary()
    finally:
        adapter.close()
        model.close()

    # Native selected detail is accounted in PRA K/V, not duplicated in the
    # sequential cache. Only the query/decode working set remains local.
    local_kv = _kv_bytes(cfg.query_tokens, cfg.layers, cfg)
    all_pra = pra_per_layer * len(profile_layers)
    if residency == "hot":
        pra_peak = all_pra
    elif residency == "hybrid":
        pra_peak = pra_per_layer * min(3, len(profile_layers))
    else:
        pra_peak = pra_per_layer
    weight_hot = cfg.weight_layer_mib * MIB
    framework = 64 * MIB
    temporary = 16 * MIB
    peak = weight_hot + local_kv + pra_peak + framework + temporary
    return {
        "execution_mode": "native_pra",
        "source_tokens": source_tokens,
        "selected_tokens": cfg.selected_tokens,
        "profile": profile,
        "consumer_layers": len(profile_layers),
        "residency": residency,
        "prefetch": prefetch,
        "tier": tier,
        "repeats": repeats,
        "latency_mean_ms": statistics.mean(durations) * 1000.0,
        "latency_p95_ms": max(durations) * 1000.0,
        "weight_stall_ms": model.weight_stall_seconds * 1000.0,
        "pra_stall_ms": sum(
            event["elapsed_seconds"] for event in summary["events"] if event["event"] == "activate"
        ) * 1000.0,
        "weight_bytes_read": model.weight_bytes_read,
        "pra_bytes_read": summary["pra_bytes_read"],
        "pra_bytes_transferred": summary["pra_bytes_transferred"],
        "prefetch_hits": summary["prefetch_hits"],
        "cache_hits": summary["cache_hits"],
        "memory_bytes": {
            "weights_hot": weight_hot,
            "local_kv": local_kv,
            "pra_hot": pra_peak,
            "pra_warm": int(all_pra * persisted_scale),
            "temporary": temporary,
            "framework": framework,
            "peak": peak,
        },
    }


def baseline_row(source_tokens: int, mode: str, cfg: BenchmarkConfig) -> dict[str, Any]:
    """Memory-only E0 baselines with the same frozen selected evidence."""

    visible = source_tokens + cfg.query_tokens if mode == "full_context" else cfg.selected_tokens + cfg.query_tokens
    weight_hot = cfg.weight_layer_mib * MIB
    local_kv = _kv_bytes(visible, cfg.layers, cfg)
    temporary = 16 * MIB
    framework = 64 * MIB
    return {
        "execution_mode": mode,
        "source_tokens": source_tokens,
        "selected_tokens": cfg.selected_tokens,
        "visible_tokens": visible,
        "profile": "none",
        "consumer_layers": 0,
        "residency": "none",
        "prefetch": "weight_only",
        "tier": "none",
        "repeats": 0,
        "latency_mean_ms": None,
        "latency_p95_ms": None,
        "weight_stall_ms": None,
        "pra_stall_ms": None,
        "weight_bytes_read": None,
        "pra_bytes_read": 0,
        "pra_bytes_transferred": 0,
        "prefetch_hits": 0,
        "cache_hits": 0,
        "memory_bytes": {
            "weights_hot": weight_hot,
            "local_kv": local_kv,
            "pra_hot": 0,
            "pra_warm": 0,
            "temporary": temporary,
            "framework": framework,
            "peak": weight_hot + local_kv + temporary + framework,
        },
    }


def run_benchmark(repeats: int = 3) -> dict[str, Any]:
    cfg = BenchmarkConfig()
    rows = []
    for source_tokens in (2048, 8192, 32768, 65536):
        rows.append(baseline_row(source_tokens, "full_context", cfg))
        rows.append(baseline_row(source_tokens, "selected_text", cfg))
        for profile in ("reference_correctness", "balanced", "economy"):
            for residency in ("hot", "layer_streamed", "hybrid"):
                for prefetch in ("none", "independent_parallel", "coordinated"):
                    rows.append(
                        run_condition(
                            source_tokens=source_tokens,
                            profile=profile,
                            residency=residency,
                            prefetch=prefetch,
                            tier="warm",
                            repeats=repeats,
                            cfg=cfg,
                        )
                    )
    for tier in ("warm", "cold", "source"):
        rows.append(
            run_condition(
                source_tokens=32768,
                profile="balanced",
                residency="layer_streamed",
                prefetch="coordinated",
                tier=tier,
                repeats=repeats,
                cfg=cfg,
            )
        )
    return {
        "schema_version": "paper6.6-layer-streaming-v1",
        "evidence_tier": "CONTROLLED_MODEL",
        "claim_boundary": "Hook lifecycle, memory accounting, and controlled I/O only; no model quality claim.",
        "config": asdict(cfg),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run_benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(report['rows'])} rows to {args.output}")


if __name__ == "__main__":
    main()
