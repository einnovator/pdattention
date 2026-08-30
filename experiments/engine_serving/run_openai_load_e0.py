"""Run a matched OpenAI-compatible E0 load sweep across serving engines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping


def _load_harness():
    source = Path(__file__).resolve().parents[2] / "src" / "pra_hf" / "serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("pra_serving_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving benchmark helpers from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gpu_memory() -> Mapping[str, object]:
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).splitlines()[0]
        used, total, utilization, power = [value.strip() for value in line.split(",")]
        return {
            "available": True,
            "device_used_mib": int(used),
            "device_total_mib": int(total),
            "utilization_percent": float(utilization),
            "power_watts": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "error": str(error)}


def _request_spec(
    harness: object,
    representation: str,
    resource_number: int,
    *,
    distractor_count: int,
    distractor_repeat: int,
) -> tuple[list[dict[str, str]], str]:
    messages = harness.benchmark_messages(
        distractor_count=distractor_count,
        distractor_repeat=distractor_repeat,
    )[representation]
    expected = f"PRA_EVIDENCE_{4821 + resource_number}"
    return (
        json.loads(json.dumps(messages).replace("PRA_EVIDENCE_4821", expected)),
        expected,
    )


def _aggregate(
    harness: object,
    rows: list[Mapping[str, object]],
    elapsed: float,
) -> Mapping[str, object]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    def percentiles(name: str) -> Mapping[str, float | None]:
        data = values(name)
        return {
            "p50": harness.percentile(data, 0.50),
            "p95": harness.percentile(data, 0.95),
            "p99": harness.percentile(data, 0.99),
        }

    output_tokens = values("completion_tokens")
    cached = values("cached_tokens")
    return {
        "sample_count": len(rows),
        "quality_success_rate": statistics.fmean(float(row["quality_ok"]) for row in rows),
        "request_throughput_s": len(rows) / max(elapsed, 1e-9),
        "output_throughput_tokens_s": (
            sum(output_tokens) / max(elapsed, 1e-9) if output_tokens else None
        ),
        "ttft_ms": percentiles("ttft_ms"),
        "itl_ms": percentiles("mean_itl_ms"),
        "completion_latency_ms": percentiles("completion_latency_ms"),
        "mean_cached_tokens": statistics.fmean(cached) if cached else None,
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    harness = _load_harness()
    samples: list[Mapping[str, object]] = []
    aggregates: list[Mapping[str, object]] = []
    for representation in ("pra_only", "full_context"):
        for workload in ("shared_resource", "independent_resources"):
            for concurrency in args.concurrency:
                specs = []
                for wave in range(args.waves):
                    for slot in range(concurrency):
                        resource_number = (
                            0
                            if workload == "shared_resource"
                            else wave * concurrency + slot + 1
                        )
                        messages, expected = _request_spec(
                            harness,
                            representation,
                            resource_number,
                            distractor_count=args.distractor_count,
                            distractor_repeat=args.distractor_repeat,
                        )
                        salt = hashlib.sha256(
                            f"{args.engine}:resource-{resource_number}:{representation}".encode(
                                "utf-8"
                            )
                        ).hexdigest()
                        specs.append(
                            (wave, slot, resource_number, messages, expected, salt)
                        )

                unique_specs = {spec[2]: spec for spec in specs}
                for _, _, _, messages, _, salt in unique_specs.values():
                    harness.stream_chat_completion(
                        args.base_url,
                        model=args.model,
                        messages=messages,
                        timeout_seconds=args.timeout_seconds,
                        cache_salt=salt if args.cache_salt else None,
                        max_tokens=args.max_tokens,
                    )
                before_gpu = _gpu_memory()
                group_started = time.perf_counter()
                group_rows = []
                for wave in range(args.waves):
                    wave_specs = [spec for spec in specs if spec[0] == wave]

                    def issue(spec):
                        _, slot, resource_number, messages, expected, salt = spec
                        result = harness.stream_chat_completion(
                            args.base_url,
                            model=args.model,
                            messages=messages,
                            timeout_seconds=args.timeout_seconds,
                            cache_salt=salt if args.cache_salt else None,
                            max_tokens=args.max_tokens,
                        )
                        return {
                            "representation": representation,
                            "workload": workload,
                            "concurrency": concurrency,
                            "wave": wave,
                            "slot": slot,
                            "resource_number": resource_number,
                            "quality_ok": expected in result["output_text"],
                            **result,
                        }

                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        rows = list(executor.map(issue, wave_specs))
                    group_rows.extend(rows)
                    samples.extend(rows)
                elapsed = time.perf_counter() - group_started
                aggregates.append(
                    {
                        "representation": representation,
                        "workload": workload,
                        "concurrency": concurrency,
                        "waves": args.waves,
                        "warmup_requests": len(unique_specs),
                        "wall_seconds": elapsed,
                        "gpu_before": before_gpu,
                        "gpu_after": _gpu_memory(),
                        **_aggregate(harness, group_rows, elapsed),
                    }
                )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_cross_engine_openai_e0_load_v1",
        "evidence_tier": "SERVING_BOUNDED_LOAD",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "engine": args.engine,
        "model_id": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "waves": args.waves,
        "cache_state": "WARM_EXPLICITLY_PRIMED",
        "max_tokens": args.max_tokens,
        "distractor_count": args.distractor_count,
        "distractor_repeat": args.distractor_repeat,
        "cache_salt": args.cache_salt,
        "samples": samples,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument("--distractor-repeat", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--cache-salt", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if any(value <= 0 for value in args.concurrency) or args.waves <= 0:
        parser.error("Concurrency and wave counts must be positive.")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}))


if __name__ == "__main__":
    main()
