"""Measure TensorRT-LLM E0 concurrency with shared and independent resources."""

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
from typing import Any, Mapping


def _load_harness():
    source = Path(__file__).resolve().parents[2] / "src" / "pra_hf" / "serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("pra_serving_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving benchmark from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gpu_memory() -> Mapping[str, object]:
    device_command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(device_command, text=True, timeout=15).splitlines()[0]
        used, total = (int(value.strip()) for value in line.split(","))
        return {"available": True, "device_used_mib": used, "device_total_mib": total}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "error": str(error)}


def _request_spec(harness: object, representation: str, resource_number: int) -> tuple[list[dict[str, str]], str]:
    messages = harness.benchmark_messages(distractor_count=6, distractor_repeat=12)[representation]
    expected = f"PRA_EVIDENCE_{4821 + resource_number}"
    text = json.dumps(messages).replace("PRA_EVIDENCE_4821", expected)
    return json.loads(text), expected


def _aggregate(harness: object, rows: list[Mapping[str, object]], elapsed: float) -> Mapping[str, object]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    ttft = values("ttft_ms")
    latency = values("completion_latency_ms")
    output_tokens = values("completion_tokens")
    cached = values("cached_tokens")
    return {
        "sample_count": len(rows),
        "quality_success_rate": statistics.fmean(float(row["quality_ok"]) for row in rows),
        "request_throughput_s": len(rows) / elapsed,
        "output_throughput_tokens_s": sum(output_tokens) / elapsed if output_tokens else None,
        "ttft_ms_p50": harness.percentile(ttft, 0.50),
        "ttft_ms_p95": harness.percentile(ttft, 0.95),
        "ttft_ms_p99": harness.percentile(ttft, 0.99),
        "completion_latency_ms_p50": harness.percentile(latency, 0.50),
        "completion_latency_ms_p95": harness.percentile(latency, 0.95),
        "completion_latency_ms_p99": harness.percentile(latency, 0.99),
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
                        resource_number = 0 if workload == "shared_resource" else wave * concurrency + slot + 1
                        messages, expected = _request_spec(harness, representation, resource_number)
                        salt = hashlib.sha256(
                            f"paper6-4:tenant-a:resource-{resource_number}".encode("utf-8")
                        ).hexdigest()
                        specs.append((wave, slot, resource_number, messages, expected, salt))

                # Prime every distinct logical resource outside the timed
                # window so all concurrency rows represent warm execution.
                unique_specs = {
                    spec[2]: spec for spec in specs
                }
                for _, _, _, messages, _, salt in unique_specs.values():
                    harness.stream_chat_completion(
                        args.base_url,
                        model=args.model,
                        messages=messages,
                        timeout_seconds=args.timeout_seconds,
                        cache_salt=salt,
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
                            cache_salt=salt,
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
        "benchmark": "paper6_4_tensorrt_llm_e0_concurrency_v1",
        "evidence_tier": "LIVE_ENGINE_SMOKE",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "model_id": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "waves": args.waves,
        "cache_state": "WARM_EXPLICITLY_PRIMED",
        "max_tokens": args.max_tokens,
        "samples": samples,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--waves", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if any(value <= 0 for value in args.concurrency) or args.waves <= 0:
        parser.error("Concurrency and wave counts must be positive.")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
