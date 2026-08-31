"""Measure OpenVINO GenAI continuous batching under fixed E0 selections."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Mapping


def _load_harness():
    source = Path(__file__).resolve().parents[2] / "src" / "pra_hf" / "serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("pra_serving_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving benchmark from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric(perf: object, method: str) -> float | None:
    operation = getattr(perf, method, None)
    if operation is None:
        return None
    try:
        value = operation()
        return float(value.mean if hasattr(value, "mean") else value)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _history(genai: object, messages: list[dict[str, str]]) -> object:
    history = genai.ChatHistory()
    for message in messages:
        history.append(message)
    return history


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
    return json.loads(json.dumps(messages).replace("PRA_EVIDENCE_4821", expected)), expected


def _decode(tokenizer: object, result: object) -> str:
    generations = getattr(result, "m_generation_ids", ())
    if not generations:
        return ""
    first = generations[0]
    # OpenVINO GenAI 2026.3 exposes decoded strings here, while older
    # releases expose token-id sequences under the same field.
    return first if isinstance(first, str) else str(tokenizer.decode(first))


def _aggregate(harness: object, rows: list[Mapping[str, object]], wall_seconds: float) -> Mapping[str, object]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    ttft = values("ttft_ms")
    tpot = values("tpot_ms")
    latency = values("generation_ms")
    input_tokens = values("input_tokens")
    output_tokens = values("output_tokens")
    return {
        "sample_count": len(rows),
        "quality_success_rate": statistics.fmean(float(row["quality_ok"]) for row in rows),
        "request_throughput_s": len(rows) / wall_seconds,
        "output_throughput_tokens_s": sum(output_tokens) / wall_seconds if output_tokens else None,
        "ttft_ms_p50": harness.percentile(ttft, 0.50),
        "ttft_ms_p95": harness.percentile(ttft, 0.95),
        "ttft_ms_p99": harness.percentile(ttft, 0.99),
        "tpot_ms_p50": harness.percentile(tpot, 0.50),
        "tpot_ms_p95": harness.percentile(tpot, 0.95),
        "tpot_ms_p99": harness.percentile(tpot, 0.99),
        "generation_ms_p50": harness.percentile(latency, 0.50),
        "generation_ms_p95": harness.percentile(latency, 0.95),
        "generation_ms_p99": harness.percentile(latency, 0.99),
        "mean_input_tokens": statistics.fmean(input_tokens) if input_tokens else None,
        "mean_output_tokens": statistics.fmean(output_tokens) if output_tokens else None,
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    import openvino_genai as genai

    harness = _load_harness()
    scheduler = genai.SchedulerConfig()
    scheduler.cache_size = args.cache_size_gb
    scheduler.enable_prefix_caching = True
    scheduler.dynamic_split_fuse = True
    scheduler.max_num_seqs = max(args.concurrency)
    scheduler.max_num_batched_tokens = args.max_num_batched_tokens
    pipeline_started = time.perf_counter()
    pipe = genai.ContinuousBatchingPipeline(
        args.model,
        scheduler,
        args.device,
    )
    compile_ms = (time.perf_counter() - pipeline_started) * 1000.0
    tokenizer = pipe.get_tokenizer()

    samples: list[Mapping[str, object]] = []
    aggregates: list[Mapping[str, object]] = []
    for representation in ("pra_only", "full_context"):
        for workload in ("shared_resource", "independent_resources"):
            for concurrency in args.concurrency:
                specs = []
                for wave in range(args.waves):
                    for slot in range(concurrency):
                        number = 0 if workload == "shared_resource" else wave * concurrency + slot + 1
                        messages, expected = _request_spec(
                            harness,
                            representation,
                            number,
                            distractor_count=args.distractor_count,
                            distractor_repeat=args.distractor_repeat,
                        )
                        specs.append((wave, slot, number, messages, expected))

                unique = {spec[2]: spec for spec in specs}
                warm_histories = [_history(genai, spec[3]) for spec in unique.values()]
                warm_configs = []
                for _ in warm_histories:
                    config = genai.GenerationConfig()
                    config.max_new_tokens = args.max_tokens
                    config.do_sample = False
                    warm_configs.append(config)
                pipe.generate(warm_histories, warm_configs)

                group_rows = []
                rss_before = _rss_bytes()
                group_started = time.perf_counter()
                for wave in range(args.waves):
                    wave_specs = [spec for spec in specs if spec[0] == wave]
                    histories = [_history(genai, spec[3]) for spec in wave_specs]
                    configs = []
                    for _ in histories:
                        config = genai.GenerationConfig()
                        config.max_new_tokens = args.max_tokens
                        config.do_sample = False
                        configs.append(config)
                    results = pipe.generate(histories, configs)
                    batch_size = len(results)
                    for spec, result in zip(wave_specs, results):
                        _, slot, number, _, expected = spec
                        perf = result.perf_metrics
                        output = _decode(tokenizer, result)
                        input_tokens = _metric(perf, "get_num_input_tokens")
                        output_tokens = _metric(perf, "get_num_generated_tokens")
                        # OpenVINO GenAI 2026.3 reports input tokens per
                        # GenerationResult, but generated tokens as the batch
                        # total on every result. Normalize only the latter.
                        if output_tokens is not None:
                            output_tokens /= batch_size
                        row = {
                            "representation": representation,
                            "workload": workload,
                            "concurrency": concurrency,
                            "wave": wave,
                            "slot": slot,
                            "resource_number": number,
                            "output_text": output,
                            "quality_ok": expected in output,
                            "ttft_ms": _metric(perf, "get_ttft"),
                            "tpot_ms": _metric(perf, "get_tpot"),
                            "generation_ms": _metric(perf, "get_generate_duration"),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        }
                        group_rows.append(row)
                        samples.append(row)
                wall_seconds = time.perf_counter() - group_started
                aggregates.append(
                    {
                        "representation": representation,
                        "workload": workload,
                        "concurrency": concurrency,
                        "waves": args.waves,
                        "warmup_requests": len(unique),
                        "wall_seconds": wall_seconds,
                        "rss_before_bytes": rss_before,
                        "rss_after_bytes": _rss_bytes(),
                        **_aggregate(harness, group_rows, wall_seconds),
                    }
                )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_3_openvino_continuous_batching_v1",
        "evidence_tier": "LIVE_ENGINE_SMOKE",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "cache_state": "WARM_EXPLICITLY_PRIMED",
        "model": str(args.model),
        "device": args.device,
        "compile_ms": compile_ms,
        "concurrency": args.concurrency,
        "waves": args.waves,
        "distractor_count": args.distractor_count,
        "distractor_repeat": args.distractor_repeat,
        "samples": samples,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--waves", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--distractor-count", type=int, default=12)
    parser.add_argument("--distractor-repeat", type=int, default=28)
    parser.add_argument("--cache-size-gb", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
