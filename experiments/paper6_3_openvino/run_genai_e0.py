"""Run the common selected-text/prefix workload through OpenVINO GenAI."""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def _load_harness():
    """Load the stdlib benchmark helpers without importing the Torch SDK."""

    source = Path(__file__).resolve().parents[2] / "src" / "pra_hf" / "serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("pra_serving_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving benchmark from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric(perf: object, method: str, *, statistic: str = "mean") -> float | None:
    operation = getattr(perf, method, None)
    if operation is None:
        return None
    try:
        value = operation()
        if hasattr(value, statistic):
            value = getattr(value, statistic)
        return float(value)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _integer_metric(perf: object, method: str) -> int | None:
    value = _metric(perf, method)
    return None if value is None else int(value)


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


def _decoded_text(result: object) -> str:
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result)


def _sample(pipe: object, genai: object, messages: list[dict[str, str]], max_tokens: int) -> Mapping[str, object]:
    config = genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.do_sample = False
    started = time.perf_counter()
    result = pipe.generate(_history(genai, messages), config)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    perf = result.perf_metrics
    text = _decoded_text(result)
    return {
        "output_text": text,
        "expected_answer_present": "PRA_EVIDENCE_4821" in text,
        "wall_latency_ms": elapsed_ms,
        "ttft_ms": _metric(perf, "get_ttft"),
        "tpot_ms": _metric(perf, "get_tpot"),
        "throughput_tokens_s": _metric(perf, "get_throughput"),
        "generate_duration_ms": _metric(perf, "get_generate_duration"),
        "tokenization_duration_ms": _metric(perf, "get_tokenization_duration"),
        "detokenization_duration_ms": _metric(perf, "get_detokenization_duration"),
        "input_tokens": _integer_metric(perf, "get_num_input_tokens"),
        "generated_tokens": _integer_metric(perf, "get_num_generated_tokens"),
        "rss_bytes": _rss_bytes(),
    }


def _aggregate(
    condition: str,
    rows: list[Mapping[str, object]],
    *,
    percentile,
) -> Mapping[str, object]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    ttft = values("ttft_ms")
    latency = values("wall_latency_ms")
    return {
        "condition": condition,
        "sample_count": len(rows),
        "quality_success_rate": statistics.fmean(
            float(bool(row["expected_answer_present"])) for row in rows
        ),
        "cold_ttft_ms": rows[0].get("ttft_ms"),
        "warm_ttft_ms_mean": statistics.fmean(ttft[1:]) if len(ttft) > 1 else None,
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "ttft_ms_p99": percentile(ttft, 0.99),
        "completion_latency_ms_p50": percentile(latency, 0.50),
        "completion_latency_ms_p95": percentile(latency, 0.95),
        "completion_latency_ms_p99": percentile(latency, 0.99),
        "mean_input_tokens": (
            statistics.fmean(values("input_tokens")) if values("input_tokens") else None
        ),
        "mean_rss_bytes": statistics.fmean(values("rss_bytes")) if values("rss_bytes") else None,
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    import openvino_genai as genai

    harness = _load_harness()

    scheduler = genai.SchedulerConfig()
    scheduler.enable_prefix_caching = args.prefix_caching
    scheduler.dynamic_split_fuse = args.dynamic_split_fuse
    scheduler.use_cache_eviction = args.cache_eviction
    scheduler.max_num_seqs = args.max_num_seqs
    scheduler.max_num_batched_tokens = args.max_num_batched_tokens
    if args.num_kv_blocks is not None:
        scheduler.num_kv_blocks = args.num_kv_blocks
    elif args.cache_size_gb is not None:
        scheduler.cache_size = args.cache_size_gb

    before_compile = _rss_bytes()
    compiled_at = time.perf_counter()
    pipe = genai.LLMPipeline(
        args.model,
        args.device,
        scheduler_config=scheduler,
    )
    compile_ms = (time.perf_counter() - compiled_at) * 1000.0
    after_compile = _rss_bytes()

    samples: list[Mapping[str, object]] = []
    conditions = harness.benchmark_messages()
    for condition, messages in conditions.items():
        for repeat in range(args.repeats):
            values = _sample(pipe, genai, messages, args.max_tokens)
            samples.append({"condition": condition, "repeat": repeat, **values})
    aggregates = [
        _aggregate(
            condition,
            [sample for sample in samples if sample["condition"] == condition],
            percentile=harness.percentile,
        )
        for condition in conditions
    ]
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_3_openvino_genai_e0_v1",
        "evidence_tier": "LIVE_ENGINE_SMOKE",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("openvino", "openvino-genai")
        },
        "model": str(args.model),
        "device": args.device,
        "scheduler": {
            "prefix_caching": args.prefix_caching,
            "dynamic_split_fuse": args.dynamic_split_fuse,
            "cache_eviction": args.cache_eviction,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "cache_size_gb": args.cache_size_gb,
            "num_kv_blocks": args.num_kv_blocks,
        },
        "compile_ms": compile_ms,
        "rss_before_compile_bytes": before_compile,
        "rss_after_compile_bytes": after_compile,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "samples": samples,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--cache-size-gb", type=int, default=1)
    parser.add_argument("--num-kv-blocks", type=int)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-split-fuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-eviction", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to separate cold and warm requests")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
