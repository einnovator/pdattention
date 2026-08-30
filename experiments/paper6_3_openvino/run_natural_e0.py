"""Run the portable natural selected-text benchmark through OpenVINO GenAI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Mapping

from experiments.engine_serving.run_openai_natural_e0 import (
    _aggregate,
    _load_serving_helpers,
    _messages,
    _quality,
)


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


def _decoded_text(result: object) -> str:
    texts = getattr(result, "texts", None)
    return str(texts[0]) if texts else str(result)


def _mean(rows: list[Mapping[str, object]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return statistics.fmean(values) if values else None


def run(args: argparse.Namespace) -> Mapping[str, object]:
    import openvino_genai as genai
    from transformers import AutoTokenizer

    helpers = _load_serving_helpers()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest["entries"]
        if not args.datasets or entry["dataset"] in args.datasets
    ]
    if args.max_examples_per_dataset > 0:
        counters: Counter[str] = Counter()
        limited = []
        for entry in entries:
            dataset = str(entry["dataset"])
            if counters[dataset] >= args.max_examples_per_dataset:
                continue
            counters[dataset] += 1
            limited.append(entry)
        entries = limited

    scheduler = genai.SchedulerConfig()
    scheduler.enable_prefix_caching = True
    scheduler.dynamic_split_fuse = True
    scheduler.max_num_seqs = max(1, args.max_num_seqs)
    scheduler.max_num_batched_tokens = args.max_num_batched_tokens
    scheduler.cache_size = args.cache_size_gb
    rss_before_compile = _rss_bytes()
    compile_started = time.perf_counter()
    pipe = genai.LLMPipeline(args.model, args.device, scheduler_config=scheduler)
    compile_ms = (time.perf_counter() - compile_started) * 1000.0
    rss_after_compile = _rss_bytes()

    config = genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    rows: list[dict[str, object]] = []
    benchmark_started = time.perf_counter()
    for entry in entries:
        for condition in args.conditions:
            messages, source_tokens, selected_tokens = _messages(
                entry,
                tokenizer,
                condition,
                selected_limit=args.max_selected_tokens,
                full_limit=args.max_full_tokens,
            )
            for repeat in range(args.repeats):
                started = time.perf_counter()
                result = pipe.generate(_history(genai, messages), config)
                wall_ms = (time.perf_counter() - started) * 1000.0
                perf = result.perf_metrics
                output = _decoded_text(result)
                exact, f1, containment = _quality(output, str(entry["answer"]))
                rows.append(
                    {
                        "dataset": entry["dataset"],
                        "seed": entry["seed"],
                        "example_id": entry["example_id"],
                        "selection_id": entry["selection_id"],
                        "condition": condition,
                        "repeat": repeat,
                        "cache_state": "COLD" if repeat == 0 else "WARM_REPEAT",
                        "answer": entry["answer"],
                        "source_tokens": source_tokens,
                        "selected_source_tokens": selected_tokens,
                        "evidence_recall_at_4": entry["evidence_recall_at_4"],
                        "exact_match": exact,
                        "token_f1": f1,
                        "answer_containment": containment,
                        "output_text": output,
                        "ttft_ms": _metric(perf, "get_ttft"),
                        "mean_itl_ms": _metric(perf, "get_tpot"),
                        "completion_latency_ms": wall_ms,
                        "prompt_tokens": _metric(perf, "get_num_input_tokens"),
                        "completion_tokens": _metric(perf, "get_num_generated_tokens"),
                        "cached_tokens": None,
                        "tokenization_ms": _metric(perf, "get_tokenization_duration"),
                        "detokenization_ms": _metric(
                            perf, "get_detokenization_duration"
                        ),
                        "generation_duration_ms": _metric(
                            perf, "get_generate_duration"
                        ),
                        "rss_bytes": _rss_bytes(),
                    }
                )
    elapsed = time.perf_counter() - benchmark_started
    aggregates = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for condition in args.conditions:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            base = dict(_aggregate(helpers, selected))
            base.update(
                {
                    "mean_tokenization_ms": _mean(selected, "tokenization_ms"),
                    "mean_detokenization_ms": _mean(selected, "detokenization_ms"),
                    "mean_rss_bytes": _mean(selected, "rss_bytes"),
                }
            )
            aggregates.append(
                {"dataset": dataset, "condition": condition, **base}
            )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_cross_engine_portable_natural_e0_v1",
        "evidence_tier": "CONTROLLED_NATURAL_QA",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "engine": "openvino-genai",
        "engine_version": importlib.metadata.version("openvino-genai"),
        "model_id": str(args.model),
        "device": args.device,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "conditions": args.conditions,
        "repeats": args.repeats,
        "max_selected_tokens": args.max_selected_tokens,
        "max_full_tokens": args.max_full_tokens,
        "max_new_tokens": args.max_new_tokens,
        "compile_ms": compile_ms,
        "rss_before_compile_bytes": rss_before_compile,
        "rss_after_compile_bytes": rss_after_compile,
        "elapsed_seconds": elapsed,
        "request_throughput_s": len(rows) / max(elapsed, 1e-9),
        "aggregates": aggregates,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/portable_e0_qa_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("qasper", "hotpotqa", "2wikimultihopqa"),
        default=[],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("no_context", "selected_context", "full_context"),
        default=["no_context", "selected_context", "full_context"],
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-selected-tokens", type=int, default=384)
    parser.add_argument("--max-full-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--cache-size-gb", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}))


if __name__ == "__main__":
    main()
