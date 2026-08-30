"""Measure live HOT/WARM/COLD MLX requests under shared and independent load."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import tempfile
import time

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source, _metrics
from experiments.paper6_2_mlx.run_matched_e0_e2 import _generate_timed


def _percentile(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for small load waves."""

    if not values:
        raise ValueError("Cannot summarize an empty latency cohort.")
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[rank])


def _concurrency_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("concurrency must contain positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--concurrency", type=_concurrency_values, default=(1, 2, 4, 8, 16)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples < max(args.concurrency):
        raise ValueError("max-examples must cover the largest independent wave")

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from pra_hf.storage_lifecycle import (
        DecodingHotBridge,
        PRARetentionClass,
        PRAStorageEntry,
        PRAStorageManager,
        PRAStoragePolicy,
        PRAStorageTier,
        PRAStorageTierConfig,
    )
    from pra_mlx import MLXNativeSegmentStore
    from pra_mlx.native import (
        MLXNativeColdCodec,
        deserialize_native_memory,
        encode_native_memory,
        make_native_prompt_cache,
        serialize_native_memory,
    )

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    examples = examples[: args.max_examples]
    model, tokenizer = load(args.model, revision=args.revision)

    with tempfile.TemporaryDirectory(prefix="pra-mlx-concurrency-") as directory:
        root = Path(directory)
        manager = PRAStorageManager(
            PRAStoragePolicy(
                profile="mlx-live-storage-concurrency",
                hot=PRAStorageTierConfig(max_bytes=8 * 1024**3),
                warm=PRAStorageTierConfig(
                    path=str(root / "warm"),
                    max_bytes=8 * 1024**3,
                    representation="mmap",
                    cold_grace_seconds=0,
                ),
                cold=PRAStorageTierConfig(
                    path=str(root / "cold"),
                    max_bytes=8 * 1024**3,
                    kv_quantization="int8",
                ),
            ),
            hot=DecodingHotBridge(deserialize_native_memory),
            warm=MLXNativeSegmentStore(root / "warm"),
            cold_codec=MLXNativeColdCodec(),
        )
        prepared = []
        for example in examples:
            source = _bounded_source(
                tokenizer, example.selected_source, args.max_source_tokens
            )
            query = list(tokenizer.encode(example.question, add_special_tokens=False))
            memory = encode_native_memory(model, source)
            payload = serialize_native_memory(memory)
            key = f"concurrency-{example.dataset}-{example.example_id}"
            manager.register(
                PRAStorageEntry(
                    logical_key=key,
                    record_type="generic_document",
                    retention_class=PRARetentionClass.RECONSTRUCTABLE,
                    tenant_id="benchmark",
                    session_id="concurrent-storage",
                    task_id=None,
                    task_status=None,
                    resource_version=example.selected_source_sha256,
                    detail_bytes=memory.nbytes,
                ),
                payload,
                hot_value=memory,
                fingerprint=f"{args.model}:{len(memory.layers)}:{len(source)}",
            )
            prepared.append(
                {
                    "example": example,
                    "key": key,
                    "query": query,
                    "payload": payload,
                    "native_bytes": memory.nbytes,
                }
            )

        baseline_outputs: dict[str, str] = {}
        for index, item in enumerate(prepared):
            key = str(item["key"])
            request_id = f"baseline-{index}"
            active = manager.promote(key, request_id=request_id)
            try:
                generated = _generate_timed(
                    model,
                    tokenizer,
                    list(item["query"]),
                    make_native_prompt_cache(model, active),
                    args.max_new_tokens,
                )
            finally:
                manager.unpin(key, request_id)
            baseline_outputs[key] = str(generated["output"])
        rows = []
        try:
            for workload in ("shared_resource", "independent_resources"):
                for tier in ("hot", "warm", "cold_int8"):
                    for concurrency in args.concurrency:
                        selected = (
                            [prepared[0]] * concurrency
                            if workload == "shared_resource"
                            else prepared[:concurrency]
                        )
                        unique = {str(item["key"]): item for item in selected}
                        for key, item in unique.items():
                            entry = manager.entries[key]
                            if entry.current_tier is not PRAStorageTier.HOT:
                                manager.promote(key)
                            if tier != "hot":
                                manager.demote_hot(key, payload=bytes(item["payload"]))
                        if tier == "cold_int8":
                            manager.run_maintenance(
                                now_ns=time.time_ns() + 8 * 86400 * 1_000_000_000
                            )

                        before = manager.metrics.to_dict()
                        wave_started = time.perf_counter()

                        def run_one(index_and_item):
                            index, item = index_and_item
                            key = str(item["key"])
                            request_id = (
                                f"{workload}-{tier}-{concurrency}-{index}"
                            )
                            started = time.perf_counter()
                            active = manager.promote(key, request_id=request_id)
                            promotion_ms = (time.perf_counter() - started) * 1000.0
                            try:
                                generated = _generate_timed(
                                    model,
                                    tokenizer,
                                    list(item["query"]),
                                    make_native_prompt_cache(model, active),
                                    args.max_new_tokens,
                                )
                            finally:
                                manager.unpin(key, request_id)
                            request_ms = (time.perf_counter() - started) * 1000.0
                            example = item["example"]
                            output = str(generated["output"])
                            exact, f1 = _metrics(output, example.answer)
                            return {
                                "key": key,
                                "request_ms": request_ms,
                                "promotion_ms": promotion_ms,
                                "output": output,
                                "exact_match": exact,
                                "token_f1": f1,
                            }

                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            requests = list(pool.map(run_one, enumerate(selected)))
                        wave_ms = (time.perf_counter() - wave_started) * 1000.0
                        after = manager.metrics.to_dict()
                        request_latencies = [float(row["request_ms"]) for row in requests]
                        promotion_latencies = [
                            float(row["promotion_ms"]) for row in requests
                        ]
                        unique_bytes = sum(
                            int(item["native_bytes"]) for item in unique.values()
                        )
                        logical_bytes = sum(
                            int(item["native_bytes"]) for item in selected
                        )
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "workload": workload,
                                "tier": tier,
                                "concurrency": concurrency,
                                "requests": len(requests),
                                "unique_resources": len(unique),
                                "wave_ms": wave_ms,
                                "requests_per_second": concurrency
                                / max(wave_ms / 1000.0, 1e-9),
                                "request_p50_ms": _percentile(request_latencies, 0.50),
                                "request_p95_ms": _percentile(request_latencies, 0.95),
                                "request_p99_ms": _percentile(request_latencies, 0.99),
                                "promotion_p50_ms": _percentile(
                                    promotion_latencies, 0.50
                                ),
                                "promotion_p95_ms": _percentile(
                                    promotion_latencies, 0.95
                                ),
                                "exact_match_rate": sum(
                                    bool(row["exact_match"]) for row in requests
                                )
                                / len(requests),
                                "mean_token_f1": sum(
                                    float(row["token_f1"]) for row in requests
                                )
                                / len(requests),
                                "exact_vs_hot_rate": sum(
                                    row["output"] == baseline_outputs[row["key"]]
                                    for row in requests
                                )
                                / len(requests),
                                "logical_native_bytes": logical_bytes,
                                "unique_physical_native_bytes": unique_bytes,
                                "duplicate_physical_kv_avoided_bytes": (
                                    logical_bytes - unique_bytes
                                ),
                                "metrics_delta": {
                                    name: int(after[name]) - int(before[name])
                                    for name in (
                                        "promotions",
                                        "reloads",
                                        "bytes_read",
                                        "evictions",
                                    )
                                },
                            }
                        )
                        for key, item in unique.items():
                            if manager.entries[key].current_tier is PRAStorageTier.HOT:
                                manager.demote_hot(key, payload=bytes(item["payload"]))
                        mx.clear_cache()
        finally:
            manager.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_live_storage_concurrency_v1",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "concurrency": list(args.concurrency),
        "workloads": ["shared_resource", "independent_resources"],
        "tiers": ["hot", "warm", "cold_int8"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
