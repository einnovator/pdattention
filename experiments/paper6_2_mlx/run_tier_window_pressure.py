"""Cross HOT/WARM resource budgets with MLX local rotating-window sizes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import tempfile
import time

from experiments.paper6_2_mlx.run_answer_quality_pressure import (
    SEEDS,
    _bounded_source,
    _examples,
    _metrics,
)
from experiments.paper6_2_mlx.run_bounded_residency import _access_sequence
from experiments.paper6_2_mlx.run_matched_e0_e2 import _generate_timed


def _oldest_other(lru: list[str], protected: str) -> str | None:
    """Return and remove the oldest HOT key that is not the active request."""

    for index, key in enumerate(lru):
        if key != protected:
            return lru.pop(index)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), default="qasper"
    )
    parser.add_argument("--resources-per-seed", type=int, default=8)
    parser.add_argument("--session-rounds", type=int, default=3)
    parser.add_argument("--hot-resource-budgets", type=int, nargs="+", default=(2, 8))
    parser.add_argument("--warm-resource-budgets", type=int, nargs="+", default=(2, 8))
    parser.add_argument("--local-kv-sizes", type=int, nargs="+", default=(64, 256))
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.hot_resource_budgets, *args.warm_resource_budgets) < 1:
        raise ValueError("HOT and WARM resource budgets must be positive.")
    if min(args.local_kv_sizes) < args.max_new_tokens:
        raise ValueError("Local K/V windows must fit the generated continuation.")

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
        deserialize_native_memory,
        encode_native_memory,
        make_native_prompt_cache,
        serialize_native_memory,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    candidates = _examples(args.dataset, args.cache_dir)
    sequence = _access_sequence(args.resources_per_seed, args.session_rounds)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="pra-mlx-tier-window-") as directory:
        root = Path(directory)
        for seed in SEEDS:
            cohort = list(candidates)
            random.Random(seed).shuffle(cohort)
            cohort = cohort[: args.resources_per_seed]
            prepared = []
            for index, example in enumerate(cohort):
                source = _bounded_source(tokenizer, example.source, args.max_source_tokens)
                query = list(tokenizer.encode(example.question, add_special_tokens=False))
                memory = encode_native_memory(model, source)
                payload = serialize_native_memory(memory)
                prepared.append(
                    {
                        "example": example,
                        "source": source,
                        "query": query,
                        "memory": memory,
                        "payload": payload,
                        "key": f"{args.dataset}-{seed}-{index}",
                    }
                )
            max_payload_bytes = max(len(item["payload"]) for item in prepared)
            max_native_bytes = max(item["memory"].nbytes for item in prepared)

            for hot_budget in args.hot_resource_budgets:
                for warm_budget in args.warm_resource_budgets:
                    for local_kv_size in args.local_kv_sizes:
                        warm_root = root / f"{seed}-{hot_budget}-{warm_budget}-{local_kv_size}"
                        manager = PRAStorageManager(
                            PRAStoragePolicy(
                                profile="mlx-tier-window-pressure",
                                hot=PRAStorageTierConfig(
                                    max_bytes=args.resources_per_seed * max_native_bytes
                                ),
                                warm=PRAStorageTierConfig(
                                    path=str(warm_root),
                                    max_bytes=warm_budget * max_payload_bytes,
                                    representation="mmap",
                                    cold_grace_seconds=0,
                                ),
                                cold=PRAStorageTierConfig(enabled=False),
                            ),
                            hot=DecodingHotBridge(deserialize_native_memory),
                            warm=MLXNativeSegmentStore(warm_root),
                        )
                        payloads = {str(item["key"]): bytes(item["payload"]) for item in prepared}
                        for item in prepared:
                            key = str(item["key"])
                            source = tuple(item["source"])

                            def rebuild(source_tokens=source):
                                return serialize_native_memory(
                                    encode_native_memory(model, source_tokens)
                                )

                            manager.register(
                                PRAStorageEntry(
                                    logical_key=key,
                                    record_type="generic_document",
                                    retention_class=PRARetentionClass.RECONSTRUCTABLE,
                                    tenant_id="benchmark",
                                    session_id=f"seed-{seed}",
                                    task_id=None,
                                    task_status=None,
                                    resource_version="1",
                                    detail_bytes=item["memory"].nbytes,
                                ),
                                payloads[key],
                                hot_value=item["memory"],
                                source_loader=rebuild,
                                fingerprint=f"{args.model}:{len(item['memory'].layers)}",
                            )
                        for item in prepared:
                            key = str(item["key"])
                            if manager.entries[key].current_tier is PRAStorageTier.HOT:
                                manager.demote_hot(key, payload=payloads[key])

                        hot_lru: list[str] = []
                        before = manager.metrics.to_dict()
                        for request_index, resource_index in enumerate(sequence):
                            item = prepared[resource_index]
                            key = str(item["key"])
                            tier_before = manager.entries[key].current_tier.value
                            request_id = f"{seed}-{hot_budget}-{warm_budget}-{local_kv_size}-{request_index}"
                            started = time.perf_counter()
                            memory = manager.promote(key, request_id=request_id)
                            resolve_ms = (time.perf_counter() - started) * 1000.0
                            if key in hot_lru:
                                hot_lru.remove(key)
                            hot_lru.append(key)
                            try:
                                generated = _generate_timed(
                                    model,
                                    tokenizer,
                                    list(item["query"]),
                                    make_native_prompt_cache(
                                        model, memory, max_kv_size=local_kv_size
                                    ),
                                    args.max_new_tokens,
                                )
                            finally:
                                manager.unpin(key, request_id)
                            while len(hot_lru) > hot_budget:
                                victim = _oldest_other(hot_lru, key)
                                if victim is None:
                                    break
                                manager.demote_hot(victim, payload=payloads[victim])
                            exact, f1 = _metrics(
                                str(generated["output"]), item["example"].answer
                            )
                            rows.append(
                                {
                                    "dataset": args.dataset,
                                    "seed": seed,
                                    "hot_resource_budget": hot_budget,
                                    "warm_resource_budget": warm_budget,
                                    "local_kv_size": local_kv_size,
                                    "request_index": request_index,
                                    "resource_index": resource_index,
                                    "final_revisit": request_index == len(sequence) - 1,
                                    "tier_before": tier_before,
                                    "resolve_ms": resolve_ms,
                                    "completion_latency_ms": generated["completion_latency_ms"],
                                    "exact_match": exact,
                                    "token_f1": f1,
                                }
                            )
                        after = manager.metrics.to_dict()
                        summaries.append(
                            {
                                "seed": seed,
                                "hot_resource_budget": hot_budget,
                                "warm_resource_budget": warm_budget,
                                "local_kv_size": local_kv_size,
                                "requests": len(sequence),
                                "metrics_delta": {
                                    name: int(after[name]) - int(before[name])
                                    for name in (
                                        "promotions",
                                        "reloads",
                                        "evictions",
                                        "bytes_read",
                                        "bytes_written",
                                    )
                                },
                            }
                        )
                        manager.close()
                        del manager
                        gc.collect()
                        mx.clear_cache()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_tier_window_pressure_v1",
        "evidence_tier": "CONTROLLED_NATURAL_QA_PRESSURE",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "seeds": list(SEEDS),
        "resources_per_seed": args.resources_per_seed,
        "session_rounds": args.session_rounds,
        "hot_resource_budgets": args.hot_resource_budgets,
        "warm_resource_budgets": args.warm_resource_budgets,
        "local_kv_sizes": args.local_kv_sizes,
        "rows": rows,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
