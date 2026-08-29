"""Measure deterministic lead-time overlap for SGLang-backed PRA promotion."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.paper6_1_sglang.run_builtin_hicache_backend import _storage_config
from experiments.paper6_1_sglang.run_hicache import SEEDS, _source


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Lead times must be non-negative integers.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--lead-ms", type=_integers, default=(0, 10, 25, 50))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import sglang
    from mlx_lm import load
    from pra_mlx.native import encode_native_memory
    from pra_sglang.hicache import PRAHiCacheTier, SGLangPRAHiCache
    from pra_sglang.hicache_backend import SGLangHiCacheStorageBackend
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

    model, tokenizer = load(args.model, revision=args.revision)
    rows = []
    with tempfile.TemporaryDirectory(prefix="pra-sglang-prefetch-") as root:
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(
            Path(root) / "storage"
        )
        storage = StorageBackendFactory.create_backend(
            "file", _storage_config(args.model), None
        )
        writer = SGLangHiCacheStorageBackend(storage, namespace="paper6-1-prefetch")
        for seed in SEEDS:
            tokens = tokenizer.encode(_source(seed), add_special_tokens=False)
            memory = encode_native_memory(model, tokens)
            for lead_ms in args.lead_ms:
                logical_key = f"resource-{seed}-lead-{lead_ms}-v1"
                writer.put(logical_key, memory)
                backend = SGLangHiCacheStorageBackend(
                    storage, namespace="paper6-1-prefetch"
                )
                cache = SGLangPRAHiCache(
                    Path(root) / f"cache-{seed}-{lead_ms}",
                    max_l1_bytes=memory.nbytes,
                    max_l2_bytes=memory.nbytes,
                    l3_backend=backend,
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    started = time.perf_counter()
                    completion_time: list[float] = []
                    future = cache.prefetch(
                        logical_key, executor, target=PRAHiCacheTier.L1
                    )
                    future.add_done_callback(
                        lambda _future: completion_time.append(time.perf_counter())
                    )
                    time.sleep(lead_ms / 1000.0)
                    demand_started = time.perf_counter()
                    ready_at_demand = future.done()
                    restored = future.result()
                    demand_finished = time.perf_counter()
                    demand_stall_ms = (
                        demand_finished - demand_started
                    ) * 1000.0
                    completed = completion_time[0] if completion_time else demand_finished
                    promotion_ms = (completed - started) * 1000.0
                    actual_lead_ms = (demand_started - started) * 1000.0
                mx.eval(*(layer.keys for layer in restored.layers))
                max_key_delta = max(
                    float(mx.max(mx.abs(left.keys - right.keys)).item())
                    for left, right in zip(memory.layers, restored.layers)
                )
                rows.append(
                    {
                        "seed": seed,
                        "lead_ms": lead_ms,
                        "source_tokens": len(tokens),
                        "native_kv_bytes": memory.nbytes,
                        "promotion_ms": promotion_ms,
                        "requested_lead_ms": lead_ms,
                        "actual_lead_ms": actual_lead_ms,
                        "ready_at_demand": ready_at_demand,
                        "demand_stall_ms": demand_stall_ms,
                        "observed_overlap_ms": max(0.0, promotion_ms - demand_stall_ms),
                        "stall_fraction": (
                            demand_stall_ms / promotion_ms
                            if promotion_ms
                            else 0.0
                        ),
                        "max_key_delta": max_key_delta,
                        "exact_tensor_recovery": max_key_delta == 0.0,
                        "hicache": cache.metrics().to_dict(),
                    }
                )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_builtin_hicache_prefetch_v1",
        "evidence_tier": "CONTROLLED_ORACLE_PREFETCH",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "storage_backend": "sglang_hicache_file",
        "prefetch_signal": "oracle_selection_available_before_demand",
        "lead_ms": list(args.lead_ms),
        "off_node_transport": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
