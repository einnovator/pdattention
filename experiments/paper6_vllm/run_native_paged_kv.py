"""Measure selected PRA K/V on the vLLM-Metal paged-attention primitive."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
TOKEN_COUNTS = (16, 64, 256, 512)
CONCURRENCY = (1, 2, 4, 8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from pra_mlx.native import MLXNativeLayerKV
    from pra_vllm.metal_native import VLLMMetalPRAStore

    rows = []
    for seed in SEEDS:
        for tokens in TOKEN_COUNTS:
            for concurrency in CONCURRENCY:
                mx.random.seed(seed)
                hq, hkv, dim = 16, 8, 128
                keys = mx.random.normal((1, hkv, tokens, dim)).astype(mx.float16)
                values = mx.random.normal((1, hkv, tokens, dim)).astype(mx.float16)
                layer = MLXNativeLayerKV(keys, values)
                blocks = (tokens + 15) // 16
                store = VLLMMetalPRAStore(
                    num_layers=1,
                    num_kv_heads=hkv,
                    head_dim=dim,
                    num_blocks=blocks,
                )
                handle = store.materialize(f"seed-{seed}-tokens-{tokens}", (layer,))
                queries = mx.random.normal((concurrency, hq, dim)).astype(mx.float16)
                scale = dim**-0.5
                cold_started = time.perf_counter()
                cold_output = store.attend(
                    0, queries, (handle,) * concurrency, scale=scale
                )
                mx.eval(cold_output)
                cold_elapsed_ms = (time.perf_counter() - cold_started) * 1000.0
                started = time.perf_counter()
                output = store.attend(
                    0, queries, (handle,) * concurrency, scale=scale
                )
                mx.eval(output)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                repeated_k = mx.repeat(keys[0].transpose(1, 0, 2), hq // hkv, axis=1)
                repeated_v = mx.repeat(values[0].transpose(1, 0, 2), hq // hkv, axis=1)
                scores = mx.einsum("bhd,thd->bht", queries * scale, repeated_k)
                reference = mx.einsum(
                    "bht,thd->bhd", mx.softmax(scores.astype(mx.float32), axis=-1), repeated_v
                ).astype(mx.float16)
                mx.eval(reference)
                error = mx.abs(output - reference)
                duplicate_bytes = handle.byte_count * concurrency
                rows.append(
                    {
                        "seed": seed,
                        "selected_tokens": tokens,
                        "logical_tokens": 8192,
                        "selected_fraction": tokens / 8192,
                        "concurrency": concurrency,
                        "max_error": float(mx.max(error).item()),
                        "mean_error": float(mx.mean(error).item()),
                        "cold_paged_attention_ms": cold_elapsed_ms,
                        "warm_paged_attention_ms": elapsed_ms,
                        "paged_attention_ms": elapsed_ms,
                        "resident_selected_kv_bytes": handle.byte_count,
                        "physical_cache_capacity_bytes": store.physical_capacity_bytes,
                        "shared_physical_bytes": handle.byte_count,
                        "duplicate_physical_bytes": duplicate_bytes,
                        "sharing_bytes_saved": duplicate_bytes - handle.byte_count,
                        "native_kv_used": True,
                    }
                )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_metal_native_paged_kv_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "vllm-metal",
        "engine_version": "0.3.0.dev20260828085745",
        "engine_revision": "14705ad974863f68d00315655514f200366441bf",
        "native_pra_status": "MEASURED_KERNEL_PATH",
        "seeds": list(SEEDS),
        "token_counts": list(TOKEN_COUNTS),
        "concurrency": list(CONCURRENCY),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
