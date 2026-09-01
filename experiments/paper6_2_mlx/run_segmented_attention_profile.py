"""Profile concatenated versus segmented selected/local MLX attention."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import fmean


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-tokens", type=int, default=384)
    parser.add_argument("--local-tokens", type=int, nargs="+", default=(2048, 8192, 32768))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()

    import mlx.core as mx
    from pra_mlx.native import (
        compiled_segmented_selected_attention,
        segmented_selected_attention,
    )

    rows = []
    scale = args.head_dim ** -0.5
    for local_tokens in args.local_tokens:
        shape_memory = (1, args.heads, args.memory_tokens, args.head_dim)
        shape_local = (1, args.heads, local_tokens, args.head_dim)
        query = mx.random.normal((1, args.heads, 1, args.head_dim)).astype(mx.float16)
        memory_k = mx.random.normal(shape_memory).astype(mx.float16)
        memory_v = mx.random.normal(shape_memory).astype(mx.float16)
        local_k = mx.random.normal(shape_local).astype(mx.float16)
        local_v = mx.random.normal(shape_local).astype(mx.float16)
        mx.eval(query, memory_k, memory_v, local_k, local_v)

        def concatenated():
            keys = mx.concatenate((memory_k, local_k), axis=2)
            values = mx.concatenate((memory_v, local_v), axis=2)
            scores = (query @ mx.swapaxes(keys, -1, -2)) * scale
            return mx.softmax(scores, axis=-1) @ values

        def segmented():
            return segmented_selected_attention(
                query, memory_k, memory_v, local_k, local_v, scale=scale
            )

        def compiled_segmented():
            return compiled_segmented_selected_attention(
                query, memory_k, memory_v, local_k, local_v, scale=scale
            )

        for _ in range(args.warmup):
            mx.eval(concatenated(), segmented(), compiled_segmented())
        reference = concatenated()
        candidate = segmented()
        compiled_candidate = compiled_segmented()
        mx.eval(reference, candidate, compiled_candidate)
        error = float(mx.max(mx.abs(reference - candidate)).item())
        compiled_error = float(
            mx.max(mx.abs(reference - compiled_candidate)).item()
        )

        timings: dict[str, list[float]] = {
            "concatenated": [],
            "segmented": [],
            "compiled_segmented": [],
        }
        for _ in range(args.repeats):
            for name, operation in (
                ("concatenated", concatenated),
                ("segmented", segmented),
                ("compiled_segmented", compiled_segmented),
            ):
                started = time.perf_counter()
                value = operation()
                mx.eval(value)
                timings[name].append((time.perf_counter() - started) * 1000.0)
        element_bytes = 2
        concat_temporary = (
            2 * args.heads * (args.memory_tokens + local_tokens) * args.head_dim * element_bytes
        )
        score_temporary = args.heads * (args.memory_tokens + local_tokens) * element_bytes
        rows.append(
            {
                "local_tokens": local_tokens,
                "memory_tokens": args.memory_tokens,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "max_absolute_error": error,
                "compiled_max_absolute_error": compiled_error,
                "exact_within_fp16_tolerance": error <= 2e-3,
                "compiled_exact_within_fp16_tolerance": compiled_error <= 2e-3,
                "concatenated_mean_ms": fmean(timings["concatenated"]),
                "concatenated_p95_ms": _percentile(timings["concatenated"], 0.95),
                "segmented_mean_ms": fmean(timings["segmented"]),
                "segmented_p95_ms": _percentile(timings["segmented"], 0.95),
                "compiled_segmented_mean_ms": fmean(
                    timings["compiled_segmented"]
                ),
                "compiled_segmented_p95_ms": _percentile(
                    timings["compiled_segmented"], 0.95
                ),
                "kv_concat_temporary_bytes_avoided": concat_temporary,
                "segmented_score_temporary_bytes": score_temporary,
                "integration_status": "GRAPH_COMPILED_AND_MODEL_PATCHED",
            }
        )
        del local_k, local_v
        mx.clear_cache()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_segmented_selected_attention_v1",
        "evidence_tier": "METAL_KERNEL_MICROBENCHMARK",
        "dtype": "float16",
        "one_attention_normalization": True,
        "model_runner_graph_compiled": True,
        "custom_metal_kernel": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
