"""Validate multi-token greedy output for host split-prefill versus native reuse."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pra_hf.subagent_mlx_native import encode_mlx_subagent_memory, make_mlx_subagent_cache
from run_mlx_live_reuse import SEEDS, shared_text


def greedy(model, initial_ids: list[int], cache: list[object], tokens: int):
    import mlx.core as mx

    started = time.perf_counter()
    logits = model(mx.array(initial_ids, dtype=mx.int32)[None], cache=cache)[0, -1]
    generated = []
    step_logits = []
    for _ in range(tokens):
        mx.eval(logits)
        step_logits.append(np.asarray(logits.astype(mx.float32)))
        token = int(mx.argmax(logits).item())
        generated.append(token)
        logits = model(mx.array([token], dtype=mx.int32)[None], cache=cache)[0, -1]
    mx.eval(logits)
    return generated, step_logits, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--shared-tokens", type=int, default=8192)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-label", default="unspecified")
    parser.add_argument("--model-revision", default="UNKNOWN")
    arguments = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(arguments.model)
    rows = []
    for seed in SEEDS:
        content = shared_text(tokenizer, arguments.shared_tokens, seed)
        source_ids = tokenizer.encode(content, add_special_tokens=False)
        query_ids = tokenizer.encode(
            "\nName the field that preserves causal order. Answer:",
            add_special_tokens=False,
        )
        host_cache = make_prompt_cache(model)
        model(mx.array(source_ids, dtype=mx.int32)[None], cache=host_cache)
        host_tokens, host_logits, host_ms = greedy(
            model, query_ids, host_cache, arguments.generation_tokens
        )

        memory = encode_mlx_subagent_memory(model, tuple(source_ids))
        native_cache = make_mlx_subagent_cache(model, memory)
        native_tokens, native_logits, native_ms = greedy(
            model, query_ids, native_cache, arguments.generation_tokens
        )
        deltas = [
            float(np.max(np.abs(host - native)))
            for host, native in zip(host_logits, native_logits)
        ]
        rows.append({
            "seed": seed,
            "source_tokens": len(source_ids),
            "query_tokens": len(query_ids),
            "generated_tokens": arguments.generation_tokens,
            "exact_sequence_match": host_tokens == native_tokens,
            "max_abs_logit_delta": max(deltas),
            "host_split_generation_ms": host_ms,
            "native_generation_ms": native_ms,
            "host_token_ids": host_tokens,
            "native_token_ids": native_tokens,
            "decoded": tokenizer.decode(native_tokens, skip_special_tokens=True),
        })
        mx.clear_cache()

    artifact = {
        "protocol": "paper9-mlx-generation-parity-v1",
        "model": arguments.model,
        "model_revision": arguments.model_revision,
        "engine": "mlx-lm",
        "hardware": arguments.hardware_label,
        "seeds": list(SEEDS),
        "shared_tokens": arguments.shared_tokens,
        "generation_tokens": arguments.generation_tokens,
        "exact_sequence_matches": sum(row["exact_sequence_match"] for row in rows),
        "max_abs_logit_delta": max(row["max_abs_logit_delta"] for row in rows),
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in artifact.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
