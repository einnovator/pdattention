"""Validate PRA selected K/V through SGLang's native MLX cache protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
EXPECTED = "7391"


def _source(seed: int) -> str:
    words = "archive telemetry routine status background note ordinary maintenance".split()
    shift = seed % len(words)
    filler = " ".join((words[shift:] + words[:shift]) * 16)
    return f"Authoritative archived fact: verification code {EXPECTED}. {filler}"


def _query() -> str:
    return " What is the authoritative archived verification code? Four-digit answer:"


def _sglang_cache(layer_count: int):
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
        ContiguousAttentionKVCache,
    )

    return [ContiguousAttentionKVCache(max_seq_len=4096) for _ in range(layer_count)]


def _generate(model, tokenizer, prompt_tokens, cache, max_tokens=12):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    generated = []
    text = ""
    for token, _ in generate_step(
        mx.array(prompt_tokens, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0),
    ):
        generated.append(int(token))
        text = tokenizer.decode(generated)
    return text, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import sglang
    from mlx_lm import load
    from pra_mlx.native import encode_native_memory
    from pra_sglang.mlx_native import (
        SGLangSelectedKVCache,
        install_selected_kv_attention,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    layer_count = len(getattr(getattr(model, "model", model), "layers"))
    rows = []
    patched_layers = None
    for seed in SEEDS:
        source = tokenizer.encode(_source(seed), add_special_tokens=False)
        query = tokenizer.encode(_query(), add_special_tokens=False)

        ordinary = _sglang_cache(layer_count)
        model(mx.array(source, dtype=mx.int32)[None], cache=ordinary)
        ordinary_logits = model(mx.array(query, dtype=mx.int32)[None], cache=ordinary)[
            :, -1, :
        ]

        encode_started = time.perf_counter()
        memory = encode_native_memory(model, source)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        if patched_layers is None:
            patched_layers = install_selected_kv_attention(model)
        native = [
            SGLangSelectedKVCache(cache, layer, position_base=memory.source_tokens)
            for cache, layer in zip(_sglang_cache(layer_count), memory.layers)
        ]
        native_logits = model(mx.array(query, dtype=mx.int32)[None], cache=native)[
            :, -1, :
        ]
        mx.eval(ordinary_logits, native_logits)
        difference = mx.abs(ordinary_logits - native_logits)

        fresh_native = [
            SGLangSelectedKVCache(cache, layer, position_base=memory.source_tokens)
            for cache, layer in zip(_sglang_cache(layer_count), memory.layers)
        ]
        output, latency_ms = _generate(model, tokenizer, query, fresh_native)
        rows.append(
            {
                "seed": seed,
                "source_tokens": len(source),
                "visible_query_tokens": len(query),
                "active_native_kv_tokens": len(source),
                "active_native_kv_bytes": memory.nbytes,
                "native_encode_ms": encode_ms,
                "completion_latency_ms": latency_ms,
                "max_logit_error_vs_sglang_split_cache": float(
                    mx.max(difference).item()
                ),
                "mean_logit_error_vs_sglang_split_cache": float(
                    mx.mean(difference).item()
                ),
                "argmax_matches_sglang_split_cache": bool(
                    mx.argmax(ordinary_logits).item() == mx.argmax(native_logits).item()
                ),
                "scheduler_local_tokens_after_query": native[0].offset,
                "attention_rope_offset_after_query": native[0].rope_offset,
                "pra_tokens_absent_from_radix_prefix": native[0].offset == len(query),
                "output": output,
                "exact_recovery": EXPECTED in output,
                "native_kv_used": True,
            }
        )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_mlx_native_selected_kv_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "engine_revision": "ef20fab38a03490e2cdf1b7377145ca3a3f2bfc5",
        "model_id": args.model,
        "model_revision": args.revision,
        "seeds": list(SEEDS),
        "expected_answer": EXPECTED,
        "native_pra_status": "MEASURED_SGLANG_CACHE_PATH",
        "attention_semantics": "one_softmax_selected_memory_plus_local_kv",
        "patched_layers": patched_layers,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
