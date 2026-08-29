"""Measure external PRA L1/L2/L3 placement and promotion on SGLang MLX."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
EXPECTED = "7391"


def _source(seed: int) -> str:
    words = "archive telemetry routine status background note maintenance".split()
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


def _timed_get(cache, key: str):
    started = time.perf_counter()
    memory = cache.get(key)
    return memory, (time.perf_counter() - started) * 1000.0


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
    from pra_sglang.hicache import PRAHiCacheTier, SGLangPRAHiCache
    from pra_sglang.mlx_native import (
        SGLangSelectedKVCache,
        install_selected_kv_attention,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    layer_count = len(getattr(getattr(model, "model", model), "layers"))
    patched_layers = install_selected_kv_attention(model)
    query = tokenizer.encode(_query(), add_special_tokens=False)
    rows = []
    with tempfile.TemporaryDirectory(prefix="pra-sglang-hicache-") as root:
        for seed in SEEDS:
            tokens = tokenizer.encode(_source(seed), add_special_tokens=False)
            memory = encode_native_memory(model, tokens)
            cache = SGLangPRAHiCache(
                Path(root) / str(seed),
                max_l1_bytes=memory.nbytes,
                max_l2_bytes=memory.nbytes,
            )
            key = f"resource-{seed}"

            cache.put(key, memory, tier=PRAHiCacheTier.L1)
            _, l1_hit_ms = _timed_get(cache, key)

            cache.remove(key)
            cache.put(key, memory, tier=PRAHiCacheTier.L2)
            _, l2_to_l1_ms = _timed_get(cache, key)

            cache.remove(key)
            cache.put(key, memory, tier=PRAHiCacheTier.L3)
            selected, l3_to_l1_ms = _timed_get(cache, key)
            _, warm_l1_ms = _timed_get(cache, key)

            native = [
                SGLangSelectedKVCache(local, layer, position_base=selected.source_tokens)
                for local, layer in zip(_sglang_cache(layer_count), selected.layers)
            ]
            output, generation_ms = _generate(model, tokenizer, query, native)
            mx.eval(*(layer.keys for layer in selected.layers))
            metrics = cache.metrics()
            rows.append(
                {
                    "seed": seed,
                    "source_tokens": len(tokens),
                    "active_native_kv_bytes": selected.nbytes,
                    "l1_hit_ms": l1_hit_ms,
                    "l2_to_l1_ms": l2_to_l1_ms,
                    "l3_to_l1_ms": l3_to_l1_ms,
                    "warm_l1_ms": warm_l1_ms,
                    "generation_ms": generation_ms,
                    "output": output,
                    "exact_recovery": EXPECTED in output,
                    "scheduler_local_tokens_after_generation": native[0].offset,
                    "pra_tokens_absent_from_radix_prefix": (
                        native[0].keys is native[0].local_cache.keys
                    ),
                    "hicache": metrics.to_dict(),
                }
            )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_external_pra_hicache_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "hardware_note": (
            "Apple unified memory: L1/L2 denote attention-ready versus host-array "
            "ownership, not physically separate GPU and CPU DRAM."
        ),
        "ordinary_radix_namespace_used": False,
        "patched_layers": patched_layers,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
