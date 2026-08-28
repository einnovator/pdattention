"""Validate real MLX selected-K/V semantics, reuse, and rotating-cache coexistence."""

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


def _generate(model, tokenizer, prompt_tokens, cache, max_tokens=12):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    output = []
    generated = []
    for token, _ in generate_step(
        mx.array(prompt_tokens, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0),
    ):
        generated.append(int(token))
        output.append(tokenizer.decode(generated))
    text = output[-1] if output else ""
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
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import encode_native_memory, make_native_prompt_cache

    model, tokenizer = load(args.model, revision=args.revision)
    rows = []
    for seed in SEEDS:
        source = tokenizer.encode(_source(seed), add_special_tokens=False)
        query = tokenizer.encode(_query(), add_special_tokens=False)

        ordinary_cache = make_prompt_cache(model)
        model(mx.array(source, dtype=mx.int32)[None], cache=ordinary_cache)
        ordinary_logits = model(
            mx.array(query, dtype=mx.int32)[None], cache=ordinary_cache
        )[:, -1, :]

        encode_started = time.perf_counter()
        memory = encode_native_memory(model, source)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        native_cache = make_native_prompt_cache(model, memory)
        native_logits = model(
            mx.array(query, dtype=mx.int32)[None], cache=native_cache
        )[:, -1, :]
        mx.eval(ordinary_logits, native_logits)
        difference = mx.abs(ordinary_logits - native_logits)
        max_error = float(mx.max(difference).item())
        mean_error = float(mx.mean(difference).item())

        for condition, cache in (
            ("ordinary_split_cache", make_prompt_cache(model)),
            ("native_selected_kv", make_native_prompt_cache(model, memory)),
            (
                "native_selected_kv_rotating_local_64",
                make_native_prompt_cache(model, memory, max_kv_size=64),
            ),
        ):
            if condition == "ordinary_split_cache":
                model(mx.array(source, dtype=mx.int32)[None], cache=cache)
            text, latency_ms = _generate(model, tokenizer, query, cache)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "source_tokens": len(source),
                    "visible_query_tokens": len(query),
                    "active_native_kv_tokens": (
                        0 if condition == "ordinary_split_cache" else len(source)
                    ),
                    "active_native_kv_bytes": (
                        0 if condition == "ordinary_split_cache" else memory.nbytes
                    ),
                    "native_encode_ms": (
                        0.0 if condition == "ordinary_split_cache" else encode_ms
                    ),
                    "completion_latency_ms": latency_ms,
                    "max_logit_error_vs_ordinary_split": (
                        0.0 if condition == "ordinary_split_cache" else max_error
                    ),
                    "mean_logit_error_vs_ordinary_split": (
                        0.0 if condition == "ordinary_split_cache" else mean_error
                    ),
                    "argmax_matches_ordinary_split": bool(
                        mx.argmax(ordinary_logits).item() == mx.argmax(native_logits).item()
                    ),
                    "output": text,
                    "exact_recovery": EXPECTED in text,
                    "native_kv_used": condition != "ordinary_split_cache",
                }
            )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_native_selected_kv_v1",
        "evidence_tier": "CONTROLLED",
        "model_id": args.model,
        "model_revision": args.revision,
        "seeds": list(SEEDS),
        "expected_answer": EXPECTED,
        "native_pra_status": "MEASURED",
        "attention_semantics": "one_softmax_selected_memory_plus_local_kv",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
