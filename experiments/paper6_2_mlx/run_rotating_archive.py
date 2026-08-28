"""Measure selected archive recovery behind MLX rotating sequential K/V."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
KV_SIZES = (64, 128, 256)
EXPECTED = "7391"


def _prompt(tokenizer, seed: int, *, selected_archive: bool) -> str:
    source = f"Authoritative archived fact: the verification code is {EXPECTED}."
    vocabulary = (
        "routine status nominal observation archive background telemetry "
        "maintenance report ordinary context unrelated note"
    ).split()
    shift = seed % len(vocabulary)
    rotated = vocabulary[shift:] + vocabulary[:shift]
    distractors = " ".join(rotated * 70)
    selected = f"\nSelected archive record:\n{source}\n" if selected_archive else ""
    content = (
        f"{source}\n\n{distractors}{selected}\n"
        "Question: what is the authoritative archived verification code? "
        "Return only its four digits."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _run(model, tokenizer, make_prompt_cache, make_sampler, seed, cache_size, selected):
    prompt = _prompt(tokenizer, seed, selected_archive=selected)
    cache = make_prompt_cache(model, max_kv_size=cache_size)
    started = time.perf_counter()
    responses = list(
        __import__("mlx_lm").stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=16,
            prompt_cache=cache,
            sampler=make_sampler(temp=0),
        )
    )
    elapsed = time.perf_counter() - started
    output = "".join(row.text for row in responses)
    last = responses[-1]
    return {
        "seed": seed,
        "cache_size": cache_size,
        "selected_archive": selected,
        "condition": (
            "full_sequential_kv"
            if cache_size is None
            else "rotating_plus_selected_archive" if selected else "rotating_only"
        ),
        "prompt_tokens": int(last.prompt_tokens),
        "generation_tokens": int(last.generation_tokens),
        "completion_latency_ms": elapsed * 1000.0,
        "prompt_tokens_per_second": float(last.prompt_tps),
        "generation_tokens_per_second": float(last.generation_tps),
        "peak_memory_gb": float(last.peak_memory),
        "cache_bytes": sum(int(item.nbytes) for item in cache),
        "cache_types": sorted({type(item).__name__ for item in cache}),
        "output": output,
        "exact_recovery": EXPECTED in output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    rows = []
    for seed in SEEDS:
        rows.append(
            _run(model, tokenizer, make_prompt_cache, make_sampler, seed, None, False)
        )
        for cache_size in KV_SIZES:
            rows.append(
                _run(model, tokenizer, make_prompt_cache, make_sampler, seed, cache_size, False)
            )
            rows.append(
                _run(model, tokenizer, make_prompt_cache, make_sampler, seed, cache_size, True)
            )
    payload = {
        "schema_version": "1.0",
        "experiment": "mlx_rotating_cache_selected_archive_v1",
        "model_id": args.model,
        "model_revision": "73e3e38d981303bc594367cd910ea6eb48349da8",
        "expected_answer": EXPECTED,
        "seeds": list(SEEDS),
        "kv_sizes": list(KV_SIZES),
        "evidence_tier": "CONTROLLED",
        "native_pra_status": "NOT_MEASURED",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
