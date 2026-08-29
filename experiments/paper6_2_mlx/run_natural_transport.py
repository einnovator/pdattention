"""Evaluate native MLX memory on QASPER and HotpotQA text transport probes."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
ANSWER = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _answer(text: str) -> str | None:
    match = ANSWER.search(text)
    return match.group(1).lower() if match else None


def _generate(model, tokenizer, query, cache, max_tokens=6):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    generated = []
    text = ""
    for token, _ in generate_step(
        mx.array(query, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0),
    ):
        generated.append(int(token))
        text = tokenizer.decode(generated)
    return text, (time.perf_counter() - started) * 1000.0


def _examples(dataset: str, seed: int, count: int, cache_dir: Path):
    from data.native_kv_benchmarks import (
        hotpotqa_native_kv_examples,
        load_qasper_papers,
        qasper_native_kv_examples,
    )

    if dataset == "qasper":
        papers = load_qasper_papers("validation", cache_dir=cache_dir)
        return qasper_native_kv_examples(papers, max_examples=count, seed=seed)
    from datasets import load_dataset

    rows = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        cache_dir=str(cache_dir),
    )
    return hotpotqa_native_kv_examples(rows, max_examples=count, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("qasper", "hotpotqa"), required=True)
    parser.add_argument("--examples-per-seed", type=int, default=8)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import encode_native_memory, make_native_prompt_cache

    model, tokenizer = load(args.model, revision=args.revision)
    layer_count = len(getattr(getattr(model, "model", model), "layers"))
    last_half = tuple(range(layer_count // 2, layer_count))
    answer_token_ids = {
        answer: int(tokenizer.encode(f" {answer}", add_special_tokens=False)[0])
        for answer in ("yes", "no")
    }
    rows = []
    for seed in SEEDS:
        examples = list(_examples(args.dataset, seed, args.examples_per_seed, args.cache_dir))
        random.Random(seed).shuffle(examples)
        memories = []
        prepared = []
        for example in examples:
            source_text = " ".join(unit.text for unit in example.source_units)
            source = tokenizer.encode(source_text, add_special_tokens=False)
            query = tokenizer.encode(example.question, add_special_tokens=False)
            memories.append(encode_native_memory(model, source))
            prepared.append((example, source, query))
        for index, ((example, source, query), memory) in enumerate(zip(prepared, memories)):
            shuffled_memory = memories[(index + 1) % len(memories)]
            conditions = (
                ("ordinary_split", memory),
                ("native_all", memory),
                (
                    "native_last_half",
                    memory,
                ),
                (
                    "native_disabled",
                    memory,
                ),
                (
                    "native_shuffled",
                    shuffled_memory,
                ),
            )

            def cache_for(condition: str, active_memory):
                if condition == "ordinary_split":
                    cache = make_prompt_cache(model)
                    model(mx.array(source, dtype=mx.int32)[None], cache=cache)
                    return cache
                if condition == "native_last_half":
                    return make_native_prompt_cache(
                        model, active_memory, selected_layers=last_half
                    )
                if condition == "native_disabled":
                    return make_native_prompt_cache(model, active_memory, selected_layers=())
                return make_native_prompt_cache(model, active_memory)

            for condition, active_memory in conditions:
                logits = model(
                    mx.array(query, dtype=mx.int32)[None],
                    cache=cache_for(condition, active_memory),
                )[:, -1, :]
                mx.eval(logits)
                scores = {
                    answer: float(logits[0, token_id].item())
                    for answer, token_id in answer_token_ids.items()
                }
                ranked_answer = max(scores, key=scores.get)
                text, latency_ms = _generate(
                    model,
                    tokenizer,
                    query,
                    cache_for(condition, active_memory),
                )
                predicted = _answer(text)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "seed": seed,
                        "example_id": example.id,
                        "condition": condition,
                        "gold_answer": example.answer,
                        "predicted_answer": predicted,
                        "exact": predicted == example.answer,
                        "ranked_answer": ranked_answer,
                        "ranked_exact": ranked_answer == example.answer,
                        "yes_logit": scores["yes"],
                        "no_logit": scores["no"],
                        "gold_answer_margin": scores[example.answer]
                        - scores["no" if example.answer == "yes" else "yes"],
                        "source_tokens": len(source),
                        "query_tokens": len(query),
                        "active_native_kv_bytes": (
                            0
                            if condition == "ordinary_split"
                            else active_memory.selected_nbytes(
                                () if condition == "native_disabled" else (
                                    last_half if condition == "native_last_half" else None
                                )
                            )
                        ),
                        "completion_latency_ms": latency_ms,
                        "output": text,
                    }
                )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_natural_transport_v1",
        "evidence_tier": "NATURAL_TEXT_TRANSPORT_CONTROL",
        "dataset": args.dataset,
        "model_id": args.model,
        "model_revision": args.revision,
        "answer_token_ids": answer_token_ids,
        "seeds": list(SEEDS),
        "examples_per_seed": args.examples_per_seed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
