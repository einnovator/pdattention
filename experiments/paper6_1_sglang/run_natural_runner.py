"""Run natural-text causal controls through SGLang's real MLX runner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.paper6_2_mlx.run_natural_transport import SEEDS, _examples


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
    import sglang
    from pra_mlx.native import encode_native_memory
    from pra_sglang.mlx_native import SGLangMLXNativeBridge, SGLangSelectedKVCache
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=True,
        enable_sampling=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    answer_token_ids = {
        answer: int(tokenizer.encode(f" {answer}", add_special_tokens=False)[0])
        for answer in ("yes", "no")
    }
    model_args = getattr(runner.model, "args", None)
    vocabulary_size = max(
        len(tokenizer),
        int(getattr(model_args, "vocab_size", 0)),
        max(answer_token_ids.values()) + 1,
    )
    answer_mask = mx.full((vocabulary_size,), -mx.inf, dtype=mx.float32)
    for token_id in answer_token_ids.values():
        answer_mask[token_id] = 0.0

    prepared = []
    for seed in SEEDS:
        examples = list(
            _examples(args.dataset, seed, args.examples_per_seed, args.cache_dir)
        )
        seed_rows = []
        for example in examples:
            source_text = " ".join(unit.text for unit in example.source_units)
            source = tokenizer.encode(source_text, add_special_tokens=False)
            query = tokenizer.encode(example.question, add_special_tokens=False)
            seed_rows.append(
                (example, source, query, encode_native_memory(runner.model, source))
            )
        prepared.append((seed, seed_rows))

    bridge = SGLangMLXNativeBridge(runner)
    rows = []
    try:
        for seed, examples in prepared:
            memories = [row[3] for row in examples]
            for index, (example, source, query, memory) in enumerate(examples):
                shuffled = memories[(index + 1) % len(memories)]
                conditions = (
                    ("ordinary_full", source + query, None),
                    ("native", query, memory),
                    ("disabled", query, None),
                    ("shuffled", query, shuffled),
                )
                for condition, prompt, active_memory in conditions:
                    req_id = f"{seed}-{index}-{condition}"
                    if active_memory is not None:
                        bridge.register(
                            req_id,
                            active_memory,
                            logical_keys=(f"{condition}-{example.id}",),
                        )
                    started = time.perf_counter()
                    pending = runner.prefill_start(
                        req_id,
                        prompt,
                        prompt,
                        [],
                        [],
                        0,
                        logit_edit_row=answer_mask,
                    )
                    runner.eval_pending(pending)
                    token = runner.prefill_finalize(pending)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    ranked_answer = next(
                        answer
                        for answer, token_id in answer_token_ids.items()
                        if token_id == token
                    )
                    cache = runner._req_caches[req_id][
                        runner._cache_layout.first_attention_layer_index
                    ]
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "seed": seed,
                            "example_id": example.id,
                            "condition": condition,
                            "gold_answer": example.answer,
                            "ranked_answer": ranked_answer,
                            "ranked_exact": ranked_answer == example.answer,
                            "visible_prompt_tokens": len(prompt),
                            "selected_native_tokens": (
                                0 if active_memory is None else active_memory.source_tokens
                            ),
                            "active_native_kv_bytes": (
                                0 if active_memory is None else active_memory.nbytes
                            ),
                            "completion_latency_ms": latency_ms,
                            "scheduler_counts_exclude_pra": (
                                cache.offset == len(query)
                                if isinstance(cache, SGLangSelectedKVCache)
                                else None
                            ),
                        }
                    )
                    runner.remove_request(req_id)
                    bridge.unregister(req_id)
    finally:
        bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_natural_runner_v1",
        "evidence_tier": "NATURAL_TEXT_TRANSPORT_CONTROL",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "engine_revision": "ef20fab38a03490e2cdf1b7377145ca3a3f2bfc5",
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "seeds": list(SEEDS),
        "examples_per_seed": args.examples_per_seed,
        "answer_token_ids": answer_token_ids,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
