"""Exercise quantized MLX PRA residency under a hard two-resource budget."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from experiments.paper6_2_mlx.run_answer_quality_pressure import (
    SEEDS,
    _answer_logprob,
    _bounded_source,
    _examples,
    _generate,
    _metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True
    )
    parser.add_argument("--resources-per-seed", type=int, default=4)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resources_per_seed < 3:
        raise ValueError("The pressure sequence requires at least three resources.")

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from pra_hf.engine_memory import (
        LogicalPRABlock,
        LogicalPRABlockId,
        LogicalPRABlockStore,
    )
    from pra_hf.engine_residency import EnginePRAResidencyManager
    from pra_mlx.native import (
        dequantize_native_memory,
        encode_native_memory,
        make_native_prompt_cache,
        quantize_native_memory,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    candidates = _examples(args.dataset, args.cache_dir)
    rows = []
    seed_summaries = []
    for seed in SEEDS:
        import random

        cohort = list(candidates)
        random.Random(seed).shuffle(cohort)
        cohort = cohort[: args.resources_per_seed]
        prepared = []
        for example in cohort:
            source = _bounded_source(tokenizer, example.source, args.max_source_tokens)
            query_text = (
                "Answer the question using the available evidence. Give only the "
                f"short answer.\nQuestion: {example.question}\nAnswer:"
            )
            prepared.append(
                (
                    example,
                    source,
                    list(tokenizer.encode(query_text, add_special_tokens=False)),
                    list(tokenizer.encode(" " + example.answer, add_special_tokens=False)),
                )
            )

        # One measured probe establishes the physical int8 block size. The
        # manager then owns freshly materialized arrays, not prebuilt aliases.
        longest_source = max((item[1] for item in prepared), key=len)
        probe = quantize_native_memory(encode_native_memory(model, longest_source))
        block_bytes = probe.nbytes
        del probe
        gc.collect()
        mx.clear_cache()

        store = LogicalPRABlockStore()
        manager = EnginePRAResidencyManager(
            store,
            max_resident_bytes=2 * block_bytes,
            policy="lru",
        )
        keys = []
        for index, (_, source, _, _) in enumerate(prepared):
            identity = LogicalPRABlockId(
                tenant_id="paper6-2",
                session_id=f"seed-{seed}",
                resource_id=f"{args.dataset}-{seed}-{index}",
                resource_version="1",
                record_type="qa_evidence",
                token_start=0,
                token_end=len(source),
                layer=0,
                model_revision=args.revision,
                dtype="int8_per_head",
                layout="mlx_all_layers_bhtd",
                materialization_profile="all_layers",
                position_policy="source_local_post_rope",
            )
            keys.append(
                store.register(
                    LogicalPRABlock(identity, address_bytes=4 * len(source), detail_bytes=0)
                )
            )

        sequence = tuple(range(len(prepared))) + (0,)
        for request_index, resource_index in enumerate(sequence):
            example, source, query, answer = prepared[resource_index]
            key = keys[resource_index]
            store.select((key,), tenant_id="paper6-2")

            def materialize(source_tokens=tuple(source)):
                started = time.perf_counter()
                full = encode_native_memory(model, source_tokens)
                compact = quantize_native_memory(full)
                del full
                return compact, compact.nbytes

            resolve_started = time.perf_counter()
            compact = manager.resolve(
                key,
                materialize,
                request_id=f"seed-{seed}-request-{request_index}",
            )
            resolve_ms = (time.perf_counter() - resolve_started) * 1000.0
            with manager.pin_request(f"seed-{seed}-request-{request_index}", (key,)):
                decode_started = time.perf_counter()
                active = dequantize_native_memory(compact)
                dequantize_ms = (time.perf_counter() - decode_started) * 1000.0
                logprob = _answer_logprob(
                    model, query, answer, make_native_prompt_cache(model, active)
                )
                output, latency_ms = _generate(
                    model,
                    tokenizer,
                    query,
                    make_native_prompt_cache(model, active),
                    args.max_new_tokens,
                )
                exact, f1 = _metrics(output, example.answer)
                active_bytes = active.nbytes
                del active
            gc.collect()
            mx.clear_cache()
            metrics = manager.metrics()
            rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "request_index": request_index,
                    "resource_index": resource_index,
                    "revisit_after_eviction": request_index == len(sequence) - 1,
                    "source_tokens": len(source),
                    "gold_answer": example.answer,
                    "output": output,
                    "exact_match": exact,
                    "token_f1": f1,
                    "gold_answer_logprob": logprob,
                    "resolve_ms": resolve_ms,
                    "dequantize_ms": dequantize_ms,
                    "completion_latency_ms": latency_ms,
                    "compact_resident_bytes": compact.nbytes,
                    "active_materialized_bytes": active_bytes,
                    "resident_bytes_after_request": metrics.resident_bytes,
                    "resident_blocks_after_request": metrics.resident_blocks,
                    "loads_after_request": metrics.loads,
                    "evictions_after_request": metrics.evictions,
                    "reloads_after_request": metrics.reloads,
                }
            )

        metrics = manager.metrics()
        seed_summaries.append(
            {
                "seed": seed,
                "budget_bytes": manager.max_resident_bytes,
                "logical_resources": len(prepared),
                **metrics.to_dict(),
                "block_store": store.snapshot().to_dict(),
            }
        )
        manager.close()
        gc.collect()
        mx.clear_cache()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_bounded_quantized_residency_v1",
        "evidence_tier": "CONTROLLED_NATURAL_QA_PRESSURE",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "seeds": list(SEEDS),
        "resources_per_seed": args.resources_per_seed,
        "resident_resource_budget": 2,
        "access_pattern": "A-B-C-D-A",
        "quantization": "symmetric int8 per-head residency; request-local dequantization",
        "rows": rows,
        "seed_summaries": seed_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
