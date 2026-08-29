"""Measure routed natural-QA discovery through native MLX selected K/V."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments.paper6_2_mlx.routed_qa import route_qa_documents
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
    parser.add_argument("--examples-per-seed", type=int, default=4)
    parser.add_argument("--route-top-k", type=int, default=4)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import encode_native_memory, make_native_prompt_cache

    model, tokenizer = load(args.model, revision=args.revision)
    candidates = _examples(args.dataset, args.cache_dir)
    rows = []
    for seed in SEEDS:
        cohort = list(candidates)
        random.Random(seed).shuffle(cohort)
        cohort = cohort[: args.examples_per_seed]
        prepared = []
        for example in cohort:
            route = route_qa_documents(example, top_k=args.route_top_k)
            oracle_source = _bounded_source(
                tokenizer, example.source, args.max_source_tokens
            )
            routed_source = _bounded_source(
                tokenizer, route.selected_source, args.max_source_tokens
            )
            query_text = (
                "Answer the question using the available evidence. Give only the "
                f"short answer.\nQuestion: {example.question}\nAnswer:"
            )
            query = list(tokenizer.encode(query_text, add_special_tokens=False))
            answer = list(tokenizer.encode(" " + example.answer, add_special_tokens=False))
            prepared.append(
                {
                    "example": example,
                    "route": route,
                    "oracle_source": oracle_source,
                    "routed_source": routed_source,
                    "query": query,
                    "answer": answer,
                    "oracle_memory": encode_native_memory(model, oracle_source),
                    "routed_memory": encode_native_memory(model, routed_source),
                }
            )

        for index, item in enumerate(prepared):
            example = item["example"]
            route = item["route"]
            shuffled_memory = (
                prepared[(index + 1) % len(prepared)]["routed_memory"]
                if len(prepared) > 1
                else None
            )

            def cache_for(condition: str):
                if condition == "routed_ordinary":
                    cache = make_prompt_cache(model)
                    model(mx.array(item["routed_source"], dtype=mx.int32)[None], cache=cache)
                    return cache
                if condition == "oracle_native":
                    return make_native_prompt_cache(model, item["oracle_memory"])
                if condition == "routed_native":
                    return make_native_prompt_cache(model, item["routed_memory"])
                if condition == "routed_shuffled":
                    return make_native_prompt_cache(model, shuffled_memory)
                return make_prompt_cache(model)

            conditions = ["oracle_native", "routed_ordinary", "routed_native"]
            if shuffled_memory is not None:
                conditions.append("routed_shuffled")
            conditions.append("no_memory")
            for condition in conditions:
                logprob = _answer_logprob(
                    model, item["query"], item["answer"], cache_for(condition)
                )
                output, latency_ms = _generate(
                    model,
                    tokenizer,
                    item["query"],
                    cache_for(condition),
                    args.max_new_tokens,
                )
                exact, f1 = _metrics(output, example.answer)
                active_memory = (
                    item["oracle_memory"]
                    if condition == "oracle_native"
                    else shuffled_memory
                    if condition == "routed_shuffled"
                    else item["routed_memory"]
                    if condition == "routed_native"
                    else None
                )
                rows.append(
                    {
                        "dataset": example.dataset,
                        "seed": seed,
                        "example_id": example.example_id,
                        "condition": condition,
                        "gold_answer": example.answer,
                        "output": output,
                        "exact_match": exact,
                        "token_f1": f1,
                        "gold_answer_logprob": logprob,
                        "completion_latency_ms": latency_ms,
                        "candidate_documents": route.candidate_count,
                        "selected_documents": len(route.selected_document_ids),
                        "selected_document_ids": route.selected_document_ids,
                        "evidence_document_ids": sorted(example.evidence_document_ids),
                        "evidence_recall_at_1": route.evidence_recall_at_1,
                        "evidence_recall_at_2": route.evidence_recall_at_2,
                        "evidence_recall_at_4": route.evidence_recall_at_4,
                        "selected_evidence_recall": route.selected_evidence_recall,
                        "oracle_source_tokens": len(item["oracle_source"]),
                        "routed_source_tokens": len(item["routed_source"]),
                        "index_build_ms": route.index_build_ms,
                        "routing_ms": route.routing_ms,
                        "index_bytes": route.index_bytes,
                        "active_materialized_kv_bytes": (
                            active_memory.nbytes if active_memory is not None else 0
                        ),
                    }
                )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_routed_answer_quality_v1",
        "evidence_tier": "NATURAL_QA_ROUTED_EVIDENCE_MATERIALIZATION",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "seeds": list(SEEDS),
        "examples_per_seed": args.examples_per_seed,
        "route_top_k": args.route_top_k,
        "max_source_tokens": args.max_source_tokens,
        "routing": "PersistentResourceIndex hybrid signed-hash/BM25/token score",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
