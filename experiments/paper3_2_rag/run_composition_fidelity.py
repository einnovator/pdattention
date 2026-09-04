"""Measure independent native-memory composition against fresh packed context.

The selector runs once per question.  Every order then keeps resource content
fixed while comparing fresh packed text, one contiguous native encoding,
independent source-local K/V, GLOBAL_PACKED RoPE-rebound K/V, and an optional
diagnostic repair ladder.  Repair rows consume precomputed fresh states and are
therefore mechanism diagnostics rather than deployable cost claims.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from experiments.rag_vs_pra.datasets import (
    controlled_fixture,
    load_multihop_rag,
    select_cohort,
)
from experiments.rag_vs_pra.run_powered_decomposition import (
    DEFAULT_RERANKER,
    PersistentMLXBackend,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
)
from pra_hf.rag_composition import permutation_orders
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    SelectionReceipt,
    StandardRAGSelector,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import (
    combine_native_memories,
    diagnostic_repair_memory,
    encode_native_memory,
    make_native_prompt_cache,
    rebind_native_memories_global_packed,
)
from pra_hf.rag_powered import answer_metrics, official_multihop_rag_score


SCHEMA_VERSION = "paper3.2-composition-fidelity-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("repair fractions must be in [0, 1]")
    return values


def _token_segments(tokenizer: object, texts: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Tokenize independent resource strings whose concatenation is exact."""

    strings = tuple(text.rstrip() + "\n\n" for text in texts)
    segments = tuple(
        tuple(tokenizer.encode(text, add_special_tokens=False)) for text in strings
    )
    packed = tuple(
        tokenizer.encode("".join(strings), add_special_tokens=False)
    )
    flattened = tuple(token for segment in segments for token in segment)
    if flattened != packed:
        raise RuntimeError(
            "tokenizer is not additive across the declared resource separator; "
            "composition would not preserve an identical token sequence"
        )
    return segments


def _execute(backend: PersistentMLXBackend, question, memory) -> tuple[str, dict[str, object]]:
    query_tokens = list(
        backend.tokenizer.encode(backend._query(question), add_special_tokens=False)
    )
    prediction, serving = backend._generate(
        query_tokens, make_native_prompt_cache(backend.model, memory)
    )
    scoring = backend._score_gold_answer(
        question, query_tokens, make_native_prompt_cache(backend.model, memory)
    )
    exact, token_f1 = answer_metrics(prediction, question.answers)
    return prediction, {
        **serving,
        **scoring,
        "exact_match": exact,
        "token_f1": token_f1,
        "official_multihop_rag_score": official_multihop_rag_score(
            prediction, question.answers
        ),
    }


def _row(
    *,
    question,
    receipt,
    selection: SelectionReceipt,
    order_name: str,
    order: Sequence[str],
    condition: str,
    memory,
    backend: PersistentMLXBackend,
    encode_ms: float,
    repair_fraction: float | None = None,
) -> dict[str, object]:
    prediction, metrics = _execute(backend, question, memory)
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": question.example_id,
        "question_type": question.question_type,
        "candidate_receipt_id": receipt.receipt_id,
        "selection_receipt_id": selection.receipt_id,
        "selected_chunk_ids": [row.chunk_id for row in selection.intervals],
        "resource_order_name": order_name,
        "resource_order": list(order),
        "condition": condition,
        "repair_fraction": repair_fraction,
        "physical_native_tokens": memory.source_tokens,
        "query_position_base": memory.position_base,
        "native_bytes": memory.nbytes,
        "encode_or_transform_ms": encode_ms,
        "prediction": prediction,
        "gold_answers": list(question.answers),
        **metrics,
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["condition"]), str(row["resource_order_name"])), []
        ).append(row)
    conditions = []
    for (condition, order), values in sorted(grouped.items()):
        conditions.append(
            {
                "condition": condition,
                "resource_order_name": order,
                "examples": len(values),
                "exact_match": _mean([float(row["exact_match"]) for row in values]),
                "token_f1": _mean([float(row["token_f1"]) for row in values]),
                "gold_answer_mean_nll": _mean(
                    [float(row["gold_answer_mean_nll"]) for row in values]
                ),
                "ttft_ms": _mean([float(row["ttft_ms"]) for row in values]),
                "total_latency_ms": _mean(
                    [float(row["total_latency_ms"]) for row in values]
                ),
            }
        )
    pairs: dict[tuple[str, str], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["example_id"]), str(row["resource_order_name"]))
        pairs.setdefault(key, {})[str(row["condition"])] = row
    comparisons = {}
    for condition in sorted({str(row["condition"]) for row in rows} - {"FRESH_PACKED"}):
        available = [
            pair for pair in pairs.values() if "FRESH_PACKED" in pair and condition in pair
        ]
        comparisons[condition] = {
            "pairs": len(available),
            "output_matches": sum(
                pair["FRESH_PACKED"]["prediction"] == pair[condition]["prediction"]
                for pair in available
            ),
            "first_step_logit_hash_matches": sum(
                pair["FRESH_PACKED"]["first_step_logits_sha256"]
                == pair[condition]["first_step_logits_sha256"]
                for pair in available
            ),
            "gold_nll_mean_abs_delta": _mean(
                [
                    abs(
                        float(pair["FRESH_PACKED"]["gold_answer_mean_nll"])
                        - float(pair[condition]["gold_answer_mean_nll"])
                    )
                    for pair in available
                ]
            ),
        }
    by_example: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if row["condition"] == "FRESH_PACKED":
            by_example.setdefault(str(row["example_id"]), []).append(row)
    order_sensitivity = {
        example_id: {
            "orders": len(values),
            "unique_outputs": len({str(row["prediction"]) for row in values}),
            "gold_nll_variance": statistics.pvariance(
                [float(row["gold_answer_mean_nll"]) for row in values]
            )
            if len(values) > 1
            else 0.0,
        }
        for example_id, values in by_example.items()
    }
    return {
        "conditions": conditions,
        "fresh_packed_comparisons": comparisons,
        "fresh_packed_order_sensitivity": order_sensitivity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fixture", "multihoprag"), default="fixture")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--max-random-orders", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--selector", choices=("bm25", "strong"), default="bm25")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--repair-fractions", type=_floats, default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "fixture":
        documents, questions, dataset_metadata = controlled_fixture(seed=args.seed)
    else:
        documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    by_id = {row.document_id: row for row in documents}
    retriever = FirstStageBM25(documents)
    revision = _resolve_hf_revision(args.model, args.revision)
    backend = PersistentMLXBackend(args.model, revision, args.max_new_tokens)
    reranker_revision = None
    if args.selector == "strong":
        reranker_revision = _resolve_hf_revision(args.reranker, args.reranker_revision)
        selector = CrossEncoderRAGSelector(
            model_id=args.reranker,
            revision=reranker_revision,
            name_prefix="composition",
        )
    else:
        selector = StandardRAGSelector()
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    rows: list[dict[str, object]] = []
    started = time.time()
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.example_id}", flush=True)
        receipt = make_candidate_receipt(
            dataset=args.dataset,
            dataset_revision=dataset_metadata["dataset_revision"],
            corpus_revision=dataset_metadata["corpus_revision"],
            corpus_sha256=dataset_metadata["corpus_sha256"],
            question=question,
            retriever=retriever,
            candidate_count=args.candidate_count,
            chunker=chunker,
            ensure_gold=False,
            seed=args.seed,
        )
        prepared = prepare_candidate_context(receipt, by_id, token_count=backend.token_count)
        ranking_started = time.perf_counter()
        ranking = selector.rank(question.question, prepared.chunks)
        ranking_ms = (time.perf_counter() - ranking_started) * 1000.0
        context = packed_context_from_ranking(
            condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
            selector_name=selector.name,
            ranked=ranking,
            prepared=prepared,
            token_budget=args.token_budget,
            selector_latency_ms=ranking_ms,
        )
        selected = context.chunks[: args.max_resources]
        if len(selected) < 2:
            continue
        context = replace(
            context,
            chunks=tuple(selected),
            packed_tokens=sum(row.chunk.token_count for row in selected),
            candidate_chunks=prepared.chunks,
        )
        selection = SelectionReceipt.from_context(
            candidate_receipt_id=receipt.receipt_id,
            example_id=question.example_id,
            context=context,
            selector_revision=selector.name,
        )
        by_chunk = {row.chunk.chunk_id: row for row in selected}
        resource_ids = tuple(by_chunk)
        orders = permutation_orders(
            resource_ids, seed=args.seed, max_random=args.max_random_orders
        )
        for order_index, order in enumerate(orders):
            order_name = (
                "canonical" if order_index == 0 else "reverse" if order_index == 1 else f"random_{order_index - 1}"
            )
            texts = tuple(by_chunk[chunk_id].chunk.text for chunk_id in order)
            token_segments = _token_segments(backend.tokenizer, texts)
            packed_tokens = tuple(token for segment in token_segments for token in segment)
            started_encode = time.perf_counter()
            fresh = encode_native_memory(backend.model, packed_tokens)
            fresh_ms = (time.perf_counter() - started_encode) * 1000.0
            started_encode = time.perf_counter()
            independent = tuple(
                encode_native_memory(backend.model, segment) for segment in token_segments
            )
            independent_ms = (time.perf_counter() - started_encode) * 1000.0
            source_local = combine_native_memories(independent)
            started_rebind = time.perf_counter()
            rebound = rebind_native_memories_global_packed(backend.model, independent)
            rebind_ms = (time.perf_counter() - started_rebind) * 1000.0
            rows.extend(
                (
                    _row(
                        question=question, receipt=receipt, selection=selection,
                        order_name=order_name, order=order, condition="FRESH_PACKED",
                        memory=fresh, backend=backend, encode_ms=fresh_ms,
                    ),
                    _row(
                        question=question, receipt=receipt, selection=selection,
                        order_name=order_name, order=order, condition="NATIVE_CONTIGUOUS",
                        memory=fresh, backend=backend, encode_ms=0.0,
                    ),
                    _row(
                        question=question, receipt=receipt, selection=selection,
                        order_name=order_name, order=order, condition="NATIVE_SOURCE_LOCAL",
                        memory=source_local, backend=backend, encode_ms=independent_ms,
                    ),
                    _row(
                        question=question, receipt=receipt, selection=selection,
                        order_name=order_name, order=order, condition="NATIVE_GLOBAL_REBOUND",
                        memory=rebound, backend=backend, encode_ms=independent_ms + rebind_ms,
                    ),
                )
            )
            for fraction in args.repair_fractions:
                repaired = diagnostic_repair_memory(rebound, fresh, fraction)
                rows.append(
                    _row(
                        question=question, receipt=receipt, selection=selection,
                        order_name=order_name, order=order,
                        condition=f"REPAIR_{fraction:g}", memory=repaired,
                        backend=backend, encode_ms=independent_ms + rebind_ms,
                        repair_fraction=fraction,
                    )
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = _summary(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "composition_fidelity",
        "dataset": args.dataset,
        "dataset_metadata": dict(dataset_metadata),
        "model": args.model,
        "model_revision": revision,
        "selector": selector.name,
        "reranker_revision": reranker_revision,
        "seed": args.seed,
        "question_ids": [question.example_id for question in questions],
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "max_resources": args.max_resources,
        "max_random_orders": args.max_random_orders,
        "repair_fractions": list(args.repair_fractions),
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "git_commit": _git_commit(),
        "started_unix": started,
        "completed_unix": time.time(),
        "rows": len(rows),
        "summary": summary,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
