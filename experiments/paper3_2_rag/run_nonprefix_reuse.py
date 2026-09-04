"""Measure changing-selection reuse against ordinary exact-prefix caching.

Natural questions are first assigned frozen strong-reranker selections. The
workload constructor then forms deterministic query sequences with overlapping
selected chunks. Each execution arm sees the same turn-level selection.
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

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _row,
    _token_segments,
)
from experiments.rag_vs_pra.datasets import load_multihop_rag, select_cohort
from experiments.rag_vs_pra.run_powered_decomposition import (
    DEFAULT_RERANKER,
    PersistentMLXBackend,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
)
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    compose_resources,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    SelectionReceipt,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)
from pra_hf.rag_materialization import exact_token_plan
from pra_hf.rag_mlx_native import (
    combine_native_memories,
    diagnostic_repair_memory,
    encode_native_memory,
    rebind_native_memories_global_packed,
)
from pra_hf.rag_powered import answer_metrics, official_multihop_rag_score
from pra_hf.rag_reuse import greedy_overlap_sequences, longest_common_prefix_length


SCHEMA_VERSION = "paper3.2-nonprefix-reuse-v1"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _digest_tokens(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    conditions = []
    for condition, values in sorted(grouped.items()):
        conditions.append(
            {
                "condition": condition,
                "turns": len(values),
                "token_f1": _mean([float(row["token_f1"]) for row in values]),
                "gold_answer_mean_nll": _mean(
                    [float(row["gold_answer_mean_nll"]) for row in values]
                ),
                "newly_encoded_tokens": _mean(
                    [float(row["newly_encoded_tokens"]) for row in values]
                ),
                "reused_tokens": _mean(
                    [float(row["reused_tokens"]) for row in values]
                ),
                "ttft_with_materialization_ms": _mean(
                    [float(row["ttft_with_materialization_ms"]) for row in values]
                ),
                "total_with_materialization_ms": _mean(
                    [float(row["total_with_materialization_ms"]) for row in values]
                ),
                "exact_output_parity_with_fresh": _mean(
                    [float(bool(row["output_matches_fresh"])) for row in values]
                ),
            }
        )
    sequences: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        sequences.setdefault(int(row["sequence_id"]), []).append(row)
    cumulative = []
    for sequence_id, sequence_rows in sorted(sequences.items()):
        for condition in sorted({str(row["condition"]) for row in sequence_rows}):
            values = [row for row in sequence_rows if row["condition"] == condition]
            cumulative.append(
                {
                    "sequence_id": sequence_id,
                    "condition": condition,
                    "turns": len(values),
                    "newly_encoded_tokens": sum(
                        int(row["newly_encoded_tokens"]) for row in values
                    ),
                    "reused_tokens": sum(int(row["reused_tokens"]) for row in values),
                    "total_with_materialization_ms": sum(
                        float(row["total_with_materialization_ms"]) for row in values
                    ),
                }
            )
    return {"conditions": conditions, "sequence_cumulative": cumulative}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--candidate-questions", type=int, default=50)
    parser.add_argument("--max-resources", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--sequence-count", type=int, default=5)
    parser.add_argument("--apc-block-tokens", type=int, default=16)
    parser.add_argument("--repair-fraction", type=float, default=0.25)
    parser.add_argument("--partial-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(
        questions, max_examples=args.candidate_questions, seed=args.seed
    )
    documents_by_id = {document.document_id: document for document in documents}
    retriever = FirstStageBM25(documents)
    revision = _resolve_hf_revision(args.model, args.revision)
    reranker_revision = _resolve_hf_revision(
        args.reranker, args.reranker_revision
    )
    backend = PersistentMLXBackend(
        args.model, revision, args.max_new_tokens, native_cache_unit="chunk"
    )
    selector = CrossEncoderRAGSelector(
        model_id=args.reranker,
        revision=reranker_revision,
        name_prefix="nonprefix_reuse",
    )
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)

    prepared_turns = []
    for index, question in enumerate(questions, 1):
        print(f"[select {index}/{len(questions)}] {question.example_id}", flush=True)
        receipt = make_candidate_receipt(
            dataset="multihoprag",
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
        prepared = prepare_candidate_context(
            receipt, documents_by_id, token_count=backend.token_count
        )
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
        )
        selection = SelectionReceipt.from_context(
            candidate_receipt_id=receipt.receipt_id,
            example_id=question.example_id,
            context=context,
            selector_revision=selector.name,
        )
        prepared_turns.append((question, receipt, context, selection))

    sequence_indices = greedy_overlap_sequences(
        [tuple(row.chunk.chunk_id for row in context.chunks) for _, _, context, _ in prepared_turns],
        sequence_length=args.sequence_length,
        sequence_count=args.sequence_count,
    )
    rows: list[dict[str, object]] = []
    started = time.time()
    for sequence_id, indices in enumerate(sequence_indices, 1):
        resource_cache: dict[str, object] = {}
        partial_cache: dict[str, object] = {}
        contiguous_cache: dict[str, object] = {}
        seen_text_sources: list[tuple[int, ...]] = []
        previous_chunk_ids: set[str] = set()
        for turn_number, turn_index in enumerate(indices, 1):
            question, receipt, context, selection = prepared_turns[turn_index]
            print(
                f"[sequence {sequence_id}/{len(sequence_indices)} turn "
                f"{turn_number}/{len(indices)}] {question.example_id}",
                flush=True,
            )
            selected = context.chunks
            order = tuple(row.chunk.chunk_id for row in selected)
            document_ids = tuple(row.chunk.document_id for row in selected)
            token_segments = _token_segments(
                backend.tokenizer, tuple(row.chunk.text for row in selected)
            )
            packed_tokens = tuple(
                token for segment in token_segments for token in segment
            )
            packed_digest = _digest_tokens(packed_tokens)
            resources = tuple(
                SelectedResource(
                    resource_id=row.chunk.chunk_id,
                    chunk_id=row.chunk.chunk_id,
                    source_sha256=hashlib.sha256(row.chunk.text.encode()).hexdigest(),
                    source_positions=tuple(range(len(segment))),
                    rank=row.rank,
                    score=row.score,
                )
                for row, segment in zip(selected, token_segments)
            )
            global_receipt = compose_resources(
                resources,
                selection_receipt_id=selection.receipt_id,
                profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
                position_policy=PositionPolicy.GLOBAL_PACKED,
                near_gap=0,
            )
            source_receipt = compose_resources(
                resources,
                selection_receipt_id=selection.receipt_id,
                profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_INDEPENDENT,
                position_policy=PositionPolicy.SOURCE_LOCAL,
                near_gap=0,
            )

            common_prefix = max(
                (longest_common_prefix_length(packed_tokens, prior) for prior in seen_text_sources),
                default=0,
            )
            exact_prefix_hit = packed_tokens in seen_text_sources
            text_regime = "PREFIX_WARM" if exact_prefix_hit else "COLD"
            text_prediction, text_metrics = backend.answer(
                question,
                context.text,
                ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
                selection_receipt_id=packed_digest,
                regime=text_regime,
            )
            text_exact, text_f1 = answer_metrics(text_prediction, question.answers)

            fresh_started = time.perf_counter()
            fresh = encode_native_memory(backend.model, packed_tokens)
            fresh_ms = (time.perf_counter() - fresh_started) * 1000.0
            fresh_row = _row(
                question=question, receipt=receipt, selection=selection,
                order_name="sequence", order=order, condition="FRESH_PACKED",
                memory=fresh, backend=backend, encode_ms=fresh_ms,
                materialization_token_counts=tuple(map(len, token_segments)),
                requested_materialization_tokens=len(packed_tokens),
                selected_document_ids=document_ids,
                composition_receipt=global_receipt,
                retain_first_step_logits=True,
            )
            fresh_logits = fresh_row.pop("_first_step_logits_f32")
            fresh_prediction = str(fresh_row["prediction"])

            text_row = {
                "schema_version": SCHEMA_VERSION,
                "condition": "ORDINARY_PREFIX_CACHE",
                "example_id": question.example_id,
                "prediction": text_prediction,
                "gold_answers": list(question.answers),
                "exact_match": text_exact,
                "token_f1": text_f1,
                "official_multihop_rag_score": official_multihop_rag_score(
                    text_prediction, question.answers
                ),
                "gold_answer_mean_nll": text_metrics["gold_answer_mean_nll"],
                "gold_answer_log_probability": text_metrics[
                    "gold_answer_log_probability"
                ],
                "first_step_logits_sha256": text_metrics[
                    "first_step_logits_sha256"
                ],
                "output_matches_fresh": text_prediction == fresh_prediction,
                "newly_encoded_tokens": 0 if exact_prefix_hit else len(packed_tokens),
                "reused_tokens": len(packed_tokens) if exact_prefix_hit else 0,
                "exact_prefix_cache_hit": exact_prefix_hit,
                "longest_reusable_prefix_tokens": common_prefix,
                "apc_block_reusable_tokens": (
                    common_prefix // args.apc_block_tokens * args.apc_block_tokens
                ),
                "active_native_tokens": 0,
                "visible_prompt_tokens": text_metrics["visible_prompt_tokens"],
                "ttft_with_materialization_ms": text_metrics["ttft_ms"],
                "total_with_materialization_ms": text_metrics["total_latency_ms"],
            }

            contiguous = contiguous_cache.get(packed_digest)
            contiguous_hit = contiguous is not None
            if contiguous is None:
                contiguous = fresh
                contiguous_cache[packed_digest] = contiguous
            contiguous_row = _row(
                question=question, receipt=receipt, selection=selection,
                order_name="sequence", order=order,
                condition="PRA_CONTIGUOUS_BLOCK",
                memory=contiguous, backend=backend,
                encode_ms=0.0 if contiguous_hit else fresh_ms,
                materialization_token_counts=tuple(map(len, token_segments)),
                requested_materialization_tokens=len(packed_tokens),
                newly_materialized_tokens=0 if contiguous_hit else len(packed_tokens),
                reused_native_tokens=len(packed_tokens) if contiguous_hit else 0,
                selected_document_ids=document_ids,
                composition_receipt=global_receipt,
                reference_first_step_logits=fresh_logits,
            )

            independent = []
            newly_encoded = 0
            reused = 0
            resource_encode_ms = 0.0
            for resource, segment in zip(resources, token_segments):
                key = f"{resource.chunk_id}:{_digest_tokens(segment)}"
                memory = resource_cache.get(key)
                if memory is None:
                    encode_started = time.perf_counter()
                    memory = encode_native_memory(backend.model, segment)
                    resource_encode_ms += (time.perf_counter() - encode_started) * 1000.0
                    resource_cache[key] = memory
                    newly_encoded += len(segment)
                else:
                    reused += len(segment)
                independent.append(memory)
            transform_started = time.perf_counter()
            source_local = combine_native_memories(tuple(independent))
            source_transform_ms = (time.perf_counter() - transform_started) * 1000.0
            transform_started = time.perf_counter()
            rebound = rebind_native_memories_global_packed(
                backend.model, tuple(independent)
            )
            rebound_transform_ms = (time.perf_counter() - transform_started) * 1000.0
            repaired = diagnostic_repair_memory(
                rebound,
                fresh,
                args.repair_fraction,
                mode="boundary",
                resource_lengths=tuple(map(len, token_segments)),
            )

            native_rows = [
                _row(
                    question=question, receipt=receipt, selection=selection,
                    order_name="sequence", order=order, condition=condition,
                    memory=memory, backend=backend,
                    encode_ms=resource_encode_ms + transform_ms,
                    materialization_token_counts=tuple(map(len, token_segments)),
                    requested_materialization_tokens=len(packed_tokens),
                    newly_materialized_tokens=newly_encoded,
                    reused_native_tokens=reused,
                    selected_document_ids=document_ids,
                    composition_receipt=composition_receipt,
                    repair_fraction=repair_fraction,
                    repair_mode="boundary" if repair_fraction is not None else None,
                    repaired_token_count=(
                        round(len(packed_tokens) * repair_fraction)
                        if repair_fraction is not None
                        else 0
                    ),
                    repaired_layer_count=(
                        len(memory.layers) if repair_fraction is not None else 0
                    ),
                    reference_first_step_logits=fresh_logits,
                )
                for condition, memory, transform_ms, composition_receipt, repair_fraction in (
                    (
                        "PRA_SOURCE_LOCAL",
                        source_local,
                        source_transform_ms,
                        source_receipt,
                        None,
                    ),
                    (
                        "PRA_GLOBAL_REBOUND",
                        rebound,
                        rebound_transform_ms,
                        global_receipt,
                        None,
                    ),
                    (
                        f"PRA_REBOUND_REPAIR_{args.repair_fraction:g}",
                        repaired,
                        rebound_transform_ms,
                        global_receipt,
                        args.repair_fraction,
                    ),
                )
            ]

            partial_plan = exact_token_plan(
                tuple(map(len, token_segments)), args.partial_fraction
            )
            partial_memories = []
            partial_new = 0
            partial_reused = 0
            partial_encode_ms = 0.0
            partial_resources = []
            partial_documents = []
            for resource_index in partial_plan.selected_indices:
                count = partial_plan.token_counts[resource_index]
                segment = token_segments[resource_index][:count]
                resource = replace(
                    resources[resource_index],
                    source_positions=resources[resource_index].source_positions[:count],
                )
                key = f"{resource.chunk_id}:{_digest_tokens(segment)}"
                memory = partial_cache.get(key)
                if memory is None:
                    encode_started = time.perf_counter()
                    memory = encode_native_memory(backend.model, segment)
                    partial_encode_ms += (time.perf_counter() - encode_started) * 1000.0
                    partial_cache[key] = memory
                    partial_new += len(segment)
                else:
                    partial_reused += len(segment)
                partial_memories.append(memory)
                partial_resources.append(resource)
                partial_documents.append(document_ids[resource_index])
            partial_receipt = compose_resources(
                tuple(partial_resources),
                selection_receipt_id=selection.receipt_id,
                profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
                position_policy=PositionPolicy.GLOBAL_PACKED,
                near_gap=0,
            )
            partial_transform_started = time.perf_counter()
            partial_memory = rebind_native_memories_global_packed(
                backend.model, tuple(partial_memories)
            )
            partial_transform_ms = (
                time.perf_counter() - partial_transform_started
            ) * 1000.0
            partial_row = _row(
                question=question, receipt=receipt, selection=selection,
                order_name="sequence", order=order,
                condition=f"PRA_PARTIAL_{args.partial_fraction:g}",
                memory=partial_memory, backend=backend,
                encode_ms=partial_encode_ms + partial_transform_ms,
                materialization_fraction=partial_plan.materialized_fraction,
                materialization_policy="score",
                materialization_token_counts=partial_plan.token_counts,
                requested_materialization_tokens=partial_plan.requested_token_budget,
                newly_materialized_tokens=partial_new,
                reused_native_tokens=partial_reused,
                selected_document_ids=partial_documents,
                composition_receipt=partial_receipt,
                reference_first_step_logits=fresh_logits,
            )

            current_chunks = set(order)
            overlap = len(current_chunks & previous_chunk_ids) / max(len(current_chunks), 1)
            for row in [fresh_row, text_row, contiguous_row, *native_rows, partial_row]:
                row.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sequence_id": sequence_id,
                        "turn": turn_number,
                        "selection_receipt_id": selection.receipt_id,
                        "selected_chunk_ids": list(order),
                        "selected_document_ids": list(document_ids),
                        "previous_turn_resource_overlap": overlap,
                        "output_matches_fresh": str(row["prediction"]) == fresh_prediction,
                        "newly_encoded_tokens": int(
                            row.get(
                                "newly_encoded_tokens",
                                row.get("newly_materialized_native_tokens", 0),
                            )
                        ),
                        "reused_tokens": int(
                            row.get("reused_tokens", row.get("reused_native_tokens", 0))
                        ),
                    }
                )
                rows.append(row)
            seen_text_sources.append(packed_tokens)
            previous_chunk_ids = current_chunks

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        args.output / "condition_results.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "multihoprag",
        "dataset_metadata": dict(dataset_metadata),
        "model": args.model,
        "model_revision": revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "sequence_length": args.sequence_length,
        "sequence_count": len(sequence_indices),
        "sequence_example_ids": [
            [prepared_turns[index][0].example_id for index in sequence]
            for sequence in sequence_indices
        ],
        "repair_fraction": args.repair_fraction,
        "partial_fraction": args.partial_fraction,
        "apc_block_tokens": args.apc_block_tokens,
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "git_commit": _git_commit(),
        "started_unix": started,
        "completed_unix": time.time(),
        "summary": _summary(rows),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
