"""Evaluate PRA-native record references with BM25 and learned rerankers.

Candidate retrieval is frozen once per question.  Each selector then feeds the
same selection receipt to packed text, contiguous native K/V, explicit record
references, and a routed root reference.  The record conditions pass through
``ReferenceTable`` and ``InMemoryResolver`` before independently cached chunk
K/V is materialized, so they exercise the public PRA record abstraction rather
than an experiment-only direct composition shortcut.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _execute,
    _token_segments,
)
from experiments.rag_vs_pra.datasets import load_multihop_rag, select_cohort
from experiments.rag_vs_pra.run_powered_decomposition import (
    PersistentMLXBackend,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    FirstStageBM25,
    PackedContext,
    RankedChunk,
    SelectionReceipt,
    StandardRAGSelector,
    context_metrics,
    failure_classification,
    make_candidate_receipt,
    pack_ranked_chunks,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import (
    combine_native_memories,
    encode_native_memory,
)
from pra_hf.rag_powered import answer_metrics, official_multihop_rag_score
from pra_hf.rag_record_runtime import (
    BM25RerankSelector,
    CrossEncoderCandidateReranker,
    PRADocumentRecordStore,
    RerankerReceipt,
    validate_frozen_record_selection,
)
from pra_hf.rag_reuse import greedy_overlap_sequences, longest_common_prefix_length


SCHEMA_VERSION = "paper3.2-native-record-reranker-v1"
DEFAULT_MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BGE = "BAAI/bge-reranker-v2-m3"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _model_prompt(backend: PersistentMLXBackend, question) -> str:
    return backend._query(question)


def _selected_context(
    *,
    receipt,
    prepared,
    ranked: Sequence[RankedChunk],
    selector_name: str,
    token_budget: int,
    selector_latency_ms: float,
) -> PackedContext:
    selected = pack_ranked_chunks(ranked, token_budget)
    return PackedContext(
        condition=ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
        chunks=selected,
        token_budget=token_budget,
        packed_tokens=sum(row.chunk.token_count for row in selected),
        candidate_tokens=prepared.candidate_tokens,
        selector_latency_ms=selector_latency_ms,
        index_build_ms=prepared.build_latency_ms,
        selector_name=selector_name,
        candidate_chunks=prepared.chunks,
    )


def _rank_selector(selector, question: str, chunks):
    started = time.perf_counter()
    if isinstance(selector, BM25RerankSelector):
        result = selector.rank_with_receipt(question, chunks)
        return result.ranked, result.receipt, (time.perf_counter() - started) * 1000.0
    ranked = selector.rank(question, chunks)
    return ranked, None, (time.perf_counter() - started) * 1000.0


def _packed_memory(backend: PersistentMLXBackend, texts: Sequence[str]):
    segments = _token_segments(backend.tokenizer, texts)
    tokens = tuple(token for segment in segments for token in segment)
    started = time.perf_counter()
    memory = encode_native_memory(backend.model, tokens)
    return memory, (time.perf_counter() - started) * 1000.0, len(tokens)


def _record_memory(
    backend: PersistentMLXBackend,
    request,
    cache: dict[str, object],
):
    memories = []
    newly_encoded = 0
    hits = 0
    encode_ms = 0.0
    for uri, text in zip(request.materialized_chunk_uris, request.materialized_texts):
        memory = cache.get(uri)
        tokens = tuple(
            backend.tokenizer.encode(text.rstrip() + "\n\n", add_special_tokens=False)
        )
        if memory is None:
            started = time.perf_counter()
            memory = encode_native_memory(backend.model, tokens)
            encode_ms += (time.perf_counter() - started) * 1000.0
            cache[uri] = memory
            newly_encoded += len(tokens)
        else:
            hits += 1
        memories.append(memory)
    if not memories:
        raise ValueError("record request selected no materializable chunks")
    return (
        combine_native_memories(memories),
        encode_ms,
        newly_encoded,
        hits,
        len(memories),
    )


def _execute_row(
    *,
    backend,
    question,
    receipt,
    context,
    selection,
    selector_key: str,
    reranker_receipt: RerankerReceipt | None,
    representation: str,
    memory,
    encode_ms: float,
    physical_native_tokens: int,
    visible_prompt_tokens: int,
    newly_encoded_tokens: int,
    reused_native_tokens: int,
    native_cache_hits: int,
    native_cache_lookups: int,
    record_request=None,
    order_name: str = "canonical",
    reference_first_step_logits=None,
    reference_prediction: str | None = None,
    reference_token_f1: float | None = None,
):
    prediction, metrics, first_logits = _execute(backend, question, memory)
    exact, token_f1 = answer_metrics(prediction, question.answers)
    retrieval_metrics = context_metrics(question, receipt, context)
    if float(retrieval_metrics["supporting_document_coverage"]) < 1.0 and reranker_receipt:
        failure_class = "PRA_RERANKER_MISS"
    elif representation.startswith("PRA_") and reference_token_f1 is not None and token_f1 < reference_token_f1:
        failure_class = "PRA_RECORD_SEMANTIC_QUALITY_DROP"
    else:
        failure_class = failure_classification(
            question=question,
            receipt=receipt,
            context=context,
            answer_correct=bool(exact),
        )
    row = {
        "schema_version": SCHEMA_VERSION,
        "seed": receipt.seed,
        "example_id": question.example_id,
        "question_type": question.question_type,
        "candidate_receipt_id": receipt.receipt_id,
        "selection_receipt_id": selection.receipt_id,
        "selector": selector_key,
        "selector_name": context.selector_name,
        "selector_latency_ms": context.selector_latency_ms,
        "reranker_receipt_id": reranker_receipt.receipt_id if reranker_receipt else None,
        "reranker_model_id": reranker_receipt.model_id if reranker_receipt else None,
        "reranker_revision": reranker_receipt.model_revision if reranker_receipt else None,
        "reranker_latency_ms": reranker_receipt.latency_ms if reranker_receipt else 0.0,
        "reranker_receipt": reranker_receipt.to_dict() if reranker_receipt else None,
        "representation": representation,
        "order_name": order_name,
        "selected_document_ids": list(context.selected_document_ids),
        "selected_chunk_ids": list(context.selected_chunk_ids),
        "gold_document_ids": sorted(question.gold_document_ids),
        "prediction": prediction,
        "gold_answers": list(question.answers),
        "exact_match": exact,
        "token_f1": token_f1,
        "official_multihop_rag_score": official_multihop_rag_score(
            prediction, question.answers
        ),
        "candidate_tokens": context.candidate_tokens,
        "selected_source_tokens": context.packed_tokens,
        "visible_prompt_tokens": visible_prompt_tokens,
        "physical_native_tokens": physical_native_tokens,
        "active_attended_context_tokens": physical_native_tokens,
        "newly_encoded_native_tokens": newly_encoded_tokens,
        "reused_native_tokens": reused_native_tokens,
        "native_cache_hits": native_cache_hits,
        "native_cache_lookups": native_cache_lookups,
        "native_encode_ms": encode_ms,
        "record_request_receipt_id": record_request.receipt_id if record_request else None,
        "record_resolution_mode": record_request.resolution_mode if record_request else None,
        "logical_reference_tokens": (
            len(record_request.reference_table) if record_request else 0
        ),
        "record_request": record_request.to_dict() if record_request else None,
        "exact_output_agreement_with_packed": (
            prediction == reference_prediction if reference_prediction is not None else None
        ),
        "first_step_logit_agreement_with_packed": (
            metrics.get("first_step_logits_sha256")
            == hashlib.sha256(reference_first_step_logits.astype("<f4").tobytes()).hexdigest()
            if reference_first_step_logits is not None
            else None
        ),
        "failure_class": failure_class,
        "retrieval_context_metrics": retrieval_metrics,
        **metrics,
        **_distribution_diagnostics(reference_first_step_logits, first_logits),
    }
    return row, first_logits


def _ordered_context(context: PackedContext, document_order: Sequence[str]) -> PackedContext:
    order = {document_id: rank for rank, document_id in enumerate(document_order)}
    rows = sorted(
        context.chunks,
        key=lambda row: (order[row.chunk.document_id], row.rank, row.chunk.chunk_id),
    )
    reranked = tuple(
        RankedChunk(row.chunk, row.score, rank, row.channel_ranks)
        for rank, row in enumerate(rows, 1)
    )
    return replace(context, chunks=reranked)


def _orders(document_ids: Sequence[str], seed: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    import random

    canonical = tuple(document_ids)
    values = [("canonical", canonical), ("reverse", tuple(reversed(canonical)))]
    rng = random.Random(seed)
    for index in range(2):
        shuffled = list(canonical)
        rng.shuffle(shuffled)
        values.append((f"random_{index + 1}", tuple(shuffled)))
    unique = []
    seen = set()
    for name, order in values:
        if order not in seen:
            unique.append((name, order))
            seen.add(order)
    return tuple(unique)


def _pairwise_js(logits: Sequence[object]) -> float | None:
    values = []
    for left, right in itertools.combinations(logits, 2):
        value = _distribution_diagnostics(left, right)["first_step_js_divergence"]
        if value is not None:
            values.append(float(value))
    return statistics.fmean(values) if values else None


def _mean_present(
    rows: Sequence[Mapping[str, object]], name: str
) -> float | None:
    """Average a nullable metric without inventing a baseline value."""

    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return statistics.fmean(values) if values else None


def _aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["selector"]), str(row["representation"]), str(row["order_name"])),
            [],
        ).append(row)
    result = []
    for (selector, representation, order_name), values in sorted(grouped.items()):
        result.append(
            {
                "selector": selector,
                "representation": representation,
                "order_name": order_name,
                "examples": len(values),
                "exact_match": statistics.fmean(float(row["exact_match"]) for row in values),
                "token_f1": statistics.fmean(float(row["token_f1"]) for row in values),
                "official_score": statistics.fmean(
                    float(row["official_multihop_rag_score"]) for row in values
                ),
                "gold_answer_mean_nll": _mean_present(values, "gold_answer_mean_nll"),
                "supporting_document_coverage": statistics.fmean(
                    float(row["retrieval_context_metrics"]["supporting_document_coverage"])
                    for row in values
                ),
                "gold_chunk_recall": statistics.fmean(
                    float(row["retrieval_context_metrics"]["gold_chunk_recall"])
                    for row in values
                ),
                "mrr": statistics.fmean(
                    float(row["retrieval_context_metrics"]["mrr"]) for row in values
                ),
                "ndcg": statistics.fmean(
                    float(row["retrieval_context_metrics"]["ndcg"]) for row in values
                ),
                "answer_string_availability": statistics.fmean(
                    float(row["retrieval_context_metrics"]["answer_string_availability"])
                    for row in values
                ),
                "false_selected_document_fraction": statistics.fmean(
                    float(row["retrieval_context_metrics"]["false_selected_document_fraction"])
                    for row in values
                ),
                "selected_source_tokens": statistics.fmean(
                    float(row["selected_source_tokens"]) for row in values
                ),
                "visible_prompt_tokens": statistics.fmean(
                    float(row["visible_prompt_tokens"]) for row in values
                ),
                "newly_encoded_native_tokens": statistics.fmean(
                    float(row["newly_encoded_native_tokens"]) for row in values
                ),
                "reused_native_tokens": statistics.fmean(
                    float(row["reused_native_tokens"]) for row in values
                ),
                "ttft_ms": statistics.fmean(float(row["ttft_ms"]) for row in values),
                "total_latency_ms": statistics.fmean(
                    float(row["total_latency_ms"]) for row in values
                ),
                "reranker_latency_ms": statistics.fmean(
                    float(row["reranker_latency_ms"]) for row in values
                ),
                "exact_output_agreement_with_packed": _mean_present(
                    values, "exact_output_agreement_with_packed"
                ),
                "first_step_logit_agreement_with_packed": _mean_present(
                    values, "first_step_logit_agreement_with_packed"
                ),
                "first_step_js_vs_packed": _mean_present(
                    values, "first_step_js_divergence"
                ),
            }
        )
    return result


def _parse_selectors(value: str) -> tuple[str, ...]:
    values = tuple(item.strip().casefold() for item in value.split(",") if item.strip())
    supported = {"bm25", "minilm", "bge"}
    if not values or set(values) - supported:
        raise argparse.ArgumentTypeError(f"selectors must be selected from {sorted(supported)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--selectors", type=_parse_selectors, default=("bm25", "minilm", "bge"))
    parser.add_argument("--minilm-model", default=DEFAULT_MINILM)
    parser.add_argument("--minilm-revision", default="main")
    parser.add_argument("--bge-model", default=DEFAULT_BGE)
    parser.add_argument("--bge-revision", default="main")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--order-selector", choices=("none", "bm25", "minilm", "bge"), default="minilm")
    parser.add_argument("--reuse-sequence-count", type=int, default=5)
    parser.add_argument("--reuse-sequence-length", type=int, default=4)
    args = parser.parse_args()

    if min(args.max_examples, args.candidate_count, args.token_budget) <= 0:
        parser.error("example, candidate, and token budgets must be positive")
    documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    documents_by_id = {row.document_id: row for row in documents}
    first_stage = FirstStageBM25(documents)
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    model_revision = _resolve_hf_revision(args.model, args.revision)
    reranker_revisions = {}
    if "minilm" in args.selectors:
        reranker_revisions["minilm"] = _resolve_hf_revision(
            args.minilm_model, args.minilm_revision
        )
    if "bge" in args.selectors:
        reranker_revisions["bge"] = _resolve_hf_revision(args.bge_model, args.bge_revision)

    receipts = []
    for question in questions:
        receipts.append(
            make_candidate_receipt(
                dataset="multihoprag",
                dataset_revision=str(dataset_metadata.get("dataset_revision", "UNKNOWN")),
                corpus_revision=str(dataset_metadata.get("corpus_revision", "UNKNOWN")),
                corpus_sha256=str(dataset_metadata.get("corpus_sha256", "UNKNOWN")),
                question=question,
                retriever=first_stage,
                candidate_count=args.candidate_count,
                chunker=chunker,
                seed=args.seed,
            )
        )
    corpus_ids = tuple(
        dict.fromkeys(
            document_id
            for receipt in receipts
            for document_id in receipt.candidate_document_ids
        )
    )
    backend = PersistentMLXBackend(
        args.model, model_revision, args.max_new_tokens, native_cache_unit="chunk"
    )
    record_store = PRADocumentRecordStore(
        "multihoprag",
        tuple(documents_by_id[value] for value in corpus_ids),
        chunker=chunker,
        token_count=backend.token_count,
    )

    selectors: dict[str, object] = {"bm25": StandardRAGSelector()}
    if "minilm" in args.selectors:
        selectors["minilm"] = BM25RerankSelector(
            CrossEncoderCandidateReranker(
                model_id=args.minilm_model,
                revision=reranker_revisions["minilm"],
                device=args.reranker_device,
                batch_size=args.reranker_batch_size,
            ),
            candidate_count=args.rerank_candidates,
        )
    if "bge" in args.selectors:
        selectors["bge"] = BM25RerankSelector(
            CrossEncoderCandidateReranker(
                model_id=args.bge_model,
                revision=reranker_revisions["bge"],
                device=args.reranker_device,
                batch_size=max(1, args.reranker_batch_size // 2),
            ),
            candidate_count=args.rerank_candidates,
        )
    selectors = {key: selectors[key] for key in args.selectors}

    prepared_rows = []
    for index, (question, receipt) in enumerate(zip(questions, receipts), 1):
        print(f"[select {index}/{len(questions)}] {question.example_id}", flush=True)
        prepared = prepare_candidate_context(
            receipt, documents_by_id, token_count=backend.token_count
        )
        contexts = {}
        rerank_receipts = {}
        for key, selector in selectors.items():
            ranked, reranker_receipt, latency_ms = _rank_selector(
                selector, question.question, prepared.chunks
            )
            contexts[key] = _selected_context(
                receipt=receipt,
                prepared=prepared,
                ranked=ranked,
                selector_name=selector.name,
                token_budget=args.token_budget,
                selector_latency_ms=latency_ms,
            )
            rerank_receipts[key] = reranker_receipt
        prepared_rows.append((question, receipt, contexts, rerank_receipts))

    rows = []
    order_sensitivity = []
    for index, (question, receipt, contexts, rerank_receipts) in enumerate(prepared_rows, 1):
        print(f"[execute {index}/{len(prepared_rows)}] {question.example_id}", flush=True)
        model_prompt = _model_prompt(backend, question)
        query_tokens = len(backend.tokenizer.encode(model_prompt, add_special_tokens=False))
        for selector_key, context in contexts.items():
            # Keep the matched main matrix independent of selector execution
            # order. Cross-request persistence is measured separately below.
            condition_record_cache: dict[str, object] = {}
            selection = SelectionReceipt.from_context(
                candidate_receipt_id=receipt.receipt_id,
                example_id=question.example_id,
                context=context,
                selector_revision="native_record_v1",
            )
            texts = tuple(row.chunk.text for row in context.chunks)
            packed, packed_ms, packed_tokens = _packed_memory(backend, texts)
            packed_row, packed_logits = _execute_row(
                backend=backend,
                question=question,
                receipt=receipt,
                context=context,
                selection=selection,
                selector_key=selector_key,
                reranker_receipt=rerank_receipts[selector_key],
                representation="PACKED_RAG_TEXT",
                memory=packed,
                encode_ms=packed_ms,
                physical_native_tokens=packed_tokens,
                visible_prompt_tokens=packed_tokens + query_tokens,
                newly_encoded_tokens=packed_tokens,
                reused_native_tokens=0,
                native_cache_hits=0,
                native_cache_lookups=0,
            )
            rows.append(packed_row)
            contiguous_row, _ = _execute_row(
                backend=backend,
                question=question,
                receipt=receipt,
                context=context,
                selection=selection,
                selector_key=selector_key,
                reranker_receipt=rerank_receipts[selector_key],
                representation="NATIVE_CONTIGUOUS",
                memory=packed,
                encode_ms=0.0,
                physical_native_tokens=packed_tokens,
                visible_prompt_tokens=query_tokens,
                newly_encoded_tokens=0,
                reused_native_tokens=packed_tokens,
                native_cache_hits=1,
                native_cache_lookups=1,
                reference_first_step_logits=packed_logits,
                reference_prediction=str(packed_row["prediction"]),
                reference_token_f1=float(packed_row["token_f1"]),
            )
            rows.append(contiguous_row)

            logical, table = record_store.build_explicit_prompt(
                request_id=question.example_id,
                document_ids=context.selected_document_ids,
                model_prompt=model_prompt,
            )
            explicit = record_store.resolve_request(
                request_id=question.example_id,
                logical_prompt=logical,
                model_prompt=model_prompt,
                reference_table=table,
                selection=selection,
            )
            validate_frozen_record_selection(explicit, selection)
            memory, encode_ms, new_tokens, hits, lookups = _record_memory(
                backend, explicit, condition_record_cache
            )
            explicit_row, _ = _execute_row(
                backend=backend,
                question=question,
                receipt=receipt,
                context=context,
                selection=selection,
                selector_key=selector_key,
                reranker_receipt=rerank_receipts[selector_key],
                representation="PRA_EXPLICIT_RECORDS",
                memory=memory,
                encode_ms=encode_ms,
                physical_native_tokens=memory.source_tokens,
                visible_prompt_tokens=query_tokens,
                newly_encoded_tokens=new_tokens,
                reused_native_tokens=memory.source_tokens - new_tokens,
                native_cache_hits=hits,
                native_cache_lookups=lookups,
                record_request=explicit,
                reference_first_step_logits=packed_logits,
                reference_prediction=str(packed_row["prediction"]),
                reference_token_f1=float(packed_row["token_f1"]),
            )
            rows.append(explicit_row)

            logical, table, root_uri = record_store.build_root_prompt(
                request_id=f"{question.example_id}:{selector_key}",
                candidate_document_ids=receipt.candidate_document_ids,
                model_prompt=model_prompt,
            )
            routed = record_store.resolve_request(
                request_id=question.example_id,
                logical_prompt=logical,
                model_prompt=model_prompt,
                reference_table=table,
                selection=selection,
                root_uri=root_uri,
            )
            validate_frozen_record_selection(routed, selection)
            memory, encode_ms, new_tokens, hits, lookups = _record_memory(
                backend, routed, condition_record_cache
            )
            routed_row, _ = _execute_row(
                backend=backend,
                question=question,
                receipt=receipt,
                context=context,
                selection=selection,
                selector_key=selector_key,
                reranker_receipt=rerank_receipts[selector_key],
                representation="PRA_ROUTED_ROOT",
                memory=memory,
                encode_ms=encode_ms,
                physical_native_tokens=memory.source_tokens,
                visible_prompt_tokens=query_tokens,
                newly_encoded_tokens=new_tokens,
                reused_native_tokens=memory.source_tokens - new_tokens,
                native_cache_hits=hits,
                native_cache_lookups=lookups,
                record_request=routed,
                reference_first_step_logits=packed_logits,
                reference_prediction=str(packed_row["prediction"]),
                reference_token_f1=float(packed_row["token_f1"]),
            )
            rows.append(routed_row)

        if args.order_selector != "none" and args.order_selector in contexts:
            base = contexts[args.order_selector]
            reranker_receipt = rerank_receipts[args.order_selector]
            order_record_cache: dict[str, object] = {}
            packed_logits_by_order = []
            record_logits_by_order = []
            packed_outputs = []
            record_outputs = []
            packed_scores = []
            record_scores = []
            for order_name, order in _orders(
                base.selected_document_ids, args.seed + index
            ):
                ordered = _ordered_context(base, order)
                selection = SelectionReceipt.from_context(
                    candidate_receipt_id=receipt.receipt_id,
                    example_id=question.example_id,
                    context=ordered,
                    selector_revision="native_record_order_v1",
                )
                texts = tuple(row.chunk.text for row in ordered.chunks)
                packed, packed_ms, packed_tokens = _packed_memory(backend, texts)
                packed_row, packed_logits = _execute_row(
                    backend=backend,
                    question=question,
                    receipt=receipt,
                    context=ordered,
                    selection=selection,
                    selector_key=args.order_selector,
                    reranker_receipt=reranker_receipt,
                    representation="PACKED_ORDER_SWEEP",
                    memory=packed,
                    encode_ms=packed_ms,
                    physical_native_tokens=packed_tokens,
                    visible_prompt_tokens=packed_tokens + query_tokens,
                    newly_encoded_tokens=packed_tokens,
                    reused_native_tokens=0,
                    native_cache_hits=0,
                    native_cache_lookups=0,
                    order_name=order_name,
                )
                rows.append(packed_row)
                packed_logits_by_order.append(packed_logits)
                packed_outputs.append(packed_row["prediction"])
                packed_scores.append(float(packed_row["token_f1"]))

                logical, table = record_store.build_explicit_prompt(
                    request_id=f"{question.example_id}:{order_name}",
                    document_ids=ordered.selected_document_ids,
                    model_prompt=model_prompt,
                )
                request = record_store.resolve_request(
                    request_id=question.example_id,
                    logical_prompt=logical,
                    model_prompt=model_prompt,
                    reference_table=table,
                    selection=selection,
                )
                memory, encode_ms, new_tokens, hits, lookups = _record_memory(
                    backend, request, order_record_cache
                )
                record_row, record_logits = _execute_row(
                    backend=backend,
                    question=question,
                    receipt=receipt,
                    context=ordered,
                    selection=selection,
                    selector_key=args.order_selector,
                    reranker_receipt=reranker_receipt,
                    representation="PRA_RECORD_ORDER_SWEEP",
                    memory=memory,
                    encode_ms=encode_ms,
                    physical_native_tokens=memory.source_tokens,
                    visible_prompt_tokens=query_tokens,
                    newly_encoded_tokens=new_tokens,
                    reused_native_tokens=memory.source_tokens - new_tokens,
                    native_cache_hits=hits,
                    native_cache_lookups=lookups,
                    record_request=request,
                    order_name=order_name,
                    reference_first_step_logits=packed_logits,
                    reference_prediction=str(packed_row["prediction"]),
                    reference_token_f1=float(packed_row["token_f1"]),
                )
                rows.append(record_row)
                record_logits_by_order.append(record_logits)
                record_outputs.append(record_row["prediction"])
                record_scores.append(float(record_row["token_f1"]))
            order_sensitivity.append(
                {
                    "example_id": question.example_id,
                    "selector": args.order_selector,
                    "order_count": len(packed_outputs),
                    "packed_unique_outputs": len(set(packed_outputs)),
                    "record_unique_outputs": len(set(record_outputs)),
                    "packed_mean_pairwise_js": _pairwise_js(packed_logits_by_order),
                    "record_mean_pairwise_js": _pairwise_js(record_logits_by_order),
                    "packed_token_f1_variance": statistics.pvariance(packed_scores),
                    "record_token_f1_variance": statistics.pvariance(record_scores),
                }
            )

    selected_ids = [
        tuple(row.chunk.chunk_id for row in contexts[args.order_selector].chunks)
        if args.order_selector in contexts
        else tuple(row.chunk.chunk_id for row in next(iter(contexts.values())).chunks)
        for _, _, contexts, _ in prepared_rows
    ]
    sequence_indices = greedy_overlap_sequences(
        selected_ids,
        sequence_length=args.reuse_sequence_length,
        sequence_count=args.reuse_sequence_count,
    )
    reuse_rows = []
    reuse_selector = args.order_selector if args.order_selector in selectors else args.selectors[0]
    for sequence_id, indices in enumerate(sequence_indices, 1):
        cache: dict[str, object] = {}
        previous_ids: set[str] = set()
        previous_document_ids: set[str] = set()
        previous_packed_tokens: tuple[int, ...] = ()
        for turn, item_index in enumerate(indices, 1):
            question, receipt, contexts, rerank_receipts = prepared_rows[item_index]
            context = contexts[reuse_selector]
            selection = SelectionReceipt.from_context(
                candidate_receipt_id=receipt.receipt_id,
                example_id=question.example_id,
                context=context,
                selector_revision="native_record_reuse_v1",
            )
            model_prompt = _model_prompt(backend, question)
            logical, table = record_store.build_explicit_prompt(
                request_id=f"reuse:{sequence_id}:{turn}",
                document_ids=context.selected_document_ids,
                model_prompt=model_prompt,
            )
            request = record_store.resolve_request(
                request_id=question.example_id,
                logical_prompt=logical,
                model_prompt=model_prompt,
                reference_table=table,
                selection=selection,
            )
            packed_segments = _token_segments(
                backend.tokenizer, tuple(row.chunk.text for row in context.chunks)
            )
            packed_tokens = tuple(
                token for segment in packed_segments for token in segment
            )
            packed_started = time.perf_counter()
            packed_memory = encode_native_memory(backend.model, packed_tokens)
            packed_encode_ms = (time.perf_counter() - packed_started) * 1000.0
            packed_prediction, packed_metrics, packed_logits = _execute(
                backend, question, packed_memory
            )
            packed_exact, packed_f1 = answer_metrics(
                packed_prediction, question.answers
            )
            memory, encode_ms, new_tokens, hits, lookups = _record_memory(
                backend, request, cache
            )
            prediction, metrics, record_logits = _execute(backend, question, memory)
            exact, token_f1 = answer_metrics(prediction, question.answers)
            current_ids = set(request.materialized_chunk_ids)
            current_document_ids = set(context.selected_document_ids)
            exact_prefix_tokens = longest_common_prefix_length(
                previous_packed_tokens, packed_tokens
            )
            reuse_rows.append(
                {
                    "sequence_id": sequence_id,
                    "turn": turn,
                    "example_id": question.example_id,
                    "selector": reuse_selector,
                    "selection_receipt_id": selection.receipt_id,
                    "record_request_receipt_id": request.receipt_id,
                    "selected_chunk_ids": list(request.materialized_chunk_ids),
                    "selected_document_ids": list(context.selected_document_ids),
                    "record_overlap_fraction": len(current_ids & previous_ids) / max(len(current_ids), 1),
                    "document_overlap_fraction": len(current_document_ids & previous_document_ids)
                    / max(len(current_document_ids), 1),
                    "reused_chunk_count": hits,
                    "reused_document_count": len(current_document_ids & previous_document_ids),
                    "exact_prefix_reusable_tokens": exact_prefix_tokens,
                    "newly_encoded_native_tokens": new_tokens,
                    "reused_native_tokens": memory.source_tokens - new_tokens,
                    "native_cache_hits": hits,
                    "native_cache_lookups": lookups,
                    "native_encode_ms": encode_ms,
                    "physical_native_tokens": memory.source_tokens,
                    "prediction": prediction,
                    "packed_prediction": packed_prediction,
                    "output_agreement_with_packed": prediction == packed_prediction,
                    "exact_match": exact,
                    "token_f1": token_f1,
                    "packed_exact_match": packed_exact,
                    "packed_token_f1": packed_f1,
                    "token_f1_delta_vs_packed": token_f1 - packed_f1,
                    "official_score": official_multihop_rag_score(prediction, question.answers),
                    "packed_official_score": official_multihop_rag_score(
                        packed_prediction, question.answers
                    ),
                    "packed_native_encode_ms": packed_encode_ms,
                    "packed_total_latency_ms": packed_encode_ms
                    + float(packed_metrics["total_latency_ms"]),
                    "first_step_js_vs_packed": _distribution_diagnostics(
                        packed_logits, record_logits
                    )["first_step_js_divergence"],
                    **metrics,
                }
            )
            previous_ids = current_ids
            previous_document_ids = current_document_ids
            previous_packed_tokens = packed_tokens

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with gzip.open(args.output / "reuse_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in reuse_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "ingestion_receipt.json").write_text(
        json.dumps(record_store.ingestion_receipt.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "multihoprag",
        "dataset_metadata": dict(dataset_metadata),
        "seed": args.seed,
        "question_ids": [row.example_id for row in questions],
        "model": {"id": args.model, "revision": model_revision},
        "rerankers": {
            "minilm": {"id": args.minilm_model, "revision": reranker_revisions.get("minilm")},
            "bge": {"id": args.bge_model, "revision": reranker_revisions.get("bge")},
        },
        "configuration": {
            "candidate_count": args.candidate_count,
            "rerank_candidates": args.rerank_candidates,
            "token_budget": args.token_budget,
            "chunker": asdict(chunker),
            "max_new_tokens": args.max_new_tokens,
            "selectors": list(args.selectors),
            "order_selector": args.order_selector,
        },
        "candidate_receipts": [receipt.to_dict() for receipt in receipts],
        "record_ingestion_receipt_id": record_store.ingestion_receipt.receipt_id,
        "condition_summary": _aggregate(rows),
        "order_sensitivity": order_sensitivity,
        "reuse_summary": {
            "sequences": len(sequence_indices),
            "turns": len(reuse_rows),
            "mean_overlap_fraction": statistics.fmean(
                float(row["record_overlap_fraction"]) for row in reuse_rows
            ) if reuse_rows else None,
            "mean_reused_native_tokens": statistics.fmean(
                float(row["reused_native_tokens"]) for row in reuse_rows
            ) if reuse_rows else None,
            "mean_newly_encoded_native_tokens": statistics.fmean(
                float(row["newly_encoded_native_tokens"]) for row in reuse_rows
            ) if reuse_rows else None,
            "mean_token_f1": statistics.fmean(float(row["token_f1"]) for row in reuse_rows)
            if reuse_rows else None,
            "mean_packed_token_f1": statistics.fmean(
                float(row["packed_token_f1"]) for row in reuse_rows
            ) if reuse_rows else None,
            "mean_token_f1_delta_vs_packed": statistics.fmean(
                float(row["token_f1_delta_vs_packed"]) for row in reuse_rows
            ) if reuse_rows else None,
            "mean_exact_prefix_reusable_tokens": statistics.fmean(
                float(row["exact_prefix_reusable_tokens"]) for row in reuse_rows
            ) if reuse_rows else None,
        },
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "git_commit": _git_commit(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
