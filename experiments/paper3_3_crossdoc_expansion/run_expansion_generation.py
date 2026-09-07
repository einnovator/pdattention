"""Replay one frozen cross-document expansion policy through native MLX K/V.

This runner is deliberately downstream of ``run_expansion_frontier``.  It
measures answer behavior for a prespecified policy; it must not be used to tune
that policy on the frozen test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper3_2_rag.run_composition_fidelity import (
    _execute,
    _resolve_hf_revision,
    _token_segments,
)
from experiments.paper3_2_rag.run_prerope_causal_decomposition import DEFAULT_RERANKER
from experiments.paper3_3_crossdoc_expansion.run_expansion_frontier import (
    CachedSemanticEncoder,
    cached_record_level_context,
    _gold_chunks,
    _selected_records,
    load_selection_cache,
    environment_metadata,
    file_sha256,
)
from experiments.paper3_3_sparse_crossdoc.run_oracle_sparsity import (
    DEFAULT_SPLIT_MANIFEST,
    _condition_row,
    _encode_sparse_plan,
    _resolve_reranker_device,
    _select_split_cohort,
)
from experiments.rag_vs_pra.datasets import load_multihop_rag
from experiments.rag_vs_pra.run_powered_decomposition import PersistentMLXBackend
from pra_hf.crossdoc_expansion import (
    CrossDocumentBudget,
    CrossDocumentDirection,
    CrossDocumentExpansionConfig,
    CrossDocumentExpansionMode,
    CrossDocumentExpansionRequest,
    CrossDocumentExpansionRuntime,
    CrossDocumentGranularity,
    build_cross_document_expansion_policy,
)
from pra_hf.rag_causal_decomposition import (
    DocumentAttentionPolicy,
    build_document_attention_mask,
)
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    compose_resources,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    SelectionReceipt,
    make_candidate_receipt,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import (
    PositionBindingMode,
    encode_native_memory,
    encode_native_memory_with_mask,
    rebind_native_memories_to_receipt,
)
from pra_hf.rag_retrieval import SentenceTransformerEmbedder
from pra_hf.sparse_crossdoc import (
    CrossDocumentAttentionCollector,
    linked_pair_interaction_plan,
    linked_top_attention_edge_plan,
)


SCHEMA_VERSION = "paper3.3-crossdoc-expansion-generation-v1"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _encode_independent(
    backend: PersistentMLXBackend,
    texts: Sequence[str],
    record_ids: Sequence[str],
    *,
    selection_receipt_id: str,
    revision: str,
) -> tuple[object, float, tuple[tuple[int, ...], ...]]:
    segments = _token_segments(backend.tokenizer, texts)
    resources = tuple(
        SelectedResource(
            resource_id=record_id,
            chunk_id=record_id,
            source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            source_positions=tuple(range(len(segment))),
            rank=rank,
            score=1.0 / rank,
        )
        for rank, (record_id, text, segment) in enumerate(
            zip(record_ids, texts, segments), 1
        )
    )
    composition = compose_resources(
        resources,
        selection_receipt_id=selection_receipt_id,
        profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
        position_policy=PositionPolicy.GLOBAL_PACKED,
        near_gap=0,
    )
    started = time.perf_counter()
    pre_rope = tuple(
        encode_native_memory(
            backend.model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision=revision,
        )
        for segment in segments
    )
    memory = rebind_native_memories_to_receipt(backend.model, pre_rope, composition)
    return memory, (time.perf_counter() - started) * 1000.0, segments


def _summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    fields = (
        "token_f1",
        "exact_match",
        "official_multihop_rag_score",
        "gold_answer_mean_nll",
        "gold_answer_log_probability",
        "first_step_js_divergence",
        "encode_ms",
        "ttft_ms",
        "itl_ms",
        "total_latency_ms",
        "tokens_per_second",
    )
    result = []
    for condition, values in sorted(grouped.items()):
        summary: dict[str, object] = {"condition": condition, "examples": len(values)}
        for field in fields:
            observed = [float(row[field]) for row in values if row.get(field) is not None]
            summary[field + "_mean"] = statistics.fmean(observed) if observed else None
        summary["initial_native_tokens_mean"] = statistics.fmean(
            float(row["initial_native_tokens"]) for row in values
        )
        summary["extra_native_tokens_mean"] = statistics.fmean(
            float(row["extra_native_tokens"]) for row in values
        )
        result.append(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-name", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=16)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--reranker-device", default="auto")
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dense-revision", default="main")
    parser.add_argument("--dense-device", default="auto")
    parser.add_argument("--dense-query-weight", type=float, default=1.0)
    parser.add_argument("--dense-selected-weight", type=float, default=1.0)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--query-relevance-weight", type=float, default=0.25)
    parser.add_argument("--minimum-pair-query-score", type=float, default=0.0)
    parser.add_argument(
        "--mode", type=CrossDocumentExpansionMode, default=CrossDocumentExpansionMode.HYBRID_RRF
    )
    parser.add_argument("--query-conditioned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--direction", type=CrossDocumentDirection, default=CrossDocumentDirection.SYMMETRIC
    )
    parser.add_argument(
        "--granularity", type=CrossDocumentGranularity, default=CrossDocumentGranularity.CHUNK
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--cross-token-budget", type=int, default=128)
    parser.add_argument("--boundary-tokens", type=int, default=8)
    parser.add_argument("--linked-edge-fraction", type=float, default=0.001)
    parser.add_argument(
        "--selection-cache",
        type=Path,
        help="Frozen first-stage selections produced by the expansion frontier.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples <= 0 or args.top_k <= 0 or args.cross_token_budget < 0:
        parser.error("example/top-k counts must be positive and budget non-negative")

    model_revision = _resolve_hf_revision(args.model, args.model_revision)
    reranker_revision = _resolve_hf_revision(args.reranker, args.reranker_revision)
    reranker_device = _resolve_reranker_device(args.reranker_device)
    uses_dense = args.mode in {
        CrossDocumentExpansionMode.DENSE,
        CrossDocumentExpansionMode.HYBRID_RRF,
        CrossDocumentExpansionMode.HYBRID_WEIGHTED,
    }
    dense_revision = (
        _resolve_hf_revision(args.dense_model, args.dense_revision)
        if uses_dense
        else None
    )
    dense_device = _resolve_reranker_device(args.dense_device) if uses_dense else None
    backend = PersistentMLXBackend(
        args.model, model_revision, args.max_new_tokens, native_cache_unit="chunk"
    )
    documents, questions, metadata = load_multihop_rag(args.cache_dir)
    questions, split_metadata = _select_split_cohort(
        questions,
        split_manifest=args.split_manifest,
        split_name=args.split_name,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    by_id = {row.document_id: row for row in documents}
    retriever = FirstStageBM25(documents)
    selector = CrossEncoderRAGSelector(
        model_id=args.reranker,
        revision=reranker_revision,
        device=reranker_device,
        # Match the frontier label so its frozen selection cache can be replayed
        # without weakening the candidate/selector provenance checks.
        name_prefix="paper3_3_crossdoc_expansion",
    )
    semantic_encoder = (
        CachedSemanticEncoder(
            SentenceTransformerEmbedder(
                args.dense_model,
                revision=dense_revision,
                device=dense_device,
                query_prefix="Represent this sentence for searching relevant passages: ",
            )
        )
        if dense_revision is not None
        else None
    )
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    selection_cache = load_selection_cache(args.selection_cache)
    policy_parameters = {
        "dense_query_weight": args.dense_query_weight,
        "dense_selected_weight": args.dense_selected_weight,
        "rrf_constant": args.rrf_constant,
        "lexical_weight": args.lexical_weight,
        "dense_weight": args.dense_weight,
        "query_relevance_weight": args.query_relevance_weight,
        "minimum_pair_query_score": args.minimum_pair_query_score,
    }
    rows: list[dict[str, object]] = []
    started = time.time()

    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.example_id}", flush=True)
        candidate = make_candidate_receipt(
            dataset="multihoprag",
            dataset_revision=metadata["dataset_revision"],
            corpus_revision=metadata["corpus_revision"],
            corpus_sha256=metadata["corpus_sha256"],
            question=question,
            retriever=retriever,
            candidate_count=args.candidate_count,
            chunker=chunker,
            seed=args.seed,
        )
        # Source intervals are tokenizer-independent. Model-native token counts
        # are measured only after the frozen text selection is encoded below.
        prepared = prepare_candidate_context(candidate, by_id)
        context = cached_record_level_context(
            selection_cache,
            cache_path=args.selection_cache,
            example_id=question.example_id,
            candidate_receipt_id=candidate.receipt_id,
            prepared=prepared,
            selector=selector,
            query=question.question,
            token_budget=args.token_budget,
            max_resources=args.max_resources,
        )
        if len(context.chunks) < 2:
            continue
        selection = SelectionReceipt.from_context(
            candidate_receipt_id=candidate.receipt_id,
            example_id=question.example_id,
            context=context,
            selector_revision=selector.name,
        )
        selected_records = _selected_records("multihoprag", context.chunks)
        uri_by_document = {
            row.spans[0].document_id: row.record_uri for row in selected_records
        }
        candidates_by_record = {
            uri: tuple(
                chunk for chunk in prepared.chunks if chunk.document_id == document_id
            )
            for document_id, uri in uri_by_document.items()
        }
        budget = CrossDocumentBudget(args.top_k, args.cross_token_budget)
        request = CrossDocumentExpansionRequest(
            query=question.question,
            selection_receipt_id=selection.receipt_id,
            selected_records=selected_records,
            candidate_chunks_by_record=candidates_by_record,
            budget=budget,
            authorized_record_uris=frozenset(candidates_by_record),
            resident_chunk_ids=frozenset(chunk.chunk_id for chunk in prepared.chunks),
            gold_chunk_ids=_gold_chunks(question, prepared.chunks),
        )
        policy = build_cross_document_expansion_policy(
            CrossDocumentExpansionConfig(
                mode=args.mode,
                query_conditioned=args.query_conditioned,
                direction=args.direction,
                granularity=args.granularity,
                budget=budget,
                **policy_parameters,
            ),
            semantic_encoder=semantic_encoder,
            allow_experimental=True,
        )
        expansion = CrossDocumentExpansionRuntime().expand(request, policy)

        initial_texts = tuple(row.chunk.text for row in context.chunks)
        initial_ids = tuple(row.chunk.chunk_id for row in context.chunks)
        initial_memory, initial_encode_ms, initial_segments = _encode_independent(
            backend,
            initial_texts,
            initial_ids,
            selection_receipt_id=selection.receipt_id,
            revision=model_revision,
        )
        initial_execution = _execute(backend, question, initial_memory)
        independent_row = _condition_row(
            condition="INDEPENDENT_PRA",
            question=question,
            backend=backend,
            memory=initial_memory,
            encode_ms=initial_encode_ms,
            selection_receipt_id=selection.receipt_id,
            reference_logits=None,
            reference_condition="SELF",
            execution=initial_execution,
        )

        packed_tokens = tuple(token for segment in initial_segments for token in segment)
        packed_started = time.perf_counter()
        packed_memory = encode_native_memory(
            backend.model, packed_tokens, model_revision=model_revision
        )
        packed_encode_ms = (time.perf_counter() - packed_started) * 1000.0
        packed_execution = _execute(backend, question, packed_memory)
        packed_row = _condition_row(
            condition="PACKED_RAG",
            question=question,
            backend=backend,
            memory=packed_memory,
            encode_ms=packed_encode_ms,
            selection_receipt_id=selection.receipt_id,
            reference_logits=None,
            reference_condition="SELF",
            execution=packed_execution,
        )

        extra_texts = tuple(row.text for row in expansion.spans)
        extra_ids = tuple(
            f"{row.target_chunk_id}:{row.start}:{row.end}" for row in expansion.spans
        )
        expanded_texts = initial_texts + extra_texts
        expanded_ids = initial_ids + extra_ids
        expanded_memory, expanded_encode_ms, expanded_segments = _encode_independent(
            backend,
            expanded_texts,
            expanded_ids,
            selection_receipt_id=selection.receipt_id,
            revision=model_revision,
        )
        expanded_execution = _execute(backend, question, expanded_memory)
        expansion_row = _condition_row(
            condition="EXPANSION_ONLY",
            question=question,
            backend=backend,
            memory=expanded_memory,
            encode_ms=expanded_encode_ms,
            selection_receipt_id=selection.receipt_id,
            reference_logits=packed_execution[2],
            reference_condition="PACKED_RAG",
            execution=expanded_execution,
        )

        expanded_tokens = tuple(token for segment in expanded_segments for token in segment)
        lengths = tuple(len(segment) for segment in expanded_segments)
        full_mask, _ = build_document_attention_mask(
            lengths, policy=DocumentAttentionPolicy.FULL_CAUSAL
        )
        blocked_mask, _ = build_document_attention_mask(
            lengths, policy=DocumentAttentionPolicy.NO_CROSS_DOC
        )
        collector = CrossDocumentAttentionCollector(
            lengths,
            record_ids=expanded_ids,
            selection_receipt_id=selection.receipt_id,
            model_revision=model_revision,
        )
        encode_native_memory_with_mask(
            backend.model,
            expanded_tokens,
            full_mask,
            model_revision=model_revision,
            attention_observer=collector.observe,
        )
        graph = collector.finalize()
        selected_id_by_uri = {
            record.record_uri: record.spans[0].chunk_id for record in selected_records
        }
        linked_pairs = tuple(
            (
                selected_id_by_uri[row.source_record_uri],
                extra_id,
            )
            for row, extra_id in zip(expansion.spans, extra_ids)
        )
        plans = (
            linked_pair_interaction_plan(graph, linked_pairs),
            linked_pair_interaction_plan(
                graph, linked_pairs, boundary_tokens=args.boundary_tokens
            ),
            linked_top_attention_edge_plan(
                graph, linked_pairs, args.linked_edge_fraction
            ),
        )
        linked_rows = []
        for sparse_plan in plans:
            memory, encode_ms = _encode_sparse_plan(
                backend=backend,
                packed_tokens=expanded_tokens,
                blocked_mask=blocked_mask,
                revision=model_revision,
                graph=graph,
                plan=sparse_plan,
            )
            linked_rows.append(
                _condition_row(
                    condition=sparse_plan.mode,
                    question=question,
                    backend=backend,
                    memory=memory,
                    encode_ms=encode_ms,
                    selection_receipt_id=selection.receipt_id,
                    reference_logits=packed_execution[2],
                    reference_condition="PACKED_RAG",
                    plan=sparse_plan,
                )
            )
        for row in (packed_row, independent_row, expansion_row, *linked_rows):
            row["schema_version"] = SCHEMA_VERSION
            row["expansion_receipt_id"] = expansion.receipt.receipt_id
            row["expansion_mode"] = args.mode.value
            row["initial_native_tokens"] = sum(len(segment) for segment in initial_segments)
            row["extra_native_tokens"] = sum(len(segment) for segment in expanded_segments) - row[
                "initial_native_tokens"
            ]
            row["requested_cross_tokens"] = expansion.receipt.requested_tokens
            row["deduplicated_cross_tokens"] = expansion.receipt.deduplicated_tokens
            row["reused_cross_kv_tokens"] = expansion.receipt.reused_native_tokens
            row["new_cross_kv_tokens"] = expansion.receipt.new_native_tokens
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    run = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "model": args.model,
        "model_revision": model_revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "reranker_device": reranker_device,
        "dense_encoder": (
            semantic_encoder.identity if semantic_encoder is not None else None
        ),
        "dense_revision": dense_revision,
        "dense_device": dense_device,
        "split": split_metadata,
        "seed": args.seed,
        "selection_cache": str(args.selection_cache) if args.selection_cache else None,
        "selection_cache_sha256": file_sha256(args.selection_cache),
        "policy": {
            "mode": args.mode.value,
            "query_conditioned": args.query_conditioned,
            "direction": args.direction.value,
            "granularity": args.granularity.value,
            "top_k": args.top_k,
            "cross_token_budget": args.cross_token_budget,
            "boundary_tokens": args.boundary_tokens,
            "linked_edge_fraction": args.linked_edge_fraction,
            **policy_parameters,
        },
        "examples": len({row["example_id"] for row in rows}),
        "elapsed_s": time.time() - started,
        "environment": environment_metadata(),
        "summary": _summarize(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
