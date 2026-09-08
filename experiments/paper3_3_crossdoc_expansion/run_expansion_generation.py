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
    _score_only,
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
    balanced_layer_bands,
    combine_interaction_plans,
    linked_pair_interaction_plan,
    linked_top_attention_edge_plan,
    region_layer_interaction_plan,
)


SCHEMA_VERSION = "paper3.3-crossdoc-expansion-generation-v4"
TOKEN_REGIONS = ("prefix", "middle", "suffix")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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


def _summarize_region_layer(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate causal cell scores without mixing them with generation rows."""

    grouped: dict[tuple[int, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            int(row["band_index"]),
            str(row["source_region"]),
            str(row["target_region"]),
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for (band, source, target), values in grouped.items():
        gains = [float(row["incremental_gold_nll_gain"]) for row in values]
        result.append(
            {
                "band_index": band,
                "layers": list(values[0]["layers"]),
                "source_region": source,
                "target_region": target,
                "examples": len(values),
                "incremental_gold_nll_gain_mean": statistics.fmean(gains),
                "positive_gain_fraction": sum(value > 0.0 for value in gains)
                / len(gains),
                "selected_physical_edge_fraction_mean": statistics.fmean(
                    float(row["selected_physical_edge_fraction"])
                    for row in values
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -float(row["incremental_gold_nll_gain_mean"]),
            int(row["band_index"]),
            str(row["source_region"]),
            str(row["target_region"]),
        ),
    )


def _region_layer_audit(
    *,
    backend: PersistentMLXBackend,
    question: object,
    packed_tokens: Sequence[int],
    blocked_mask: Sequence[Sequence[bool]],
    revision: str,
    graph: object,
    linked_pairs: Sequence[tuple[str, str]],
    selection_receipt_id: str,
    packed_logits: object,
    region_tokens: int,
    layer_band_count: int,
    oracle_cells: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score matched region/layer cells, then decode a gold-NLL oracle union.

    Cell selection uses the gold answer and is therefore an oracle headroom
    measurement, not a deployable selector. Every consumed cell executes the
    frozen model's original attention under an explicit sparse logical mask.
    """

    empty_plan = combine_interaction_plans(
        graph, (), mode="NO_CROSS_DOC_PACKED"
    )
    blocked_memory, blocked_encode_ms = _encode_sparse_plan(
        backend=backend,
        packed_tokens=packed_tokens,
        blocked_mask=blocked_mask,
        revision=revision,
        graph=graph,
        plan=empty_plan,
    )
    blocked_scoring, _ = _score_only(backend, question, blocked_memory)
    blocked_nll = float(blocked_scoring["gold_answer_mean_nll"])
    bands = balanced_layer_bands(graph.layer_count, layer_band_count)
    diagnostics: list[dict[str, object]] = []
    candidates: list[tuple[float, object, dict[str, object]]] = []
    total_cells = len(bands) * len(TOKEN_REGIONS) ** 2
    cell_index = 0
    print(f"  measuring {total_cells} region/layer cells", flush=True)
    for band_index, layers in enumerate(bands):
        for source_region in TOKEN_REGIONS:
            for target_region in TOKEN_REGIONS:
                cell_index += 1
                if cell_index == 1 or cell_index % 6 == 0 or cell_index == total_cells:
                    print(f"    region/layer {cell_index}/{total_cells}", flush=True)
                plan = region_layer_interaction_plan(
                    graph,
                    linked_pairs,
                    layer_indices=layers,
                    source_region=source_region,
                    target_region=target_region,
                    region_tokens=region_tokens,
                    mode="REGION_LAYER_CELL",
                )
                memory, encode_ms = _encode_sparse_plan(
                    backend=backend,
                    packed_tokens=packed_tokens,
                    blocked_mask=blocked_mask,
                    revision=revision,
                    graph=graph,
                    plan=plan,
                )
                scoring, _ = _score_only(backend, question, memory)
                nll = float(scoring["gold_answer_mean_nll"])
                gain = blocked_nll - nll
                identity = {
                    "band_index": band_index,
                    "layers": list(layers),
                    "source_region": source_region,
                    "target_region": target_region,
                    "region_tokens": region_tokens,
                }
                diagnostics.append(
                    {
                        "schema_version": "paper3.3-region-layer-cell-v1",
                        "example_id": getattr(question, "example_id"),
                        "selection_receipt_id": selection_receipt_id,
                        "graph_digest": graph.graph_digest,
                        "plan_digest": plan.plan_digest,
                        **identity,
                        "selected_physical_head_edges": (
                            plan.selected_physical_head_edges
                        ),
                        "selected_physical_edge_fraction": (
                            plan.selected_physical_edge_fraction
                        ),
                        "blocked_gold_answer_mean_nll": blocked_nll,
                        "cell_gold_answer_mean_nll": nll,
                        "incremental_gold_nll_gain": gain,
                        "encode_ms": encode_ms,
                        "gold_scoring_ms": scoring.get("gold_scoring_ms"),
                    }
                )
                candidates.append((gain, plan, identity))
    candidates.sort(
        key=lambda row: (
            -row[0],
            row[2]["band_index"],
            row[2]["source_region"],
            row[2]["target_region"],
        )
    )
    positive = [row for row in candidates if row[0] > 0.0]
    selected = positive[:oracle_cells]

    def _consume(
        *, condition: str, chosen: Sequence[tuple[float, object, dict[str, object]]]
    ) -> dict[str, object]:
        plan = combine_interaction_plans(
            graph, [row[1] for row in chosen], mode=condition
        )
        memory, encode_ms = _encode_sparse_plan(
            backend=backend,
            packed_tokens=packed_tokens,
            blocked_mask=blocked_mask,
            revision=revision,
            graph=graph,
            plan=plan,
        )
        row = _condition_row(
            condition=condition,
            question=question,
            backend=backend,
            memory=memory,
            encode_ms=encode_ms,
            selection_receipt_id=selection_receipt_id,
            reference_logits=packed_logits,
            reference_condition="PACKED_RAG",
            plan=plan,
        )
        row.update(
            {
                "selection_signal": "per_example_gold_answer_nll",
                "selection_scope": "oracle_only",
                "region_tokens": region_tokens,
                "layer_band_count": layer_band_count,
                "maximum_oracle_cells": (
                    1 if condition.endswith("_SINGLETON") else oracle_cells
                ),
                "selected_oracle_cell_count": len(chosen),
                "selected_oracle_cells": [candidate[2] for candidate in chosen],
                "singleton_predicted_gold_nll_gain": (
                    float(chosen[0][0]) if chosen else 0.0
                ),
                "additive_predicted_gold_nll_gain": sum(
                    float(candidate[0]) for candidate in chosen
                ),
                "positive_candidate_cells": len(positive),
                "blocked_gold_answer_mean_nll": blocked_nll,
            }
        )
        return row

    # The best singleton measures selection headroom without assuming that
    # independently useful cells compose. The ranked union is a separate
    # consumption test; its realized NLL can be worse through interaction.
    singleton = positive[:1]
    singleton_row = _consume(
        condition="TASK_ORACLE_REGION_LAYER_SINGLETON", chosen=singleton
    )
    union_row = _consume(
        condition="TASK_ORACLE_REGION_LAYER_UNION", chosen=selected
    )
    blocked_row = _condition_row(
        condition="NO_CROSS_DOC_PACKED",
        question=question,
        backend=backend,
        memory=blocked_memory,
        encode_ms=blocked_encode_ms,
        selection_receipt_id=selection_receipt_id,
        reference_logits=packed_logits,
        reference_condition="PACKED_RAG",
        plan=empty_plan,
    )
    return [blocked_row, singleton_row, union_row], diagnostics


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
        "--region-layer-audit",
        action="store_true",
        help=(
            "Score matched prefix/middle/suffix windows by layer band and decode "
            "a per-example gold-NLL oracle union."
        ),
    )
    parser.add_argument("--region-tokens", type=int, default=8)
    parser.add_argument("--layer-band-count", type=int, default=4)
    parser.add_argument("--oracle-cells", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only configuration-matched per-question checkpoints.",
    )
    parser.add_argument(
        "--interaction-audit-only",
        action="store_true",
        help="Emit packed, independent, pair-SA, and boundary-SA without encoding expanded spans.",
    )
    parser.add_argument(
        "--selection-cache",
        type=Path,
        help="Frozen first-stage selections produced by the expansion frontier.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples <= 0 or args.top_k <= 0 or args.cross_token_budget < 0:
        parser.error("example/top-k counts must be positive and budget non-negative")
    if args.region_tokens <= 0 or args.layer_band_count <= 0 or args.oracle_cells <= 0:
        parser.error("region tokens, layer bands, and oracle cells must be positive")
    if args.region_layer_audit and not args.interaction_audit_only:
        parser.error("--region-layer-audit requires --interaction-audit-only")

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
    run_configuration = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "model_revision": model_revision,
        "split_name": args.split_name,
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "max_resources": args.max_resources,
        "max_new_tokens": args.max_new_tokens,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "mode": args.mode.value,
        "query_conditioned": args.query_conditioned,
        "direction": args.direction.value,
        "granularity": args.granularity.value,
        "top_k": args.top_k,
        "cross_token_budget": args.cross_token_budget,
        "boundary_tokens": args.boundary_tokens,
        "linked_edge_fraction": args.linked_edge_fraction,
        "interaction_audit_only": args.interaction_audit_only,
        "region_layer_audit": args.region_layer_audit,
        "region_tokens": args.region_tokens,
        "layer_band_count": args.layer_band_count,
        "oracle_cells": args.oracle_cells,
        "selection_cache": str(args.selection_cache) if args.selection_cache else None,
        **policy_parameters,
    }
    run_configuration_digest = hashlib.sha256(
        json.dumps(
            run_configuration, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    rows: list[dict[str, object]] = []
    region_layer_diagnostics: list[dict[str, object]] = []
    started = time.time()

    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.example_id}", flush=True)
        checkpoint_path = checkpoint_dir / (
            hashlib.sha256(question.example_id.encode()).hexdigest()[:16] + ".json"
        )
        if args.resume and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("run_configuration_digest") != run_configuration_digest:
                raise RuntimeError(
                    f"checkpoint configuration mismatch for {question.example_id}; "
                    "use a different output directory"
                )
            rows.extend(checkpoint["rows"])
            region_layer_diagnostics.extend(
                checkpoint.get("region_layer_diagnostics", [])
            )
            print("  resumed completed question", flush=True)
            continue
        row_start = len(rows)
        diagnostic_start = len(region_layer_diagnostics)
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

        # Isolate neural interaction from retrieval expansion.  These plans use
        # only the original selected records and therefore form the no-expansion,
        # interaction-present cells of the factorial comparison.
        initial_lengths = tuple(len(segment) for segment in initial_segments)
        initial_full_mask, _ = build_document_attention_mask(
            initial_lengths, policy=DocumentAttentionPolicy.FULL_CAUSAL
        )
        initial_blocked_mask, _ = build_document_attention_mask(
            initial_lengths, policy=DocumentAttentionPolicy.NO_CROSS_DOC
        )
        initial_collector = CrossDocumentAttentionCollector(
            initial_lengths,
            record_ids=initial_ids,
            selection_receipt_id=selection.receipt_id,
            model_revision=model_revision,
        )
        encode_native_memory_with_mask(
            backend.model,
            packed_tokens,
            initial_full_mask,
            model_revision=model_revision,
            attention_observer=initial_collector.observe,
        )
        initial_graph = initial_collector.finalize()
        selected_id_by_uri = {
            record.record_uri: record.spans[0].chunk_id for record in selected_records
        }
        original_linked_pairs = tuple(
            dict.fromkeys(
                (
                    selected_id_by_uri[row.source_record_uri],
                    selected_id_by_uri[row.target_record_uri],
                )
                for row in expansion.spans
            )
        )
        interaction_only_rows = []
        for sparse_plan in (
            linked_pair_interaction_plan(
                initial_graph,
                original_linked_pairs,
                mode="PAIR_SA_ONLY",
            ),
            linked_pair_interaction_plan(
                initial_graph,
                original_linked_pairs,
                boundary_tokens=args.boundary_tokens,
                mode="PAIR_SA_ONLY",
            ),
        ):
            memory, encode_ms = _encode_sparse_plan(
                backend=backend,
                packed_tokens=packed_tokens,
                blocked_mask=initial_blocked_mask,
                revision=model_revision,
                graph=initial_graph,
                plan=sparse_plan,
            )
            interaction_only_rows.append(
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

        region_layer_rows: list[dict[str, object]] = []
        if args.region_layer_audit:
            region_layer_rows, diagnostics = _region_layer_audit(
                backend=backend,
                question=question,
                packed_tokens=packed_tokens,
                blocked_mask=initial_blocked_mask,
                revision=model_revision,
                graph=initial_graph,
                linked_pairs=original_linked_pairs,
                selection_receipt_id=selection.receipt_id,
                packed_logits=packed_execution[2],
                region_tokens=args.region_tokens,
                layer_band_count=args.layer_band_count,
                oracle_cells=args.oracle_cells,
            )
            region_layer_diagnostics.extend(diagnostics)

        if args.interaction_audit_only:
            for row in (
                packed_row,
                independent_row,
                *interaction_only_rows,
                *region_layer_rows,
            ):
                row["schema_version"] = SCHEMA_VERSION
                row["expansion_receipt_id"] = expansion.receipt.receipt_id
                row["expansion_mode"] = args.mode.value
                row["initial_native_tokens"] = sum(
                    len(segment) for segment in initial_segments
                )
                row["extra_native_tokens"] = 0
                row["requested_cross_tokens"] = expansion.receipt.requested_tokens
                row["deduplicated_cross_tokens"] = 0
                row["reused_cross_kv_tokens"] = 0
                row["new_cross_kv_tokens"] = 0
                row["source_token_budget"] = args.token_budget
                rows.append(row)
            _atomic_json(
                checkpoint_path,
                {
                    "run_configuration_digest": run_configuration_digest,
                    "rows": rows[row_start:],
                    "region_layer_diagnostics": region_layer_diagnostics[
                        diagnostic_start:
                    ],
                },
            )
            continue

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
        for row in (
            packed_row,
            independent_row,
            *interaction_only_rows,
            expansion_row,
            *linked_rows,
        ):
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
            row["source_token_budget"] = args.token_budget
            rows.append(row)

        _atomic_json(
            checkpoint_path,
            {
                "run_configuration_digest": run_configuration_digest,
                "rows": rows[row_start:],
                "region_layer_diagnostics": region_layer_diagnostics[
                    diagnostic_start:
                ],
            },
        )

    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    run = {
        "schema_version": SCHEMA_VERSION,
        "run_configuration_digest": run_configuration_digest,
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
            "interaction_audit_only": args.interaction_audit_only,
            "region_layer_audit": args.region_layer_audit,
            "region_tokens": args.region_tokens,
            "layer_band_count": args.layer_band_count,
            "oracle_cells": args.oracle_cells,
            **policy_parameters,
        },
        "examples": len({row["example_id"] for row in rows}),
        "elapsed_s": time.time() - started,
        "environment": environment_metadata(),
        "summary": _summarize(rows),
        "region_layer_diagnostic_rows": len(region_layer_diagnostics),
        "region_layer_summary": _summarize_region_layer(region_layer_diagnostics),
    }
    if region_layer_diagnostics:
        (args.output / "region_layer_diagnostics.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in region_layer_diagnostics
            ),
            encoding="utf-8",
        )
    (args.output / "summary.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
