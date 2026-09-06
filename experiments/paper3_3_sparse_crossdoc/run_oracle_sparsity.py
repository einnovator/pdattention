"""Run the Paper 3.3 packed-teacher oracle sparsity gate on MLX.

Retrieval, reranking, selected records, record order, and prompt semantics are
frozen once per question. The only intervention is which causal
document-to-document token pairs remain visible at each transformer layer.
Every selected pair executes the frozen host model's original attention across
all heads; no synthetic K/V is appended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _execute,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
    _token_segments,
)
from experiments.paper3_2_rag.run_prerope_causal_decomposition import (
    DEFAULT_RERANKER,
)
from experiments.rag_vs_pra.datasets import (
    controlled_fixture,
    load_multihop_rag,
    select_cohort,
)
from experiments.rag_vs_pra.run_powered_decomposition import PersistentMLXBackend
from pra_hf.context_records import ContextRecord, RecordType
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
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    SelectionReceipt,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import (
    PositionBindingMode,
    encode_native_memory,
    encode_native_memory_with_mask,
    native_memory_diagnostics,
    rebind_native_memories_to_receipt,
)
from pra_hf.sparse_crossdoc import (
    CrossDocumentAttentionCollector,
    cumulative_attention_mass_plan,
    interaction_localization,
    ranked_physical_indices,
    top_attention_edge_plan,
)


SCHEMA_VERSION = "paper3.3-oracle-sparsity-run-v1"
DEFAULT_EDGE_PERCENTAGES = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
DEFAULT_MASS_PERCENTAGES = (50.0, 75.0, 90.0, 95.0, 99.0)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _percentages(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(item < 0.0 or item > 100.0 for item in values):
        raise argparse.ArgumentTypeError("percentages must lie in [0, 100]")
    return tuple(dict.fromkeys(values))


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate conditions without treating questions as independent seeds."""

    grouped: dict[tuple[str, float | None], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), row.get("target_percentage"))
        grouped.setdefault(key, []).append(row)
    result = []
    for (condition, target), values in grouped.items():
        result.append(
            {
                "condition": condition,
                "target_percentage": target,
                "examples": len(values),
                "token_f1": _mean(values, "token_f1"),
                "official_score": _mean(values, "official_multihop_rag_score"),
                "exact_match": _mean(values, "exact_match"),
                "gold_answer_mean_nll": _mean(values, "gold_answer_mean_nll"),
                "first_step_js_vs_reference": _mean(
                    values, "first_step_js_divergence"
                ),
                "selected_logical_edge_fraction": _mean(
                    values, "selected_logical_edge_fraction"
                ),
                "selected_physical_edge_fraction": _mean(
                    values, "selected_physical_edge_fraction"
                ),
                "retained_attention_mass": _mean(values, "retained_attention_mass"),
                "selected_physical_head_edges": _mean(
                    values, "selected_physical_head_edges"
                ),
                "encode_ms": _mean(values, "encode_ms"),
                "ttft_ms": _mean(values, "ttft_ms"),
                "total_latency_ms": _mean(values, "total_latency_ms"),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            str(row["condition"]),
            float(row["target_percentage"] or -1.0),
        ),
    )


def oracle_gate(summary: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the prespecified inception gate to top-edge conditions only."""

    eligible = [
        row
        for row in summary
        if row["condition"] == "ORACLE_TOP_ATTENTION"
        and float(row["target_percentage"] or 0.0) <= 5.0
        and row["token_f1"] is not None
        and row["official_score"] is not None
    ]
    passing = [
        row
        for row in eligible
        if float(row["token_f1"]) >= 0.19
        and float(row["official_score"]) >= 0.67
    ]
    return {
        "schema_version": "paper3.3-oracle-headroom-gate-v1",
        "status": "PASS_SMOKE" if passing else "FAIL_SMOKE",
        "criteria": {
            "maximum_dense_edge_fraction": 0.05,
            "minimum_token_f1": 0.19,
            "minimum_official_score": 0.67,
        },
        "qualifier": (
            "A small natural cohort can establish mechanism headroom only; it "
            "cannot qualify the learned-selector claim."
        ),
        "passing_conditions": passing,
    }


def _condition_row(
    *,
    condition: str,
    question: object,
    backend: PersistentMLXBackend,
    memory: object,
    encode_ms: float,
    selection_receipt_id: str,
    reference_logits: object | None,
    reference_condition: str,
    plan: object | None = None,
    execution: tuple[str, dict[str, object], object | None] | None = None,
) -> dict[str, object]:
    prediction, metrics, logits = execution or _execute(backend, question, memory)
    distribution = (
        {
            "first_step_logit_max_abs_delta": 0.0,
            "first_step_logit_mean_abs_delta": 0.0,
            "first_step_js_divergence": 0.0,
            "first_step_kl_reference_to_condition": 0.0,
        }
        if reference_logits is None
        else _distribution_diagnostics(reference_logits, logits)
    )
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "example_id": getattr(question, "example_id"),
        "condition": condition,
        "selection_receipt_id": selection_receipt_id,
        "distribution_reference_condition": reference_condition,
        "prediction": prediction,
        "encode_ms": encode_ms,
        **metrics,
        **distribution,
    }
    if plan is not None:
        receipt = plan.to_dict()
        row.update(receipt)
        row["target_percentage"] = float(receipt["target"]) * 100.0
    return row


def _plot(summary: Sequence[Mapping[str, object]], localization: Sequence[Mapping[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    top = [row for row in summary if row["condition"] == "ORACLE_TOP_ATTENTION"]
    if top:
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        percentages = [float(row["target_percentage"]) for row in top]
        axes[0].plot(
            percentages,
            [float(row["token_f1"] or 0.0) for row in top],
            marker="o",
            label="Token F1",
        )
        axes[0].plot(
            percentages,
            [float(row["official_score"] or 0.0) for row in top],
            marker="s",
            label="Official",
        )
        axes[0].axhline(0.19, color="black", linestyle="--", linewidth=1, label="F1 gate")
        axes[0].set_xscale("symlog", linthresh=0.01)
        axes[0].set_xlabel("Dense cross-document edges retained (%)")
        axes[0].set_ylabel("Task score")
        axes[0].legend(fontsize=8)
        axes[1].plot(percentages, [100.0 * float(row["retained_attention_mass"]) for row in top], marker="o")
        axes[1].set_xscale("symlog", linthresh=0.01)
        axes[1].set_xlabel("Dense cross-document edges retained (%)")
        axes[1].set_ylabel("Teacher attention mass retained (%)")
        figure.tight_layout()
        figure.savefig(output / "oracle_quality_frontier.pdf", bbox_inches="tight")
        figure.savefig(output / "oracle_quality_frontier.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    if localization:
        layer_count = max(
            int(row["layer"])
            for item in localization
            for row in item["layers"]
        ) + 1
        layer_mass = [0.0] * layer_count
        for item in localization:
            for row in item["layers"]:
                layer_mass[int(row["layer"])] += float(row["attention_mass_fraction"])
        total = sum(layer_mass) or 1.0
        layer_mass = [value / total for value in layer_mass]
        figure, axis = plt.subplots(figsize=(7.2, 3.4))
        axis.bar(range(layer_count), layer_mass, color="#2f6f9f")
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("Cross-document attention mass fraction")
        figure.tight_layout()
        figure.savefig(output / "oracle_layer_localization.pdf", bbox_inches="tight")
        figure.savefig(output / "oracle_layer_localization.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fixture", "multihoprag"), default="multihoprag")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=16)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--edge-percentages", type=_percentages, default=DEFAULT_EDGE_PERCENTAGES)
    parser.add_argument("--mass-percentages", type=_percentages, default=DEFAULT_MASS_PERCENTAGES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "fixture":
        documents, questions, dataset_metadata = controlled_fixture(seed=args.seed)
    else:
        documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    by_id = {document.document_id: document for document in documents}
    retriever = FirstStageBM25(documents)
    revision = _resolve_hf_revision(args.model, args.revision)
    reranker_revision = _resolve_hf_revision(args.reranker, args.reranker_revision)
    selector = CrossEncoderRAGSelector(
        model_id=args.reranker,
        revision=reranker_revision,
        name_prefix="paper3_3_oracle",
    )
    backend = PersistentMLXBackend(args.model, revision, args.max_new_tokens)
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    args.output.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    graph_summaries: list[dict[str, object]] = []
    localizations: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    started = time.time()
    for question_index, question in enumerate(questions, 1):
        print(f"[{question_index}/{len(questions)}] {question.example_id}", flush=True)
        candidate = make_candidate_receipt(
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
        prepared = prepare_candidate_context(candidate, by_id, token_count=backend.token_count)
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
        selected = tuple(context.chunks[: args.max_resources])
        if len(selected) < 2:
            continue
        context = replace(
            context,
            chunks=selected,
            packed_tokens=sum(row.chunk.token_count for row in selected),
            candidate_chunks=prepared.chunks,
        )
        selection = SelectionReceipt.from_context(
            candidate_receipt_id=candidate.receipt_id,
            example_id=question.example_id,
            context=context,
            selector_revision=selector.name,
        )
        texts = tuple(row.chunk.text for row in selected)
        document_ids = tuple(row.chunk.document_id for row in selected)
        record_ids = tuple(row.chunk.chunk_id for row in selected)
        segments = _token_segments(backend.tokenizer, texts)
        lengths = tuple(len(segment) for segment in segments)
        packed_tokens = tuple(token for segment in segments for token in segment)
        records = tuple(
            ContextRecord(
                record_id="pra://multihoprag/chunk/" + quote(record_id, safe=""),
                record_type=RecordType.RAG_CHUNK,
                payload=text,
                selection_provenance={
                    "selection_receipt_id": selection.receipt_id,
                    "rank": row.rank,
                    "score": row.score,
                    "document_id": row.chunk.document_id,
                },
                version=dataset_metadata["dataset_revision"],
            )
            for row, record_id, text in zip(selected, record_ids, texts)
        )
        resources = tuple(
            SelectedResource(
                resource_id=row.chunk.chunk_id,
                chunk_id=row.chunk.chunk_id,
                source_sha256=hashlib.sha256(row.chunk.text.encode("utf-8")).hexdigest(),
                source_positions=tuple(range(len(segment))),
                rank=row.rank,
                score=row.score,
            )
            for row, segment in zip(selected, segments)
        )
        composition = compose_resources(
            resources,
            selection_receipt_id=selection.receipt_id,
            profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
            position_policy=PositionPolicy.GLOBAL_PACKED,
            near_gap=0,
        )
        full_mask, full_receipt = build_document_attention_mask(
            lengths, policy=DocumentAttentionPolicy.FULL_CAUSAL
        )
        blocked_mask, blocked_receipt = build_document_attention_mask(
            lengths, policy=DocumentAttentionPolicy.NO_CROSS_DOC
        )

        collector = CrossDocumentAttentionCollector(
            lengths,
            record_ids=record_ids,
            selection_receipt_id=selection.receipt_id,
            model_revision=revision,
        )
        encode_started = time.perf_counter()
        instrumented = encode_native_memory_with_mask(
            backend.model,
            packed_tokens,
            full_mask,
            model_revision=revision,
            attention_observer=collector.observe,
        )
        instrumented_ms = (time.perf_counter() - encode_started) * 1000.0
        graph = collector.finalize()
        graph_path = graph_dir / f"{hashlib.sha256(question.example_id.encode()).hexdigest()[:16]}.npz"
        graph.save(graph_path)
        localization = interaction_localization(graph)
        localization["example_id"] = question.example_id
        localizations.append(localization)
        graph_summary = graph.summary()
        graph_summary.update({"example_id": question.example_id, "path": str(graph_path.relative_to(args.output))})
        graph_summaries.append(graph_summary)
        ranked_edges = ranked_physical_indices(graph)

        encode_started = time.perf_counter()
        packed = encode_native_memory(backend.model, packed_tokens, model_revision=revision)
        packed_ms = (time.perf_counter() - encode_started) * 1000.0
        host_diagnostic = native_memory_diagnostics(packed, instrumented)
        encode_started = time.perf_counter()
        blocked = encode_native_memory_with_mask(
            backend.model, packed_tokens, blocked_mask, model_revision=revision
        )
        blocked_ms = (time.perf_counter() - encode_started) * 1000.0
        encode_started = time.perf_counter()
        independent_pre = tuple(
            encode_native_memory(
                backend.model,
                segment,
                position_binding_mode=PositionBindingMode.PRE_ROPE,
                model_revision=revision,
            )
            for segment in segments
        )
        independent = rebind_native_memories_to_receipt(
            backend.model, independent_pre, composition
        )
        independent_ms = (time.perf_counter() - encode_started) * 1000.0

        packed_execution = _execute(backend, question, packed)
        packed_logits = packed_execution[2]
        instrumented_execution = _execute(backend, question, instrumented)
        explicit_teacher_logits = instrumented_execution[2]
        packed_row = _condition_row(
            condition="PACKED_RAG_HOST",
            question=question,
            backend=backend,
            memory=packed,
            encode_ms=packed_ms,
            selection_receipt_id=selection.receipt_id,
            reference_logits=None,
            reference_condition="SELF",
            execution=packed_execution,
        )
        rows.append(packed_row)
        rows.append(
            _condition_row(
                condition="PACKED_RAG_INSTRUMENTED",
                question=question,
                backend=backend,
                memory=instrumented,
                encode_ms=instrumented_ms,
                selection_receipt_id=selection.receipt_id,
                reference_logits=packed_logits,
                reference_condition="PACKED_RAG_HOST",
                execution=instrumented_execution,
            )
        )
        rows.append(
            _condition_row(
                condition="NO_CROSS_DOC_PACKED",
                question=question,
                backend=backend,
                memory=blocked,
                encode_ms=blocked_ms,
                selection_receipt_id=selection.receipt_id,
                reference_logits=explicit_teacher_logits,
                reference_condition="PACKED_RAG_INSTRUMENTED",
            )
        )
        rows.append(
            _condition_row(
                condition="INDEPENDENT_PRA",
                question=question,
                backend=backend,
                memory=independent,
                encode_ms=independent_ms,
                selection_receipt_id=selection.receipt_id,
                reference_logits=packed_logits,
                reference_condition="PACKED_RAG_HOST",
            )
        )

        full_oracle_diagnostic: dict[str, object] | None = None
        for percentage in args.edge_percentages:
            plan = top_attention_edge_plan(
                graph, percentage / 100.0, ranked=ranked_edges
            )
            encode_started = time.perf_counter()
            memory = encode_native_memory_with_mask(
                backend.model,
                packed_tokens,
                blocked_mask,
                model_revision=revision,
                sparse_mask_provider=lambda layer, _heads, current=plan: current.mask_for_layer(
                    layer,
                    base_mask=blocked_mask,
                    source_tokens=graph.source_tokens,
                    target_tokens=graph.target_tokens,
                ),
            )
            encode_ms = (time.perf_counter() - encode_started) * 1000.0
            if percentage == 100.0:
                full_oracle_diagnostic = native_memory_diagnostics(
                    instrumented, memory
                )
            rows.append(
                _condition_row(
                    condition="ORACLE_TOP_ATTENTION",
                    question=question,
                    backend=backend,
                    memory=memory,
                    encode_ms=encode_ms,
                    selection_receipt_id=selection.receipt_id,
                    reference_logits=explicit_teacher_logits,
                    reference_condition="PACKED_RAG_INSTRUMENTED",
                    plan=plan,
                )
            )
        for percentage in args.mass_percentages:
            plan = cumulative_attention_mass_plan(
                graph, percentage / 100.0, ranked=ranked_edges
            )
            encode_started = time.perf_counter()
            memory = encode_native_memory_with_mask(
                backend.model,
                packed_tokens,
                blocked_mask,
                model_revision=revision,
                sparse_mask_provider=lambda layer, _heads, current=plan: current.mask_for_layer(
                    layer,
                    base_mask=blocked_mask,
                    source_tokens=graph.source_tokens,
                    target_tokens=graph.target_tokens,
                ),
            )
            encode_ms = (time.perf_counter() - encode_started) * 1000.0
            rows.append(
                _condition_row(
                    condition="ORACLE_CUMULATIVE_MASS",
                    question=question,
                    backend=backend,
                    memory=memory,
                    encode_ms=encode_ms,
                    selection_receipt_id=selection.receipt_id,
                    reference_logits=explicit_teacher_logits,
                    reference_condition="PACKED_RAG_INSTRUMENTED",
                    plan=plan,
                )
            )
        receipts.append(
            {
                "example_id": question.example_id,
                "candidate_receipt": candidate.to_dict(),
                "selection_receipt": selection.to_dict(),
                "record_contracts": [
                    {
                        "record_id": record.record_id,
                        "record_type": record.record_type.value,
                        "version": record.version,
                        "source_fingerprint": record.source_fingerprint,
                    }
                    for record in records
                ],
                "document_ids": list(document_ids),
                "record_ids": list(record_ids),
                "document_lengths": list(lengths),
                "full_mask_receipt": full_receipt.to_dict(),
                "blocked_mask_receipt": blocked_receipt.to_dict(),
                "instrumented_vs_host_diagnostic": host_diagnostic,
                "instrumented_host_parity": (
                    float(host_diagnostic["max_key_abs_delta"]) < 1e-5
                    and float(host_diagnostic["max_value_abs_delta"]) < 1e-5
                ),
                "full_oracle_vs_instrumented_diagnostic": full_oracle_diagnostic,
                "full_oracle_replay_parity": bool(
                    full_oracle_diagnostic
                    and float(full_oracle_diagnostic["max_key_abs_delta"]) < 1e-5
                    and float(full_oracle_diagnostic["max_value_abs_delta"]) < 1e-5
                ),
            }
        )

    summary = summarize_rows(rows)
    gate = oracle_gate(summary)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper3.3_oracle_cross_document_sparsity",
        "scope": "small_natural_mechanism_gate" if args.dataset == "multihoprag" else "fixture_smoke",
        "dataset": dataset_metadata,
        "model": args.model,
        "model_revision": revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "seed": args.seed,
        "questions": len({str(row["example_id"]) for row in rows}),
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "max_resources": args.max_resources,
        "edge_percentages": list(args.edge_percentages),
        "mass_percentages": list(args.mass_percentages),
        "selector_frozen": True,
        "oracle_granularity": "layer_head_token_pair",
        "base_model_weights_frozen": True,
        "persistent_records_mutated": False,
        "git_commit": _git_commit(),
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "elapsed_seconds": time.time() - started,
        "inherited_residual_baseline": {
            "source": "docs/papers/shared/results/paper3_2_rag/crossdoc_adapter/qwen3_1_7b_rank8_five_seed/manifest.json",
            "token_f1": 0.153518,
            "official_score": 0.533333,
            "status": "INHERITED_NOT_RERUN",
        },
        "gate": gate,
        "conditions": summary,
        "graphs": graph_summaries,
        "localization": localizations,
        "receipts": receipts,
        "rows": rows,
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _plot(summary, localizations, args.output)
    print(json.dumps({"output": str(args.output), "gate": gate["status"], "questions": result["questions"]}, indent=2))


if __name__ == "__main__":
    main()
