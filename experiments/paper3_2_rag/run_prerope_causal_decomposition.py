"""Run the Paper 3.2 packed-context causal decomposition on MLX.

For every question the BM25 candidate receipt, BGE ranking, selected records,
token sequence, separators, order, and positions are frozen once. The runner
then compares ordinary causal packing (A), block-isolated packing (B), and
independent pre-RoPE records rebound to B's exact positions (C).
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
from urllib.parse import quote

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _execute,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
    _token_segments,
)
from experiments.rag_vs_pra.datasets import (
    controlled_fixture,
    load_multihop_rag,
    select_cohort,
)
from experiments.rag_vs_pra.run_powered_decomposition import PersistentMLXBackend
from pra_hf.rag_causal_decomposition import (
    CausalDecompositionReceipt,
    DocumentAttentionPolicy,
    build_document_attention_mask,
    request_positions_digest,
    token_sequence_digest,
    validate_matched_abc_receipts,
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


SCHEMA_VERSION = "paper3.2-prerope-causal-decomposition-v1"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _policies(value: str) -> tuple[DocumentAttentionPolicy, ...]:
    try:
        policies = tuple(
            DocumentAttentionPolicy(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not policies:
        raise argparse.ArgumentTypeError("at least one mask policy is required")
    return tuple(dict.fromkeys(policies))


def _windows(value: str) -> tuple[int, ...]:
    windows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not windows or any(window <= 0 for window in windows):
        raise argparse.ArgumentTypeError("boundary windows must be positive")
    return windows


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _condition_row(
    *,
    condition: str,
    question,
    backend: PersistentMLXBackend,
    memory,
    candidate_receipt_id: str,
    selection_receipt_id: str,
    decomposition_receipt: CausalDecompositionReceipt,
    mask_receipt,
    encode_ms: float,
    transform_ms: float,
    selected_document_ids: Sequence[str],
    reference_logits=None,
    reference_condition: str | None = None,
    retain_logits: bool = False,
) -> dict[str, object]:
    prediction, metrics, logits = _execute(backend, question, memory)
    support = len(
        set(selected_document_ids).intersection(question.gold_document_ids)
    ) / max(len(question.gold_document_ids), 1)
    row = {
        "schema_version": SCHEMA_VERSION,
        "example_id": question.example_id,
        "question_type": question.question_type,
        "condition": condition,
        "candidate_receipt_id": candidate_receipt_id,
        "selection_receipt_id": selection_receipt_id,
        "decomposition_receipt_id": decomposition_receipt.receipt_id,
        "token_sequence_digest": decomposition_receipt.token_sequence_digest,
        "document_order_digest": decomposition_receipt.document_order_digest,
        "request_positions_digest": decomposition_receipt.request_positions_digest,
        "attention_mask_policy": mask_receipt.policy.value,
        "attention_mask_digest": mask_receipt.attention_mask_digest,
        "document_token_boundaries": [
            {"start": boundary.start, "end": boundary.end}
            for boundary in mask_receipt.document_token_boundaries
        ],
        "query_token_boundary": {
            "start": mask_receipt.query_token_boundary.start,
            "end": mask_receipt.query_token_boundary.end,
        },
        "cross_document_attention_edges_allowed": (
            mask_receipt.cross_document_attention_edges_allowed
        ),
        "boundary_window_size": mask_receipt.boundary_window_size,
        "position_binding_mode": decomposition_receipt.position_binding_mode,
        "pre_rope_storage": decomposition_receipt.position_binding_mode == "PRE_ROPE",
        "request_position_policy": "EXACT_PACKED_REQUEST_POSITIONS",
        "rope_frequency_digest": decomposition_receipt.rope_frequency_digest,
        "selected_document_ids": list(selected_document_ids),
        "supporting_document_coverage": support,
        "physical_native_tokens": memory.source_tokens,
        "native_bytes": memory.nbytes,
        "encode_ms": encode_ms,
        "request_rope_transform_ms": transform_ms,
        "prediction": prediction,
        "gold_answers": list(question.answers),
        "numerical_reference_condition": reference_condition,
        **_distribution_diagnostics(reference_logits, logits),
        **metrics,
    }
    row["ttft_with_materialization_ms"] = (
        float(metrics["ttft_ms"]) + encode_ms + transform_ms
    )
    row["total_with_materialization_ms"] = (
        float(metrics["total_latency_ms"]) + encode_ms + transform_ms
    )
    if retain_logits:
        row["_first_step_logits_f32"] = logits
    return row


def _summary(
    rows: Sequence[Mapping[str, object]],
    bc_diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    conditions = []
    for condition, values in sorted(grouped.items()):
        conditions.append(
            {
                "condition": condition,
                "examples": len(values),
                "token_f1": _mean([float(row["token_f1"]) for row in values]),
                "exact_match": _mean([float(row["exact_match"]) for row in values]),
                "official_multihop_rag_score": _mean(
                    [float(row["official_multihop_rag_score"]) for row in values]
                ),
                "gold_answer_mean_nll": _mean(
                    [float(row["gold_answer_mean_nll"]) for row in values]
                ),
                "first_step_js_from_declared_reference": _mean(
                    [
                        float(row["first_step_js_divergence"])
                        for row in values
                        if row["first_step_js_divergence"] is not None
                    ]
                ),
                "encode_ms": _mean([float(row["encode_ms"]) for row in values]),
                "request_rope_transform_ms": _mean(
                    [float(row["request_rope_transform_ms"]) for row in values]
                ),
                "ttft_ms": _mean([float(row["ttft_ms"]) for row in values]),
            }
        )

    paired: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        paired.setdefault(str(row["example_id"]), {})[str(row["condition"])] = row
    bc = [
        pair
        for pair in paired.values()
        if "B_NO_CROSS_DOC_RAG" in pair and "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS" in pair
    ]
    ab = [
        pair
        for pair in paired.values()
        if "A_FULL_CAUSAL_RAG" in pair and "B_NO_CROSS_DOC_RAG" in pair
    ]
    return {
        "conditions": conditions,
        "a_minus_b": {
            "pairs": len(ab),
            "mean_token_f1_delta": _mean(
                [
                    float(pair["A_FULL_CAUSAL_RAG"]["token_f1"])
                    - float(pair["B_NO_CROSS_DOC_RAG"]["token_f1"])
                    for pair in ab
                ]
            ),
            "mean_gold_nll_delta": _mean(
                [
                    float(pair["A_FULL_CAUSAL_RAG"]["gold_answer_mean_nll"])
                    - float(pair["B_NO_CROSS_DOC_RAG"]["gold_answer_mean_nll"])
                    for pair in ab
                ]
            ),
        },
        "b_minus_c": {
            "pairs": len(bc),
            "output_matches": sum(
                pair["B_NO_CROSS_DOC_RAG"]["prediction"]
                == pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["prediction"]
                for pair in bc
            ),
            "first_step_logit_hash_matches": sum(
                pair["B_NO_CROSS_DOC_RAG"]["first_step_logits_sha256"]
                == pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["first_step_logits_sha256"]
                for pair in bc
            ),
            "mean_token_f1_delta": _mean(
                [
                    float(pair["B_NO_CROSS_DOC_RAG"]["token_f1"])
                    - float(pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["token_f1"])
                    for pair in bc
                ]
            ),
            "mean_gold_nll_abs_delta": _mean(
                [
                    abs(
                        float(pair["B_NO_CROSS_DOC_RAG"]["gold_answer_mean_nll"])
                        - float(
                            pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"][
                                "gold_answer_mean_nll"
                            ]
                        )
                    )
                    for pair in bc
                ]
            ),
            "first_step_js_divergence_mean": _mean(
                [
                    float(
                        pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"][
                            "first_step_js_divergence"
                        ]
                    )
                    for pair in bc
                ]
            ),
            "first_step_logit_max_abs_delta_mean": _mean(
                [
                    float(
                        pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"][
                            "first_step_logit_max_abs_delta"
                        ]
                    )
                    for pair in bc
                ]
            ),
            "max_layer_key_rmse": max(
                (float(row["max_key_rmse"]) for row in bc_diagnostics), default=None
            ),
            "max_layer_value_rmse": max(
                (float(row["max_value_rmse"]) for row in bc_diagnostics), default=None
            ),
            "max_key_abs_delta": max(
                (float(row["max_key_abs_delta"]) for row in bc_diagnostics), default=None
            ),
            "max_value_abs_delta": max(
                (float(row["max_value_abs_delta"]) for row in bc_diagnostics), default=None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fixture", "multihoprag"), default="fixture")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument(
        "--mask-policies",
        type=_policies,
        default=(
            DocumentAttentionPolicy.PREVIOUS_DOC_ONLY,
            DocumentAttentionPolicy.TOP_RANKED_TO_ALL,
        ),
    )
    parser.add_argument("--boundary-windows", type=_windows, default=(8, 16, 32, 64))
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
        name_prefix="prerope_causal",
    )
    backend = PersistentMLXBackend(args.model, revision, args.max_new_tokens)
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    rows: list[dict[str, object]] = []
    bc_diagnostics: list[dict[str, object]] = []
    frozen_receipts: list[dict[str, object]] = []
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
        prepared = prepare_candidate_context(
            candidate, by_id, token_count=backend.token_count
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
        query_tokens = tuple(
            backend.tokenizer.encode(backend._query(question), add_special_tokens=False)
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
        positions = tuple(range(len(packed_tokens)))
        common = {
            "selection_receipt_id": selection.receipt_id,
            "ordered_record_ids": record_ids,
            "token_sequence_digest": token_sequence_digest(packed_tokens),
            "document_order_digest": _digest(record_ids),
            "request_positions_digest": request_positions_digest(positions),
        }

        full_mask, full_mask_receipt = build_document_attention_mask(
            lengths,
            query_tokens=len(query_tokens),
            policy=DocumentAttentionPolicy.FULL_CAUSAL,
        )
        blocked_mask, blocked_mask_receipt = build_document_attention_mask(
            lengths,
            query_tokens=len(query_tokens),
            policy=DocumentAttentionPolicy.NO_CROSS_DOC,
        )
        source_tokens = len(packed_tokens)
        blocked_prefix_mask = tuple(
            row[:source_tokens] for row in blocked_mask[:source_tokens]
        )

        encode_started = time.perf_counter()
        a_memory = encode_native_memory(
            backend.model, packed_tokens, model_revision=revision
        )
        a_encode_ms = (time.perf_counter() - encode_started) * 1000.0
        encode_started = time.perf_counter()
        b_memory = encode_native_memory_with_mask(
            backend.model,
            packed_tokens,
            blocked_prefix_mask,
            model_revision=revision,
        )
        b_encode_ms = (time.perf_counter() - encode_started) * 1000.0
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
        c_encode_ms = (time.perf_counter() - encode_started) * 1000.0
        transform_started = time.perf_counter()
        c_memory = rebind_native_memories_to_receipt(
            backend.model, independent_pre, composition
        )
        c_transform_ms = (time.perf_counter() - transform_started) * 1000.0

        if b_memory.source_positions != c_memory.source_positions:
            raise RuntimeError("B/C effective source positions differ")
        diagnostic = native_memory_diagnostics(b_memory, c_memory)
        diagnostic.update(
            {
                "example_id": question.example_id,
                "selection_receipt_id": selection.receipt_id,
            }
        )
        bc_diagnostics.append(diagnostic)
        rope_digest = c_memory.rope_contract.layer_frequency_digest
        receipts = {
            "A": CausalDecompositionReceipt(
                **common,
                attention_mask_receipt_id=full_mask_receipt.receipt_id,
                position_binding_mode="POST_ROPE",
                rope_frequency_digest=rope_digest,
            ),
            "B": CausalDecompositionReceipt(
                **common,
                attention_mask_receipt_id=blocked_mask_receipt.receipt_id,
                position_binding_mode="POST_ROPE",
                rope_frequency_digest=rope_digest,
            ),
            "C": CausalDecompositionReceipt(
                **common,
                attention_mask_receipt_id=blocked_mask_receipt.receipt_id,
                position_binding_mode="PRE_ROPE",
                rope_frequency_digest=rope_digest,
            ),
        }
        validate_matched_abc_receipts(receipts)
        mask_receipts = {
            "A_FULL_CAUSAL_RAG": full_mask_receipt.to_dict(),
            "B_NO_CROSS_DOC_RAG": blocked_mask_receipt.to_dict(),
            "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS": blocked_mask_receipt.to_dict(),
        }

        a_row = _condition_row(
            condition="A_FULL_CAUSAL_RAG",
            question=question,
            backend=backend,
            memory=a_memory,
            candidate_receipt_id=candidate.receipt_id,
            selection_receipt_id=selection.receipt_id,
            decomposition_receipt=receipts["A"],
            mask_receipt=full_mask_receipt,
            encode_ms=a_encode_ms,
            transform_ms=0.0,
            selected_document_ids=document_ids,
            retain_logits=True,
        )
        a_logits = a_row.pop("_first_step_logits_f32")
        rows.append(a_row)
        b_row = _condition_row(
            condition="B_NO_CROSS_DOC_RAG",
            question=question,
            backend=backend,
            memory=b_memory,
            candidate_receipt_id=candidate.receipt_id,
            selection_receipt_id=selection.receipt_id,
            decomposition_receipt=receipts["B"],
            mask_receipt=blocked_mask_receipt,
            encode_ms=b_encode_ms,
            transform_ms=0.0,
            selected_document_ids=document_ids,
            reference_logits=a_logits,
            reference_condition="A_FULL_CAUSAL_RAG",
            retain_logits=True,
        )
        b_logits = b_row.pop("_first_step_logits_f32")
        rows.append(b_row)
        rows.append(
            _condition_row(
                condition="C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS",
                question=question,
                backend=backend,
                memory=c_memory,
                candidate_receipt_id=candidate.receipt_id,
                selection_receipt_id=selection.receipt_id,
                decomposition_receipt=receipts["C"],
                mask_receipt=blocked_mask_receipt,
                encode_ms=c_encode_ms,
                transform_ms=c_transform_ms,
                selected_document_ids=document_ids,
                reference_logits=b_logits,
                reference_condition="B_NO_CROSS_DOC_RAG",
            )
        )

        ladder = [(policy, 0) for policy in args.mask_policies]
        ladder.extend(
            (DocumentAttentionPolicy.BOUNDARY_ONLY, window)
            for window in args.boundary_windows
        )
        for policy, window in ladder:
            policy_mask, policy_receipt = build_document_attention_mask(
                lengths,
                query_tokens=len(query_tokens),
                policy=policy,
                boundary_window_size=window,
            )
            prefix_mask = tuple(
                row[:source_tokens] for row in policy_mask[:source_tokens]
            )
            encode_started = time.perf_counter()
            memory = encode_native_memory_with_mask(
                backend.model,
                packed_tokens,
                prefix_mask,
                model_revision=revision,
            )
            encode_ms = (time.perf_counter() - encode_started) * 1000.0
            condition = (
                f"M4_BOUNDARY_ONLY_{window}"
                if policy is DocumentAttentionPolicy.BOUNDARY_ONLY
                else f"M_{policy.value}"
            )
            policy_decomposition = CausalDecompositionReceipt(
                **common,
                attention_mask_receipt_id=policy_receipt.receipt_id,
                position_binding_mode="POST_ROPE",
                rope_frequency_digest=rope_digest,
            )
            mask_receipts[condition] = policy_receipt.to_dict()
            rows.append(
                _condition_row(
                    condition=condition,
                    question=question,
                    backend=backend,
                    memory=memory,
                    candidate_receipt_id=candidate.receipt_id,
                    selection_receipt_id=selection.receipt_id,
                    decomposition_receipt=policy_decomposition,
                    mask_receipt=policy_receipt,
                    encode_ms=encode_ms,
                    transform_ms=0.0,
                    selected_document_ids=document_ids,
                    reference_logits=a_logits,
                    reference_condition="A_FULL_CAUSAL_RAG",
                )
            )

        frozen_receipts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "example_id": question.example_id,
                "candidate_receipt": candidate.to_dict(),
                "selection_receipt": selection.to_dict(),
                "decomposition_receipts": {
                    arm: receipt.to_dict() for arm, receipt in receipts.items()
                },
                "attention_mask_receipts": mask_receipts,
                "packed_token_ids": list(packed_tokens),
                "record_position_bindings": [
                    {
                        "record_uri": (
                            "pra://multihoprag/chunk/" + quote(record_id, safe="")
                        ),
                        "record_id": record_id,
                        "token_index": list(range(length)),
                        "local_position": list(range(length)),
                        "packed_request_position": list(
                            range(boundary.start, boundary.end)
                        ),
                        "applied_position": list(range(boundary.start, boundary.end)),
                        "phase_metadata": {
                            "position_binding_mode": "PRE_ROPE",
                            "request_position_policy": "EXACT_PACKED_REQUEST_POSITIONS",
                            "rope_frequency_digest": rope_digest,
                        },
                    }
                    for record_id, length, boundary in zip(
                        record_ids, lengths, blocked_mask_receipt.document_token_boundaries
                    )
                ],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with gzip.open(args.output / "bc_layer_diagnostics.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in bc_diagnostics:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with gzip.open(args.output / "frozen_receipts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in frozen_receipts:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "prerope_causal_decomposition",
        "dataset": args.dataset,
        "dataset_metadata": dict(dataset_metadata),
        "model": args.model,
        "model_revision": revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "selector": selector.name,
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "max_resources": args.max_resources,
        "seed": args.seed,
        "question_ids": [question.example_id for question in questions],
        "mask_policies": [policy.value for policy in args.mask_policies],
        "boundary_windows": list(args.boundary_windows),
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "git_commit": _git_commit(),
        "started_unix": started,
        "completed_unix": time.time(),
        "rows": len(rows),
        "summary": _summary(rows, bc_diagnostics),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
