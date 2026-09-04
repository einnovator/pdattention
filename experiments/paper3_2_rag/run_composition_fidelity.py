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
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    compose_resources,
    permutation_orders,
)
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
    rebind_native_memories_to_receipt,
)
from pra_hf.rag_materialization import (
    TokenMaterializationPlan,
    evidence_oracle_plan,
    exact_token_plan,
    wrong_memory_plan,
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


def _repair_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    supported = {"even", "prefix", "boundary", "later_prefix"}
    unknown = set(modes) - supported
    if not modes or unknown:
        raise argparse.ArgumentTypeError(
            f"repair modes must be selected from {sorted(supported)}; got {sorted(unknown)}"
        )
    return modes


def _position_policies(value: str) -> tuple[PositionPolicy, ...]:
    try:
        return tuple(
            PositionPolicy(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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


def _execute(
    backend: PersistentMLXBackend, question, memory
) -> tuple[str, dict[str, object], object | None]:
    query_tokens = list(
        backend.tokenizer.encode(backend._query(question), add_special_tokens=False)
    )
    prediction, serving = backend._generate(
        query_tokens, make_native_prompt_cache(backend.model, memory)
    )
    scoring = backend._score_gold_answer(
        question,
        query_tokens,
        make_native_prompt_cache(backend.model, memory),
        include_first_step_logits=True,
    )
    first_step_logits = scoring.pop("_first_step_logits_f32", None)
    exact, token_f1 = answer_metrics(prediction, question.answers)
    return prediction, {
        **serving,
        **scoring,
        "exact_match": exact,
        "token_f1": token_f1,
        "official_multihop_rag_score": official_multihop_rag_score(
            prediction, question.answers
        ),
    }, first_step_logits


def _distribution_diagnostics(reference, candidate) -> dict[str, float | None]:
    """Compare two first-step distributions without persisting vocabulary logits."""

    if reference is None or candidate is None:
        return {
            "first_step_logit_max_abs_delta": None,
            "first_step_logit_mean_abs_delta": None,
            "first_step_js_divergence": None,
            "first_step_kl_reference_to_condition": None,
        }
    import numpy as np

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    delta = np.abs(left - right)
    left_probability = np.exp(left - left.max())
    left_probability /= left_probability.sum()
    right_probability = np.exp(right - right.max())
    right_probability /= right_probability.sum()
    middle = 0.5 * (left_probability + right_probability)
    epsilon = np.finfo(np.float64).tiny
    left_log = np.log(np.maximum(left_probability, epsilon))
    right_log = np.log(np.maximum(right_probability, epsilon))
    middle_log = np.log(np.maximum(middle, epsilon))
    return {
        "first_step_logit_max_abs_delta": float(delta.max()),
        "first_step_logit_mean_abs_delta": float(delta.mean()),
        "first_step_js_divergence": float(
            0.5
            * (
                np.sum(left_probability * (left_log - middle_log))
                + np.sum(right_probability * (right_log - middle_log))
            )
        ),
        "first_step_kl_reference_to_condition": float(
            np.sum(left_probability * (left_log - right_log))
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
    repair_mode: str | None = None,
    repaired_token_count: int = 0,
    repaired_layer_count: int = 0,
    materialization_fraction: float = 1.0,
    materialization_policy: str = "full",
    materialization_token_counts: Sequence[int] = (),
    requested_materialization_tokens: int | None = None,
    newly_materialized_tokens: int | None = None,
    reused_native_tokens: int = 0,
    selected_document_ids: Sequence[str] = (),
    composition_receipt=None,
    reference_first_step_logits=None,
    retain_first_step_logits: bool = False,
) -> dict[str, object]:
    try:
        import mlx.core as mx

        reset_peak = getattr(mx, "reset_peak_memory", None)
        if reset_peak is not None:
            reset_peak()
    except ImportError:
        mx = None
    prediction, metrics, first_step_logits = _execute(backend, question, memory)
    peak_memory = (
        int(getattr(mx, "get_peak_memory", lambda: 0)()) or None
        if mx is not None
        else None
    )
    query_tokens = len(
        backend.tokenizer.encode(backend._query(question), add_special_tokens=False)
    )
    support_coverage = (
        len(set(selected_document_ids).intersection(question.gold_document_ids))
        / max(len(question.gold_document_ids), 1)
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "example_id": question.example_id,
        "question_type": question.question_type,
        "candidate_receipt_id": receipt.receipt_id,
        "selection_receipt_id": selection.receipt_id,
        "composition_receipt_id": (
            composition_receipt.receipt_id if composition_receipt is not None else None
        ),
        "position_policy": (
            composition_receipt.position_policy.value
            if composition_receipt is not None
            else None
        ),
        "selected_chunk_ids": [row.chunk_id for row in selection.intervals],
        "resource_order_name": order_name,
        "resource_order": list(order),
        "condition": condition,
        "repair_fraction": repair_fraction,
        "repair_mode": repair_mode,
        "repaired_token_count": repaired_token_count,
        "repaired_layer_count": repaired_layer_count,
        "materialization_fraction": materialization_fraction,
        "materialization_policy": materialization_policy,
        "materialization_token_counts": list(materialization_token_counts),
        "requested_materialization_tokens": requested_materialization_tokens,
        "materialized_document_ids": list(selected_document_ids),
        "supporting_document_coverage": support_coverage,
        "evidence_density_per_1k_native_tokens": (
            1000.0 * support_coverage / max(memory.source_tokens, 1)
        ),
        "physical_native_tokens": memory.source_tokens,
        "newly_materialized_native_tokens": (
            memory.source_tokens
            if newly_materialized_tokens is None
            else newly_materialized_tokens
        ),
        "reused_native_tokens": reused_native_tokens,
        "query_tokens": query_tokens,
        "active_attended_tokens_at_first_decode": memory.source_tokens + query_tokens,
        "query_position_base": memory.position_base,
        "native_bytes": memory.nbytes,
        "peak_memory_bytes": peak_memory,
        "encode_or_transform_ms": encode_ms,
        "prediction": prediction,
        "gold_answers": list(question.answers),
        **_distribution_diagnostics(reference_first_step_logits, first_step_logits),
        **metrics,
    }
    if retain_first_step_logits:
        row["_first_step_logits_f32"] = first_step_logits
    return row


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
                "supporting_document_coverage": _mean(
                    [float(row["supporting_document_coverage"]) for row in values]
                ),
                "active_native_tokens": _mean(
                    [float(row["physical_native_tokens"]) for row in values]
                ),
                "newly_materialized_native_tokens": _mean(
                    [float(row["newly_materialized_native_tokens"]) for row in values]
                ),
                "peak_memory_bytes": _mean(
                    [float(row["peak_memory_bytes"]) for row in values if row["peak_memory_bytes"] is not None]
                ),
                "first_step_logit_max_abs_delta": _mean(
                    [
                        float(row["first_step_logit_max_abs_delta"])
                        for row in values
                        if row["first_step_logit_max_abs_delta"] is not None
                    ]
                ),
                "first_step_js_divergence": _mean(
                    [
                        float(row["first_step_js_divergence"])
                        for row in values
                        if row["first_step_js_divergence"] is not None
                    ]
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
            "first_step_logit_max_abs_delta_mean": _mean(
                [
                    float(pair[condition]["first_step_logit_max_abs_delta"])
                    for pair in available
                ]
            ),
            "first_step_logit_mean_abs_delta_mean": _mean(
                [
                    float(pair[condition]["first_step_logit_mean_abs_delta"])
                    for pair in available
                ]
            ),
            "first_step_js_divergence_mean": _mean(
                [
                    float(pair[condition]["first_step_js_divergence"])
                    for pair in available
                ]
            ),
            "first_step_kl_reference_to_condition_mean": _mean(
                [
                    float(pair[condition]["first_step_kl_reference_to_condition"])
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
    parser.add_argument(
        "--repair-fractions", type=_floats, default=(0.05, 0.1, 0.25, 0.5, 1.0)
    )
    parser.add_argument(
        "--repair-modes",
        type=_repair_modes,
        default=("even", "prefix", "boundary", "later_prefix"),
    )
    parser.add_argument(
        "--materialization-fractions", type=_floats, default=(0.125, 0.25, 0.5, 0.75)
    )
    parser.add_argument(
        "--position-policies",
        type=_position_policies,
        default=(),
        help="Optional comma-separated additional position policies",
    )
    parser.add_argument("--position-near-gap", type=int, default=4)
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
            document_ids = tuple(by_chunk[chunk_id].chunk.document_id for chunk_id in order)
            token_segments = _token_segments(backend.tokenizer, texts)
            packed_tokens = tuple(token for segment in token_segments for token in segment)
            resources = tuple(
                SelectedResource(
                    resource_id=chunk_id,
                    chunk_id=chunk_id,
                    source_sha256=hashlib.sha256(
                        by_chunk[chunk_id].chunk.text.encode("utf-8")
                    ).hexdigest(),
                    source_positions=tuple(range(len(segment))),
                    rank=by_chunk[chunk_id].rank,
                    score=by_chunk[chunk_id].score,
                )
                for chunk_id, segment in zip(order, token_segments)
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
            resource_token_counts = tuple(len(row) for row in token_segments)
            fresh_row = _row(
                question=question, receipt=receipt, selection=selection,
                order_name=order_name, order=order, condition="FRESH_PACKED",
                memory=fresh, backend=backend, encode_ms=fresh_ms,
                materialization_token_counts=resource_token_counts,
                requested_materialization_tokens=len(packed_tokens),
                selected_document_ids=document_ids,
                composition_receipt=global_receipt,
                retain_first_step_logits=True,
            )
            fresh_logits = fresh_row.pop("_first_step_logits_f32")
            rows.append(fresh_row)
            rows.extend(
                _row(
                    question=question, receipt=receipt, selection=selection,
                    order_name=order_name, order=order, condition=condition,
                    memory=memory, backend=backend, encode_ms=encode_ms,
                    materialization_token_counts=resource_token_counts,
                    requested_materialization_tokens=len(packed_tokens),
                    newly_materialized_tokens=newly_materialized,
                    reused_native_tokens=reused,
                    selected_document_ids=document_ids,
                    composition_receipt=composition_receipt,
                    reference_first_step_logits=fresh_logits,
                )
                for condition, memory, encode_ms, newly_materialized, reused, composition_receipt in (
                    ("NATIVE_CONTIGUOUS", fresh, 0.0, 0, len(packed_tokens), global_receipt),
                    ("NATIVE_SOURCE_LOCAL", source_local, independent_ms, len(packed_tokens), 0, source_receipt),
                    ("NATIVE_GLOBAL_REBOUND", rebound, independent_ms + rebind_ms, len(packed_tokens), 0, global_receipt),
                )
            )
            for policy in args.position_policies:
                if policy in {PositionPolicy.SOURCE_LOCAL, PositionPolicy.GLOBAL_PACKED}:
                    continue
                policy_receipt = compose_resources(
                    resources,
                    selection_receipt_id=selection.receipt_id,
                    profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
                    position_policy=policy,
                    near_gap=args.position_near_gap,
                    random_seed=args.seed + order_index,
                )
                policy_memory = rebind_native_memories_to_receipt(
                    backend.model, independent, policy_receipt
                )
                rows.append(
                    _row(
                        question=question, receipt=receipt, selection=selection,
                        order_name=order_name, order=order,
                        condition=f"POSITION_{policy.value}",
                        memory=policy_memory, backend=backend,
                        encode_ms=independent_ms,
                        materialization_token_counts=resource_token_counts,
                        requested_materialization_tokens=len(packed_tokens),
                        selected_document_ids=document_ids,
                        composition_receipt=policy_receipt,
                        reference_first_step_logits=fresh_logits,
                    )
                )
            for fraction in args.repair_fractions:
                modes = ("even",) if fraction == 1.0 else args.repair_modes
                for mode in modes:
                    repaired = diagnostic_repair_memory(
                        rebound,
                        fresh,
                        fraction,
                        mode=mode,
                        resource_lengths=tuple(len(row) for row in token_segments),
                    )
                    repaired_tokens = max(1, round(len(packed_tokens) * fraction))
                    rows.append(
                        _row(
                            question=question, receipt=receipt, selection=selection,
                            order_name=order_name, order=order,
                            condition=f"REPAIR_{mode.upper()}_{fraction:g}",
                            memory=repaired,
                            backend=backend, encode_ms=independent_ms + rebind_ms,
                            repair_fraction=fraction,
                            repair_mode=mode,
                            repaired_token_count=repaired_tokens,
                            repaired_layer_count=len(repaired.layers),
                            materialization_token_counts=tuple(
                                len(row) for row in token_segments
                            ),
                            requested_materialization_tokens=len(packed_tokens),
                            selected_document_ids=document_ids,
                            composition_receipt=compose_resources(
                                resources,
                                selection_receipt_id=selection.receipt_id,
                                profile=RAGPRAProfile.RAG_PLUS_PRA_REPAIR,
                                position_policy=PositionPolicy.GLOBAL_PACKED,
                                near_gap=0,
                                repair_fraction=fraction,
                            ),
                            reference_first_step_logits=fresh_logits,
                        )
                    )
            if order_index == 0:
                for fraction in args.materialization_fractions:
                    counts = tuple(len(segment) for segment in token_segments)
                    gold_indices = tuple(
                        index
                        for index, document_id in enumerate(document_ids)
                        if document_id in question.gold_document_ids
                    )
                    plans: tuple[tuple[str, TokenMaterializationPlan], ...] = (
                        ("SCORE", exact_token_plan(counts, fraction)),
                        ("ORACLE", evidence_oracle_plan(counts, gold_indices, fraction)),
                        ("WRONG", wrong_memory_plan(counts, gold_indices, fraction)),
                    )
                    for policy, plan in plans:
                        partial_segments = tuple(
                            token_segments[index][:count]
                            for index, count in enumerate(plan.token_counts)
                            if count
                        )
                        partial_tokens = tuple(
                            token for segment in partial_segments for token in segment
                        )
                        started_partial = time.perf_counter()
                        partial = encode_native_memory(backend.model, partial_tokens)
                        partial_ms = (time.perf_counter() - started_partial) * 1000.0
                        partial_documents = tuple(
                            document_ids[index] for index in plan.selected_indices
                        )
                        partial_resources = tuple(
                            replace(
                                resources[index],
                                source_positions=resources[index].source_positions[
                                    : plan.token_counts[index]
                                ],
                            )
                            for index in plan.selected_indices
                        )
                        rows.append(
                            _row(
                                question=question, receipt=receipt, selection=selection,
                                order_name=order_name, order=order,
                                condition=f"PARTIAL_{policy}_{fraction:g}",
                                memory=partial, backend=backend, encode_ms=partial_ms,
                                materialization_fraction=plan.materialized_fraction,
                                materialization_policy=policy.lower(),
                                materialization_token_counts=plan.token_counts,
                                requested_materialization_tokens=plan.requested_token_budget,
                                selected_document_ids=partial_documents,
                                composition_receipt=compose_resources(
                                    partial_resources,
                                    selection_receipt_id=selection.receipt_id,
                                    profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_CONTIGUOUS,
                                    position_policy=PositionPolicy.GLOBAL_PACKED,
                                    near_gap=0,
                                ),
                                reference_first_step_logits=fresh_logits,
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
        "repair_modes": list(args.repair_modes),
        "materialization_fractions": list(args.materialization_fractions),
        "position_policies": [policy.value for policy in args.position_policies],
        "position_near_gap": args.position_near_gap,
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
