"""Run frozen-selection lexical, dense, hybrid, entity, and oracle expansion.

This first runner measures evidence discovery and context cost.  It does not
substitute support recall for answer quality; generation and sparse-SA replay
consume its frozen plans in the next experiment stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from experiments.paper3_2_rag.run_composition_fidelity import _resolve_hf_revision
from experiments.paper3_2_rag.run_prerope_causal_decomposition import DEFAULT_RERANKER
from experiments.paper3_3_sparse_crossdoc.run_oracle_sparsity import (
    DEFAULT_SPLIT_MANIFEST,
    _resolve_reranker_device,
    _select_split_cohort,
)
from experiments.rag_vs_pra.datasets import load_multihop_rag
from pra_hf.crossdoc_expansion import (
    CrossDocumentBudget,
    CrossDocumentDirection,
    CrossDocumentExpansionConfig,
    CrossDocumentExpansionMode,
    CrossDocumentExpansionRequest,
    CrossDocumentExpansionRuntime,
    CrossDocumentGranularity,
    CrossDocumentSelectedRecord,
    build_cross_document_expansion_policy,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    PackedContext,
    RAGChunk,
    RAGDocument,
    RAGQuestion,
    RankedChunk,
    SelectionReceipt,
    StandardRAGSelector,
    context_metrics,
    make_candidate_receipt,
    prepare_candidate_context,
)
from pra_hf.rag_record_runtime import document_record_uri
from pra_hf.rag_retrieval import SentenceTransformerEmbedder


SCHEMA_VERSION = "paper3.3-crossdoc-expansion-frontier-v1"
SELECTION_CACHE_SCHEMA_VERSION = "paper3.3-frozen-selection-cache-v1"


class _FrozenProposalPolicy:
    """Replay one scored candidate tuple across counterfactual budgets."""

    def __init__(self, policy, candidates, proposal_latency_ms: float) -> None:
        self.name = policy.name
        self.revision = policy.revision
        self.candidates = tuple(candidates)
        self.proposal_latency_ms = proposal_latency_ms

    def propose(self, request) -> tuple:
        del request
        return self.candidates


def _controlled_crossdoc_fixture(
    *, seed: int = 11,
) -> tuple[tuple[RAGDocument, ...], tuple[RAGQuestion, ...], Mapping[str, str]]:
    """Build cases where frozen selected spans omit one recoverable peer span.

    Each question starts with chunk zero from a short profile and a longer
    registry record.  The answer-bearing registry interval starts after 128
    neutral tokens, so expansion must follow the profile's device alias into
    an unselected chunk rather than receive the answer in its frozen input.
    """

    cases = (
        ("Ada Rivera", "amber cipher", "Vault Seven"),
        ("Mina Chen", "cobalt compass", "Archive Twelve"),
        ("Jon Bell", "silver relay", "Chamber Nine"),
        ("Ravi Shah", "violet sensor", "Depot Four"),
        ("Lena Ortiz", "green beacon", "Vault Twenty"),
    )
    documents: list[RAGDocument] = []
    questions: list[RAGQuestion] = []
    neutral = " ".join(f"registry filler{index}" for index in range(150))
    for index, (person, device, destination) in enumerate(cases):
        profile_id = f"crossdoc:profile:{index}"
        registry_id = f"crossdoc:registry:{index}"
        profile_text = (
            f"{person} designed the {device}. The prototype passed its public review."
        )
        registry_prefix = (
            f"The device registered for {person} is known as the {device}. {neutral} "
        )
        support_text = (
            f"The final {device} authorization record names {destination} as its destination."
        )
        registry_text = registry_prefix + support_text
        documents.extend(
            (
                RAGDocument(
                    profile_id,
                    f"Profile of {person}",
                    profile_text,
                    source="controlled_crossdoc_fixture",
                    uri=f"fixture://crossdoc/profile/{index}",
                ),
                RAGDocument(
                    registry_id,
                    f"Device registry for {person}",
                    registry_text,
                    source="controlled_crossdoc_fixture",
                    uri=f"fixture://crossdoc/registry/{index}",
                ),
            )
        )
        questions.append(
            RAGQuestion(
                example_id=f"crossdoc:bridge:{index}",
                question=f"What destination is authorized for the device designed by {person}?",
                answers=(destination,),
                gold_document_ids=frozenset((profile_id, registry_id)),
                gold_spans={
                    profile_id: ((0, len(profile_text)),),
                    registry_id: (
                        (len(registry_prefix), len(registry_prefix) + len(support_text)),
                    ),
                },
                question_type="controlled_cross_document_bridge",
            )
        )
    # Distractors preserve realistic first-stage competition without sharing
    # the person/device aliases used by the recoverable links.
    for index in range(15):
        documents.append(
            RAGDocument(
                f"crossdoc:distractor:{index}",
                f"General archive {index}",
                "This unrelated catalog entry discusses routine inventory and maintenance. " * 12,
                source="controlled_crossdoc_fixture",
                uri=f"fixture://crossdoc/distractor/{index}",
            )
        )
    corpus_sha = hashlib.sha256(
        "".join(document.fingerprint for document in documents).encode("ascii")
    ).hexdigest()
    metadata = {
        "dataset_revision": "controlled_crossdoc_fixture_v1",
        "corpus_revision": "controlled_crossdoc_fixture_v1",
        "corpus_sha256": corpus_sha,
        "questions_sha256": hashlib.sha256(
            "".join(question.example_id for question in questions).encode("ascii")
        ).hexdigest(),
        "upstream_revision": "local",
        "license": "synthetic",
        "seed": str(seed),
    }
    return tuple(documents), tuple(questions), metadata


def _fixture_initial_context(
    question: RAGQuestion,
    prepared,
    *,
    token_budget: int,
) -> PackedContext:
    """Freeze chunk zero from both support records, omitting the target fact."""

    selected_chunks = tuple(
        chunk
        for document_id in sorted(question.gold_document_ids)
        for chunk in prepared.chunks
        if chunk.document_id == document_id and chunk.ordinal == 0
    )
    if len(selected_chunks) != len(question.gold_document_ids):
        raise RuntimeError("controlled cross-document fixture lost a selected support record")
    ranked = tuple(
        RankedChunk(chunk, 1.0 / rank, rank, {"fixture_frozen": rank})
        for rank, chunk in enumerate(selected_chunks, 1)
    )
    return PackedContext(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        chunks=ranked,
        token_budget=token_budget,
        packed_tokens=sum(row.chunk.token_count for row in ranked),
        candidate_tokens=prepared.candidate_tokens,
        selector_latency_ms=0.0,
        index_build_ms=prepared.build_latency_ms,
        selector_name="controlled_crossdoc_frozen_v1",
        candidate_chunks=prepared.chunks,
    )


def _record_level_context(
    ranking: Sequence[RankedChunk],
    prepared,
    *,
    selector_name: str,
    selector_latency_ms: float,
    token_budget: int,
    max_resources: int,
) -> PackedContext:
    """Select one address span per ranked document under a physical budget."""

    selected: list[RankedChunk] = []
    seen_documents: set[str] = set()
    packed_tokens = 0
    for row in ranking:
        if row.chunk.document_id in seen_documents:
            continue
        if packed_tokens + row.chunk.token_count > token_budget:
            continue
        selected.append(row)
        seen_documents.add(row.chunk.document_id)
        packed_tokens += row.chunk.token_count
        if len(selected) >= max_resources:
            break
    return PackedContext(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        chunks=tuple(selected),
        token_budget=token_budget,
        packed_tokens=packed_tokens,
        candidate_tokens=prepared.candidate_tokens,
        selector_latency_ms=selector_latency_ms,
        index_build_ms=prepared.build_latency_ms,
        selector_name=selector_name,
        candidate_chunks=prepared.chunks,
    )


def load_selection_cache(path: Path | None) -> dict[str, dict[str, object]]:
    """Load resumable frozen first-stage selections keyed by example identity."""

    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != SELECTION_CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported selection cache schema at line {line_number}")
        example_id = str(row["example_id"])
        previous = result.get(example_id)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting frozen selections for {example_id}")
        result[example_id] = row
    return result


def cached_record_level_context(
    cache: MutableMapping[str, dict[str, object]],
    *,
    cache_path: Path | None,
    example_id: str,
    candidate_receipt_id: str,
    prepared,
    selector,
    query: str,
    token_budget: int,
    max_resources: int,
) -> PackedContext:
    """Replay or persist one provenance-checked frozen first-stage selection."""

    cached = cache.get(example_id)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in prepared.chunks}
    if cached is not None:
        expected = {
            "candidate_receipt_id": candidate_receipt_id,
            "selector_name": selector.name,
            "token_budget": token_budget,
            "max_resources": max_resources,
        }
        mismatches = [key for key, value in expected.items() if cached.get(key) != value]
        if mismatches:
            raise ValueError(
                f"frozen selection cache mismatch for {example_id}: {', '.join(mismatches)}"
            )
        selected = tuple(
            RankedChunk(
                chunks_by_id[str(row["chunk_id"])],
                float(row["score"]),
                int(row["rank"]),
                {str(key): int(value) for key, value in dict(row["channel_ranks"]).items()},
            )
            for row in cached["selected"]
        )
        return PackedContext(
            condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
            chunks=selected,
            token_budget=token_budget,
            packed_tokens=sum(row.chunk.token_count for row in selected),
            candidate_tokens=prepared.candidate_tokens,
            selector_latency_ms=float(cached["selector_latency_ms"]),
            index_build_ms=prepared.build_latency_ms,
            selector_name=selector.name,
            candidate_chunks=prepared.chunks,
        )

    ranking_started = time.perf_counter()
    ranking = selector.rank(query, prepared.chunks)
    ranking_ms = (time.perf_counter() - ranking_started) * 1000.0
    context = _record_level_context(
        ranking,
        prepared,
        selector_name=selector.name,
        selector_latency_ms=ranking_ms,
        token_budget=token_budget,
        max_resources=max_resources,
    )
    if cache_path is not None:
        row: dict[str, object] = {
            "schema_version": SELECTION_CACHE_SCHEMA_VERSION,
            "example_id": example_id,
            "candidate_receipt_id": candidate_receipt_id,
            "selector_name": selector.name,
            "selector_latency_ms": ranking_ms,
            "token_budget": token_budget,
            "max_resources": max_resources,
            "selected": [
                {
                    "chunk_id": selected.chunk.chunk_id,
                    "score": selected.score,
                    "rank": selected.rank,
                    "channel_ranks": dict(selected.channel_ranks),
                }
                for selected in context.chunks
            ],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        cache[example_id] = row
    return context


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _csv_enum(value: str, enum_type) -> tuple:
    try:
        return tuple(dict.fromkeys(enum_type(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _csv_int(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return rows


def _csv_bool(value: str) -> tuple[bool, ...]:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    try:
        rows = tuple(dict.fromkeys(mapping[item.strip().casefold()] for item in value.split(",") if item.strip()))
    except KeyError as error:
        raise argparse.ArgumentTypeError("boolean list must use true,false") from error
    if not rows:
        raise argparse.ArgumentTypeError("boolean list cannot be empty")
    return rows


def _gold_chunks(question: RAGQuestion, chunks: Sequence[RAGChunk]) -> frozenset[str]:
    result = set()
    for chunk in chunks:
        spans = question.gold_spans.get(chunk.document_id, ())
        if spans:
            if any(chunk.start < end and start < chunk.end for start, end in spans):
                result.add(chunk.chunk_id)
        elif chunk.document_id in question.gold_document_ids:
            result.add(chunk.chunk_id)
    return frozenset(result)


def _selected_records(
    dataset: str,
    selected: Sequence[RankedChunk],
) -> tuple[CrossDocumentSelectedRecord, ...]:
    by_document: dict[str, list[RankedChunk]] = {}
    for row in selected:
        by_document.setdefault(row.chunk.document_id, []).append(row)
    records = []
    for rank, (document_id, rows) in enumerate(by_document.items(), 1):
        records.append(
            CrossDocumentSelectedRecord(
                record_uri=document_record_uri(dataset, document_id),
                rank=rank,
                spans=tuple(row.chunk for row in rows),
                query_score=max(row.score for row in rows),
            )
        )
    return tuple(records)


def _expanded_context(
    context: PackedContext,
    records: Sequence[CrossDocumentSelectedRecord],
    plan,
) -> PackedContext:
    document_by_uri = {
        row.record_uri: row.spans[0].document_id for row in records
    }
    additions = tuple(
        RankedChunk(
            RAGChunk(
                chunk_id=row.target_chunk_id,
                document_id=document_by_uri[row.target_record_uri],
                ordinal=10_000 + index,
                start=row.start,
                end=row.end,
                text=row.text,
                token_count=row.token_count,
            ),
            row.score,
            len(context.chunks) + index,
            {"cross_document_expansion": index},
        )
        for index, row in enumerate(plan.spans, 1)
    )
    return replace(
        context,
        chunks=context.chunks + additions,
        packed_tokens=context.packed_tokens + sum(row.chunk.token_count for row in additions),
    )


def _span_hits(question: RAGQuestion, chunks: Sequence[RAGChunk]) -> int:
    return sum(
        any(
            chunk.document_id == document_id
            and chunk.start < end
            and start < chunk.end
            for chunk in chunks
        )
        for document_id, spans in question.gold_spans.items()
        for start, end in spans
    )


def evaluate_question(
    *,
    dataset: str,
    question: RAGQuestion,
    candidate,
    prepared,
    context: PackedContext,
    selection: SelectionReceipt,
    modes: Sequence[CrossDocumentExpansionMode],
    directions: Sequence[CrossDocumentDirection],
    granularity: CrossDocumentGranularity,
    query_conditioning: Sequence[bool],
    top_ks: Sequence[int],
    budgets: Sequence[int],
    all_candidate_kv_resident: bool,
    semantic_encoder=None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records = _selected_records(dataset, context.chunks)
    if len(records) < 2:
        return [], []
    uri_by_document = {row.spans[0].document_id: row.record_uri for row in records}
    candidates_by_record = {
        uri: tuple(row for row in prepared.chunks if row.document_id == document_id)
        for document_id, uri in uri_by_document.items()
    }
    gold = _gold_chunks(question, prepared.chunks)
    initial_metrics = context_metrics(question, candidate, context)
    initial_span_hits = _span_hits(question, tuple(row.chunk for row in context.chunks))
    rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    seen_conditions: set[tuple[object, ...]] = set()
    proposal_cache: dict[tuple[object, ...], _FrozenProposalPolicy] = {}
    for mode in modes:
        mode_queries = (True,) if mode in {CrossDocumentExpansionMode.NONE, CrossDocumentExpansionMode.ORACLE} else query_conditioning
        for query_conditioned in mode_queries:
            for direction in directions:
                for top_k in top_ks:
                    for budget_tokens in budgets:
                        condition = (
                            mode.value,
                            query_conditioned,
                            direction.value,
                            granularity.value,
                            top_k,
                            budget_tokens,
                        )
                        if mode is CrossDocumentExpansionMode.NONE:
                            condition = (mode.value,)
                        if condition in seen_conditions:
                            continue
                        seen_conditions.add(condition)
                        budget = CrossDocumentBudget(top_k, budget_tokens)
                        request = CrossDocumentExpansionRequest(
                            query=question.question,
                            selection_receipt_id=selection.receipt_id,
                            selected_records=records,
                            candidate_chunks_by_record=candidates_by_record,
                            budget=budget,
                            authorized_record_uris=frozenset(candidates_by_record),
                            resident_chunk_ids=(
                                frozenset(row.chunk_id for row in prepared.chunks)
                                if all_candidate_kv_resident
                                else frozenset(row.chunk.chunk_id for row in context.chunks)
                            ),
                            gold_chunk_ids=gold,
                        )
                        config = CrossDocumentExpansionConfig(
                            mode=mode,
                            query_conditioned=query_conditioned,
                            direction=direction,
                            granularity=granularity,
                            budget=budget,
                        )
                        policy = build_cross_document_expansion_policy(
                            config,
                            semantic_encoder=semantic_encoder,
                            allow_experimental=True,
                        )
                        proposal_key = (
                            mode.value,
                            query_conditioned,
                            direction.value,
                            granularity.value,
                        )
                        cached_policy = proposal_cache.get(proposal_key)
                        if cached_policy is None:
                            proposal_started = time.perf_counter()
                            candidates = policy.propose(request)
                            proposal_latency_ms = (
                                time.perf_counter() - proposal_started
                            ) * 1000.0
                            cached_policy = _FrozenProposalPolicy(
                                policy, candidates, proposal_latency_ms
                            )
                            proposal_cache[proposal_key] = cached_policy
                        plan = CrossDocumentExpansionRuntime().expand(
                            request, cached_policy
                        )
                        expanded = _expanded_context(context, records, plan)
                        metrics = context_metrics(question, candidate, expanded)
                        expanded_chunks = tuple(row.chunk for row in expanded.chunks)
                        final_span_hits = _span_hits(question, expanded_chunks)
                        expanded_gold = sum(
                            row.target_chunk_id in gold for row in plan.spans
                        )
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "example_id": question.example_id,
                            "question_type": question.question_type,
                            "condition": mode.value.upper(),
                            "mode": mode.value,
                            "query_conditioned": query_conditioned,
                            "direction": direction.value,
                            "granularity": granularity.value,
                            "top_k_per_pair": top_k,
                            "max_extra_tokens": budget_tokens,
                            "selection_receipt_id": selection.receipt_id,
                            "expansion_receipt_id": plan.receipt.receipt_id,
                            "selected_record_count": len(records),
                            "initial_native_tokens": context.packed_tokens,
                            "requested_cross_tokens": plan.receipt.requested_tokens,
                            "deduplicated_cross_tokens": plan.receipt.deduplicated_tokens,
                            "reused_cross_kv_tokens": plan.receipt.reused_native_tokens,
                            "newly_materialized_cross_tokens": plan.receipt.new_native_tokens,
                            "extra_native_fraction": (
                                plan.receipt.deduplicated_tokens / context.packed_tokens
                                if context.packed_tokens else 0.0
                            ),
                            "cross_search_latency_ms": plan.receipt.search_latency_ms,
                            "initial_supporting_document_coverage": initial_metrics[
                                "supporting_document_coverage"
                            ],
                            "supporting_document_coverage": metrics[
                                "supporting_document_coverage"
                            ],
                            "initial_supporting_span_coverage": initial_metrics[
                                "supporting_span_coverage"
                            ],
                            "supporting_span_coverage": metrics[
                                "supporting_span_coverage"
                            ],
                            "initial_gold_chunk_recall": initial_metrics["gold_chunk_recall"],
                            "gold_chunk_recall": metrics["gold_chunk_recall"],
                            "initial_answer_string_availability": initial_metrics[
                                "answer_string_availability"
                            ],
                            "answer_string_availability": metrics[
                                "answer_string_availability"
                            ],
                            "newly_exposed_supporting_spans": max(0, final_span_hits - initial_span_hits),
                            "expanded_supporting_chunks": expanded_gold,
                            "expanded_span_count": len(plan.spans),
                            "distractor_fraction": (
                                1.0 - expanded_gold / len(plan.spans) if plan.spans else 0.0
                            ),
                            "failures": list(plan.receipt.failures),
                            "evidence_scope": "retrieval_mechanism_not_answer_quality",
                        }
                        rows.append(row)
                        receipts.append(plan.receipt.to_dict())
    return rows, receipts


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            row["mode"], row["query_conditioned"], row["direction"],
            row["granularity"], row["top_k_per_pair"], row["max_extra_tokens"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    mean_fields = (
        "initial_native_tokens",
        "requested_cross_tokens",
        "deduplicated_cross_tokens",
        "reused_cross_kv_tokens",
        "newly_materialized_cross_tokens",
        "extra_native_fraction",
        "cross_search_latency_ms",
        "initial_supporting_document_coverage",
        "supporting_document_coverage",
        "initial_supporting_span_coverage",
        "supporting_span_coverage",
        "initial_gold_chunk_recall",
        "gold_chunk_recall",
        "initial_answer_string_availability",
        "answer_string_availability",
        "newly_exposed_supporting_spans",
        "distractor_fraction",
    )
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = {
            "mode": key[0],
            "query_conditioned": key[1],
            "direction": key[2],
            "granularity": key[3],
            "top_k_per_pair": key[4],
            "max_extra_tokens": key[5],
            "examples": len(values),
        }
        for field in mean_fields:
            row[field + "_mean"] = statistics.fmean(float(value[field]) for value in values)
        row["supporting_span_coverage_delta"] = (
            row["supporting_span_coverage_mean"]
            - row["initial_supporting_span_coverage_mean"]
        )
        row["gold_chunk_recall_delta"] = (
            row["gold_chunk_recall_mean"] - row["initial_gold_chunk_recall_mean"]
        )
        result.append(row)
    return result


def _write_plots(summary: Sequence[Mapping[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in summary if row["mode"] != "none"]
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    for mode in sorted({str(row["mode"]) for row in rows}):
        selected = sorted(
            (
                row for row in rows
                if row["mode"] == mode
                and row["direction"] == "symmetric"
                and int(row["top_k_per_pair"]) == 1
                and bool(row["query_conditioned"])
            ),
            key=lambda row: float(row["deduplicated_cross_tokens_mean"]),
        )
        if selected:
            axis.plot(
                [float(row["deduplicated_cross_tokens_mean"]) for row in selected],
                [float(row["supporting_span_coverage_delta"]) for row in selected],
                marker="o",
                label=mode,
            )
    axis.set_xlabel("Mean deduplicated cross-document tokens")
    axis.set_ylabel("Supporting-span coverage gain")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "support_gain_vs_cross_tokens.pdf")
    fig.savefig(output / "support_gain_vs_cross_tokens.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    for mode in sorted({str(row["mode"]) for row in rows}):
        selected = [
            row for row in rows
            if row["mode"] == mode
            and row["direction"] == "symmetric"
            and int(row["top_k_per_pair"]) == 1
            and bool(row["query_conditioned"])
        ]
        if selected:
            axis.scatter(
                [100 * float(row["distractor_fraction_mean"]) for row in selected],
                [100 * float(row["supporting_span_coverage_delta"]) for row in selected],
                label=mode,
                alpha=0.8,
            )
    axis.set_xlabel("Expanded-span distractor fraction (%)")
    axis.set_ylabel("Supporting-span coverage gain (percentage points)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "support_gain_vs_distractors.pdf")
    fig.savefig(output / "support_gain_vs_distractors.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fixture", "multihoprag"), default="fixture")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-name", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=16)
    parser.add_argument("--max-resources", type=int, default=3)
    parser.add_argument("--selector", choices=("bm25", "cross_encoder"), default="bm25")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision")
    parser.add_argument("--reranker-device", default="auto")
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dense-revision", default="main")
    parser.add_argument("--dense-device", default="auto")
    parser.add_argument(
        "--modes",
        type=lambda value: _csv_enum(value, CrossDocumentExpansionMode),
        default=tuple(
            CrossDocumentExpansionMode(value)
            for value in "none,lexical,dense,hybrid_rrf,entity,oracle".split(",")
        ),
    )
    parser.add_argument(
        "--directions",
        type=lambda value: _csv_enum(value, CrossDocumentDirection),
        default=(CrossDocumentDirection.SYMMETRIC,),
    )
    parser.add_argument(
        "--granularity",
        type=CrossDocumentGranularity,
        choices=tuple(CrossDocumentGranularity),
        default=CrossDocumentGranularity.CHUNK,
    )
    parser.add_argument("--query-conditioning", type=_csv_bool, default=(False, True))
    parser.add_argument("--top-k", type=_csv_int, default=(1, 2, 4, 8))
    parser.add_argument("--cross-token-budgets", type=_csv_int, default=(32, 64, 128, 256, 512))
    parser.add_argument("--all-candidate-kv-resident", action="store_true")
    parser.add_argument(
        "--selection-cache",
        type=Path,
        help="Resumable cache that freezes first-stage selections across experiment arms.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples <= 0 or args.max_resources < 2:
        parser.error("max-examples must be positive and max-resources must be at least two")

    if args.dataset == "fixture":
        documents, questions, metadata = _controlled_crossdoc_fixture(seed=args.seed)
        split_metadata = None
        questions = questions[: args.max_examples]
    else:
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
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    if args.selector == "cross_encoder":
        reranker_revision = _resolve_hf_revision(
            args.reranker, args.reranker_revision or "main"
        )
        selector = CrossEncoderRAGSelector(
            model_id=args.reranker,
            revision=reranker_revision,
            device=_resolve_reranker_device(args.reranker_device),
            name_prefix="paper3_3_crossdoc_expansion",
        )
    else:
        reranker_revision = None
        selector = StandardRAGSelector()
    uses_dense = any(
        mode
        in {
            CrossDocumentExpansionMode.DENSE,
            CrossDocumentExpansionMode.HYBRID_RRF,
            CrossDocumentExpansionMode.HYBRID_WEIGHTED,
        }
        for mode in args.modes
    )
    if uses_dense:
        dense_revision = _resolve_hf_revision(args.dense_model, args.dense_revision)
        semantic_encoder = SentenceTransformerEmbedder(
            args.dense_model,
            revision=dense_revision,
            device=_resolve_reranker_device(args.dense_device),
            query_prefix="Represent this sentence for searching relevant passages: ",
        )
    else:
        dense_revision = None
        semantic_encoder = None

    rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    selection_cache = load_selection_cache(args.selection_cache)
    started = time.time()
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.example_id}", flush=True)
        candidate = make_candidate_receipt(
            dataset=args.dataset,
            dataset_revision=metadata["dataset_revision"],
            corpus_revision=metadata["corpus_revision"],
            corpus_sha256=metadata["corpus_sha256"],
            question=question,
            retriever=retriever,
            candidate_count=args.candidate_count,
            chunker=chunker,
            seed=args.seed,
        )
        prepared = prepare_candidate_context(candidate, by_id)
        if args.dataset == "fixture":
            context = _fixture_initial_context(
                question, prepared, token_budget=args.token_budget
            )
        else:
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
        if len({row.chunk.document_id for row in context.chunks}) < 2:
            continue
        selection = SelectionReceipt.from_context(
            candidate_receipt_id=candidate.receipt_id,
            example_id=question.example_id,
            context=context,
            selector_revision=selector.name,
        )
        question_rows, question_receipts = evaluate_question(
            dataset=args.dataset,
            question=question,
            candidate=candidate,
            prepared=prepared,
            context=context,
            selection=selection,
            modes=args.modes,
            directions=args.directions,
            granularity=args.granularity,
            query_conditioning=args.query_conditioning,
            top_ks=args.top_k,
            budgets=args.cross_token_budgets,
            all_candidate_kv_resident=args.all_candidate_kv_resident,
            semantic_encoder=semantic_encoder,
        )
        rows.extend(question_rows)
        receipts.extend(question_receipts)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output / "receipts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts), encoding="utf-8"
    )
    run = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "dataset": args.dataset,
        "dataset_metadata": dict(metadata),
        "split_name": args.split_name if args.dataset != "fixture" else None,
        "split_digest": split_metadata.get("split_digest") if split_metadata else None,
        "seed": args.seed,
        "question_count_requested": args.max_examples,
        "question_count_evaluated": len({row["example_id"] for row in rows}),
        "selector": selector.name,
        "reranker_revision": reranker_revision,
        "dense_encoder": (
            semantic_encoder.identity if semantic_encoder is not None else None
        ),
        "dense_revision": dense_revision,
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "chunker": asdict(chunker),
        "max_resources": args.max_resources,
        "modes": [row.value for row in args.modes],
        "directions": [row.value for row in args.directions],
        "granularity": args.granularity.value,
        "query_conditioning": list(args.query_conditioning),
        "top_k": list(args.top_k),
        "cross_token_budgets": list(args.cross_token_budgets),
        "all_candidate_kv_resident": args.all_candidate_kv_resident,
        "selection_cache": str(args.selection_cache) if args.selection_cache else None,
        "elapsed_s": time.time() - started,
        "evidence_scope": "retrieval_mechanism_not_answer_quality",
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_plots(summary, args.output)
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "output": str(args.output)}))


if __name__ == "__main__":
    main()
