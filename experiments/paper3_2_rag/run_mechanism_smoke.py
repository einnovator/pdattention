"""Run the five-seed model-free RAG+PRA receipt and geometry smoke."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

from experiments.rag_vs_pra.datasets import controlled_fixture
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    SelectorRole,
    compose_resources,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    PRAHybridSelector,
    PackedContext,
    RankedChunk,
    SelectionReceipt,
    packed_context_from_ranking,
    prepare_candidate_context,
    select_context,
)
from pra_hf.rag_retrieval import (
    ExactDenseRetriever,
    HybridRetriever,
    make_backend_candidate_receipt,
)
from pra_hf.rag_evaluation import FirstStageBM25


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper3_2_rag/mechanism_smoke"
SEEDS = (11, 23, 37, 71, 101)


def _external_order_context(receipt, documents, token_budget: int) -> PackedContext:
    """Pack chunks in frozen first-stage document order without reranking."""

    prepared = prepare_candidate_context(receipt, documents)
    candidates = {row.document_id: row for row in receipt.candidates}
    ordered_chunks = sorted(
        prepared.chunks,
        key=lambda row: (candidates[row.document_id].rank, row.ordinal, row.chunk_id),
    )
    ranked = tuple(
        RankedChunk(
            chunk=chunk,
            score=float(candidates[chunk.document_id].score),
            rank=rank,
            channel_ranks={"external": candidates[chunk.document_id].rank},
        )
        for rank, chunk in enumerate(ordered_chunks, 1)
    )
    return packed_context_from_ranking(
        condition=ContextCondition.NO_PRA_STANDARD_RAG,
        selector_name=f"{receipt.retriever}:frozen_document_order",
        ranked=ranked,
        prepared=prepared,
        token_budget=token_budget,
        selector_latency_ms=0.0,
    )


def _selected_resources(context: PackedContext) -> tuple[SelectedResource, ...]:
    output = []
    stride = 14
    for rank, row in enumerate(context.chunks, 1):
        logical_start = row.chunk.ordinal * stride
        positions = tuple(range(logical_start, logical_start + row.chunk.token_count))
        output.append(
            SelectedResource(
                resource_id=row.chunk.document_id,
                chunk_id=row.chunk.chunk_id,
                source_sha256=hashlib.sha256(row.chunk.text.encode("utf-8")).hexdigest(),
                source_positions=positions,
                rank=rank,
                score=float(row.score),
            )
        )
    return tuple(output)


def _profile_policies(profile: RAGPRAProfile) -> tuple[PositionPolicy, ...]:
    if profile in {
        RAGPRAProfile.RAG_ONLY_TEXT,
        RAGPRAProfile.RAG_PLUS_PRA_SELECTED,
        RAGPRAProfile.RAG_PLUS_PRA_NATIVE_CONTIGUOUS,
    }:
        return (PositionPolicy.GLOBAL_PACKED,)
    if profile is RAGPRAProfile.RAG_PLUS_PRA_NATIVE_INDEPENDENT:
        return (PositionPolicy.SOURCE_LOCAL,)
    return tuple(PositionPolicy)


def _collision_fraction(receipt) -> float:
    positions = [
        position
        for placement in receipt.placements
        for position in placement.effective_positions
    ]
    return 1.0 - len(set(positions)) / max(len(positions), 1)


def _coverage(selected_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    return len(set(selected_ids) & gold) / max(len(gold), 1)


def run(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    chunker = ChunkerConfig(max_tokens=16, overlap_tokens=2)
    for seed in SEEDS:
        documents, questions, metadata = controlled_fixture(seed=seed, document_count=60)
        by_id = {row.document_id: row for row in documents}
        bm25 = FirstStageBM25(documents)
        dense = ExactDenseRetriever(documents, dimensions=256)
        hybrid = HybridRetriever({"bm25": bm25, "dense": dense}, documents)
        retrievers = {"bm25": bm25, "dense_exact": dense, "hybrid_rrf": hybrid}

        for question in questions:
            for backend_name, retriever in retrievers.items():
                candidate = make_backend_candidate_receipt(
                    dataset="controlled_fixture",
                    dataset_revision=str(metadata["dataset_revision"]),
                    corpus_revision=str(metadata["corpus_revision"]),
                    corpus_sha256=str(metadata["corpus_sha256"]),
                    question=question,
                    retriever=retriever,
                    retriever_id=backend_name,
                    documents=by_id,
                    candidate_count=10,
                    chunker=chunker,
                    seed=seed,
                )
                contexts = {
                    SelectorRole.EXTERNAL_ONLY: _external_order_context(
                        candidate, by_id, token_budget=64
                    ),
                    SelectorRole.PRA_SECOND_STAGE: select_context(
                        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
                        selector=PRAHybridSelector(),
                        query=question.question,
                        receipt=candidate,
                        documents=by_id,
                        token_budget=64,
                    ),
                }
                selections = {
                    role: SelectionReceipt.from_context(
                        candidate_receipt_id=candidate.receipt_id,
                        example_id=question.example_id,
                        context=context,
                    )
                    for role, context in contexts.items()
                }

                for profile in RAGPRAProfile:
                    role = (
                        SelectorRole.PRA_SECOND_STAGE
                        if profile is RAGPRAProfile.RAG_PLUS_PRA_SELECTED
                        else SelectorRole.EXTERNAL_ONLY
                    )
                    context = contexts[role]
                    selection = selections[role]
                    resources = _selected_resources(context)
                    for policy in _profile_policies(profile):
                        composition = compose_resources(
                            resources,
                            selection_receipt_id=selection.receipt_id,
                            profile=profile,
                            selector_role=role,
                            position_policy=policy,
                            random_seed=seed,
                            repair_fraction=(
                                0.1
                                if profile is RAGPRAProfile.RAG_PLUS_PRA_REPAIR
                                else 0.0
                            ),
                        )
                        rows.append(
                            {
                                "status": "MECHANISM_ONLY",
                                "answer_quality_publishable": False,
                                "seed": seed,
                                "example_id": question.example_id,
                                "question_type": question.question_type,
                                "retrieval_backend": backend_name,
                                "profile": profile.value,
                                "selector_role": role.value,
                                "position_policy": policy.value,
                                "candidate_receipt_id": candidate.receipt_id,
                                "selection_receipt_id": selection.receipt_id,
                                "composition_receipt_id": composition.receipt_id,
                                "candidate_document_recall": _coverage(
                                    candidate.candidate_document_ids,
                                    question.gold_document_ids,
                                ),
                                "selected_document_recall": _coverage(
                                    context.selected_document_ids,
                                    question.gold_document_ids,
                                ),
                                "candidate_documents": len(candidate.candidates),
                                "selected_resources": len(resources),
                                "selected_native_tokens": sum(
                                    len(row.source_positions) for row in resources
                                ),
                                "position_collision_fraction": _collision_fraction(
                                    composition
                                ),
                                "exact_match": None,
                                "token_f1": None,
                                "gold_log_probability": None,
                            }
                        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mechanism_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(output_dir / "mechanism_rows.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def summarize(rows: Sequence[Mapping[str, object]], output_dir: Path) -> dict[str, object]:
    candidates: dict[str, Mapping[str, object]] = {}
    selections: dict[str, Mapping[str, object]] = {}
    policies: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        candidates.setdefault(str(row["candidate_receipt_id"]), row)
        selections.setdefault(str(row["selection_receipt_id"]), row)
        policies[str(row["position_policy"])].append(row)
    retrieval: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    selected: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in candidates.values():
        retrieval[str(row["retrieval_backend"])].append(row)
    for row in selections.values():
        selected[(str(row["retrieval_backend"]), str(row["selector_role"]))].append(row)
    summary = {
        "status": "MECHANISM_ONLY",
        "answer_quality_publishable": False,
        "seeds": list(SEEDS),
        "examples_per_seed": len(
            {(row["seed"], row["example_id"]) for row in rows}
        )
        // len(SEEDS),
        "rows": len(rows),
        "retrieval": {
            name: {
                "candidate_document_recall": statistics.fmean(
                    float(row["candidate_document_recall"]) for row in values
                ),
                "external_selected_document_recall": statistics.fmean(
                    float(row["selected_document_recall"])
                    for row in selected[(name, SelectorRole.EXTERNAL_ONLY.value)]
                ),
                "pra_second_stage_document_recall": statistics.fmean(
                    float(row["selected_document_recall"])
                    for row in selected[(name, SelectorRole.PRA_SECOND_STAGE.value)]
                ),
            }
            for name, values in sorted(retrieval.items())
        },
        "position_collision_fraction": {
            name: statistics.fmean(
                float(row["position_collision_fraction"]) for row in values
            )
            for name, values in sorted(policies.items())
        },
        "profile_count": len({row["profile"] for row in rows}),
        "position_policy_count": len({row["position_policy"] for row in rows}),
        "candidate_receipts": len({row["candidate_receipt_id"] for row in rows}),
        "selection_receipts": len({row["selection_receipt_id"] for row in rows}),
        "composition_receipts": len({row["composition_receipt_id"] for row in rows}),
    }
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def plot(summary: Mapping[str, object], output_dir: Path) -> None:
    retrieval = summary["retrieval"]
    names = list(retrieval)
    candidate = [retrieval[name]["candidate_document_recall"] for name in names]
    external = [retrieval[name]["external_selected_document_recall"] for name in names]
    pra = [retrieval[name]["pra_second_stage_document_recall"] for name in names]
    x = range(len(names))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([value - 0.26 for value in x], candidate, width=0.26, label="Candidate Recall@10")
    axis.bar(list(x), external, width=0.26, label="External-order selected")
    axis.bar([value + 0.26 for value in x], pra, width=0.26, label="PRA second-stage selected")
    axis.set_xticks(list(x), names)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Document recall")
    axis.set_title("Synthetic retrieval and frozen-selection mechanism smoke")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "retrieval_recall.png", dpi=180)
    figure.savefig(output_dir / "retrieval_recall.pdf")
    plt.close(figure)

    collisions = summary["position_collision_fraction"]
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(list(collisions), list(collisions.values()), color="#357a6b")
    axis.set_ylabel("Effective-position collision fraction")
    axis.set_title("Declared overlap by composition policy")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_dir / "position_collisions.png", dpi=180)
    figure.savefig(output_dir / "position_collisions.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = run(args.output_dir)
    summary = summarize(rows, args.output_dir)
    plot(summary, args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "paper3_2_rag_mechanism_smoke_v1",
                "status": "MECHANISM_ONLY",
                "answer_quality_publishable": False,
                "seeds": list(SEEDS),
                "retrieval_backends": ["bm25", "dense_exact", "hybrid_rrf"],
                "profiles": [row.value for row in RAGPRAProfile],
                "position_policies": [row.value for row in PositionPolicy],
                "summary": "mechanism_summary.json",
                "rows": ["mechanism_rows.csv", "mechanism_rows.jsonl.gz"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
