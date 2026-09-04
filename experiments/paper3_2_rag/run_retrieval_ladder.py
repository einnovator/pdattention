"""Evaluate lexical, modern dense/hybrid, and two reranker generations.

This runner stops at frozen candidate receipts.  It measures whether the RAG
baseline retrieves gold documents before any PRA routing or native transport,
preventing weak retrieval from being misattributed to realization.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from experiments.rag_vs_pra.datasets import load_multihop_rag, select_cohort
from experiments.rag_vs_pra.run_powered_decomposition import (
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
)
from pra_hf.rag_evaluation import CandidateDocument, FirstStageBM25
from pra_hf.rag_retrieval import (
    CrossEncoderRerankedRetriever,
    FaissDenseRetriever,
    HybridRetriever,
    SentenceTransformerEmbedder,
)


DEFAULT_DENSE = "BAAI/bge-base-en-v1.5"
HISTORICAL_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STRONG_RERANKER = "BAAI/bge-reranker-v2-m3"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _recall(rows: Sequence[CandidateDocument], gold: frozenset[str]) -> float:
    if not gold:
        return 1.0
    return len({row.document_id for row in rows}.intersection(gold)) / len(gold)


def _mrr(rows: Sequence[CandidateDocument], gold: frozenset[str]) -> float:
    return next((1.0 / row.rank for row in rows if row.document_id in gold), 0.0)


def _aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["top_k"])), []).append(row)
    result = []
    for (method, top_k), values in sorted(grouped.items()):
        latencies = [float(row["latency_ms"]) for row in values]
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))]
        result.append(
            {
                "method": method,
                "top_k": top_k,
                "examples": len(values),
                "supporting_document_recall": statistics.fmean(
                    float(row["supporting_document_recall"]) for row in values
                ),
                "all_supporting_documents_recalled": statistics.fmean(
                    float(row["all_supporting_documents_recalled"]) for row in values
                ),
                "mrr": statistics.fmean(float(row["mrr"]) for row in values),
                "latency_ms_mean": statistics.fmean(latencies),
                "latency_ms_p95": p95,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--dense-model", default=DEFAULT_DENSE)
    parser.add_argument("--dense-revision", default="main")
    parser.add_argument("--historical-reranker", default=HISTORICAL_RERANKER)
    parser.add_argument("--historical-revision", default="main")
    parser.add_argument("--strong-reranker", default=STRONG_RERANKER)
    parser.add_argument("--strong-revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    args = parser.parse_args()

    top_ks = tuple(int(value) for value in args.top_k.split(","))
    if not top_ks or any(value <= 0 for value in top_ks):
        parser.error("--top-k must contain positive integers")
    documents, questions, metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    revisions = {
        "dense": _resolve_hf_revision(args.dense_model, args.dense_revision),
        "historical_reranker": _resolve_hf_revision(
            args.historical_reranker, args.historical_revision
        ),
        "strong_reranker": _resolve_hf_revision(
            args.strong_reranker, args.strong_revision
        ),
    }

    build_times: dict[str, float] = {}
    started = time.perf_counter()
    bm25 = FirstStageBM25(documents)
    build_times["bm25"] = (time.perf_counter() - started) * 1000.0
    embedder = SentenceTransformerEmbedder(
        args.dense_model,
        revision=revisions["dense"],
        device=args.device,
        batch_size=args.batch_size,
        query_prefix="Represent this sentence for searching relevant passages: ",
    )
    started = time.perf_counter()
    dense = FaissDenseRetriever(
        documents,
        dimensions=embedder.dimensions,
        embedder=embedder,
        embedder_revision=embedder.identity,
    )
    build_times["faiss_dense"] = (time.perf_counter() - started) * 1000.0
    hybrid = HybridRetriever({"bm25": bm25, "dense": dense}, documents)
    historical = CrossEncoderRerankedRetriever(
        hybrid,
        documents,
        model_id=args.historical_reranker,
        revision=revisions["historical_reranker"],
        candidate_count=args.rerank_candidates,
        device=args.device,
        batch_size=args.batch_size,
    )
    strong = CrossEncoderRerankedRetriever(
        hybrid,
        documents,
        model_id=args.strong_reranker,
        revision=revisions["strong_reranker"],
        candidate_count=args.rerank_candidates,
        device=args.device,
        batch_size=max(1, args.batch_size // 2),
    )
    methods = {
        "BM25": bm25,
        "FAISS_BGE_BASE_EN_V1_5": dense,
        "HYBRID_RRF": hybrid,
        "HISTORICAL_MINILM_RERANK": historical,
        "STRONG_BGE_V2_M3_RERANK": strong,
    }
    rows: list[dict[str, object]] = []
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.example_id}", flush=True)
        for method, retriever in methods.items():
            started = time.perf_counter()
            ranked = retriever.retrieve(question.question, max(top_ks))
            elapsed = (time.perf_counter() - started) * 1000.0
            for top_k in top_ks:
                selected = ranked[:top_k]
                recall = _recall(selected, question.gold_document_ids)
                rows.append(
                    {
                        "example_id": question.example_id,
                        "method": method,
                        "top_k": top_k,
                        "supporting_document_recall": recall,
                        "all_supporting_documents_recalled": float(recall == 1.0),
                        "mrr": _mrr(selected, question.gold_document_ids),
                        "latency_ms": elapsed,
                        "selected_document_ids": [row.document_id for row in selected],
                        "gold_document_ids": sorted(question.gold_document_ids),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_query.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "paper3.2-retrieval-ladder-v1",
        "dataset": "multihoprag",
        "dataset_metadata": dict(metadata),
        "seed": args.seed,
        "question_ids": [row.example_id for row in questions],
        "models": {
            "dense": {"id": args.dense_model, "revision": revisions["dense"]},
            "historical_reranker": {
                "id": args.historical_reranker,
                "revision": revisions["historical_reranker"],
                "role": "historical Paper-4.5 control",
            },
            "strong_reranker": {
                "id": args.strong_reranker,
                "revision": revisions["strong_reranker"],
            },
        },
        "build_time_ms": build_times,
        "conditions": _aggregate(rows),
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "git_commit": _git_commit(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
