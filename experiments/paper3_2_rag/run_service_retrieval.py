"""Run matched Elasticsearch BM25 and Qdrant dense/hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from experiments.rag_vs_pra.datasets import load_multihop_rag, select_cohort
from experiments.rag_vs_pra.run_powered_decomposition import (
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
)
from pra_hf.rag_retrieval import (
    ElasticsearchBM25Retriever,
    HybridRetriever,
    QdrantDenseRetriever,
    SentenceTransformerEmbedder,
    index_elasticsearch_documents,
    index_qdrant_documents,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elasticsearch", default="http://127.0.0.1:9200")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--index-prefix", default="paper32-multihoprag")
    parser.add_argument("--index-revision", required=True)
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dense-revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    top_ks = tuple(int(value) for value in args.top_k.split(","))
    documents, questions, metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    dense_revision = _resolve_hf_revision(args.dense_model, args.dense_revision)
    embedder = SentenceTransformerEmbedder(
        args.dense_model,
        revision=dense_revision,
        device=args.device,
        batch_size=args.batch_size,
        query_prefix="Represent this sentence for searching relevant passages: ",
    )
    elastic_index = f"{args.index_prefix}-bm25"
    qdrant_collection = f"{args.index_prefix}-dense"
    build_ms = {"elasticsearch": 0.0, "qdrant": 0.0}
    if args.rebuild:
        started = time.perf_counter()
        index_elasticsearch_documents(
            documents,
            endpoint=args.elasticsearch,
            index_name=elastic_index,
        )
        build_ms["elasticsearch"] = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        index_qdrant_documents(
            documents,
            embedder=embedder,
            endpoint=args.qdrant,
            collection_name=qdrant_collection,
        )
        build_ms["qdrant"] = (time.perf_counter() - started) * 1000.0

    elastic = ElasticsearchBM25Retriever(
        documents,
        endpoint=args.elasticsearch,
        index_name=elastic_index,
        index_revision=args.index_revision,
    )
    qdrant = QdrantDenseRetriever(
        documents,
        endpoint=args.qdrant,
        collection_name=qdrant_collection,
        collection_revision=args.index_revision,
        embedder=embedder,
        dimensions=embedder.dimensions,
    )
    methods = {
        "ELASTICSEARCH_BM25": elastic,
        "QDRANT_BGE_DENSE": qdrant,
        "SERVICE_HYBRID_RRF": HybridRetriever(
            {"elasticsearch": elastic, "qdrant": qdrant}, documents
        ),
    }
    rows = []
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.example_id}", flush=True)
        for method, retriever in methods.items():
            started = time.perf_counter()
            ranked = retriever.retrieve(question.question, max(top_ks))
            latency_ms = (time.perf_counter() - started) * 1000.0
            if method == "ELASTICSEARCH_BM25":
                embedding_ms = 0.0
                service_ms = elastic.last_service_ms
            elif method == "QDRANT_BGE_DENSE":
                embedding_ms = qdrant.last_embedding_ms
                service_ms = qdrant.last_service_ms
            else:
                embedding_ms = qdrant.last_embedding_ms
                service_ms = elastic.last_service_ms + qdrant.last_service_ms
            for top_k in top_ks:
                selected = ranked[:top_k]
                recovered = {row.document_id for row in selected}.intersection(
                    question.gold_document_ids
                )
                recall = len(recovered) / max(len(question.gold_document_ids), 1)
                rows.append(
                    {
                        "example_id": question.example_id,
                        "method": method,
                        "top_k": top_k,
                        "supporting_document_recall": recall,
                        "all_supporting_documents_recalled": float(recall == 1.0),
                        "latency_ms": latency_ms,
                        "query_embedding_ms": embedding_ms,
                        "service_search_ms": service_ms,
                        "selected_document_ids": [row.document_id for row in selected],
                        "gold_document_ids": sorted(question.gold_document_ids),
                    }
                )
    aggregate = []
    for method in methods:
        for top_k in top_ks:
            values = [
                row for row in rows if row["method"] == method and row["top_k"] == top_k
            ]
            latencies = [float(row["latency_ms"]) for row in values]
            embedding = [float(row["query_embedding_ms"]) for row in values]
            service = [float(row["service_search_ms"]) for row in values]
            aggregate.append(
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
                    "latency_ms_mean": statistics.fmean(latencies),
                    "latency_ms_p50": _percentile(latencies, 0.50),
                    "latency_ms_p95": _percentile(latencies, 0.95),
                    "latency_ms_p99": _percentile(latencies, 0.99),
                    "query_embedding_ms_mean": statistics.fmean(embedding),
                    "service_search_ms_mean": statistics.fmean(service),
                    "other_client_ms_mean": statistics.fmean(
                        total - embed - search
                        for total, embed, search in zip(latencies, embedding, service)
                    ),
                }
            )
    result = {
        "schema_version": "paper3.2-service-retrieval-v2",
        "dataset": "multihoprag",
        "dataset_metadata": dict(metadata),
        "seed": args.seed,
        "question_ids": [row.example_id for row in questions],
        "dense_model": args.dense_model,
        "dense_revision": dense_revision,
        "service_index_revision": args.index_revision,
        "build_ms": build_ms,
        "conditions": aggregate,
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_query.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
