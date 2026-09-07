"""Reconcile the Paper 3.2 and Paper 3.3 packed-versus-PRA endpoints.

The experiment introduces no new PRA mechanism.  It freezes BM25-v2 candidate
retrieval and cross-encoder ranking once, then realizes the same selected chunks
as either one packed causal prefix or independently encoded source-local K/V.
The common 128-token geometry is evaluated at 512 and 1,024 source-token
budgets on both historical cohorts.  A final old-cohort cell keeps Paper 3.2's
256-token chunk geometry while changing only BM25 v1 to v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _execute,
    _token_segments,
)
from experiments.rag_vs_pra.datasets import load_multihop_rag
from experiments.rag_vs_pra.run_powered_decomposition import (
    PersistentMLXBackend,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25V2,
    PackedContext,
    RankedChunk,
    SelectionReceipt,
    make_candidate_receipt,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import combine_native_memories, encode_native_memory


SCHEMA_VERSION = "paper3.2-paper3.3-protocol-reconciliation-v1"
SELECTION_SCHEMA_VERSION = "paper3.2-protocol-selection-v1"
PAPER33_SELECTION_SCHEMA_VERSION = "paper3.3-frozen-selection-cache-v1"
PAPER33_MODEL_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
PAPER33_RERANKER = "BAAI/bge-reranker-v2-m3"
PAPER33_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@dataclass(frozen=True)
class Geometry:
    """Candidate and chunk geometry shared by one or more audit cells."""

    name: str
    candidate_count: int
    chunk_tokens: int
    chunk_overlap: int
    build_token_budget: int
    build_max_resources: int


@dataclass(frozen=True)
class AuditCell:
    """One cohort, source-token budget, and frozen-selection realization."""

    name: str
    cohort: str
    token_budget: int
    max_resources: int
    geometry: Geometry


COMMON_GEOMETRY = Geometry("common_128", 10, 128, 16, 1024, 8)
LEGACY_GEOMETRY_V2 = Geometry("paper32_256_bm25v2", 50, 256, 32, 1024, 4)
CELLS = (
    AuditCell("paper33_test_512", "paper33_test", 512, 4, COMMON_GEOMETRY),
    AuditCell("paper33_test_1024", "paper33_test", 1024, 8, COMMON_GEOMETRY),
    AuditCell("paper32_eval_512", "paper32_eval", 512, 4, COMMON_GEOMETRY),
    AuditCell("paper32_eval_1024", "paper32_eval", 1024, 8, COMMON_GEOMETRY),
    AuditCell(
        "paper32_eval_legacy_geometry_bm25v2",
        "paper32_eval",
        1024,
        4,
        LEGACY_GEOMETRY_V2,
    ),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_cohort_ids(
    paper32_manifest: Path, paper33_anchor: Path
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, object]]]:
    paper32 = json.loads(paper32_manifest.read_text(encoding="utf-8"))
    old_ids = tuple(str(value) for value in paper32["eval_example_ids"])
    anchor_rows = _load_jsonl(paper33_anchor)
    if any(row.get("schema_version") != PAPER33_SELECTION_SCHEMA_VERSION for row in anchor_rows):
        raise ValueError("Paper 3.3 anchor has an unsupported schema")
    anchor = {str(row["example_id"]): row for row in anchor_rows}
    if len(anchor) != len(anchor_rows):
        raise ValueError("Paper 3.3 anchor contains duplicate question identities")
    return {
        "paper32_eval": old_ids,
        "paper33_test": tuple(anchor),
    }, anchor


def _select_under_budget(
    ranking: Sequence[RankedChunk], *, token_budget: int, max_resources: int
) -> tuple[RankedChunk, ...]:
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
    return tuple(selected)


def _load_selection_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows = _load_jsonl(path)
    if any(row.get("schema_version") != SELECTION_SCHEMA_VERSION for row in rows):
        raise ValueError(f"unsupported audit selection schema in {path}")
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate audit selections in {path}")
    return result


def _ranked_from_cache(
    row: Mapping[str, object], prepared
) -> tuple[RankedChunk, ...]:
    chunks = {chunk.chunk_id: chunk for chunk in prepared.chunks}
    result = []
    for selected in row["selected"]:
        chunk_id = str(selected["chunk_id"])
        if chunk_id not in chunks:
            raise ValueError(f"cached chunk {chunk_id!r} is absent from candidate receipt")
        result.append(
            RankedChunk(
                chunks[chunk_id],
                float(selected["score"]),
                int(selected["rank"]),
                {
                    str(key): int(value)
                    for key, value in dict(selected["channel_ranks"]).items()
                },
            )
        )
    return tuple(result)


def _frozen_ranking(
    *,
    path: Path,
    cache: dict[str, dict[str, object]],
    question,
    candidate,
    prepared,
    selector,
    geometry: Geometry,
) -> tuple[tuple[RankedChunk, ...], float]:
    cached = cache.get(question.example_id)
    expected = {
        "candidate_receipt_id": candidate.receipt_id,
        "selector_name": selector.name,
        "geometry": geometry.name,
        "build_token_budget": geometry.build_token_budget,
        "build_max_resources": geometry.build_max_resources,
    }
    if cached is not None:
        mismatches = [key for key, value in expected.items() if cached.get(key) != value]
        if mismatches:
            raise ValueError(
                f"selection cache mismatch for {question.example_id}: {', '.join(mismatches)}"
            )
        return _ranked_from_cache(cached, prepared), float(cached["selector_latency_ms"])

    started = time.perf_counter()
    ranking = selector.rank(question.question, prepared.chunks)
    latency_ms = (time.perf_counter() - started) * 1000.0
    selected = _select_under_budget(
        ranking,
        token_budget=geometry.build_token_budget,
        max_resources=geometry.build_max_resources,
    )
    row: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "example_id": question.example_id,
        **expected,
        "selector_latency_ms": latency_ms,
        "selected": [
            {
                "chunk_id": value.chunk.chunk_id,
                "score": value.score,
                "rank": value.rank,
                "channel_ranks": dict(value.channel_ranks),
            }
            for value in selected
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    cache[question.example_id] = row
    return selected, latency_ms


def _context(
    *,
    ranked: Sequence[RankedChunk],
    prepared,
    selector_name: str,
    selector_latency_ms: float,
    token_budget: int,
    max_resources: int,
) -> PackedContext:
    selected = _select_under_budget(
        ranked, token_budget=token_budget, max_resources=max_resources
    )
    return PackedContext(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        chunks=selected,
        token_budget=token_budget,
        packed_tokens=sum(row.chunk.token_count for row in selected),
        candidate_tokens=prepared.candidate_tokens,
        selector_latency_ms=selector_latency_ms,
        index_build_ms=prepared.build_latency_ms,
        selector_name=selector_name,
        candidate_chunks=prepared.chunks,
    )


def _selection_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(value["chunk_id"]) for value in row["selected"])


def _validate_paper33_anchor(
    *, cell: AuditCell, question_id: str, context: PackedContext, anchor: Mapping[str, object]
) -> None:
    if cell.name != "paper33_test_512":
        return
    observed = tuple(row.chunk.chunk_id for row in context.chunks)
    expected = _selection_ids(anchor)
    if observed != expected:
        raise ValueError(
            f"Paper 3.3 512-token selection changed for {question_id}: "
            f"expected {expected}, observed {observed}"
        )


def _condition_row(
    *,
    cell: AuditCell,
    question,
    candidate,
    selection: SelectionReceipt,
    context: PackedContext,
    condition: str,
    memory,
    encode_ms: float,
    execution,
    reference_logits,
) -> dict[str, object]:
    prediction, metrics, first_step_logits = execution
    selected_documents = tuple(row.chunk.document_id for row in context.chunks)
    return {
        "schema_version": SCHEMA_VERSION,
        "cell": cell.name,
        "cohort": cell.cohort,
        "geometry": cell.geometry.name,
        "source_token_budget": cell.token_budget,
        "max_resources": cell.max_resources,
        "condition": condition,
        "example_id": question.example_id,
        "question_type": question.question_type,
        "candidate_receipt_id": candidate.receipt_id,
        "selection_receipt_id": selection.receipt_id,
        "selected_chunk_ids": [row.chunk.chunk_id for row in context.chunks],
        "selected_document_ids": list(selected_documents),
        "selected_source_tokens": context.packed_tokens,
        "selected_native_tokens": memory.source_tokens,
        "selected_native_bytes": memory.nbytes,
        "supporting_document_coverage": len(
            set(selected_documents).intersection(question.gold_document_ids)
        )
        / max(len(question.gold_document_ids), 1),
        "prediction": prediction,
        "gold_answers": list(question.answers),
        "encode_ms": encode_ms,
        **_distribution_diagnostics(reference_logits, first_step_logits),
        **metrics,
    }


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    cells = sorted({str(row["cell"]) for row in rows})
    for cell in cells:
        for condition in ("PACKED_RAG", "INDEPENDENT_PRA"):
            values = [
                row
                for row in rows
                if row["cell"] == cell and row["condition"] == condition
            ]
            if not values:
                continue
            result.append(
                {
                    "cell": cell,
                    "cohort": values[0]["cohort"],
                    "geometry": values[0]["geometry"],
                    "source_token_budget": values[0]["source_token_budget"],
                    "condition": condition,
                    "examples": len(values),
                    "token_f1": _mean(values, "token_f1"),
                    "official_score": _mean(values, "official_multihop_rag_score"),
                    "gold_answer_mean_nll": _mean(values, "gold_answer_mean_nll"),
                    "selected_source_tokens": _mean(values, "selected_source_tokens"),
                    "selected_native_tokens": _mean(values, "selected_native_tokens"),
                    "supporting_document_coverage": _mean(
                        values, "supporting_document_coverage"
                    ),
                    "encode_ms": _mean(values, "encode_ms"),
                    "ttft_ms": _mean(values, "ttft_ms"),
                    "total_latency_ms": _mean(values, "total_latency_ms"),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument(
        "--paper32-manifest",
        type=Path,
        default=Path(
            "docs/papers/shared/results/paper3_2_rag/crossdoc_adapter/"
            "qwen3_1_7b_rank8_five_seed/manifest.json"
        ),
    )
    parser.add_argument("--paper33-selection-anchor", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--model-revision", default=PAPER33_MODEL_REVISION)
    parser.add_argument("--reranker", default=PAPER33_RERANKER)
    parser.add_argument("--reranker-revision", default=PAPER33_RERANKER_REVISION)
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-paper32-examples", type=int, default=30)
    parser.add_argument("--max-paper33-examples", type=int, default=150)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort_ids, paper33_anchor = _load_cohort_ids(
        args.paper32_manifest, args.paper33_selection_anchor
    )
    cohort_ids["paper32_eval"] = cohort_ids["paper32_eval"][: args.max_paper32_examples]
    cohort_ids["paper33_test"] = cohort_ids["paper33_test"][: args.max_paper33_examples]
    documents, questions, metadata = load_multihop_rag(args.cache_dir)
    question_by_id = {question.example_id: question for question in questions}
    document_by_id = {document.document_id: document for document in documents}
    missing = sorted(
        {
            example_id
            for values in cohort_ids.values()
            for example_id in values
            if example_id not in question_by_id
        }
    )
    if missing:
        raise ValueError(f"cohort identities absent from pinned dataset: {missing[:5]}")

    retriever = FirstStageBM25V2(documents)
    reranker_revision = _resolve_hf_revision(args.reranker, args.reranker_revision)
    selector = CrossEncoderRAGSelector(
        model_id=args.reranker,
        revision=reranker_revision,
        device=args.reranker_device,
        name_prefix="paper3_3_crossdoc_expansion",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rows.jsonl"
    rows = _load_jsonl(rows_path) if rows_path.exists() else []
    completed = {
        (str(row["cell"]), str(row["example_id"]), str(row["condition"]))
        for row in rows
    }
    backend = None
    model_revision = None
    started = time.time()

    geometry_cache: dict[tuple[str, str], tuple[Path, dict[str, dict[str, object]]]] = {}
    for cell in CELLS:
        ids = cohort_ids[cell.cohort]
        cache_key = (cell.cohort, cell.geometry.name)
        if cache_key not in geometry_cache:
            cache_path = args.output / "selections" / f"{cell.cohort}_{cell.geometry.name}.jsonl"
            geometry_cache[cache_key] = (cache_path, _load_selection_cache(cache_path))
        cache_path, selection_cache = geometry_cache[cache_key]
        chunker = ChunkerConfig(cell.geometry.chunk_tokens, cell.geometry.chunk_overlap)

        for index, example_id in enumerate(ids, 1):
            print(f"[{cell.name} {index}/{len(ids)}] {example_id}", flush=True)
            question = question_by_id[example_id]
            candidate = make_candidate_receipt(
                dataset="multihoprag",
                dataset_revision=metadata["dataset_revision"],
                corpus_revision=metadata["corpus_revision"],
                corpus_sha256=metadata["corpus_sha256"],
                question=question,
                retriever=retriever,
                candidate_count=cell.geometry.candidate_count,
                chunker=chunker,
                seed=11,
            )
            prepared = prepare_candidate_context(candidate, document_by_id)
            ranked, selector_latency_ms = _frozen_ranking(
                path=cache_path,
                cache=selection_cache,
                question=question,
                candidate=candidate,
                prepared=prepared,
                selector=selector,
                geometry=cell.geometry,
            )
            context = _context(
                ranked=ranked,
                prepared=prepared,
                selector_name=selector.name,
                selector_latency_ms=selector_latency_ms,
                token_budget=cell.token_budget,
                max_resources=cell.max_resources,
            )
            if len(context.chunks) < 2:
                raise RuntimeError(f"{example_id} selected fewer than two records")
            if cell.cohort == "paper33_test":
                _validate_paper33_anchor(
                    cell=cell,
                    question_id=example_id,
                    context=context,
                    anchor=paper33_anchor[example_id],
                )
            if args.selection_only:
                continue
            if all(
                (cell.name, example_id, condition) in completed
                for condition in ("PACKED_RAG", "INDEPENDENT_PRA")
            ):
                continue
            if backend is None:
                model_revision = _resolve_hf_revision(args.model, args.model_revision)
                backend = PersistentMLXBackend(
                    args.model, model_revision, args.max_new_tokens, native_cache_unit="chunk"
                )

            selection = SelectionReceipt.from_context(
                candidate_receipt_id=candidate.receipt_id,
                example_id=example_id,
                context=context,
                selector_revision=selector.name,
            )
            texts = tuple(row.chunk.text for row in context.chunks)
            segments = _token_segments(backend.tokenizer, texts)
            packed_tokens = tuple(token for segment in segments for token in segment)

            packed_started = time.perf_counter()
            packed_memory = encode_native_memory(
                backend.model, packed_tokens, model_revision=model_revision
            )
            packed_encode_ms = (time.perf_counter() - packed_started) * 1000.0
            packed_execution = _execute(backend, question, packed_memory)
            packed_logits = packed_execution[2]

            independent_started = time.perf_counter()
            independent_memory = combine_native_memories(
                tuple(
                    encode_native_memory(
                        backend.model, segment, model_revision=model_revision
                    )
                    for segment in segments
                )
            )
            independent_encode_ms = (
                time.perf_counter() - independent_started
            ) * 1000.0
            independent_execution = _execute(backend, question, independent_memory)

            new_rows = (
                _condition_row(
                    cell=cell,
                    question=question,
                    candidate=candidate,
                    selection=selection,
                    context=context,
                    condition="PACKED_RAG",
                    memory=packed_memory,
                    encode_ms=packed_encode_ms,
                    execution=packed_execution,
                    reference_logits=packed_logits,
                ),
                _condition_row(
                    cell=cell,
                    question=question,
                    candidate=candidate,
                    selection=selection,
                    context=context,
                    condition="INDEPENDENT_PRA",
                    memory=independent_memory,
                    encode_ms=independent_encode_ms,
                    execution=independent_execution,
                    reference_logits=packed_logits,
                ),
            )
            with rows_path.open("a", encoding="utf-8") as stream:
                for row in new_rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    completed.add((cell.name, example_id, str(row["condition"])))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper3_2_paper3_3_protocol_reconciliation",
        "introduces_new_pra_mechanism": False,
        "git_commit": _git_commit(),
        "dataset": dict(metadata),
        "model": args.model if not args.selection_only else None,
        "model_revision": model_revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "retriever_revision": retriever.revision,
        "paper32_manifest": str(args.paper32_manifest),
        "paper32_manifest_sha256": _file_sha256(args.paper32_manifest),
        "paper33_selection_anchor": str(args.paper33_selection_anchor),
        "paper33_selection_anchor_sha256": _file_sha256(args.paper33_selection_anchor),
        "cohort_ids": {key: list(value) for key, value in cohort_ids.items()},
        "cells": [
            {
                "name": cell.name,
                "cohort": cell.cohort,
                "source_token_budget": cell.token_budget,
                "max_resources": cell.max_resources,
                "geometry": vars(cell.geometry),
            }
            for cell in CELLS
        ],
        "selection_only": args.selection_only,
        "rows": len(rows),
        "summary": _summarize(rows),
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "elapsed_s": time.time() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
