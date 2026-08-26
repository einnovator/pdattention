"""Run the frozen Paper 3.1 summary-index study on inherited QA identities.

The runner consumes Paper 2.8 feature tensors but reconstructs source text from
the original datasets.  Generated summaries are persisted as routing sidecars;
selected records are translated back to the exact parent source spans.  No
summary text is passed to the native-K/V materializer or answer model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import file_sha256
from experiments.paper2_8_qk_compression.run_multidataset_extension import (
    _lowrank_scores,
    _selector_from_checkpoint,
)
from experiments.paper2_8_qk_compression.run_gated_study import _project_native_queries
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from experiments.paper3_1_summary_index.ollama_sidecar import (
    JSONLGenerationCache,
    OllamaClient,
    PROMPT_VERSION,
    generate_cached,
)
from pra_hf.multihop_routing_data import load_multihop_routing_examples
from pra_hf.summary_index import (
    BM25SummaryScorer,
    FrozenEmbeddingScorer,
    SummaryFacet,
    SummaryIndex,
    SummaryIndexRecord,
    exact_summary_scores,
    hybrid_scores,
    retrieval_metrics,
    source_sha256,
)


SEED = 20260826
SEEDS = (11, 23, 37, 53, 71)
SELECTION_BUDGET = 4
EMBEDDING_MODEL = "all-minilm:latest"
DEFAULT_FEATURE_ROOT = Path(
    r"D:/git/rd/pdattention-paper2-8/docs/papers/shared/results/paper2_8_qk_compression"
)
DEFAULT_DATA_ROOT = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
OUTPUT_ROOT = ROOT / "docs/papers/shared/results/paper3_1_summary_index"
MODEL_PROFILES = {
    "tiny_135m": "smollm2:135m",
    "subb_600m": "qwen3:0.6b",
    "candidate_1b": "llama3.2:1b",
    "mid_4b": "gemma3:4b-it-qat",
    "teacher_8b": "llama3.1:8b",
}
INHERITED_PROVENANCE = {
    "paper2_5_iterative_pra": {
        "branch": "research/paper2-5-iter-gist",
        "commit": "e81d434501db5e67dcb8e2b6043f6b798f7b7688",
    },
    "paper2_6_hybrid": {
        "branch": "hybrid-pra",
        "commit": "e30510cb691feffa243d55c14a2a39cef21cef50",
    },
    "paper2_7_graph_query": {
        "branch": "research/paper2-7-graph-query",
        "commit": "e44285c5444b8dc49a372cd8da459e3f24298518",
    },
    "paper2_8_qk_compression": {
        "branch": "research/paper2-8-qk-compression",
        "commit": "81fe76d35852c8563c039878dadd0ec7bfb0e4d1",
    },
    "paper2_9_temporal_query": {
        "branch": "research/paper2-9-look-ahead-back",
        "commit": "8f83bb486d1695ea229bf3f64b72972f7a3d83c3",
    },
    "paper3_native_kv_materialization": {
        "branch": "research/paper3-kv-materialization",
        "commit": "94d5446383ce4569a518a8b5bbf9d7c00b4c79ec",
    },
}


@dataclass(frozen=True)
class ConditionSpec:
    """One frozen summary generator, prompt, and equal-token geometry."""

    profile: str
    prompt_id: str
    facet_count: int
    token_budget: int = 32

    @property
    def name(self) -> str:
        return f"{self.profile}_{self.prompt_id}_{self.facet_count}x{self.token_budget // self.facet_count}"


@dataclass
class StudyCase:
    """One query, source, and inherited feature row aligned at parent chunks."""

    dataset: str
    split: str
    example_id: str
    question: str
    source: str
    feature: dict
    chunk_texts: tuple[str, ...]
    positive_indices: tuple[int, ...]

    @property
    def uri(self) -> str:
        return f"benchmark://{self.dataset}/{self.example_id}"

    def item_id(self, chunk_index: int) -> str:
        return f"{self.dataset}:{self.split}:{self.example_id}:parent-{chunk_index}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_feature_group(feature_root: Path, dataset: str, split: str) -> list[dict]:
    if dataset in {"hotpotqa", "qasper"}:
        path = feature_root / f"native_qk_features_{split}.pt"
        rows = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return [row for row in rows if row["dataset"] == dataset]
    directory = {"2wikimultihopqa": "2wiki", "musique": "musique"}[dataset]
    path = feature_root / "multi_dataset" / directory / f"native_qk_features_{split}.pt"
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _raw_examples(args, split: str) -> dict[tuple[str, str], object]:
    offset = 0 if split == "validation" else 8
    count = 8 if split == "validation" else 16
    qa = load_split_examples(args.cache_dir, count, offset, args.dataset_seed)
    output: dict[tuple[str, str], object] = {
        (str(row["dataset"]), str(row["id"])): row for row in qa
    }
    multi = load_multihop_routing_examples(
        args.annotations,
        args.twowiki_dev,
        args.musique_dev,
    )
    output.update(
        {
            (row.dataset, row.example_id): row
            for row in multi
            if row.split == split
        }
    )
    return output


def _example_fields(example: object) -> tuple[str, str, str]:
    if isinstance(example, dict):
        return str(example["question"]), str(example["source"]), str(example["id"])
    return str(example.question), str(example.source), str(example.example_id)


def _chunk_texts(tokenizer, source: str, spans: list[tuple[int, int]]) -> tuple[str, ...]:
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.offset_mapping
    if spans and spans[-1][1] != len(encoded.input_ids):
        raise ValueError(
            f"Tokenizer/source parity failed: spans end at {spans[-1][1]}, tokens={len(encoded.input_ids)}"
        )
    chunks = []
    for start, end in spans:
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        text = source[char_start:char_end].strip()
        if not text:
            text = tokenizer.decode(encoded.input_ids[start:end], skip_special_tokens=True).strip()
        chunks.append(text)
    return tuple(chunks)


def load_cases(args, tokenizer) -> tuple[list[StudyCase], list[dict]]:
    """Reconstruct source text and assert exact inherited identity/chunk parity."""

    raw = _raw_examples(args, args.split)
    cases = []
    parity_rows = []
    for dataset in args.datasets:
        features = _load_feature_group(args.feature_root, dataset, args.split)
        if args.max_per_dataset is not None:
            features = features[: args.max_per_dataset]
        for feature in features:
            key = (dataset, str(feature["example_id"]))
            if key not in raw:
                raise ValueError(f"Missing raw source for inherited identity {key}")
            question, source, example_id = _example_fields(raw[key])
            spans = [(int(start), int(end)) for start, end in feature["parent_spans"]]
            chunks = _chunk_texts(tokenizer, source, spans)
            positives = tuple(
                int(index)
                for index in torch.nonzero(feature["parent_positive_mask"], as_tuple=False)
                .flatten()
                .tolist()
            )
            if not positives:
                raise ValueError(f"No positive parent for {dataset}/{example_id}")
            case = StudyCase(
                dataset=dataset,
                split=args.split,
                example_id=example_id,
                question=question,
                source=source,
                feature=feature,
                chunk_texts=chunks,
                positive_indices=positives,
            )
            cases.append(case)
            parity_rows.append(
                {
                    "dataset": dataset,
                    "split": args.split,
                    "example_id": example_id,
                    "source_tokens": int(feature["source_tokens"]),
                    "parent_chunks": len(chunks),
                    "positive_parent_chunks": len(positives),
                    "identity_match": True,
                    "chunk_boundary_match": True,
                    "native_kv_rewritten": False,
                }
            )
    return cases, parity_rows


def _clip_text(tokenizer, text: str, budget: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids[:budget]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _records_from_outputs(case, outputs, tokenizer, spec, model) -> SummaryIndex:
    records = []
    per_facet_budget = max(spec.token_budget // spec.facet_count, 1)
    for chunk_index, (text, output, span) in enumerate(
        zip(case.chunk_texts, outputs, case.feature["parent_spans"])
    ):
        raw_facets = list(output.facets)
        if spec.facet_count == 1:
            facets = ()
            summary = _clip_text(tokenizer, output.summary, spec.token_budget)
            token_count = len(tokenizer(summary, add_special_tokens=False).input_ids)
        else:
            if len(raw_facets) < spec.facet_count:
                summary_ids = tokenizer(output.summary, add_special_tokens=False).input_ids
                raw_facets = [
                    (
                        f"segment-{index + 1}",
                        tokenizer.decode(
                            summary_ids[index * per_facet_budget : (index + 1) * per_facet_budget],
                            skip_special_tokens=True,
                        ),
                    )
                    for index in range(spec.facet_count)
                ]
            facets = tuple(
                SummaryFacet(label=label or f"facet-{index + 1}", text=clipped)
                for index, (label, value) in enumerate(raw_facets[: spec.facet_count])
                if (clipped := _clip_text(tokenizer, value, per_facet_budget))
            )
            if not facets:
                clipped = _clip_text(tokenizer, output.summary, spec.token_budget)
                facets = (SummaryFacet(label="fallback", text=clipped),)
            summary = " ".join(facet.text for facet in facets)
            token_count = sum(
                len(tokenizer(facet.text, add_special_tokens=False).input_ids) for facet in facets
            )
        records.append(
            SummaryIndexRecord(
                uri=case.uri,
                chunk_id=f"parent-{chunk_index}",
                token_start=int(span[0]),
                token_end=int(span[1]),
                source_sha256=source_sha256(text),
                summary=summary,
                facets=facets,
                summary_token_count=token_count,
                generation_model=model,
                prompt_id=spec.prompt_id,
            )
        )
    index = SummaryIndex(records)
    index.assert_source_alignment(
        (
            case.uri,
            f"parent-{chunk_index}",
            int(span[0]),
            int(span[1]),
            source_sha256(text),
        )
        for chunk_index, (span, text) in enumerate(zip(case.feature["parent_spans"], case.chunk_texts))
    )
    return index


def _source_index(case: StudyCase) -> SummaryIndex:
    return SummaryIndex(
        SummaryIndexRecord(
            uri=case.uri,
            chunk_id=f"parent-{index}",
            token_start=int(span[0]),
            token_end=int(span[1]),
            source_sha256=source_sha256(text),
            summary=text,
            summary_token_count=int(span[1]) - int(span[0]),
            generation_model="none",
            prompt_id="source-text-control",
        )
        for index, (text, span) in enumerate(zip(case.chunk_texts, case.feature["parent_spans"]))
    )


def _parent_mean_scores(case: StudyCase) -> np.ndarray:
    query = case.feature["query_hidden"].float()
    parents = case.feature["parent_hidden"].float()
    return F.cosine_similarity(parents, query.unsqueeze(0), dim=-1).numpy()


def _checkpoint_path(feature_root: Path, dataset: str, rank: int, seed: int) -> Path:
    if dataset in {"hotpotqa", "qasper"}:
        return feature_root / "low_rank_frontier" / "checkpoints" / f"direct_lowrank_r{rank}_seed{seed}.pt"
    directory = "2wiki" if dataset == "2wikimultihopqa" else "musique"
    checkpoint_name = (
        f"retrained_2wikimultihopqa_r{rank}_seed{seed}.pt"
        if dataset == "2wikimultihopqa"
        else f"retrained_musique_r{rank}_seed{seed}.pt"
    )
    return feature_root / "multi_dataset" / directory / "checkpoints" / checkpoint_name


def _load_lowrank(feature_root: Path, dataset: str, rank: int, device: torch.device):
    models = []
    for seed in SEEDS:
        checkpoint = torch.load(
            _checkpoint_path(feature_root, dataset, rank, seed),
            map_location="cpu",
            weights_only=False,
        )
        models.append((_selector_from_checkpoint(checkpoint, device), checkpoint))
    return models


@torch.no_grad()
def _lowrank_parent_scores(case, models, *, centroids, device) -> np.ndarray:
    local_scores = []
    for selector, checkpoint in models:
        scores, _ = _lowrank_scores(
            selector,
            checkpoint,
            case.feature,
            centroids=centroids,
            device=device,
        )
        local_scores.append(scores)
    local = torch.stack(local_scores).mean(0)
    parent_ids = case.feature["local_parent_indices"].long()
    parent = torch.full((len(case.feature["parent_spans"]),), -torch.inf)
    for index in range(len(parent)):
        values = local[parent_ids == index]
        if len(values):
            parent[index] = values.max()
    return parent.numpy()


def _embedding_scores(client, index: SummaryIndex, question: str):
    texts = [text for record in index.records for text in record.address_texts]
    embeddings = client.embed(EMBEDDING_MODEL, [*texts, question])
    nested = []
    cursor = 0
    for record in index.records:
        count = len(record.address_texts)
        nested.append(embeddings[cursor : cursor + count])
        cursor += count
    scorer = FrozenEmbeddingScorer(index, nested)
    return scorer.score(embeddings[-1]), len(embeddings[-1])


def _evaluate_scores(
    case,
    condition,
    scores,
    *,
    index_bytes,
    routing_seconds,
    ingestion_seconds=0.0,
    summary_tokens=0,
    embedding_bytes=0,
):
    metrics = retrieval_metrics(scores, case.positive_indices, k=SELECTION_BUDGET)
    selected = metrics["selected_indices"]
    materialized = sum(
        int(case.feature["parent_spans"][index][1])
        - int(case.feature["parent_spans"][index][0])
        for index in selected
    )
    return {
        "dataset": case.dataset,
        "split": case.split,
        "example_id": case.example_id,
        "condition": condition,
        "evidence_recall": metrics["evidence_recall"],
        "complete_recovery": metrics["complete_recovery"],
        "precision": metrics["precision"],
        "reciprocal_rank": metrics["reciprocal_rank"],
        "selected_indices": " ".join(map(str, selected)),
        "recovered_indices": " ".join(map(str, metrics["recovered_indices"])),
        "positive_indices": " ".join(map(str, case.positive_indices)),
        "candidate_chunks": len(scores),
        "selected_chunks": len(selected),
        "native_kv_tokens_materialized": materialized,
        "routing_index_bytes": int(index_bytes),
        "embedding_index_bytes": int(embedding_bytes),
        "summary_tokens": int(summary_tokens),
        "ingestion_seconds": float(ingestion_seconds),
        "routing_seconds": float(routing_seconds),
        "native_kv_rewritten": False,
    }


def _timed_scores(score_fn) -> tuple[np.ndarray, float]:
    """Evaluate one routing channel inside the measured timing boundary."""

    started = time.perf_counter()
    scores = np.asarray(score_fn())
    return scores, time.perf_counter() - started


def _bootstrap(deltas: list[float], seed: int = SEED, draws: int = 10000):
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    summary = []
    for (dataset, condition), group in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "condition": condition,
                "examples": len(group),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in (
                        "evidence_recall",
                        "complete_recovery",
                        "precision",
                        "reciprocal_rank",
                        "native_kv_tokens_materialized",
                        "routing_index_bytes",
                        "embedding_index_bytes",
                        "summary_tokens",
                        "ingestion_seconds",
                        "routing_seconds",
                    )
                },
            }
        )
    by_key = {(row["dataset"], row["example_id"], row["condition"]): row for row in rows}
    paired = []
    for dataset in sorted({row["dataset"] for row in rows}):
        identities = sorted({row["example_id"] for row in rows if row["dataset"] == dataset})
        conditions = sorted({row["condition"] for row in rows if row["dataset"] == dataset})
        for condition in conditions:
            if condition == "native_mean":
                continue
            deltas = [
                float(by_key[(dataset, identity, condition)]["evidence_recall"])
                - float(by_key[(dataset, identity, "native_mean")]["evidence_recall"])
                for identity in identities
                if (dataset, identity, condition) in by_key
                and (dataset, identity, "native_mean") in by_key
            ]
            if not deltas:
                continue
            mean, low, high = _bootstrap(deltas)
            paired.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "baseline": "native_mean",
                    "paired_identities": len(deltas),
                    "recall_delta": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return summary, paired


def _channel_overlap(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["example_id"])][row["condition"]] = row
    output = []
    for (dataset, example_id), conditions in grouped.items():
        if "source_bm25" not in conditions:
            continue
        lexical = set(map(int, conditions["source_bm25"]["selected_indices"].split()))
        for name, row in conditions.items():
            if name in {"source_bm25", "oracle_identity"}:
                continue
            selected = set(map(int, row["selected_indices"].split()))
            positive = set(map(int, row["positive_indices"].split()))
            union = lexical | selected
            output.append(
                {
                    "dataset": dataset,
                    "example_id": example_id,
                    "condition": name,
                    "selection_jaccard_with_source_bm25": len(lexical & selected) / max(len(union), 1),
                    "unique_evidence_vs_source_bm25": len((selected - lexical) & positive),
                    "lexical_unique_evidence": len((lexical - selected) & positive),
                }
            )
    return output


def _plot(summary: list[dict], output_dir: Path) -> None:
    datasets = sorted({row["dataset"] for row in summary})
    priority = [
        "native_mean",
        "source_bm25",
        "rank16",
        "rank8_centroid8",
        "oracle_identity",
    ]
    generated = sorted(
        {row["condition"] for row in summary if row["condition"] not in priority},
        key=lambda value: ("shuffled" in value, value),
    )
    conditions = [name for name in priority if any(row["condition"] == name for row in summary)] + generated
    figure, axes = plt.subplots(1, len(datasets), figsize=(max(7.2, 3.5 * len(datasets)), 4.2), squeeze=False)
    lookup = {(row["dataset"], row["condition"]): row for row in summary}
    for axis, dataset in zip(axes[0], datasets):
        present = [condition for condition in conditions if (dataset, condition) in lookup]
        values = [lookup[(dataset, condition)]["evidence_recall"] for condition in present]
        colors = ["#277DA1" if "summary" not in name and not name.startswith(tuple(MODEL_PROFILES)) else "#43AA8B" for name in present]
        axis.barh(range(len(present)), values, color=colors)
        axis.set_yticks(range(len(present)), [name.replace("_", " ") for name in present], fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_title(dataset)
        axis.set_xlabel("Evidence recall @ 4 parent chunks")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "summary_index_recall.png", dpi=180)
    figure.savefig(output_dir / "summary_index_recall.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    markers = {dataset: marker for dataset, marker in zip(datasets, ("o", "s", "^", "D"))}
    for dataset in datasets:
        rows = [row for row in summary if row["dataset"] == dataset and row["condition"] != "oracle_identity"]
        axis.scatter(
            [max(float(row["routing_index_bytes"]), 1.0) for row in rows],
            [row["evidence_recall"] for row in rows],
            label=dataset,
            marker=markers[dataset],
            alpha=0.8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Persistent routing-index bytes per source")
    axis.set_ylabel("Evidence recall @ 4")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "recall_vs_index_bytes.png", dpi=180)
    figure.savefig(output_dir / "recall_vs_index_bytes.pdf")
    plt.close(figure)


def _select_validation_policy(summary: list[dict]) -> dict:
    generated = [
        row
        for row in summary
        if row["condition"].startswith(tuple(MODEL_PROFILES))
        and "shuffled" not in row["condition"]
    ]
    output = {}
    for dataset in sorted({row["dataset"] for row in generated}):
        candidates = [row for row in generated if row["dataset"] == dataset]
        best = max(
            candidates,
            key=lambda row: (
                row["evidence_recall"],
                row["complete_recovery"],
                row["reciprocal_rank"],
                -row["routing_index_bytes"],
                row["condition"],
            ),
        )
        output[dataset] = {
            "condition": best["condition"],
            "validation_evidence_recall": best["evidence_recall"],
            "selection_rule": "recall, complete recovery, MRR, smaller index, lexical name",
        }
    return output


def run(args) -> dict:
    output_dir = args.output_root / (args.run_name or args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    client = OllamaClient(args.ollama_url, timeout=args.ollama_timeout)
    cases, parity_rows = load_cases(args, tokenizer)
    _write_csv(output_dir / "parity_rows.csv", parity_rows)

    specs = [ConditionSpec(*value) for value in args.conditions]
    generated_by_spec: dict[str, dict[str, object]] = {}
    model_metadata = {}
    for spec in specs:
        model = MODEL_PROFILES[spec.profile]
        if model not in model_metadata:
            info = client.model_info(model)
            model_metadata[model] = {
                "modified_at": info.get("modified_at"),
                "details": info.get("details", {}),
                "capabilities": info.get("capabilities", []),
                "model_info_sha256": hashlib.sha256(
                    json.dumps(info, sort_keys=True, default=str).encode()
                ).hexdigest(),
            }
        items = [
            (case.item_id(index), text)
            for case in cases
            for index, text in enumerate(case.chunk_texts)
        ]
        cache = JSONLGenerationCache(
            args.output_root / "summary_cache" / f"{spec.name}.jsonl"
        )
        outputs = generate_cached(
            client,
            cache,
            items,
            model=model,
            prompt_id=spec.prompt_id,
            token_budget=spec.token_budget,
            facet_count=spec.facet_count,
            seed=args.generation_seed,
            batch_size=args.generation_batch_size,
            structured_batches=spec.facet_count > 1,
        )
        generated_by_spec[spec.name] = {
            output.item_id: output for output in outputs
        }

    # Ollama may retain several gigabytes after teacher generation.  Release
    # those weights before loading Qwen for inherited native-Q/K replay.
    for model in model_metadata:
        client.unload(model)

    device = torch.device(args.device)
    lowrank_models = {}
    if not args.skip_lowrank:
        missing_queries = [case.feature for case in cases if "query_pre_query" not in case.feature]
        if missing_queries:
            _project_native_queries({args.split: missing_queries}, device)
        for dataset in args.datasets:
            lowrank_models[(dataset, 16)] = _load_lowrank(args.feature_root, dataset, 16, device)
            lowrank_models[(dataset, 8)] = _load_lowrank(args.feature_root, dataset, 8, device)

    rows = []
    address_rows = []
    for case_number, case in enumerate(cases, start=1):
        source_index = _source_index(case)
        native_mean, native_mean_seconds = _timed_scores(lambda: _parent_mean_scores(case))
        source_bm25, source_bm25_seconds = _timed_scores(
            lambda: BM25SummaryScorer(source_index).score(case.question)
        )
        source_exact, source_exact_seconds = _timed_scores(
            lambda: exact_summary_scores(source_index, case.question)
        )
        oracle, oracle_seconds = _timed_scores(
            lambda: [
                float(index in case.positive_indices)
                for index in range(len(case.chunk_texts))
            ]
        )
        baseline_channels = {
            "native_mean": (native_mean, len(case.chunk_texts) * 1024 * 4, native_mean_seconds),
            "source_bm25": (source_bm25, source_index.text_bytes, source_bm25_seconds),
            "source_exact": (source_exact, source_index.text_bytes, source_exact_seconds),
            "oracle_identity": (oracle, 0, oracle_seconds),
        }
        if not args.skip_lowrank:
            rank16, rank16_seconds = _timed_scores(
                lambda: _lowrank_parent_scores(
                    case, lowrank_models[(case.dataset, 16)], centroids=None, device=device
                )
            )
            rank8, rank8_seconds = _timed_scores(
                lambda: _lowrank_parent_scores(
                    case, lowrank_models[(case.dataset, 8)], centroids=8, device=device
                )
            )
            baseline_channels["rank16"] = (
                rank16,
                len(case.chunk_texts) * 8 * 32 * 16 * 4,
                rank16_seconds,
            )
            baseline_channels["rank8_centroid8"] = (
                rank8,
                len(case.chunk_texts) * 8 * 8 * 8 * 4,
                rank8_seconds,
            )
        for condition, (scores, index_bytes, routing_seconds) in baseline_channels.items():
            rows.append(
                _evaluate_scores(
                    case,
                    condition,
                    scores,
                    index_bytes=index_bytes,
                    routing_seconds=routing_seconds,
                )
            )

        for spec in specs:
            outputs = generated_by_spec[spec.name]
            case_outputs = [outputs[case.item_id(index)] for index in range(len(case.chunk_texts))]
            model = MODEL_PROFILES[spec.profile]
            index = _records_from_outputs(case, case_outputs, tokenizer, spec, model)
            ingestion_seconds = sum(output.generation_seconds for output in case_outputs)
            summary_tokens = sum(record.summary_token_count for record in index.records)
            started = time.perf_counter()
            bm25 = BM25SummaryScorer(index).score(case.question)
            bm25_seconds = time.perf_counter() - started
            started = time.perf_counter()
            exact = exact_summary_scores(index, case.question)
            exact_seconds = time.perf_counter() - started
            started = time.perf_counter()
            embedding, width = _embedding_scores(client, index, case.question)
            embedding_seconds = time.perf_counter() - started
            channels = {
                f"{spec.name}_summary_bm25": (bm25, bm25_seconds, 0),
                f"{spec.name}_summary_exact": (exact, exact_seconds, 0),
                f"{spec.name}_summary_embedding": (
                    embedding,
                    embedding_seconds,
                    len(index.records) * width * 4,
                ),
            }
            for alpha in (0.25, 0.5, 0.75):
                channels[f"{spec.name}_summary_hybrid_a{alpha:.2f}"] = (
                    hybrid_scores(bm25, embedding, alpha),
                    bm25_seconds + embedding_seconds,
                    len(index.records) * width * 4,
                )
            shuffled = index.shuffled_addresses(SEED)
            started = time.perf_counter()
            shuffled_scores = BM25SummaryScorer(shuffled).score(case.question)
            channels[f"{spec.name}_summary_bm25_shuffled"] = (
                shuffled_scores,
                time.perf_counter() - started,
                0,
            )
            for condition, (scores, routing_seconds, embedding_bytes) in channels.items():
                rows.append(
                    _evaluate_scores(
                        case,
                        condition,
                        scores,
                        index_bytes=index.text_bytes + embedding_bytes,
                        routing_seconds=routing_seconds,
                        ingestion_seconds=ingestion_seconds,
                        summary_tokens=summary_tokens,
                        embedding_bytes=embedding_bytes,
                    )
                )
            for record, output in zip(index.records, case_outputs):
                address_rows.append(
                    {
                        "dataset": case.dataset,
                        "split": case.split,
                        "example_id": case.example_id,
                        **record.to_dict(),
                        "generation_seconds": output.generation_seconds,
                        "prompt_eval_tokens": output.prompt_eval_tokens,
                        "eval_tokens": output.eval_tokens,
                        "generation_mode": output.generation_mode,
                    }
                )
        print(f"[evaluate {case_number}/{len(cases)}] {case.dataset} {case.example_id}", flush=True)

    summary, paired = _aggregate(rows)
    overlap = _channel_overlap(rows)
    _write_csv(output_dir / "per_example.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "paired_effects.csv", paired)
    _write_csv(output_dir / "channel_overlap.csv", overlap)
    with (output_dir / "summary_addresses.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in address_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _plot(summary, output_dir)
    policy = _select_validation_policy(summary) if args.split == "validation" else {}
    if policy:
        _write_json(output_dir / "validation_policy.json", policy)

    feature_artifacts = {}
    for dataset in args.datasets:
        if dataset in {"hotpotqa", "qasper"}:
            path = args.feature_root / f"native_qk_features_{args.split}.pt"
        else:
            directory = "2wiki" if dataset == "2wikimultihopqa" else "musique"
            path = args.feature_root / "multi_dataset" / directory / f"native_qk_features_{args.split}.pt"
        feature_artifacts[dataset] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest = {
        "schema_version": "1.0",
        "paper": "3.1",
        "split": args.split,
        "run_name": args.run_name or args.split,
        "datasets": list(args.datasets),
        "examples": len(cases),
        "max_per_dataset": args.max_per_dataset,
        "selection_budget_parent_chunks": SELECTION_BUDGET,
        "parent_chunk_tokens": 256,
        "source_model": MODEL_ID,
        "source_model_revision": MODEL_REVISION,
        "generation_seed": args.generation_seed,
        "summary_prompt_version": PROMPT_VERSION,
        "dataset_seed": args.dataset_seed,
        "embedding_model": EMBEDDING_MODEL,
        "conditions": [spec.__dict__ | {"name": spec.name, "model": MODEL_PROFILES[spec.profile]} for spec in specs],
        "model_metadata": model_metadata,
        "feature_artifacts": feature_artifacts,
        "source_native_kv_rewritten": False,
        "materialization_policy_changed": False,
        "summary_only_answering": False,
        "validation_policy": policy,
        "inherited_provenance": INHERITED_PROVENANCE,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _condition(value: str) -> tuple[str, str, int, int]:
    parts = value.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("condition must be PROFILE:PROMPT:FACETS[:TOKENS]")
    profile, prompt, facets = parts[:3]
    if profile not in MODEL_PROFILES:
        raise argparse.ArgumentTypeError(f"unknown profile {profile}")
    return profile, prompt, int(facets), int(parts[3]) if len(parts) == 4 else 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpotqa", "qasper", "2wikimultihopqa", "musique"),
        default=("hotpotqa", "qasper", "2wikimultihopqa", "musique"),
    )
    parser.add_argument("--max-per-dataset", type=int, default=4)
    parser.add_argument("--condition", dest="conditions", action="append", type=_condition)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-seed", type=int, default=SEED)
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-lowrank", action="store_true")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=float, default=900.0)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:/git/rd/pdattention-paper2-8/data/.hf_cache"))
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=DEFAULT_DATA_ROOT / "2wiki/dev.json")
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=DEFAULT_DATA_ROOT / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if not args.conditions:
        args.conditions = [
            ("tiny_135m", "generic", 1, 32),
            ("tiny_135m", "retrieval", 1, 32),
            ("subb_600m", "generic", 1, 32),
            ("subb_600m", "retrieval", 1, 32),
            ("candidate_1b", "generic", 1, 32),
            ("candidate_1b", "retrieval", 1, 32),
            ("mid_4b", "retrieval", 1, 32),
            ("teacher_8b", "generic", 1, 32),
            ("teacher_8b", "retrieval", 1, 32),
            ("teacher_8b", "faceted", 2, 32),
            ("teacher_8b", "faceted", 4, 32),
            ("teacher_8b", "faceted", 8, 32),
        ]
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
