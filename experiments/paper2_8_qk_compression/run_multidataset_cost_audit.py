"""Measure cold and warm routing costs for the Paper 2.8 dataset extension."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import file_sha256, write_csv, write_json
from experiments.paper2_8_qk_compression.run_gated_study import (
    MODEL_ID,
    MODEL_REVISION,
    _case_tensors,
    _score_compact,
)
from experiments.paper2_8_qk_compression.run_multidataset_extension import (
    DATASET_DIR,
    DATASET_LABEL,
    OUTPUT_ROOT,
    _load_features,
    _query_feature,
    _selector_from_checkpoint,
    _semantic_unit,
    _token_index,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy
from pra_hf.multihop_routing_data import load_multihop_routing_examples
from pra_hf.qk_compression import (
    LowRankRoutingIndex,
    masked_mean_keys,
    routing_metrics,
    stable_topk_indices,
)


LEXICAL_MODES = {
    "exact": "token_exact",
    "bm25": "bm25",
    "approximate": "token_approx",
    "inherited_hybrid": "iterative_hybrid",
}
LOW_RANK_MODES = {
    "rank16_retrained": (16, None),
    "rank8_centroid8_retrained": (8, 8),
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warm_time(callback, repeats: int, device: torch.device):
    for _ in range(2):
        result = callback()
    _sync(device)
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = callback()
        _sync(device)
        elapsed.append(1000.0 * (time.perf_counter() - started))
    return result, statistics.fmean(elapsed), sorted(elapsed)[int(0.95 * (len(elapsed) - 1))]


def _materialization(feature: dict, selected: list[int]) -> dict[str, float]:
    tokens = sum(int(feature["local_token_mask"][index].sum()) for index in selected)
    native_width = feature["local_pre_key"].shape[2] * feature["local_pre_key"].shape[3]
    return {
        "materialized_native_kv_tokens": tokens,
        "active_memory_fraction": tokens / max(int(feature["source_tokens"]), 1),
        "backing_native_kv_bytes": int(feature["source_tokens"]) * native_width * 4,
        "transfer_bytes": tokens * native_width * 4,
    }


def _row(
    feature: dict,
    *,
    condition: str,
    selected: list[int],
    construction_ms: float,
    cached_ms: float,
    cached_p95_ms: float,
    index_bytes: int,
    native_dots: int = 0,
    low_rank_dots: int = 0,
    lexical_scoring_ms: float = 0.0,
) -> dict:
    return {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "condition": condition,
        "candidate_chunks": len(feature["local_spans"]),
        "index_construction_ms": construction_ms,
        "cached_routing_ms": cached_ms,
        "cached_routing_p95_ms": cached_p95_ms,
        "index_bytes": index_bytes,
        "index_bytes_per_chunk": index_bytes / len(feature["local_spans"]),
        "native_dots": native_dots,
        "low_rank_dots": low_rank_dots,
        "lexical_scoring_ms": lexical_scoring_ms,
        **_materialization(feature, selected),
        **routing_metrics(selected, feature["local_positive_mask"], budget=4),
    }


def _checkpoint(output_root: Path, dataset: str, rank: int, device: torch.device):
    path = (
        output_root
        / DATASET_DIR[dataset]
        / "checkpoints"
        / f"retrained_{dataset}_r{rank}_seed11.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _selector_from_checkpoint(payload, device), payload, path


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    examples = load_multihop_routing_examples(
        args.annotations, args.twowiki_dev, args.musique_dev
    )
    example_lookup = {(row.dataset, row.example_id): row for row in examples}
    checkpoints = {}
    selectors = {}
    for dataset in DATASET_DIR:
        for rank in (8, 16):
            selector, checkpoint, path = _checkpoint(args.output_root, dataset, rank, device)
            selectors[(dataset, rank)] = selector
            checkpoints[(dataset, rank)] = (checkpoint, path)

    rows = []
    audited_ids = {}
    for dataset in DATASET_DIR:
        features = _load_features(args.output_root, dataset, "test")[: args.examples]
        audited_ids[dataset] = [feature["example_id"] for feature in features]
        for index, feature in enumerate(features, start=1):
            example = example_lookup[(dataset, feature["example_id"])]
            query, native_keys, native_mask, _ = _case_tensors(feature, device)

            _sync(device)
            started = time.perf_counter()
            mean_keys = masked_mean_keys(native_keys, native_mask)
            mean_mask = torch.ones(
                len(native_keys), 1, dtype=torch.bool, device=device
            )
            _sync(device)
            construction_ms = 1000.0 * (time.perf_counter() - started)

            def score_mean():
                return _score_compact(
                    query,
                    mean_keys,
                    mean_mask,
                    function="top_r_mean",
                    head_reduction="mean",
                )

            mean_scores, cached_ms, cached_p95 = _warm_time(
                score_mean, args.repeats, device
            )
            selected = stable_topk_indices(mean_scores, 4).tolist()
            rows.append(
                _row(
                    feature,
                    condition="native_mean",
                    selected=selected,
                    construction_ms=construction_ms,
                    cached_ms=cached_ms,
                    cached_p95_ms=cached_p95,
                    index_bytes=mean_keys.numel() * mean_keys.element_size() + mean_mask.numel(),
                    native_dots=len(native_keys) * query.shape[1],
                )
            )

            lexical_started = time.perf_counter()
            lexical_index = _token_index(tokenizer, example, feature)
            lexical_construction_ms = 1000.0 * (time.perf_counter() - lexical_started)
            lexical_bytes = len(pickle.dumps(lexical_index, protocol=5))
            query_ids = tokenizer(example.question, add_special_tokens=False).input_ids
            semantic = _semantic_unit(mean_scores.cpu())
            for condition, mode in LEXICAL_MODES.items():
                policy = HybridDiscoveryPolicy(mode=mode)

                def score_lexical():
                    return lexical_index.score(
                        query_ids,
                        semantic,
                        tokenizer,
                        policy,
                        hop=1,
                        parent_id="query",
                    )

                candidates, cached_ms, cached_p95 = _warm_time(
                    score_lexical, args.repeats, torch.device("cpu")
                )
                scores = torch.tensor(
                    [candidate.selected_score for candidate in candidates]
                )
                selected = stable_topk_indices(scores, 4).tolist()
                rows.append(
                    _row(
                        feature,
                        condition=condition,
                        selected=selected,
                        construction_ms=lexical_construction_ms,
                        cached_ms=cached_ms,
                        cached_p95_ms=cached_p95,
                        index_bytes=lexical_bytes,
                        lexical_scoring_ms=cached_ms,
                    )
                )

            raw_keys = feature["local_pre_key"].to(device).float().flatten(2)
            token_mask = feature["local_token_mask"].to(device)
            for condition, (rank, representatives) in LOW_RANK_MODES.items():
                selector = selectors[(dataset, rank)]
                checkpoint, _ = checkpoints[(dataset, rank)]
                _sync(device)
                started = time.perf_counter()
                projected = selector.feature_projection(
                    raw_keys / float(checkpoint["native_key_rms_scale"])
                )
                routing_index = LowRankRoutingIndex.build(
                    projected,
                    token_mask,
                    storage_dtype="float32",
                    representatives=representatives,
                )
                _sync(device)
                construction_ms = 1000.0 * (time.perf_counter() - started)
                projected_query = selector.query_projection(
                    _query_feature(feature["query_pre_query"]).to(device)
                )

                def score_low_rank():
                    return routing_index.search(projected_query, 4)

                result, cached_ms, cached_p95 = _warm_time(
                    score_low_rank, args.repeats, device
                )
                selected = result[1].reshape(-1).tolist()
                rows.append(
                    _row(
                        feature,
                        condition=condition,
                        selected=selected,
                        construction_ms=construction_ms,
                        cached_ms=cached_ms,
                        cached_p95_ms=cached_p95,
                        index_bytes=routing_index.storage_bytes,
                        low_rank_dots=(
                            routing_index.chunk_count
                            * routing_index.tokens.shape[1]
                            * routing_index.rank
                        ),
                    )
                )
            print(f"[cost {dataset} {index}/{len(features)}]", flush=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    metrics = (
        "candidate_chunks",
        "index_construction_ms",
        "cached_routing_ms",
        "cached_routing_p95_ms",
        "index_bytes",
        "index_bytes_per_chunk",
        "native_dots",
        "low_rank_dots",
        "lexical_scoring_ms",
        "materialized_native_kv_tokens",
        "active_memory_fraction",
        "backing_native_kv_bytes",
        "transfer_bytes",
        "evidence_recall",
        "evidence_precision",
        "any_evidence",
        "chain_completion",
        "mrr",
    )
    summary = []
    for (dataset, condition), group in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "condition": condition,
                "examples": len(group),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in metrics
                },
            }
        )

    args.cost_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.cost_root / "per_example.csv", rows)
    write_csv(args.cost_root / "summary.csv", summary)
    quality = {}
    with (args.output_root / "summary.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test" and int(row["budget_chunks"]) == 4:
                quality[(row["dataset"], row["condition"])] = float(row["evidence_recall"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    display = {
        "native_mean": "native mean",
        "exact": "exact",
        "bm25": "BM25",
        "approximate": "approximate",
        "inherited_hybrid": "inherited hybrid",
        "rank16_retrained": "rank 16",
        "rank8_centroid8_retrained": "rank 8 / 8 centroids",
    }
    for axis, dataset in zip(axes, sorted(DATASET_DIR)):
        group = [row for row in summary if row["dataset"] == dataset]
        for row in group:
            recall = quality[(dataset, row["condition"])]
            axis.scatter(row["cached_routing_ms"], recall, s=42)
            axis.annotate(
                display[row["condition"]],
                (row["cached_routing_ms"], recall),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_xscale("log")
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_xlabel("Warm cached routing (ms)")
        axis.set_ylabel("Full-test evidence recall@4")
        axis.margins(x=0.12, y=0.12)
        axis.grid(alpha=0.25)
    figure.savefig(args.cost_root / "recall_vs_cached_latency.pdf")
    figure.savefig(args.cost_root / "recall_vs_cached_latency.png", dpi=180)
    plt.close(figure)

    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": 11,
        "examples_per_dataset": args.examples,
        "repeats": args.repeats,
        "audited_ids": audited_ids,
        "quality_summary_sha256": file_sha256(args.output_root / "summary.csv"),
        "checkpoint_sha256": {
            f"{dataset}_r{rank}": file_sha256(path)
            for (dataset, rank), (_, path) in checkpoints.items()
        },
        "backing_native_kv_unchanged": True,
        "timing_scope": "component-only; source encoding and native-K/V materialization excluded",
    }
    write_json(args.cost_root / "manifest.json", manifest)
    return {"rows": len(rows), "summary_rows": len(summary)}


def parse_args() -> argparse.Namespace:
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cost-root", type=Path, default=OUTPUT_ROOT / "cost_audit")
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl"
    )
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json")
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
