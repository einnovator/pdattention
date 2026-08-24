"""Benchmark cached, quantized, vectorized Paper 2.8 routing indexes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_confirmation import (
    CHECKPOINT_ROOT,
    RESULT_ROOT,
    _load_selector,
    _row,
)
from experiments.paper2_8_qk_compression.run_gated_study import (
    _project_native_queries,
    _sha256,
    _write_csv,
)
from experiments.paper2_8_qk_compression.run_query_conditioned_study import _query_feature
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.qk_compression import LowRankRoutingIndex


INDEX_MODES = (
    ("float32", "float32", None),
    ("float16", "float16", None),
    ("bfloat16", "bfloat16", None),
    ("int8", "int8", None),
    ("float16_centroid8", "float16", 8),
    ("int8_centroid8", "int8", 8),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    features = features[: args.examples] if args.examples else features
    if any("query_pre_query" not in row for row in features):
        _project_native_queries({"benchmark": features}, device)
    selector, checkpoint = _load_selector(args.checkpoint, device)
    rows = []
    for example_index, feature in enumerate(features, start=1):
        raw = feature["local_pre_key"].to(device).float().flatten(2)
        mask = feature["local_token_mask"].to(device)
        query = selector.query_projection(
            _query_feature(feature["query_pre_query"]).to(device)
        )
        _sync(device)
        projection_started = time.perf_counter()
        projected = selector.feature_projection(
            raw / float(checkpoint["native_key_rms_scale"])
        )
        _sync(device)
        projection_ms = (time.perf_counter() - projection_started) * 1000
        for name, storage_dtype, representatives in INDEX_MODES:
            _sync(device)
            started = time.perf_counter()
            index = LowRankRoutingIndex.build(
                projected,
                mask,
                storage_dtype=storage_dtype,
                representatives=representatives,
            )
            _sync(device)
            build_ms = (time.perf_counter() - started) * 1000
            index.search(query, args.top_k)
            _sync(device)
            warm = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                index.search(query, args.top_k)
                _sync(device)
                warm.append((time.perf_counter() - started) * 1000)
            batch = query.unsqueeze(0).expand(args.query_batch, -1)
            _sync(device)
            started = time.perf_counter()
            index.search(batch, args.top_k)
            _sync(device)
            batch_ms = (time.perf_counter() - started) * 1000
            if index.chunk_count > 1:
                left = LowRankRoutingIndex(
                    index.tokens[:-1], index.token_mask[:-1], index.storage_dtype, index.scales
                )
                right = LowRankRoutingIndex(
                    index.tokens[-1:], index.token_mask[-1:], index.storage_dtype, index.scales
                )
                started = time.perf_counter()
                updated = left.append(right)
                update_ms = (time.perf_counter() - started) * 1000
                assert updated.chunk_count == index.chunk_count
            else:
                update_ms = 0.0
            full_scores = index.score(query)[0].cpu()
            quality = _row(
                feature,
                condition=name,
                seed=int(args.seed),
                scores=full_scores,
            )
            rows.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "mode": name,
                    "chunks": index.chunk_count,
                    "representatives": int(index.tokens.shape[1]),
                    "rank": index.rank,
                    "index_bytes": index.storage_bytes,
                    "bytes_per_chunk": index.storage_bytes / index.chunk_count,
                    "projection_ms": projection_ms,
                    "cold_build_ms": build_ms,
                    "warm_search_ms": statistics.fmean(warm),
                    "warm_search_p95_ms": sorted(warm)[int(0.95 * (len(warm) - 1))],
                    "batch_queries": args.query_batch,
                    "batch_search_ms": batch_ms,
                    "queries_per_second": args.query_batch / max(batch_ms / 1000, 1e-9),
                    "append_update_ms": update_ms,
                    **{key: quality[key] for key in ("evidence_recall", "any_evidence", "mrr")},
                }
            )
        print(f"[index benchmark {example_index}/{len(features)}] {feature['dataset']} {feature['example_id']}", flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["mode"])].append(row)
    summary = []
    for (dataset, mode), group in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "mode": mode,
                "examples": len(group),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in (
                        "chunks", "representatives", "index_bytes", "bytes_per_chunk",
                        "projection_ms", "cold_build_ms", "warm_search_ms",
                        "warm_search_p95_ms", "batch_search_ms", "queries_per_second",
                        "append_update_ms", "evidence_recall", "any_evidence", "mrr",
                    )
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_example.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        group = [row for row in summary if row["dataset"] == dataset]
        axis.scatter([row["bytes_per_chunk"] for row in group], [row["warm_search_ms"] for row in group])
        for row in group:
            axis.annotate(row["mode"], (row["bytes_per_chunk"], row["warm_search_ms"]), fontsize=7)
        axis.set_xscale("log")
        axis.set_title(dataset.upper())
        axis.set_xlabel("Persistent bytes/chunk")
        axis.set_ylabel("Warm vectorized search (ms)")
        axis.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(args.output_dir / f"production_index.{suffix}", dpi=190)
    plt.close(figure)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "feature_sha256": _sha256(args.features),
        "modes": [row[0] for row in INDEX_MODES],
        "repeats": args.repeats,
        "query_batch": args.query_batch,
        "vectorized_topk": True,
        "backing_native_kv_unchanged": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"rows": len(rows), "summary_rows": len(summary)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--features", type=Path, default=RESULT_ROOT / "native_qk_features_confirmation.pt")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_ROOT / "direct_lowrank_r16_seed11.pt")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "production_index")
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--query-batch", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
