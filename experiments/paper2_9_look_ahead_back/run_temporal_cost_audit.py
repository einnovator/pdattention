"""Measure cached temporal-routing component cost on fixed test identities."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_9_look_ahead_back.precompute_temporal_queries import DATASETS
from experiments.paper2_9_look_ahead_back.run_temporal_study import (
    RESULT_28,
    aligned_rows,
    build_scorer,
    load_checkpoints,
    load_source_features,
    temporal_rows,
    write_csv,
)


def timed(call, repeats: int) -> float:
    for _ in range(3):
        call()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    return 1000.0 * (time.perf_counter() - started) / repeats


def run(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    policies = json.loads(
        (args.output_root / "study_manifest.json").read_text(encoding="utf-8")
    )["validation_selected_temporal_policies"]
    source = load_source_features(args.paper2_8_root)
    device = torch.device(args.device)
    rows = []
    for dataset in args.datasets:
        checkpoints = {rank: load_checkpoints(dataset, rank, device) for rank in (8, 16)}
        pairs = aligned_rows(
            source[(dataset, "test")],
            temporal_rows(args.temporal_root, dataset, "test"),
        )[: args.examples]
        policy = policies[dataset]
        for source_row, temporal in pairs:
            scorer = build_scorer(source_row, temporal, checkpoints, device)
            started = time.perf_counter()
            scorer.routing_memory("rank16")
            construction_ms = 1000.0 * (time.perf_counter() - started)
            tokens = len(temporal["prompt_token_ids"])
            calls = {
                "b1_current": lambda: scorer.score(
                    "rank16", layer=27, start=tokens - 1, stop=tokens, reducer="current"
                ),
                "selected_temporal": lambda: scorer.score(
                    "rank16",
                    layer=27,
                    start=max(0, tokens - policy["look_behind"]),
                    stop=tokens,
                    reducer=policy["reducer"],
                ),
            }
            for condition, call in calls.items():
                _, _, index_bytes, dots = call()
                milliseconds = timed(call, args.repeats)
                for stride in ((1, 2, 4, 8) if condition == "selected_temporal" else (1,)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "example_id": source_row["example_id"],
                            "condition": condition,
                            "stride": stride,
                            "candidate_chunks": len(source_row["local_positive_mask"]),
                            "cached_ensemble_routing_ms_per_call": milliseconds,
                            "amortized_routing_ms_per_token": milliseconds / stride,
                            "routing_dots_per_seed_per_call": dots,
                            "routing_dots_five_seed_ensemble_per_call": dots * 5,
                            "routing_index_bytes_per_chunk_per_seed": index_bytes,
                            "routing_index_bytes_per_chunk_five_seed_ensemble": index_bytes * 5,
                            "temporal_buffer_bytes_per_seed": policy["look_behind"] * 16 * 4 if condition == "selected_temporal" else 16 * 4,
                            "index_construction_ms_five_seed_ensemble": construction_ms,
                        }
                    )
    write_csv(args.output_root / "cost_per_example.csv", rows)
    summary = []
    for dataset in args.datasets:
        for condition in ("b1_current", "selected_temporal"):
            for stride in ((1, 2, 4, 8) if condition == "selected_temporal" else (1,)):
                group = [
                    row for row in rows
                    if row["dataset"] == dataset and row["condition"] == condition and row["stride"] == stride
                ]
                summary.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "stride": stride,
                        "examples": len(group),
                        "mean_cached_ensemble_ms_per_call": statistics.fmean(row["cached_ensemble_routing_ms_per_call"] for row in group),
                        "median_cached_ensemble_ms_per_call": statistics.median(row["cached_ensemble_routing_ms_per_call"] for row in group),
                        "mean_amortized_ms_per_token": statistics.fmean(row["amortized_routing_ms_per_token"] for row in group),
                        "mean_index_construction_ms": statistics.fmean(row["index_construction_ms_five_seed_ensemble"] for row in group),
                        "mean_candidate_chunks": statistics.fmean(row["candidate_chunks"] for row in group),
                        "routing_index_bytes_per_chunk_per_seed": group[0]["routing_index_bytes_per_chunk_per_seed"],
                        "temporal_buffer_bytes_per_seed": group[0]["temporal_buffer_bytes_per_seed"],
                    }
                )
    write_csv(args.output_root / "cost_summary.csv", summary)
    manifest = {
        "schema_version": "1.0",
        "device": str(device),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "examples_per_dataset": args.examples,
        "warmups": 3,
        "repeats": args.repeats,
        "component_timing_only": True,
        "backbone_execution_included": False,
        "native_kv_transfer_included": False,
        "quality_uses_five_seed_score_ensemble": True,
    }
    (args.output_root / "cost_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--paper2-8-root", type=Path, default=RESULT_28)
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_9_look_ahead_back",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_9_look_ahead_back",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
