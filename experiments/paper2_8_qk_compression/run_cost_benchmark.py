"""Measure uncached controller construction and QK scoring costs on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_gated_study import (
    M_VALUES,
    SEEDS,
    _case_tensors,
    _project_native_queries,
)
from pra_hf.qk_compression import (
    NativeLandmarkSelector,
    farthest_first_indices,
    gather_landmarks,
    greedy_qk_landmarks,
    landmark_features,
    last_token_indices,
    masked_mean_keys,
    qk_response_scores,
    random_token_indices,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation):
    _sync(device)
    started = time.perf_counter()
    value = operation()
    _sync(device)
    return value, 1000 * (time.perf_counter() - started)


def _write(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_selector(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    selector = NativeLandmarkSelector(hidden_width=32).to(device)
    selector.load_state_dict(checkpoint["state_dict"])
    selector.eval()
    return selector, checkpoint["feature_mean"].to(device), checkpoint["feature_scale"].to(device)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    _project_native_queries({"cost": features}, device)
    selector, feature_mean, feature_scale = _load_selector(args.checkpoint, device)
    rows = []
    for example_index, feature in enumerate(features, start=1):
        queries, keys, mask, _ = _case_tensors(feature, device)
        methods = {
            "full_k": lambda: (keys, mask, int(mask.sum(dim=1).float().mean().item())),
            "mean": lambda: (
                masked_mean_keys(keys, mask),
                torch.ones(keys.shape[0], 1, dtype=torch.bool, device=device),
                1,
            ),
            "last_m8": lambda: (*gather_landmarks(keys, last_token_indices(mask, 8)), 8),
            "random_m8": lambda: (
                *gather_landmarks(
                    keys,
                    random_token_indices(
                        mask,
                        8,
                        generator=torch.Generator().manual_seed(
                            args.seed + example_index * 1009
                        ),
                    ),
                ),
                8,
            ),
            "farthest_m8": lambda: (
                *gather_landmarks(keys, farthest_first_indices(keys, mask, 8)),
                8,
            ),
            "greedy_oracle_m8": lambda: (
                *gather_landmarks(
                    keys,
                    greedy_qk_landmarks(
                        queries,
                        keys,
                        mask,
                        8,
                        function="top_r_mean",
                        top_r=4,
                        head_reduction="mean",
                    ),
                ),
                8,
            ),
            "learned_m8": lambda: (
                *gather_landmarks(
                    keys,
                    [
                        row.topk(min(8, int(row_mask.sum().item())))
                        .indices.sort()
                        .values.tolist()
                        for row, row_mask in zip(
                            selector(
                                (landmark_features(keys, mask) - feature_mean)
                                / feature_scale,
                                mask,
                            ),
                            mask,
                        )
                    ],
                ),
                8,
            ),
        }
        for method, operation in methods.items():
            with torch.no_grad():
                (compact, compact_mask, landmarks), construction_ms = _timed(
                    device, operation
                )
                _, scoring_ms = _timed(
                    device,
                    lambda: qk_response_scores(
                        queries,
                        compact,
                        compact_mask,
                        function="top_r_mean",
                        top_r=4,
                        head_reduction="mean",
                    ),
                )
            rows.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "method": method,
                    "candidate_chunks": int(keys.shape[0]),
                    "valid_key_tokens": int(mask.sum().item()),
                    "landmarks_per_chunk": landmarks,
                    "controller_construction_ms": construction_ms,
                    "qk_scoring_ms": scoring_ms,
                    "total_routing_ms": construction_ms + scoring_ms,
                    "native_dots": int(keys.shape[0] * landmarks * queries.shape[1]),
                    "device": str(device),
                }
            )
        print(
            f"[cost {example_index}/{len(features)}] "
            f"{feature['dataset']} {feature['example_id']}",
            flush=True,
        )
    _write(args.output_dir / "controller_cost_rows.csv", rows)
    summary = []
    for dataset in ("hotpotqa", "qasper"):
        for method in methods:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "examples": len(group),
                    **{
                        metric: sum(float(row[metric]) for row in group) / len(group)
                        for metric in (
                            "candidate_chunks",
                            "valid_key_tokens",
                            "landmarks_per_chunk",
                            "controller_construction_ms",
                            "qk_scoring_ms",
                            "total_routing_ms",
                            "native_dots",
                        )
                    },
                }
            )
    _write(args.output_dir / "controller_cost_summary.csv", summary)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    shown = ("mean", "farthest_m8", "greedy_oracle_m8", "learned_m8")
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        values = {row["method"]: row for row in summary if row["dataset"] == dataset}
        axis.bar(
            range(len(shown)),
            [float(values[method]["total_routing_ms"]) for method in shown],
            color=("#6c757d", "#457b9d", "#2a9d8f", "#6a4c93"),
        )
        axis.set_yscale("log")
        axis.set_xticks(
            range(len(shown)),
            ("mean", "farthest m=8", "greedy oracle m=8", "learned m=8"),
            rotation=25,
            ha="right",
        )
        axis.set_ylabel("Controller + QK scoring (ms, log scale)")
        axis.set_title(dataset.upper())
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(args.output_dir / f"controller_cost.{suffix}", dpi=180)
    plt.close(figure)
    result = {
        "device": str(device),
        "examples": len(features),
        "checkpoint": args.checkpoint.name,
        "teacher_function": "top_r_mean",
        "m": max(M_VALUES),
        "seeds_available": SEEDS,
    }
    (args.output_dir / "controller_cost_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    output = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--features", type=Path, default=output / "native_qk_features_test.pt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=output / "selector_checkpoints/native_landmark_selector_seed11.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
