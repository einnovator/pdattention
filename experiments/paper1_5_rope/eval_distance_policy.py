"""Measure deferred-RoPE position semantics and warm materialization cost."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    refresh_manifest,
    set_seed,
    write_csv,
    write_json,
)
from experiments.paper1_5_rope.instrumented_model import materialize_raw_rope_key  # noqa: E402
from experiments.paper1_5_rope.position_policies import POLICIES  # noqa: E402
from pra_torch.positions import RotaryPositionEncoding  # noqa: E402


NATIVE_CONTEXT = 192


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    return scores, F.softmax(scores, dim=-1) @ value


def evaluate_semantics(device: str) -> list[dict]:
    """Compare approximate chunk policies with exact logical source positions."""
    rows = []
    for tier, settings in TIERS.items():
        heads = settings["n_heads"]
        head_dim = settings["d_model"] // heads
        rope = RotaryPositionEncoding(head_dim).to(device)
        for seed in SEEDS:
            set_seed(seed)
            raw_query = torch.randn(1, heads, 1, head_dim, device=device)
            raw_key = torch.randn(1, heads, 32, head_dim, device=device)
            value = torch.randn_like(raw_key)
            source_positions = torch.arange(32, device=device)
            for ratio in (1, 2, 4, 8, 16, 32):
                query_position = ratio * NATIVE_CONTEXT
                query = rope.apply_rotary(
                    raw_query,
                    torch.tensor([query_position], device=device),
                )
                exact_key, _ = materialize_raw_rope_key(
                    raw_key,
                    source_positions,
                    query_position,
                    policy="exact_logical",
                    distance_limit=NATIVE_CONTEXT,
                    rope=rope,
                )
                exact_scores, exact_output = _attention(query, exact_key, value)
                for policy in POLICIES:
                    key, assigned = materialize_raw_rope_key(
                        raw_key,
                        source_positions,
                        query_position,
                        policy=policy,
                        distance_limit=NATIVE_CONTEXT,
                        rope=rope,
                    )
                    scores, output = _attention(query, key, value)
                    rows.append(
                        {
                            "seed": seed,
                            "model_tier": tier,
                            "heads": heads,
                            "head_dim": head_dim,
                            "policy": policy,
                            "logical_native_ratio": ratio,
                            "query_position": query_position,
                            "source_last_position": int(source_positions[-1]),
                            "assigned_last_position": int(assigned[-1]),
                            "effective_nearest_distance": query_position - int(assigned[-1]),
                            "score_rmse_vs_exact": float((scores - exact_scores).square().mean().sqrt()),
                            "output_rmse_vs_exact": float((output - exact_output).square().mean().sqrt()),
                            "top_token_agreement_vs_exact": float(
                                (scores.argmax(dim=-1) == exact_scores.argmax(dim=-1)).float().mean()
                            ),
                        }
                    )
    return rows


def _time_cuda(operation, *, warmup: int, iterations: int, device: str) -> float:
    for _ in range(warmup):
        operation()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    if device == "cuda":
        torch.cuda.synchronize()
    return 1_000.0 * (time.perf_counter() - start) / iterations


def evaluate_performance(device: str, iterations: int) -> list[dict]:
    """Benchmark selected-key gather, deferred rotation, and attention separately."""
    rows = []
    warmup = max(10, iterations // 10)
    for tier, settings in TIERS.items():
        heads = settings["n_heads"]
        head_dim = settings["d_model"] // heads
        rope = RotaryPositionEncoding(head_dim).to(device)
        for seed in SEEDS:
            set_seed(seed)
            raw_cache = torch.randn(1, heads, 192, head_dim, device=device)
            value_cache = torch.randn_like(raw_cache)
            all_positions = torch.arange(192, device=device)
            post_cache = rope.apply_rotary(raw_cache, all_positions)
            raw_query = torch.randn(1, heads, 32, head_dim, device=device)
            query_positions = torch.arange(32, device=device) + 4 * NATIVE_CONTEXT
            query = rope.apply_rotary(raw_query, query_positions)
            for memory_tokens in (32, 64, 128, 160):
                indices = torch.arange(memory_tokens, device=device)

                def post_materialize():
                    return post_cache.index_select(2, indices)

                def pre_materialize():
                    selected = raw_cache.index_select(2, indices)
                    return rope.apply_rotary(selected, all_positions.index_select(0, indices))

                post_key = post_materialize()
                pre_key = pre_materialize()
                selected_value = value_cache.index_select(2, indices)

                def post_warm_query():
                    key = post_materialize()
                    return _attention(query, key, selected_value)[1]

                def pre_warm_query():
                    key = pre_materialize()
                    return _attention(query, key, selected_value)[1]

                attention_ms = _time_cuda(
                    lambda: _attention(query, post_key, selected_value)[1],
                    warmup=warmup,
                    iterations=iterations,
                    device=device,
                )
                common = {
                    "seed": seed,
                    "model_tier": tier,
                    "device": device,
                    "heads": heads,
                    "head_dim": head_dim,
                    "direct_tokens": 32,
                    "stored_tokens": 192,
                    "selected_tokens": memory_tokens,
                    "stored_kv_bytes": int(2 * raw_cache.numel() * raw_cache.element_size()),
                    "position_metadata_bytes": int(all_positions.numel() * all_positions.element_size()),
                    "selected_materialized_kv_bytes": int(
                        2 * pre_key.numel() * pre_key.element_size()
                    ),
                    "active_attention_kv_bytes": int(
                        2 * (pre_key.numel() + query.numel()) * pre_key.element_size()
                    ),
                    "attention_ms": attention_ms,
                }
                rows.append(
                    {
                        **common,
                        "storage_mode": "post_position",
                        "rotation_materialization_ms": _time_cuda(
                            post_materialize,
                            warmup=warmup,
                            iterations=iterations,
                            device=device,
                        ),
                        "warm_query_ms": _time_cuda(
                            post_warm_query,
                            warmup=warmup,
                            iterations=iterations,
                            device=device,
                        ),
                    }
                )
                rows.append(
                    {
                        **common,
                        "storage_mode": "pre_position_deferred",
                        "rotation_materialization_ms": _time_cuda(
                            pre_materialize,
                            warmup=warmup,
                            iterations=iterations,
                            device=device,
                        ),
                        "warm_query_ms": _time_cuda(
                            pre_warm_query,
                            warmup=warmup,
                            iterations=iterations,
                            device=device,
                        ),
                    }
                )
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, values in sorted(grouped.items()):
        result = dict(zip(keys, identity))
        result["seed_count"] = len(values)
        for metric in metrics:
            observed = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = statistics.fmean(observed)
            result[f"{metric}_std"] = statistics.pstdev(observed)
        output.append(result)
    return output


def plot_results(semantic: list[dict], performance: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    styles = {
        "local_chunk": ("#A34832", "o"),
        "clipped": ("#D08B32", "s"),
        "log_compressed": ("#327A5A", "^"),
        "bucketed": ("#245A8D", "D"),
        "remote_past": ("#6C5A8E", "v"),
    }
    for policy, (color, marker) in styles.items():
        values = [row for row in semantic if row["model_tier"] == "small" and row["policy"] == policy]
        axes[0].plot(
            [row["logical_native_ratio"] for row in values],
            [max(row["output_rmse_vs_exact_mean"], 1e-12) for row in values],
            color=color,
            marker=marker,
            label=policy.replace("_", " "),
        )
    for mode, color, marker in (
        ("post_position", "#245A8D", "o"),
        ("pre_position_deferred", "#A34832", "s"),
    ):
        values = [row for row in performance if row["model_tier"] == "small" and row["storage_mode"] == mode]
        axes[1].plot(
            [row["selected_tokens"] for row in values],
            [row["warm_query_ms_mean"] for row in values],
            color=color,
            marker=marker,
            label=mode.replace("_", " "),
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Logical / native distance")
    axes[0].set_ylabel("Attention-output RMSE vs exact")
    axes[1].set_xlabel("Selected memory tokens")
    axes[1].set_ylabel("Warm query latency (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(device: str, iterations: int) -> Path:
    metadata = environment_metadata()
    semantic_rows = evaluate_semantics(device)
    performance_rows = evaluate_performance(device, iterations)
    semantic_aggregate = aggregate(
        semantic_rows,
        ("model_tier", "policy", "logical_native_ratio"),
        ("score_rmse_vs_exact", "output_rmse_vs_exact", "top_token_agreement_vs_exact"),
    )
    performance_aggregate = aggregate(
        performance_rows,
        ("model_tier", "storage_mode", "selected_tokens"),
        ("rotation_materialization_ms", "attention_ms", "warm_query_ms"),
    )
    payload = {
        "metadata": metadata,
        "seeds": list(SEEDS),
        "native_context": NATIVE_CONTEXT,
        "semantic_rows": semantic_rows,
        "semantic_aggregate": semantic_aggregate,
        "performance_rows": performance_rows,
        "performance_aggregate": performance_aggregate,
        "timing_iterations": iterations,
    }
    write_json(RESULTS / "rope_distance_policy.json", payload)
    write_csv(RESULTS / "rope_distance_policy.csv", semantic_rows)
    write_csv(RESULTS / "rope_storage_performance.csv", performance_rows)
    plot_results(
        semantic_aggregate,
        performance_aggregate,
        RESULTS / "rope_distance_policy.png",
    )
    return refresh_manifest(metadata=metadata)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(args.device, args.iterations))
