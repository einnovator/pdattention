"""Run the controlled Paper 5 logical-memory and active-K/V scaling study.

The primary benchmark plants a fixed four-region working set in progressively
larger pools of distractor chunks.  It measures retrieval and systems behavior;
it does not substitute retrieval recall for language-model quality.  Rows that
come from prior papers or analytical accounting are labeled accordingly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper5_scaling_laws.scaling_core import percentile


ADAPTIVE_DIFFICULTIES = (
    ("medium", 8, 0.040),
    ("hard", 24, 0.015),
)


@dataclass(frozen=True)
class ScalingConfig:
    """Complete, serializable configuration for the controlled pilot."""

    seeds: tuple[int, ...] = (17, 29, 41, 53, 67)
    logical_tokens: tuple[int, ...] = (32_768, 131_072, 524_288, 2_097_152, 8_388_608)
    active_kv_budgets: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    chunk_tokens: int = 32
    materialized_tokens_per_node: int = 8
    routing_dimension: int = 64
    queries_per_seed: int = 8
    evidence_regions: int = 4
    consumer_layers: int = 2
    total_layers: int = 8
    indexed_probes: tuple[int, ...] = (1, 4, 16)
    search_repeats: int = 7
    confidence_thresholds: tuple[float, ...] = (0.00, 0.03, 0.06, 0.10)
    dispersion_regions: tuple[int, ...] = (1, 2, 4, 8)
    dispersion_depths: tuple[int, ...] = (1, 2, 4)

    def validate(self) -> None:
        if any(tokens % self.chunk_tokens for tokens in self.logical_tokens):
            raise ValueError("logical-token ladder must divide exactly into search chunks")
        if any(budget % self.materialized_tokens_per_node for budget in self.active_kv_budgets):
            raise ValueError("active budgets must divide into materialized node spans")
        if self.evidence_regions > min(tokens // self.chunk_tokens for tokens in self.logical_tokens):
            raise ValueError("evidence working set exceeds the smallest logical memory")


@dataclass
class MemoryPool:
    keys: torch.Tensor
    centroids: torch.Tensor
    evidence: torch.Tensor
    queries: torch.Tensor
    clusters: int
    nodes_per_cluster: int
    build_seconds: float

    @property
    def index_bytes(self) -> int:
        return (
            self.keys.numel() * self.keys.element_size()
            + self.centroids.numel() * self.centroids.element_size()
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
        ).strip()
    )


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cluster_count(nodes: int) -> int:
    target = 2 ** round(math.log2(math.sqrt(nodes)))
    return min(512, max(32, target))


def build_memory_pool(
    nodes: int,
    *,
    dimension: int,
    queries: int,
    evidence_regions: int,
    seed: int,
    device: torch.device,
    hard_negatives: int = 0,
    hard_negative_noise: float = 0.040,
) -> MemoryPool:
    """Build native-key surrogates with planted evidence and optional decoys.

    Hard negatives share the query's coarse region and may be closer to the
    query than evidence. They make adaptive stopping observable without
    changing the fixed evidence count or consulting truth during selection.
    """

    clusters = _cluster_count(nodes)
    if nodes % clusters:
        raise ValueError("controlled memory sizes must divide into power-of-two clusters")
    per_cluster = nodes // clusters
    if per_cluster < evidence_regions + hard_negatives:
        raise ValueError("a cluster cannot hold the requested evidence working set")
    generator = torch.Generator(device=device).manual_seed(seed + nodes)
    _synchronize(device)
    started = time.perf_counter()
    coarse = F.normalize(
        torch.randn(clusters, dimension, generator=generator, device=device), dim=1
    )
    noise = torch.randn(nodes, dimension, generator=generator, device=device)
    keys = F.normalize(coarse.repeat_interleave(per_cluster, dim=0) + 0.70 * noise, dim=1)

    query_rows, evidence_rows = [], []
    for query_index in range(queries):
        cluster = (query_index * 7 + seed) % clusters
        unique = F.normalize(
            torch.randn(dimension, generator=generator, device=device), dim=0
        )
        query = F.normalize(1.6 * coarse[cluster] + 0.55 * unique, dim=0)
        occupied = evidence_regions + hard_negatives
        local_start = (query_index * (occupied + 3)) % (per_cluster - occupied + 1)
        indices = torch.arange(
            cluster * per_cluster + local_start,
            cluster * per_cluster + local_start + evidence_regions,
            device=device,
        )
        keys[indices] = F.normalize(
            query.unsqueeze(0)
            + 0.025
            * torch.randn(
                evidence_regions, dimension, generator=generator, device=device
            ),
            dim=1,
        )
        if hard_negatives:
            decoys = torch.arange(
                int(indices[-1]) + 1,
                int(indices[-1]) + 1 + hard_negatives,
                device=device,
            )
            keys[decoys] = F.normalize(
                query.unsqueeze(0)
                + hard_negative_noise
                * torch.randn(
                    hard_negatives, dimension, generator=generator, device=device
                ),
                dim=1,
            )
        query_rows.append(query)
        evidence_rows.append(indices)

    # Recompute the coarse index after planting evidence, as an index builder would.
    centroids = F.normalize(keys.view(clusters, per_cluster, dimension).mean(dim=1), dim=1)
    _synchronize(device)
    build_seconds = time.perf_counter() - started
    return MemoryPool(
        keys,
        centroids,
        torch.stack(evidence_rows),
        torch.stack(query_rows),
        clusters,
        per_cluster,
        build_seconds,
    )


def _timed(function, repeats: int, device: torch.device):
    values, result = [], None
    function()  # warm-up
    _synchronize(device)
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        result = function()
        _synchronize(device)
        values.append(time.perf_counter() - started)
    return result, values


def exact_search(pool: MemoryPool, k: int, repeats: int) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    k = min(k, len(pool.keys))

    def run():
        scores = pool.queries @ pool.keys.T
        return torch.topk(scores, k, dim=1, sorted=True)

    result, timings = _timed(run, repeats, pool.keys.device)
    assert result is not None
    return result.indices, result.values, timings


def indexed_search(
    pool: MemoryPool,
    k: int,
    probes: int,
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float], int]:
    probes = min(probes, pool.clusters)
    effective_k = min(k, probes * pool.nodes_per_cluster)

    def run():
        coarse_ids = torch.topk(pool.queries @ pool.centroids.T, probes, dim=1).indices
        row_indices, row_scores = [], []
        offsets = torch.arange(pool.nodes_per_cluster, device=pool.keys.device)
        for query, selected in zip(pool.queries, coarse_ids):
            candidates = (selected[:, None] * pool.nodes_per_cluster + offsets[None, :]).reshape(-1)
            scores = query @ pool.keys[candidates].T
            values, local = torch.topk(scores, effective_k, sorted=True)
            row_indices.append(candidates[local])
            row_scores.append(values)
        return torch.stack(row_indices), torch.stack(row_scores)

    result, timings = _timed(run, repeats, pool.keys.device)
    assert result is not None
    comparisons = len(pool.queries) * (
        pool.clusters + probes * pool.nodes_per_cluster
    )
    return result[0], result[1], timings, comparisons


def retrieval_metrics(indices: torch.Tensor, evidence: torch.Tensor) -> dict[str, float]:
    recalls, roots, complete = [], [], []
    for selected, truth in zip(indices, evidence):
        selected_set = set(int(value) for value in selected.detach().cpu().tolist())
        truth_values = [int(value) for value in truth.detach().cpu().tolist()]
        found = sum(value in selected_set for value in truth_values)
        recalls.append(found / len(truth_values))
        roots.append(float(truth_values[0] in selected_set))
        complete.append(float(found == len(truth_values)))
    return {
        "root_recall": statistics.fmean(roots),
        "evidence_recall": statistics.fmean(recalls),
        "path_recall": statistics.fmean(complete),
        "task_accuracy": statistics.fmean(complete),
    }


def _row_timing(timings: Sequence[float], queries: int) -> dict[str, float]:
    return {
        "search_latency_p50_seconds": percentile(timings, 0.50),
        "search_latency_p95_seconds": percentile(timings, 0.95),
        "queries_per_second": queries / max(percentile(timings, 0.50), 1e-12),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    values = list(rows)
    if fields is None:
        fields = sorted({field for row in values for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _base_row(
    *,
    config: ScalingConfig,
    logical_tokens: int,
    seed: int,
    device: torch.device,
    pool: MemoryPool,
    backend: str,
    probes: int,
    requested_budget: int,
    actual_nodes: int,
) -> dict[str, Any]:
    active_tokens = actual_nodes * config.materialized_tokens_per_node
    bytes_per_layer = active_tokens * config.routing_dimension * 2 * 4
    return {
        "seed": seed,
        "training_regime": "PRA-native routing surrogate; no LM checkpoint",
        "measurement_scope": "controlled fixed-working-set retrieval",
        "measured": True,
        "logical_tokens": logical_tokens,
        "memory_nodes": logical_tokens // config.chunk_tokens,
        "encode_granularity_tokens": config.chunk_tokens,
        "search_granularity_tokens": config.chunk_tokens,
        "materialize_granularity_tokens": config.materialized_tokens_per_node,
        "requested_active_kv_tokens": requested_budget,
        "active_native_kv_tokens": active_tokens,
        "consumer_layers": config.consumer_layers,
        "total_layers": config.total_layers,
        "layer_token_kv_states": active_tokens * config.consumer_layers,
        "cpu_reference_bytes": logical_tokens * 4,
        "gpu_active_kv_bytes": bytes_per_layer * config.consumer_layers,
        "h2d_bytes": bytes_per_layer * config.consumer_layers,
        "routing_index_bytes": pool.index_bytes,
        "backend": backend,
        "probes": probes,
        "device": str(device),
        "dtype": str(pool.keys.dtype).replace("torch.", ""),
    }


def run_primary(config: ScalingConfig, device: torch.device) -> dict[str, list[dict[str, Any]]]:
    tables = {
        "model_scaling_runs": [],
        "logical_memory_scaling": [],
        "active_kv_scaling": [],
        "search_index_scaling": [],
        "adaptive_effort_scaling": [],
        "evidence_dispersion_scaling": [],
        "parameter_memory_tradeoff": [],
        "hardware_scaling": [],
        "serving_scaling": [],
        "baseline_scaling": [],
        "training_compute_scaling": [],
    }
    max_k = max(config.active_kv_budgets) // config.materialized_tokens_per_node
    primary_budget = 128

    for seed in config.seeds:
        for logical_tokens in config.logical_tokens:
            nodes = logical_tokens // config.chunk_tokens
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            pool = build_memory_pool(
                nodes,
                dimension=config.routing_dimension,
                queries=config.queries_per_seed,
                evidence_regions=config.evidence_regions,
                seed=seed,
                device=device,
            )
            peak_index_bytes = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else pool.index_bytes
            )
            exact_cache: dict[int, tuple[torch.Tensor, torch.Tensor, list[float]]] = {}
            indexed_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, list[float], int]] = {}
            for budget in config.active_kv_budgets:
                top_nodes = budget // config.materialized_tokens_per_node
                exact_cache[top_nodes] = exact_search(pool, top_nodes, config.search_repeats)
                indices, scores, timings = exact_cache[top_nodes]
                metrics = retrieval_metrics(indices, pool.evidence)
                base = _base_row(
                    config=config,
                    logical_tokens=logical_tokens,
                    seed=seed,
                    device=device,
                    pool=pool,
                    backend="exact_gemm",
                    probes=pool.clusters,
                    requested_budget=budget,
                    actual_nodes=indices.shape[1],
                )
                row = {
                    **base,
                    **metrics,
                    **_row_timing(timings, len(pool.queries)),
                    "comparisons": len(pool.queries) * nodes,
                    "index_build_seconds": pool.build_seconds,
                    "peak_device_bytes": peak_index_bytes,
                    "logical_to_active_ratio": logical_tokens / base["active_native_kv_tokens"],
                }
                tables["active_kv_scaling"].append(row)
                if budget == primary_budget:
                    tables["logical_memory_scaling"].append(dict(row))

                for probes in config.indexed_probes:
                    key = (top_nodes, probes)
                    indexed_cache[key] = indexed_search(pool, top_nodes, probes, config.search_repeats)
                    idx, idx_scores, idx_timings, comparisons = indexed_cache[key]
                    idx_metrics = retrieval_metrics(idx, pool.evidence)
                    idx_base = _base_row(
                        config=config,
                        logical_tokens=logical_tokens,
                        seed=seed,
                        device=device,
                        pool=pool,
                        backend="coarse_to_fine",
                        probes=probes,
                        requested_budget=budget,
                        actual_nodes=idx.shape[1],
                    )
                    idx_row = {
                        **idx_base,
                        **idx_metrics,
                        **_row_timing(idx_timings, len(pool.queries)),
                        "comparisons": comparisons,
                        "index_build_seconds": pool.build_seconds,
                        "peak_device_bytes": peak_index_bytes,
                        "logical_to_active_ratio": logical_tokens / idx_base["active_native_kv_tokens"],
                    }
                    tables["active_kv_scaling"].append(idx_row)
                    if budget == primary_budget:
                        exact_metrics = retrieval_metrics(exact_cache[top_nodes][0], pool.evidence)
                        tables["search_index_scaling"].append(
                            {
                                **idx_row,
                                "recall_at_k_vs_exact": statistics.fmean(
                                    len(set(a.tolist()) & set(b.tolist())) / max(len(b), 1)
                                    for a, b in zip(idx.cpu(), exact_cache[top_nodes][0].cpu())
                                ),
                                "evidence_recall_delta_vs_exact": idx_metrics["evidence_recall"]
                                - exact_metrics["evidence_recall"],
                            }
                        )
                        if probes == 4:
                            tables["logical_memory_scaling"].append(dict(idx_row))

            # Adaptive retry is oracle-free: confidence is the score gap after the
            # current boundary; truth is consulted only for evaluation.
            full_indices, full_scores, _, _ = indexed_search(pool, max_k + 1, 4, config.search_repeats)
            for threshold in config.confidence_thresholds:
                selected_widths = []
                selected_rows = []
                for query_index in range(len(pool.queries)):
                    chosen = max_k
                    for budget in config.active_kv_budgets:
                        width = min(budget // config.materialized_tokens_per_node, full_scores.shape[1] - 1)
                        gap = float(full_scores[query_index, width - 1] - full_scores[query_index, width])
                        if gap >= threshold:
                            chosen = width
                            break
                    selected_widths.append(chosen)
                    selected_rows.append(full_indices[query_index, :chosen])
                widest = max(len(row) for row in selected_rows)
                padded_rows = [
                    F.pad(row, (0, widest - len(row)), value=-1)
                    for row in selected_rows
                ]
                adaptive_metrics = retrieval_metrics(torch.stack(padded_rows), pool.evidence)
                mean_tokens = statistics.fmean(selected_widths) * config.materialized_tokens_per_node
                tables["adaptive_effort_scaling"].append(
                    {
                        "seed": seed,
                        "logical_tokens": logical_tokens,
                        "memory_nodes": nodes,
                        "threshold": threshold,
                        "difficulty": "easy",
                        "hard_negative_nodes": 0,
                        "backend": "coarse_to_fine",
                        "probes": 4,
                        "expected_active_kv_tokens": mean_tokens,
                        "expected_layer_token_kv_states": mean_tokens * config.consumer_layers,
                        "escalation_rate": statistics.fmean(
                            width > config.active_kv_budgets[0] // config.materialized_tokens_per_node
                            for width in selected_widths
                        ),
                        "max_budget_rate": statistics.fmean(width == max_k for width in selected_widths),
                        **adaptive_metrics,
                        "measured": True,
                        "measurement_scope": "oracle-free score-gap controller on controlled retrieval",
                        "training_regime": "PRA-native routing surrogate; no LM checkpoint",
                        "device": str(device),
                    }
                )

            # Serving-style batched exact search isolates routing throughput. It
            # excludes tokenizer, model prefill, generation, and network service.
            for concurrency in (1, 4, 8):
                queries = pool.queries[:concurrency]

                def serve_once():
                    return torch.topk(queries @ pool.keys.T, min(16, nodes), dim=1)

                _, timings = _timed(serve_once, config.search_repeats, device)
                tables["serving_scaling"].append(
                    {
                        "seed": seed,
                        "logical_tokens": logical_tokens,
                        "memory_nodes": nodes,
                        "concurrency": len(queries),
                        "routing_latency_p50_seconds": percentile(timings, 0.5),
                        "routing_latency_p95_seconds": percentile(timings, 0.95),
                        "routing_queries_per_second": len(queries) / max(percentile(timings, 0.5), 1e-12),
                        "ttft_seconds": "",
                        "tpot_seconds": "",
                        "end_to_end_throughput": "",
                        "measured": True,
                        "measurement_scope": "routing-only batched GEMM; not end-to-end serving",
                        "device": str(device),
                    }
                )

            del pool
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return tables


def run_adaptive_difficulty(
    config: ScalingConfig, device: torch.device
) -> list[dict[str, Any]]:
    """Stress adaptive stopping with semantically confusable planted decoys."""

    rows: list[dict[str, Any]] = []
    max_k = max(config.active_kv_budgets) // config.materialized_tokens_per_node
    for difficulty, hard_negatives, noise in ADAPTIVE_DIFFICULTIES:
        for seed in config.seeds:
            for logical_tokens in config.logical_tokens:
                nodes = logical_tokens // config.chunk_tokens
                pool = build_memory_pool(
                    nodes,
                    dimension=config.routing_dimension,
                    queries=config.queries_per_seed,
                    evidence_regions=config.evidence_regions,
                    seed=seed,
                    device=device,
                    hard_negatives=hard_negatives,
                    hard_negative_noise=noise,
                )
                indices, scores, timings, comparisons = indexed_search(
                    pool, max_k + 1, 4, config.search_repeats
                )
                for threshold in config.confidence_thresholds:
                    selected_widths: list[int] = []
                    selected_rows: list[torch.Tensor] = []
                    for query_index in range(len(pool.queries)):
                        chosen = max_k
                        for budget in config.active_kv_budgets:
                            width = min(
                                budget // config.materialized_tokens_per_node,
                                scores.shape[1] - 1,
                            )
                            gap = float(
                                scores[query_index, width - 1]
                                - scores[query_index, width]
                            )
                            if gap >= threshold:
                                chosen = width
                                break
                        selected_widths.append(chosen)
                        selected_rows.append(indices[query_index, :chosen])
                    widest = max(map(len, selected_rows))
                    padded = torch.stack(
                        [F.pad(row, (0, widest - len(row)), value=-1) for row in selected_rows]
                    )
                    metrics = retrieval_metrics(padded, pool.evidence)
                    mean_tokens = (
                        statistics.fmean(selected_widths)
                        * config.materialized_tokens_per_node
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "logical_tokens": logical_tokens,
                            "memory_nodes": nodes,
                            "threshold": threshold,
                            "difficulty": difficulty,
                            "hard_negative_nodes": hard_negatives,
                            "hard_negative_noise": noise,
                            "backend": "coarse_to_fine",
                            "probes": 4,
                            "expected_active_kv_tokens": mean_tokens,
                            "expected_layer_token_kv_states": (
                                mean_tokens * config.consumer_layers
                            ),
                            "escalation_rate": statistics.fmean(
                                width
                                > config.active_kv_budgets[0]
                                // config.materialized_tokens_per_node
                                for width in selected_widths
                            ),
                            "max_budget_rate": statistics.fmean(
                                width == max_k for width in selected_widths
                            ),
                            "search_latency_p50_seconds": percentile(timings, 0.50),
                            "search_latency_p95_seconds": percentile(timings, 0.95),
                            "comparisons": comparisons,
                            **metrics,
                            "measured": True,
                            "measurement_scope": (
                                "oracle-free score-gap controller with planted "
                                "semantically confusable neighbors"
                            ),
                            "training_regime": (
                                "PRA-native routing surrogate; no LM checkpoint"
                            ),
                            "device": str(device),
                        }
                    )
                del pool
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return rows


def run_dispersion(config: ScalingConfig, device: torch.device) -> list[dict[str, Any]]:
    """Measure active K/V needed as evidence working-set complexity grows."""

    rows = []
    logical_tokens = config.logical_tokens[2]
    nodes = logical_tokens // config.chunk_tokens
    for seed in config.seeds:
        for regions in config.dispersion_regions:
            for depth in config.dispersion_depths:
                evidence_count = regions * depth
                pool = build_memory_pool(
                    nodes,
                    dimension=config.routing_dimension,
                    queries=config.queries_per_seed,
                    evidence_regions=evidence_count,
                    seed=seed + regions * 100 + depth,
                    device=device,
                )
                achieved = False
                for budget in config.active_kv_budgets:
                    top_nodes = budget // config.materialized_tokens_per_node
                    indices, _, timings = exact_search(pool, top_nodes, config.search_repeats)
                    metrics = retrieval_metrics(indices, pool.evidence)
                    meets = metrics["evidence_recall"] >= 0.90
                    rows.append(
                        {
                            "seed": seed,
                            "logical_tokens": logical_tokens,
                            "evidence_regions": regions,
                            "chain_depth": depth,
                            "evidence_nodes": evidence_count,
                            "source_dispersion_fraction": (regions - 1) / max(regions, 1),
                            "max_evidence_distance_tokens": int(
                                logical_tokens * (regions - 1) / max(regions, 1)
                            ),
                            "active_native_kv_tokens": budget,
                            "layer_token_kv_states": budget * config.consumer_layers,
                            "evidence_recall": metrics["evidence_recall"],
                            "task_accuracy": metrics["task_accuracy"],
                            "meets_90pct_recall": meets,
                            "first_budget_meeting_target": bool(meets and not achieved),
                            "search_latency_p50_seconds": percentile(timings, 0.5),
                            "measured": True,
                            "measurement_scope": "controlled difficulty scaling",
                            "training_regime": "PRA-native routing surrogate; no LM checkpoint",
                            "device": str(device),
                        }
                    )
                    achieved |= meets
                del pool
    return rows


def run_hardware(config: ScalingConfig, devices: Sequence[torch.device]) -> list[dict[str, Any]]:
    rows = []
    hardware_points = tuple(
        dict.fromkeys(
            (
                config.logical_tokens[min(1, len(config.logical_tokens) - 1)],
                config.logical_tokens[min(3, len(config.logical_tokens) - 1)],
            )
        )
    )
    for device in devices:
        for logical_tokens in hardware_points:
            nodes = logical_tokens // config.chunk_tokens
            pool = build_memory_pool(
                nodes,
                dimension=config.routing_dimension,
                queries=config.queries_per_seed,
                evidence_regions=config.evidence_regions,
                seed=config.seeds[0],
                device=device,
            )
            indices, _, timings = exact_search(pool, 16, max(config.search_repeats + 2, 11))
            rows.append(
                {
                    "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
                    "device": str(device),
                    "logical_tokens": logical_tokens,
                    "memory_nodes": nodes,
                    "backend": "exact_gemm",
                    "search_latency_p50_seconds": percentile(timings, 0.5),
                    "search_latency_p95_seconds": percentile(timings, 0.95),
                    "queries_per_second": len(pool.queries) / max(percentile(timings, 0.5), 1e-12),
                    "evidence_recall": retrieval_metrics(indices, pool.evidence)["evidence_recall"],
                    "measured": True,
                    "measurement_scope": "routing microbenchmark",
                }
            )
            del pool
    rows.append(
        {
            "hardware": "Apple Silicon",
            "device": "mps",
            "logical_tokens": "",
            "memory_nodes": "",
            "backend": "",
            "search_latency_p50_seconds": "",
            "search_latency_p95_seconds": "",
            "queries_per_second": "",
            "evidence_recall": "",
            "measured": False,
            "measurement_scope": "planned; hardware unavailable in this run",
        }
    )
    return rows


def inherited_model_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load Paper 4 calibration without treating it as a matched scaling curve."""

    result_dir = root / "docs/papers/shared/results/paper4_training/tier0"
    with (result_dir / "adaptation_ladder_seed_results.csv").open(newline="", encoding="utf-8") as stream:
        seed_rows = list(csv.DictReader(stream))
    with (result_dir / "lora_configs.csv").open(newline="", encoding="utf-8") as stream:
        config_rows = list(csv.DictReader(stream))
    parameter_lookup = {row["model"]: row for row in config_rows if row["seed"] == "17"}
    models, tradeoffs = [], []
    regime_names = {
        "frozen": "Frozen",
        "consumer_lora": "LoRA",
        "interface_lora": "LoRA",
        "broad_lora": "LoRA",
        "full_weight": "Full-weight continued",
        "native_scratch": "PRA-native",
    }
    for model, regime in regime_names.items():
        values = [row for row in seed_rows if row["model"] == model and row["condition"] == "evidence_only"]
        parameter = parameter_lookup[model]
        accuracy = statistics.fmean(float(row["accuracy"]) for row in values)
        row = {
            "checkpoint": f"paper4_tier0_{model}",
            "model_parameters": int(parameter["total_parameters"]),
            "trainable_parameters": int(parameter["trainable_parameters"]),
            "training_regime": regime,
            "seeds": len(values),
            "quality_metric": "controlled answer accuracy",
            "quality": accuracy,
            "logical_tokens": "",
            "active_native_kv_tokens": "evidence-only oracle spans",
            "measured": True,
            "measurement_scope": "inherited Paper 4 calibration; excluded from Paper 5 scaling fits",
            "source": "paper4_training/tier0 at train branch b7db316",
        }
        models.append(row)
        tradeoffs.append(dict(row))
    for checkpoint, parameters in (
        ("google/gemma-3-270m", 270_000_000),
        ("google/gemma-3-1b", 999_885_955),
        ("google/gemma-3-4b", 4_000_000_000),
    ):
        models.append(
            {
                "checkpoint": checkpoint,
                "model_parameters": parameters,
                "trainable_parameters": "",
                "training_regime": "planned",
                "seeds": 0,
                "quality_metric": "",
                "quality": "",
                "logical_tokens": "",
                "active_native_kv_tokens": "",
                "measured": False,
                "measurement_scope": "pending matched-task model ladder",
                "source": "Paper 5 sponsorship-gated plan",
            }
        )

    training_rows = []
    for filename in ("paper4_hardware_benchmarks.csv",):
        with (result_dir / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                training_rows.append(
                    {
                        **row,
                        "training_regime": regime_names.get(row["model"], row["model"]),
                        "trainable_fraction": parameter_lookup.get(row["model"], {}).get("trainable_fraction", ""),
                        "measured": True,
                        "measurement_scope": "inherited Paper 4 controlled training",
                        "training_flops": "",
                        "training_cost_usd": "",
                    }
                )
    return models, tradeoffs, training_rows


def modeled_baselines(config: ScalingConfig) -> list[dict[str, Any]]:
    rows = []
    for logical_tokens in config.logical_tokens:
        for name, active, search in (
            ("native_dense_context", logical_tokens, "none"),
            ("text_rag_top4_chunks", 4 * config.chunk_tokens, "external top-4"),
            ("fixed_pra_top16_spans", 16 * config.materialized_tokens_per_node, "exact native-key"),
        ):
            layer_tokens = active * config.total_layers if name == "native_dense_context" else active * config.consumer_layers
            rows.append(
                {
                    "baseline": name,
                    "logical_tokens": logical_tokens,
                    "active_tokens": active,
                    "layer_token_kv_states": layer_tokens,
                    "search_backend": search,
                    "quality": "",
                    "latency_seconds": "",
                    "cost_per_million_tokens_usd": "",
                    "measured": False,
                    "measurement_scope": "analytical state accounting only; no quality or latency claim",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper5_scaling",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Use two seeds and the first three memory sizes.")
    args = parser.parse_args()

    config = ScalingConfig()
    if args.quick:
        config = ScalingConfig(
            seeds=config.seeds[:2],
            logical_tokens=config.logical_tokens[:3],
            active_kv_budgets=(32, 128, 512),
            indexed_probes=(1, 4),
            search_repeats=2,
            dispersion_regions=(1, 4),
            dispersion_depths=(1, 2),
        )
    config.validate()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **asdict(config),
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_tree_dirty_at_run": _git_dirty(),
        "experiment_source_sha256": _source_hash(
            [
                Path(__file__).resolve(),
                Path(__file__).with_name("scaling_core.py").resolve(),
                Path(__file__).with_name("summarize_scaling_study.py").resolve(),
            ]
        ),
        "primary_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "scope": "controlled retrieval and systems pilot; no language-model output evaluation",
    }
    (output / "scaling_configs.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    tables = run_primary(config, device)
    tables["adaptive_effort_scaling"].extend(
        run_adaptive_difficulty(config, device)
    )
    tables["evidence_dispersion_scaling"] = run_dispersion(config, device)
    hardware_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        hardware_devices.append(torch.device("cuda"))
    tables["hardware_scaling"] = run_hardware(config, hardware_devices)
    models, tradeoffs, training = inherited_model_rows(ROOT)
    tables["model_scaling_runs"] = models
    tables["parameter_memory_tradeoff"] = tradeoffs
    tables["training_compute_scaling"] = training
    tables["baseline_scaling"] = modeled_baselines(config)

    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)

    print(json.dumps({name: len(rows) for name, rows in tables.items()}, indent=2))
    print(f"Wrote Paper 5 raw artifacts to {output}")


if __name__ == "__main__":
    main()
