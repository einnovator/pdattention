"""Controlled systems and matched-budget baseline benchmarks for Paper 3.5."""

from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pra_hf.serving_runtime import NativeQKIndex, PagedKVCache, fused_gather_kv, merge_token_intervals


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({field for row in values for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _median_runtime(function: Callable[[], Any], repeats: int) -> tuple[float, Any, list[float]]:
    timings, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings), result, timings


def benchmark_indexed_search() -> list[dict[str, Any]]:
    """Measure exact loop/GEMM and deterministic coarse-to-fine recall/latency."""

    rows = []
    for count in (256, 1024, 4096):
        generator = torch.Generator().manual_seed(1000 + count)
        keys = torch.nn.functional.normalize(torch.randn(count, 64, generator=generator), dim=1)
        targets = torch.randint(0, count, (32,), generator=generator)
        queries = torch.nn.functional.normalize(
            keys[targets] + 0.08 * torch.randn(32, 64, generator=generator), dim=1
        )
        build_started = time.perf_counter()
        index = NativeQKIndex(keys, coarse_clusters=min(32, count))
        build_seconds = time.perf_counter() - build_started
        exact = index.search(queries, 8, backend="gemm")
        exact_sets = [set(row.tolist()) for row in exact.indices]
        configs = [("brute_force", 0), ("gemm", 0), ("coarse_to_fine", 2), ("coarse_to_fine", 4), ("coarse_to_fine", 8)]
        for backend, probes in configs:
            def run():
                return index.search(queries, 8, backend=backend, probes=max(probes, 1))

            seconds, result, timings = _median_runtime(run, 3)
            recall = statistics.fmean(
                len(set(actual.tolist()) & expected) / len(expected)
                for actual, expected in zip(result.indices, exact_sets)
            )
            target_recall = statistics.fmean(
                int(int(target) in set(actual.tolist()))
                for target, actual in zip(targets, result.indices)
            )
            rows.append(
                {
                    "memory_vectors": count,
                    "queries": len(queries),
                    "dimension": keys.shape[1],
                    "top_k": 8,
                    "backend": backend,
                    "probes": probes,
                    "recall_at_8_vs_exact": recall,
                    "target_recall": target_recall,
                    "median_search_seconds": seconds,
                    "p95_search_seconds": sorted(timings)[-1],
                    "queries_per_second": len(queries) / max(seconds, 1e-12),
                    "comparisons": result.comparisons,
                    "index_build_seconds": build_seconds,
                    "device": "cpu",
                    "measured": True,
                }
            )
    return rows


def _slice_cat_gather(
    key: torch.Tensor, value: torch.Tensor, intervals: list[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    merged = merge_token_intervals(intervals, key.shape[2])
    return (
        torch.cat([key[:, :, start:end, :] for start, end in merged], dim=2),
        torch.cat([value[:, :, start:end, :] for start, end in merged], dim=2),
    )


def benchmark_kernels() -> list[dict[str, Any]]:
    """Compare Python slice concatenation with one index-select gather."""

    rows = []
    generator = torch.Generator().manual_seed(31)
    for tokens in (512, 2048, 8192):
        key = torch.randn(1, 8, tokens, 32, generator=generator)
        value = torch.randn(1, 8, tokens, 32, generator=generator)
        starts = torch.linspace(0, tokens - 24, 12).long().tolist()
        intervals = [(int(start), min(int(start) + 16, tokens)) for start in starts]
        expected_key, expected_value = _slice_cat_gather(key, value, intervals)
        def fused_once():
            gathered = fused_gather_kv(key, value, intervals)
            return gathered.key, gathered.value

        methods = {
            "python_slice_cat": lambda: _slice_cat_gather(key, value, intervals),
            "fused_index_select": fused_once,
        }
        for name, function in methods.items():
            seconds, result, timings = _median_runtime(function, 15)
            parity = torch.equal(result[0], expected_key) and torch.equal(result[1], expected_value)
            rows.append(
                {
                    "source_tokens": tokens,
                    "materialized_tokens": expected_key.shape[2],
                    "heads": key.shape[1],
                    "head_dim": key.shape[3],
                    "method": name,
                    "median_seconds": seconds,
                    "p95_seconds": sorted(timings)[int(0.95 * (len(timings) - 1))],
                    "parity": parity,
                    "source_kv_bytes": key.numel() * key.element_size() * 2,
                    "active_kv_bytes": expected_key.numel() * expected_key.element_size() * 2,
                    "device": "cpu",
                    "measured": True,
                    "production_kernel": False,
                }
            )
    return rows


def benchmark_paged_kv() -> list[dict[str, Any]]:
    rows = []
    generator = torch.Generator().manual_seed(37)
    key = torch.randn(8, 1537, 32, generator=generator)
    value = torch.randn(8, 1537, 32, generator=generator)
    patterns = {
        "clustered": list(range(128, 192)) + list(range(900, 964)),
        "dispersed": torch.linspace(0, 1536, 128).long().tolist(),
    }
    for page_size in (16, 32, 64, 128):
        for pattern, selected in patterns.items():
            cache = PagedKVCache(
                page_size,
                math.ceil(key.shape[1] / page_size),
                policy="lru",
            )
            put_started = time.perf_counter()
            cache.put("request", key, value)
            put_seconds = time.perf_counter() - put_started
            page_seconds, gathered, timings = _median_runtime(
                lambda: cache.gather("request", selected), 5
            )
            contiguous_seconds, contiguous, _ = _median_runtime(
                lambda: (
                    key.index_select(1, torch.tensor(selected)),
                    value.index_select(1, torch.tensor(selected)),
                ),
                5,
            )
            parity = torch.equal(gathered[0], contiguous[0]) and torch.equal(gathered[1], contiguous[1])
            touched_pages = len({int(token) // page_size for token in selected})
            rows.append(
                {
                    "page_size": page_size,
                    "selection_pattern": pattern,
                    "source_tokens": key.shape[1],
                    "selected_tokens": len(selected),
                    "resident_pages": int(cache.stats()["resident_pages"]),
                    "touched_pages": touched_pages,
                    "fragmentation_tokens": cache.fragmentation_tokens,
                    "resident_bytes": cache.resident_bytes,
                    "put_seconds": put_seconds,
                    "paged_gather_seconds": page_seconds,
                    "contiguous_index_select_seconds": contiguous_seconds,
                    "paged_overhead_ratio": page_seconds / max(contiguous_seconds, 1e-12),
                    "parity": parity,
                    "cache_hit_rate": cache.hit_rate,
                    "measured": True,
                    "integration_scope": "standalone_page_table_prototype_not_vllm",
                }
            )
    return rows


class _PolicyCache:
    def __init__(self, capacity: int, policy: str) -> None:
        self.capacity, self.policy, self.clock = capacity, policy, 0
        self.values: dict[int, tuple[int, int]] = {}

    def access(self, identity: int) -> bool:
        self.clock += 1
        if identity in self.values:
            frequency, _ = self.values[identity]
            self.values[identity] = (frequency + 1, self.clock)
            return True
        if len(self.values) >= self.capacity:
            if self.policy == "lru":
                victim = min(self.values, key=lambda key: (self.values[key][1], key))
            elif self.policy == "lfu":
                victim = min(self.values, key=lambda key: (self.values[key][0], self.values[key][1], key))
            else:
                victim = min(
                    self.values,
                    key=lambda key: (
                        self.values[key][0] / max(self.clock - self.values[key][1] + 1, 1),
                        self.values[key][1],
                        key,
                    ),
                )
            del self.values[victim]
        self.values[identity] = (1, self.clock)
        return False


def benchmark_cache_policies() -> list[dict[str, Any]]:
    """Replay a fixed locality/path workload with cache and prefetch policies."""

    requests = []
    for round_index in range(30):
        hot = [0, 1, 2, 3, 4]
        path = [10 + round_index % 12, 11 + round_index % 12, 12 + round_index % 12]
        requests.extend(hot + path)
    rows = []
    for capacity in (8, 16, 32):
        for policy in ("lru", "lfu", "hybrid"):
            for prefetch in ("none", "graph_neighbor", "oracle_next_upper_bound"):
                cache = _PolicyCache(capacity, policy)
                hits = misses = useful = wasted = 0
                prefetched: set[int] = set()
                started = time.perf_counter()
                for index, identity in enumerate(requests):
                    hit = cache.access(identity)
                    hits += int(hit)
                    misses += int(not hit)
                    if identity in prefetched:
                        useful += 1
                        prefetched.remove(identity)
                    candidates = []
                    if prefetch == "graph_neighbor":
                        candidates = [identity + 1]
                    elif prefetch == "oracle_next_upper_bound" and index + 1 < len(requests):
                        candidates = [requests[index + 1]]
                    for candidate in candidates:
                        if candidate not in cache.values:
                            cache.access(candidate)
                            prefetched.add(candidate)
                wasted = len(prefetched)
                seconds = time.perf_counter() - started
                rows.append(
                    {
                        "capacity_pages": capacity,
                        "cache_policy": policy,
                        "prefetch_policy": prefetch,
                        "requests": len(requests),
                        "hit_rate": hits / len(requests),
                        "miss_rate": misses / len(requests),
                        "useful_prefetches": useful,
                        "wasted_prefetches_at_end": wasted,
                        "replay_seconds": seconds,
                        "measured": True,
                        "workload": "controlled_hotset_plus_path",
                    }
                )
    return rows


def _batch_operation(keys: list[torch.Tensor], queries: list[torch.Tensor], strategy: str) -> None:
    if strategy == "packed_ragged":
        for key, query in zip(keys, queries):
            torch.mv(key, query)
    elif strategy == "pad_to_max":
        maximum = max(len(key) for key in keys)
        padded = torch.zeros(len(keys), maximum, keys[0].shape[1])
        query = torch.stack(queries).unsqueeze(2)
        for index, key in enumerate(keys):
            padded[index, : len(key)] = key
        torch.bmm(padded, query)
    elif strategy in {"bucketed", "page_based"}:
        buckets: dict[int, list[int]] = defaultdict(list)
        quantum = 128 if strategy == "bucketed" else 64
        for index, key in enumerate(keys):
            buckets[math.ceil(len(key) / quantum) * quantum].append(index)
        for width, indices in buckets.items():
            padded = torch.zeros(len(indices), width, keys[0].shape[1])
            query = torch.stack([queries[index] for index in indices]).unsqueeze(2)
            for local, index in enumerate(indices):
                padded[local, : len(keys[index])] = keys[index]
            torch.bmm(padded, query)
    else:
        raise ValueError(strategy)


def benchmark_batching_concurrency() -> list[dict[str, Any]]:
    rows = []
    generator = torch.Generator().manual_seed(41)
    base_lengths = (256, 384, 512, 768, 1024)
    for concurrency in (1, 2, 4, 8, 16, 32):
        lengths = [base_lengths[index % len(base_lengths)] for index in range(concurrency)]
        keys = [torch.randn(length, 64, generator=generator) for length in lengths]
        queries = [torch.randn(64, generator=generator) for _ in lengths]
        for strategy in ("pad_to_max", "bucketed", "packed_ragged", "page_based"):
            seconds, _, timings = _median_runtime(
                lambda: _batch_operation(keys, queries, strategy), 9
            )
            if strategy == "pad_to_max":
                allocated_tokens = max(lengths) * concurrency
            elif strategy == "bucketed":
                allocated_tokens = sum(math.ceil(length / 128) * 128 for length in lengths)
            elif strategy == "page_based":
                allocated_tokens = sum(math.ceil(length / 64) * 64 for length in lengths)
            else:
                allocated_tokens = sum(lengths)
            sorted_times = sorted(timings)
            rows.append(
                {
                    "concurrent_requests": concurrency,
                    "strategy": strategy,
                    "active_kv_tokens": sum(lengths),
                    "allocated_kv_tokens": allocated_tokens,
                    "padding_or_fragmentation_fraction": 1.0 - sum(lengths) / allocated_tokens,
                    "hbm_bytes_extrapolated": allocated_tokens * 64 * 2 * 2,
                    "p50_latency_seconds": statistics.median(timings),
                    "p95_latency_seconds": sorted_times[int(0.95 * (len(sorted_times) - 1))],
                    "throughput_requests_per_second": concurrency / max(seconds, 1e-12),
                    "device": "cpu",
                    "measured_compute": True,
                    "hbm_extrapolated": True,
                }
            )
    return rows


def extract_serving_metrics(gate3_path: Path) -> list[dict[str, Any]]:
    """Normalize inherited held-out Qwen measurements into the serving schema."""

    mapping = {"graph_sparse": "E0_low", "graph_balanced": "E1_medium", "graph_high": "E2_high"}
    rows = []
    for row in read_csv(gate3_path):
        if (
            row["partition"] != "test"
            or row["phase"] != "heldout"
            or row["oracle_condition"] != "False"
            or row["condition"] not in mapping
        ):
            continue
        generated = max(int(float(row["generated_tokens"])), 1)
        total = float(row["total_generation_seconds"])
        rows.append(
            {
                "dataset": row["dataset"],
                "example_id": row["example_id"],
                "effort": mapping[row["condition"]],
                "condition": row["condition"],
                "active_kv_tokens": int(float(row["active_native_kv_token_states"])),
                "active_kv_bytes": int(float(row["active_native_kv_bytes"])),
                "peak_gpu_allocated_bytes": int(float(row["peak_gpu_allocated_bytes"])),
                "peak_gpu_reserved_bytes": int(float(row["peak_gpu_reserved_bytes"])),
                "ttft_seconds": float(row["ttft_seconds"]),
                "tpot_seconds": float(row["tpot_seconds"]),
                "total_latency_seconds": total,
                "routing_latency_seconds": float(row["routing_search_seconds"]),
                "graph_search_latency_seconds": float(row["routing_search_seconds"]),
                "gather_materialization_seconds": float(row["materialization_seconds"]),
                "h2d_bytes": int(float(row["native_kv_bytes"])),
                "h2d_seconds": float(row["selected_kv_transfer_seconds"]),
                "tokens_per_second": float(row["tokens_per_second"]),
                "throughput_requests_per_second": 1.0 / max(total, 1e-12),
                "generated_tokens": generated,
                "gpu_utilization": "not_captured",
                "cost_per_million_tokens": "not_measurable_local_hardware",
                "measured": True,
                "measurement_scope": "inherited_qwen_gate3_single_request",
            }
        )
    return rows


def benchmark_rag_and_long_context() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a controlled two-hop corpus with matched retrieval/token budgets."""

    generator = torch.Generator().manual_seed(47)
    corpus_size, dimension, examples = 1024, 48, 200
    documents = torch.nn.functional.normalize(
        torch.randn(corpus_size, dimension, generator=generator), dim=1
    )
    targets = torch.randint(256, corpus_size, (examples,), generator=generator)
    bridges = torch.randint(0, 256, (examples,), generator=generator)
    queries = torch.nn.functional.normalize(
        documents[bridges] + 0.06 * torch.randn(examples, dimension, generator=generator), dim=1
    )
    index = NativeQKIndex(documents, coarse_clusters=32)
    exact = index.search(queries, 8, backend="gemm")
    methods: dict[str, list[set[int]]] = {
        "single_shot_rag_top4": [set(row[:4].tolist()) for row in exact.indices],
        "multi_query_rag_top8": [set(row.tolist()) for row in exact.indices],
        "iterative_rag_top4_plus_links": [],
        "fixed_pra_low": [set(row[:2].tolist()) for row in exact.indices],
        "adaptive_pra_medium": [],
    }
    for index_value, root_set in enumerate(methods["single_shot_rag_top4"]):
        expanded = set(root_set)
        if int(bridges[index_value]) in expanded:
            expanded.add(int(targets[index_value]))
        methods["iterative_rag_top4_plus_links"].append(expanded)
        adaptive = set(methods["fixed_pra_low"][index_value])
        if int(bridges[index_value]) in adaptive:
            adaptive.add(int(targets[index_value]))
        methods["adaptive_pra_medium"].append(adaptive)
    rows = []
    for method, selections in methods.items():
        quality = statistics.fmean(
            int(int(target) in selected) for target, selected in zip(targets, selections)
        )
        selected_count = statistics.fmean(len(selected) for selected in selections)
        iterative = "iterative" in method or "adaptive" in method
        rows.append(
            {
                "method": method,
                "corpus_documents": corpus_size,
                "examples": examples,
                "retrieval_recall": quality,
                "answer_quality_proxy": quality,
                "mean_selected_documents": selected_count,
                "prompt_tokens": selected_count * 64 if "rag" in method else 0,
                "active_kv_tokens": selected_count * 64,
                "retrieval_seconds_per_query": exact.seconds / examples * (2 if iterative else 1),
                "ttft_proxy_seconds": exact.seconds / examples * (2 if iterative else 1),
                "cost_units": selected_count * 64 + (32 if iterative else 0),
                "controlled_proxy": True,
                "backbone_generation_held_constant": True,
            }
        )
    long_rows = []
    conditions = {
        "native_full_context": [set(range(corpus_size)) for _ in range(examples)],
        "truncate_first_128": [set(range(128)) for _ in range(examples)],
        "matched_budget_8_documents": [set(range(8)) for _ in range(examples)],
    }
    for method, selections in conditions.items():
        quality = statistics.fmean(
            int(int(target) in selected) for target, selected in zip(targets, selections)
        )
        selected = statistics.fmean(len(value) for value in selections)
        long_rows.append(
            {
                "method": method,
                "logical_context_documents": corpus_size,
                "active_documents": selected,
                "active_kv_tokens": selected * 64,
                "answer_quality_proxy": quality,
                "controlled_proxy": True,
                "generation_cost_units": selected * 64,
            }
        )
    return rows, long_rows


def benchmark_kv_baselines() -> list[dict[str, Any]]:
    """Compare mechanism-faithful controlled selectors at matched K/V budgets."""

    generator = torch.Generator().manual_seed(53)
    sequence, examples = 512, 200
    rows = []
    for budget in (32, 64, 128):
        recalls: dict[str, list[int]] = defaultdict(list)
        started = time.perf_counter()
        for _ in range(examples):
            target = int(torch.randint(0, sequence, (1,), generator=generator))
            query_scores = torch.randn(sequence, generator=generator)
            query_scores[target] += 4.0
            heavy_scores = torch.randn(sequence, generator=generator)
            heavy_scores[target] += 3.0
            selectors = {
                "H2O_proxy": torch.topk(heavy_scores, budget).indices.tolist(),
                "SnapKV_proxy": torch.topk(query_scores + 0.2 * heavy_scores, budget).indices.tolist(),
                "StreamingLLM_proxy": list(range(min(4, budget))) + list(range(sequence - (budget - min(4, budget)), sequence)),
                "ClusterKV_proxy": torch.linspace(0, sequence - 1, budget).long().tolist(),
                "PRA_native_qk_proxy": torch.topk(query_scores, budget).indices.tolist(),
            }
            for method, selected in selectors.items():
                recalls[method].append(int(target in selected))
        elapsed = time.perf_counter() - started
        for method, values in recalls.items():
            rows.append(
                {
                    "method": method,
                    "active_kv_budget": budget,
                    "salient_token_recall": statistics.fmean(values),
                    "examples": examples,
                    "benchmark_seconds_shared": elapsed,
                    "matched_budget": True,
                    "controlled_proxy": True,
                    "third_party_kernel_used": False,
                }
            )
    return rows


def baseline_taxonomy() -> str:
    return """# Baseline taxonomy

The controlled benchmark separates mechanisms that are often conflated:

| Family | Representative rows | What is active | Status in this study |
|---|---|---|---|
| Dense/native long context | full context, truncation, matched budget | prompt/native K/V | controlled matched-corpus proxy |
| External text retrieval | single-shot and multi-query RAG | retrieved text retokenized in prompt | controlled matched-corpus proxy |
| Iterative retrieval | iterative RAG | text reached after a second retrieval step | controlled matched-corpus proxy |
| K/V eviction or compression | H2O, SnapKV, StreamingLLM, ClusterKV | subset/summary of an existing sequence cache | mechanism-faithful selector proxy, not third-party kernels |
| PRA | fixed/adaptive native-Q/K search | contextual native K/V selected from addressable memory | package implementation plus controlled corpus |
| Sparse architectural attention | Longformer, BigBird, Reformer, Routing Transformer | checkpoint-specific sparse pattern | taxonomy only; no compatible checkpoint comparison |
| Recurrent/persistent memory | Compressive Transformer, Infini-Attention, Memorizing Transformer, RETRO, Titans | recurrent state or external datastore | taxonomy only; no compatible checkpoint comparison |

The proxy rows test accounting and frontier construction. They are not substitutes
for upstream H2O/SnapKV/ClusterKV kernels or a production RAG stack. The paper marks
those external integrations as pending and makes no production-speed claim from
the standalone Python page table or gather paths.
"""


def run_systems_benchmarks(output_dir: Path, gate3_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed = benchmark_indexed_search()
    kernels = benchmark_kernels()
    pages = benchmark_paged_kv()
    cache = benchmark_cache_policies()
    batching = benchmark_batching_concurrency()
    serving = extract_serving_metrics(gate3_path)
    rag, long_context = benchmark_rag_and_long_context()
    kv = benchmark_kv_baselines()
    write_csv(output_dir / "indexed_search_benchmarks.csv", indexed)
    write_csv(output_dir / "kernel_benchmarks.csv", kernels)
    write_csv(output_dir / "paged_kv_benchmarks.csv", pages)
    write_csv(output_dir / "cache_policy_benchmarks.csv", cache)
    write_csv(output_dir / "batching_concurrency.csv", batching)
    write_csv(output_dir / "serving_metrics.csv", serving)
    write_csv(output_dir / "rag_baselines.csv", rag)
    write_csv(output_dir / "kv_cache_baselines.csv", kv)
    write_csv(output_dir / "long_context_baselines.csv", long_context)
    (output_dir / "baseline_taxonomy.md").write_text(baseline_taxonomy(), encoding="utf-8")
    return {
        "indexed_search_rows": len(indexed),
        "kernel_rows": len(kernels),
        "paged_rows": len(pages),
        "cache_rows": len(cache),
        "batching_rows": len(batching),
        "serving_rows": len(serving),
        "rag_rows": len(rag),
        "kv_baseline_rows": len(kv),
        "long_context_rows": len(long_context),
        "gemm_speedup_vs_loop_at_4096": _search_speedup(indexed, 4096),
        "best_coarse_recall_at_4096": max(
            row["recall_at_8_vs_exact"]
            for row in indexed
            if row["memory_vectors"] == 4096 and row["backend"] == "coarse_to_fine"
        ),
        "best_nonoracle_cache_policy": max(
            (
                row for row in cache
                if row["prefetch_policy"] != "oracle_next_upper_bound"
            ),
            key=lambda row: row["hit_rate"],
        ),
        "oracle_prefetch_upper_bound": max(
            (row for row in cache if row["prefetch_policy"] == "oracle_next_upper_bound"),
            key=lambda row: row["hit_rate"],
        ),
    }


def _search_speedup(rows: list[dict[str, Any]], count: int) -> float:
    brute = next(
        row for row in rows if row["memory_vectors"] == count and row["backend"] == "brute_force"
    )
    gemm = next(row for row in rows if row["memory_vectors"] == count and row["backend"] == "gemm")
    return brute["median_search_seconds"] / max(gemm["median_search_seconds"], 1e-12)
