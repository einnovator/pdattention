"""Measure cold and persistent exact PRA routing indexes on real encoded caches."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pra_torch.cache_services import build_cache_from_metadata  # noqa: E402
from pra_torch.model import convert_sa_model_to_pra  # noqa: E402
from run_native_kv_benchmark import SEEDS, _native_config, _set_seed, train_full_context_sa  # noqa: E402
from run_pra_scale_sensitivity import (  # noqa: E402
    DATASET_SETTINGS,
    GENERATION_VERSION,
    MAX_SEQ_LEN,
    MODEL_TIERS,
    prepare_scale_data,
)


VERSION = "routing_index_reuse_v1"


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _elapsed_ms(device: str, function) -> tuple[float, object]:
    _sync(device)
    start = time.perf_counter()
    value = function()
    _sync(device)
    return 1_000.0 * (time.perf_counter() - start), value


def _trace(rows) -> list[list[tuple]]:
    return [
        [
            (
                hit.reference_uri,
                hit.chunk_id,
                hit.reference_rank,
                hit.rank_within_reference,
                hit.reference_score,
                hit.chunk_score,
            )
            for hit in row
        ]
        for row in rows
    ]


def _assert_trace_equal(expected, actual) -> None:
    assert len(expected) == len(actual)
    for expected_row, actual_row in zip(expected, actual):
        assert len(expected_row) == len(actual_row)
        for left, right in zip(expected_row, actual_row):
            assert left[:4] == right[:4]
            assert abs(left[4] - right[4]) <= 1e-6
            assert abs(left[5] - right[5]) <= 1e-6


def _time_queries(cache, queries, layer_ids, cfg, device, *, mode: str) -> tuple[float, int, int]:
    if mode == "warm":
        cache.invalidate_routing_indexes()
        for layer_id in layer_ids:
            cache.prepare_routing_index(layer_id, queries[layer_id][0])

    def run_all():
        for query_index in range(len(next(iter(queries.values())))):
            if mode == "cold":
                cache.invalidate_routing_indexes()
            for layer_id in layer_ids:
                cache.search(queries[layer_id][query_index], layer_id, cfg)

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    duration_ms, _ = _elapsed_ms(device, run_all)
    count = len(next(iter(queries.values())))
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if str(device).startswith("cuda") else 0
    )
    return duration_ms / count, peak_allocated, peak_reserved


def _profile_warm_gpu_phases(cache, queries, layer_ids, cfg, device) -> dict[str, float]:
    scoring_ms = reference_topk_ms = chunk_topk_ms = 0.0
    for query_index in range(len(next(iter(queries.values())))):
        for layer_id in layer_ids:
            query = queries[layer_id][query_index]
            index = cache._tensorized_chunk_index(layer_id, query, reuse=True)
            if index is None:
                continue

            elapsed, scored = _elapsed_ms(
                device,
                lambda: (
                    cache._reduce_gist_scores(query, index, cfg.gist_score_aggregation)
                ),
            )
            scoring_ms += elapsed
            chunk_scores, winning_indices, winning_scores = scored
            elapsed, reference_scores = _elapsed_ms(
                device,
                lambda: cache._reference_score_tensor(
                    chunk_scores, index, cfg.reference_score_aggregation
                ),
            )
            scoring_ms += elapsed
            reference_k = min(cfg.top_k_references, len(index.reference_entries))
            elapsed, top_references = _elapsed_ms(
                device,
                lambda: torch.topk(
                    reference_scores,
                    k=reference_k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ),
            )
            reference_topk_ms += elapsed
            del winning_indices, winning_scores
            reference_indices = top_references.indices
            grouped = chunk_scores[:, index.chunk_indices_by_reference]
            grouped = grouped.masked_fill(
                ~index.chunk_mask_by_reference.unsqueeze(0), float("-inf")
            )
            selected_grouped = torch.gather(
                grouped,
                1,
                reference_indices.unsqueeze(-1).expand(-1, -1, grouped.shape[-1]),
            )
            chunk_k = min(
                cfg.top_k_chunks_per_reference,
                int(index.chunk_indices_by_reference.shape[1]),
            )
            elapsed, _ = _elapsed_ms(
                device,
                lambda: torch.topk(
                    selected_grouped,
                    k=chunk_k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ),
            )
            chunk_topk_ms += elapsed
    count = len(next(iter(queries.values())))
    return {
        "query_scoring_ms": scoring_ms / count,
        "reference_topk_ms": reference_topk_ms / count,
        "chunk_topk_ms": chunk_topk_ms / count,
    }


def benchmark_cache(cache, model, *, seed: int, query_count: int, warmups: int, device: str) -> dict:
    cfg = model.cfg
    layer_ids = sorted(
        layer_id
        for layer_id in range(cfg.n_layers)
        if any(layer_id in entry.layer_memory for entry in cache.all_entries())
    )
    generator = torch.Generator(device=device).manual_seed(seed + 10_003)
    queries = {
        layer_id: [
            torch.randn((1, cfg.d_model), generator=generator, device=device)
            for _ in range(query_count)
        ]
        for layer_id in layer_ids
    }
    legacy_cfg = type(cfg)(**{**cfg.__dict__, "routing_backend": "legacy"})
    tensorized_cfg = type(cfg)(**{**cfg.__dict__, "routing_backend": "tensorized"})

    for _ in range(warmups):
        for layer_id in layer_ids:
            cache.search(queries[layer_id][0], layer_id, legacy_cfg)
            cache.invalidate_routing_indexes()
            cache.search(queries[layer_id][0], layer_id, tensorized_cfg)

    legacy_trace = []
    cold_trace = []
    warm_trace = []
    cache.invalidate_routing_indexes()
    for layer_id in layer_ids:
        legacy_trace.append(_trace(cache.search(queries[layer_id][0], layer_id, legacy_cfg)))
        cache.invalidate_routing_indexes()
        cold_trace.append(_trace(cache.search(queries[layer_id][0], layer_id, tensorized_cfg)))
        warm_trace.append(_trace(cache.search(queries[layer_id][0], layer_id, tensorized_cfg)))
    for expected, actual, repeated in zip(legacy_trace, cold_trace, warm_trace):
        _assert_trace_equal(expected, actual)
        _assert_trace_equal(actual, repeated)

    legacy_ms, legacy_peak, legacy_reserved = _time_queries(
        cache, queries, layer_ids, legacy_cfg, device, mode="legacy"
    )
    cold_ms, cold_peak, cold_reserved = _time_queries(
        cache, queries, layer_ids, tensorized_cfg, device, mode="cold"
    )
    warm_ms, warm_peak, warm_reserved = _time_queries(
        cache, queries, layer_ids, tensorized_cfg, device, mode="warm"
    )

    cache.invalidate_routing_indexes()
    index_build_ms = 0.0
    index_bytes = candidate_chunks = candidate_gists = 0
    for layer_id in layer_ids:
        elapsed, _ = _elapsed_ms(
            device,
            lambda layer_id=layer_id: cache._tensorized_chunk_index(
                layer_id, queries[layer_id][0], reuse=True
            ),
        )
        index_build_ms += elapsed
        stats = cache.prepare_routing_index(layer_id, queries[layer_id][0])
        index_bytes += int(stats["index_bytes"])
        candidate_chunks += int(stats["candidate_chunks"])
        candidate_gists += int(stats["candidate_gists"])
    phases = _profile_warm_gpu_phases(
        cache, queries, layer_ids, tensorized_cfg, device
    )
    measured_gpu_ms = sum(phases.values())
    return {
        "layer_count": len(layer_ids),
        "measured_queries": query_count,
        "candidate_chunks": candidate_chunks,
        "candidate_gists": candidate_gists,
        "index_bytes": index_bytes,
        "index_build_ms": index_build_ms,
        "legacy_scalar_ms": legacy_ms,
        "tensorized_cold_index_ms": cold_ms,
        "tensorized_warm_index_ms": warm_ms,
        **phases,
        "selected_hit_serialization_and_dispatch_ms": max(warm_ms - measured_gpu_ms, 0.0),
        "cold_speedup": legacy_ms / max(cold_ms, 1e-12),
        "warm_speedup": legacy_ms / max(warm_ms, 1e-12),
        "index_fraction_of_cold": index_build_ms / max(cold_ms, 1e-12),
        "legacy_peak_allocated": legacy_peak,
        "legacy_peak_reserved": legacy_reserved,
        "cold_peak_allocated": cold_peak,
        "cold_peak_reserved": cold_reserved,
        "warm_peak_allocated": warm_peak,
        "warm_peak_reserved": warm_reserved,
        "selection_parity": True,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    keys = (
        "candidate_chunks",
        "candidate_gists",
        "index_bytes",
        "index_build_ms",
        "legacy_scalar_ms",
        "tensorized_cold_index_ms",
        "tensorized_warm_index_ms",
        "query_scoring_ms",
        "reference_topk_ms",
        "chunk_topk_ms",
        "selected_hit_serialization_and_dispatch_ms",
        "cold_speedup",
        "warm_speedup",
        "index_fraction_of_cold",
        "legacy_peak_allocated",
        "legacy_peak_reserved",
        "cold_peak_allocated",
        "cold_peak_reserved",
        "warm_peak_allocated",
        "warm_peak_reserved",
    )
    groups = {}
    for row in rows:
        group = (row["dataset"], row["model_tier"], row["split_count"])
        groups.setdefault(group, []).append(row)
    output = []
    for (dataset, tier, splits), members in sorted(groups.items()):
        item = {"dataset": dataset, "model_tier": tier, "split_count": splits, "seeds": len(members)}
        for key in keys:
            values = [float(row[key]) for row in members]
            item[f"{key}_mean"] = statistics.fmean(values)
            item[f"{key}_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, aggregate: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in aggregate})
    figure, axes = plt.subplots(2, len(datasets), figsize=(5.4 * len(datasets), 7.2), squeeze=False)
    colors = {"legacy_scalar_ms_mean": "#3366a8", "tensorized_cold_index_ms_mean": "#d05a3a", "tensorized_warm_index_ms_mean": "#2f8567"}
    labels = {"legacy_scalar_ms_mean": "legacy scalar", "tensorized_cold_index_ms_mean": "tensorized cold", "tensorized_warm_index_ms_mean": "tensorized warm"}
    for column, dataset in enumerate(datasets):
        rows = [row for row in aggregate if row["dataset"] == dataset]
        for tier in ("tiny", "small"):
            tier_rows = sorted((row for row in rows if row["model_tier"] == tier), key=lambda row: row["split_count"])
            for metric in colors:
                axes[0][column].plot(
                    [row["split_count"] for row in tier_rows],
                    [row[metric] for row in tier_rows],
                    marker="o",
                    linestyle="-" if tier == "tiny" else "--",
                    color=colors[metric],
                    label=f"{tier} {labels[metric]}",
                )
            axes[1][column].plot(
                [row["split_count"] for row in tier_rows],
                [row["warm_speedup_mean"] for row in tier_rows],
                marker="o",
                label=tier,
            )
        axes[0][column].set_title(dataset)
        axes[0][column].set_yscale("log")
        axes[0][column].set_ylabel("Routing latency (ms/query)")
        axes[0][column].grid(alpha=0.25)
        axes[0][column].legend(fontsize=7)
        axes[1][column].set_xlabel("Nominal split count")
        axes[1][column].set_ylabel("Legacy / warm exact speedup")
        axes[1][column].grid(alpha=0.25)
        axes[1][column].legend(fontsize=8)
    figure.suptitle("Exact PRA routing with reusable packed indexes (five seeds, CUDA)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    published = REPO / "docs" / "papers" / "shared"
    result_root = REPO / "out" / "pra_routing_index_reuse"
    (published / "results").mkdir(parents=True, exist_ok=True)
    (published / "figures").mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset in args.datasets:
        tokenizer, training_module, modules = prepare_scale_data(dataset)
        for tier in args.model_tiers:
            settings = {
                **DATASET_SETTINGS[dataset],
                **MODEL_TIERS[tier],
                "max_seq_len": MAX_SEQ_LEN,
                "scale_source_unit_count": 255,
                "generation_version": GENERATION_VERSION,
                "model_tier": tier,
            }
            training_module.batch_size = int(settings["batch_size"])
            for seed in args.seeds:
                output = result_root / dataset / tier / f"seed-{seed}.json"
                if output.exists() and not args.force:
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    if payload.get("version") == VERSION and payload.get("query_count") == args.queries:
                        all_rows.extend(payload["results"])
                        print(f"reuse {output.relative_to(REPO)}", flush=True)
                        continue
                _set_seed(seed)
                source, _ = train_full_context_sa(
                    seed=seed,
                    tokenizer=tokenizer,
                    datamodule=training_module,
                    settings=settings,
                    run_dir=REPO / "out" / "pra_scale_sensitivity" / dataset / tier / f"seed-{seed}",
                    device=device,
                    force=False,
                )
                cfg = _native_config(
                    source,
                    device,
                    {
                        "reference_encoding_strategy": "native_slice",
                        "reference_position_mode": "global",
                        "prompt_position_mode": "historical",
                        "top_k_references": 8,
                        "top_k_chunks_per_reference": 1,
                        "collect_detailed_timing": False,
                        "collect_routing_metrics": False,
                        "collect_rank_diagnostics": False,
                        "recursive_max_total_references": 512,
                        "recursive_max_total_tokens": 65_536,
                    },
                )
                model = convert_sa_model_to_pra(source, cfg).to(device).eval()
                seed_rows = []
                for split_count in args.splits:
                    batch = next(iter(modules[split_count].test_loader()))
                    cache = build_cache_from_metadata(
                        model, tokenizer, [batch["metadata"][0]], device
                    )
                    row = benchmark_cache(
                        cache,
                        model,
                        seed=seed + split_count,
                        query_count=args.queries,
                        warmups=args.warmups,
                        device=device,
                    )
                    seed_rows.append(
                        {
                            "dataset": dataset,
                            "model_tier": tier,
                            "seed": seed,
                            "split_count": split_count,
                            **row,
                        }
                    )
                    print(f"done {dataset}/{tier}/seed-{seed}/split-{split_count}", flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {"version": VERSION, "query_count": args.queries, "results": seed_rows},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                all_rows.extend(seed_rows)
                del model, source
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

    aggregate = _aggregate(all_rows)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    payload = {
        "manifest": {
            "version": VERSION,
            "git_sha": git_sha,
            "device": device,
            "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "datasets": args.datasets,
            "model_tiers": args.model_tiers,
            "seeds": args.seeds,
            "splits": args.splits,
            "warmup_count": args.warmups,
            "measured_query_count": args.queries,
            "top_k_references": 8,
            "timing_scope": "sum across all PRA layers; real encoded caches; routing only",
        },
        "aggregate": aggregate,
        "raw": all_rows,
    }
    json_path = published / "results" / "pra_routing_index_reuse.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(published / "results" / "pra_routing_index_reuse.csv", aggregate)
    _plot(published / "figures" / "pra_routing_index_reuse.pdf", aggregate)
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SETTINGS), default=sorted(DATASET_SETTINGS))
    parser.add_argument("--model-tiers", nargs="+", choices=sorted(MODEL_TIERS), default=["tiny", "small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
