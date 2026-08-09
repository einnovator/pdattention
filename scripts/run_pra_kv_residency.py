"""Compare GPU- and CPU-resident native PRA K/V with selective transfer."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pra_torch.model import convert_sa_model_to_pra  # noqa: E402
from pra_torch.cache_services import build_cache_from_metadata  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402
from run_native_kv_benchmark import SEEDS, _native_config, _set_seed, train_full_context_sa  # noqa: E402
from run_pra_scale_sensitivity import (  # noqa: E402
    DATASET_SETTINGS,
    GENERATION_VERSION,
    MAX_SEQ_LEN,
    MODEL_TIERS,
    prepare_scale_data,
)


VERSION = "kv_residency_v2"


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row.get(key, 0.0)) for row in rows)


def _condition(result: dict, residency: str, warm: dict) -> dict:
    examples = result["per_example"]
    return {
        "kv_cache_residency": residency,
        "loss": float(result["loss"]),
        "token_accuracy": float(result["token_accuracy"]),
        "cached_kv_bytes": _mean(examples, "cached_kv_bytes"),
        "gpu_cached_kv_bytes": _mean(examples, "gpu_cached_kv_bytes"),
        "selected_kv_transfer_bytes": _mean(examples, "kv_transfer_bytes"),
        "selected_kv_transfer_ms": 1_000.0 * _mean(examples, "kv_transfer_latency"),
        "routing_ms": 1_000.0 * _mean(examples, "routing_latency"),
        "materialization_ms": 1_000.0 * _mean(examples, "kv_materialization_latency"),
        "cold_request_ms": 1_000.0 * _mean(examples, "example_latency"),
        **warm,
        "peak_allocated_bytes": _mean(examples, "peak_cuda_memory"),
        "peak_reserved_bytes": _mean(examples, "peak_cuda_memory_reserved"),
        "retrieved_physical_kv_tokens": _mean(examples, "retrieved_physical_kv_tokens"),
        "selected_chunk_ids": [row["routed_selected_chunk_ids"] for row in examples],
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "loss",
        "token_accuracy",
        "cached_kv_bytes",
        "gpu_cached_kv_bytes",
        "selected_kv_transfer_bytes",
        "selected_kv_transfer_ms",
        "routing_ms",
        "materialization_ms",
        "cold_request_ms",
        "warm_request_ms",
        "warm_peak_allocated_bytes",
        "warm_peak_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "retrieved_physical_kv_tokens",
        "loss_delta_vs_gpu",
    )
    groups = {}
    for row in rows:
        key = (
            row["dataset"],
            row["model_tier"],
            row["split_count"],
            row["kv_cache_residency"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (dataset, tier, splits, residency), members in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "model_tier": tier,
            "split_count": splits,
            "kv_cache_residency": residency,
            "seeds": len(members),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in members]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _measure_warm_request(model, datamodule, tokenizer, device: str, repeats: int = 8) -> dict:
    """Time repeated forwards after source K/V and packed indexes already exist."""
    batch = next(iter(datamodule.test_loader()))
    metadata = batch["metadata"][0]
    cache = build_cache_from_metadata(model, tokenizer, [metadata], device)
    model.set_pra_cache(cache)
    input_ids = batch["input_ids"][0:1].to(device)
    position_offset = sum(
        len(tokenizer.encode(str(reference.metadata.get("text", ""))))
        for reference in metadata.get("references") or []
    )
    with torch.no_grad():
        model(input_ids, position_offset=position_offset)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            model(input_ids, position_offset=position_offset)
        end.record()
        end.synchronize()
    return {
        "warm_request_ms": float(start.elapsed_time(end)) / repeats,
        "warm_peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
        "warm_peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, aggregate: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in aggregate})
    figure, axes = plt.subplots(2, len(datasets), figsize=(5.4 * len(datasets), 7.2), squeeze=False)
    for column, dataset in enumerate(datasets):
        for tier, line in (("tiny", "-"), ("small", "--")):
            for residency, color in (("gpu", "#d05a3a"), ("cpu", "#2f8567")):
                rows = sorted(
                    (
                        row for row in aggregate
                        if row["dataset"] == dataset
                        and row["model_tier"] == tier
                        and row["kv_cache_residency"] == residency
                    ),
                    key=lambda row: row["split_count"],
                )
                axes[0][column].plot(
                    [row["split_count"] for row in rows],
                    [row["gpu_cached_kv_bytes_mean"] / 2**20 for row in rows],
                    marker="o",
                    linestyle=line,
                    color=color,
                    label=f"{tier} {residency}",
                )
                axes[1][column].plot(
                    [row["split_count"] for row in rows],
                    [row["peak_allocated_bytes_mean"] / 2**20 for row in rows],
                    marker="o",
                    linestyle=line,
                    color=color,
                    label=f"{tier} {residency}",
                )
        axes[0][column].set_title(dataset)
        axes[0][column].set_ylabel("GPU-resident source K/V (MiB)")
        axes[1][column].set_ylabel("Peak CUDA allocated (MiB)")
        axes[1][column].set_xlabel("Nominal split count")
        for axis in (axes[0][column], axes[1][column]):
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    figure.suptitle("Native K/V residency with top-k=8 selective transfer (five seeds)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(device).startswith("cuda"):
        raise ValueError("The residency benchmark requires CUDA.")
    published = REPO / "docs" / "papers" / "shared"
    result_root = REPO / "out" / "pra_kv_residency"
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
                    if payload.get("version") == VERSION and payload.get("eval_examples") == args.eval_examples:
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
                seed_rows = []
                for split_count in args.splits:
                    datamodule = modules[split_count]
                    original_test = datamodule.test_dataset
                    datamodule.test_dataset = Subset(
                        datamodule.dataset,
                        range(min(args.eval_examples, len(datamodule.dataset))),
                    )
                    conditions = {}
                    residency_order = (
                        ("gpu", "cpu")
                        if args.seeds.index(seed) % 2 == 0
                        else ("cpu", "gpu")
                    )
                    measured_test = datamodule.test_dataset
                    for residency in residency_order:
                        cfg = _native_config(
                            source,
                            device,
                            {
                                "reference_encoding_strategy": "native_slice",
                                "reference_position_mode": "global",
                                "prompt_position_mode": "historical",
                                "top_k_references": 8,
                                "top_k_chunks_per_reference": 1,
                                "collect_detailed_timing": True,
                                "collect_routing_metrics": False,
                                "collect_rank_diagnostics": False,
                                "kv_cache_residency": residency,
                                "kv_cache_pin_memory": residency == "cpu",
                                "kv_cache_non_blocking": False,
                                "recursive_max_total_references": 512,
                                "recursive_max_total_tokens": 65_536,
                            },
                        )
                        model = convert_sa_model_to_pra(source, cfg).to(device).eval()
                        datamodule.test_dataset = Subset(datamodule.dataset, range(1))
                        evaluate_reference_ablation(
                            model=model,
                            loader=datamodule.test_loader(),
                            tokenizer=tokenizer,
                            device=device,
                            condition="valid",
                            collect_per_example=False,
                        )
                        datamodule.test_dataset = measured_test
                        result = evaluate_reference_ablation(
                            model=model,
                            loader=datamodule.test_loader(),
                            tokenizer=tokenizer,
                            device=device,
                            condition="valid",
                            collect_per_example=True,
                        )
                        warm = _measure_warm_request(
                            model, datamodule, tokenizer, device
                        )
                        conditions[residency] = _condition(result, residency, warm)
                        del model
                        torch.cuda.empty_cache()
                    datamodule.test_dataset = original_test
                    gpu = conditions["gpu"]
                    cpu = conditions["cpu"]
                    if abs(gpu["loss"] - cpu["loss"]) > 1e-6:
                        raise AssertionError("CPU residency changed loss.")
                    if gpu["selected_chunk_ids"] != cpu["selected_chunk_ids"]:
                        raise AssertionError("CPU residency changed selected chunks.")
                    for condition in conditions.values():
                        condition["loss_delta_vs_gpu"] = condition["loss"] - gpu["loss"]
                        condition.pop("selected_chunk_ids")
                        seed_rows.append(
                            {
                                "dataset": dataset,
                                "model_tier": tier,
                                "seed": seed,
                                "split_count": split_count,
                                **condition,
                            }
                        )
                    print(f"done {dataset}/{tier}/seed-{seed}/split-{split_count}", flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {"version": VERSION, "eval_examples": args.eval_examples, "results": seed_rows},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                all_rows.extend(seed_rows)
                del source
                torch.cuda.empty_cache()

    aggregate = _aggregate(all_rows)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    payload = {
        "manifest": {
            "version": VERSION,
            "git_sha": git_sha,
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "datasets": args.datasets,
            "model_tiers": args.model_tiers,
            "seeds": args.seeds,
            "splits": args.splits,
            "eval_examples_per_seed": args.eval_examples,
            "top_k_references": 8,
            "cpu_mode": "pinned CPU source K/V; selected blocking host-to-device transfer",
            "timing_scope": "cold per-example cache build plus model forward",
            "warm_timing_scope": "eight repeated forwards after source encoding and packed-index construction",
        },
        "aggregate": aggregate,
        "raw": all_rows,
    }
    path = published / "results" / "pra_kv_residency.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(published / "results" / "pra_kv_residency.csv", aggregate)
    _plot(published / "figures" / "pra_kv_residency.pdf", aggregate)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SETTINGS), default=sorted(DATASET_SETTINGS))
    parser.add_argument("--model-tiers", nargs="+", choices=sorted(MODEL_TIERS), default=["tiny", "small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--eval-examples", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
