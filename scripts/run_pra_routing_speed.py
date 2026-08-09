"""Benchmark legacy scalar and exact tensorized PRA routing on trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pra_torch.model import convert_sa_model_to_pra  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402
from run_native_kv_benchmark import SEEDS, _native_config, _set_seed, train_full_context_sa  # noqa: E402
from run_pra_scale_sensitivity import (  # noqa: E402
    DATASET_SETTINGS,
    GENERATION_VERSION,
    MAX_SEQ_LEN,
    MODEL_TIERS,
    prepare_scale_data,
)


VERSION = "exact_tensorized_v2"
BACKENDS = ("legacy", "tensorized")


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _condition_row(
    result: dict,
    backend: str,
    split_count: int,
    expected_gist_comparisons: int,
) -> dict:
    examples = result["per_example"]
    return {
        "routing_backend": backend,
        "split_count": split_count,
        "source_unit_count": split_count - 1,
        "loss": float(result["loss"]),
        "token_accuracy": float(result["token_accuracy"]),
        "duration_seconds": float(result["duration_seconds"]),
        "routing_latency_seconds": _mean(examples, "routing_latency"),
        "materialization_latency_seconds": _mean(examples, "kv_materialization_latency"),
        "attention_latency_seconds": _mean(examples, "attention_latency"),
        "gist_comparisons": expected_gist_comparisons,
        "retrieved_physical_kv_tokens": _mean(examples, "retrieved_physical_kv_tokens"),
    }


def benchmark_seed(
    *,
    source,
    tokenizer,
    modules: dict,
    splits: list[int],
    eval_examples: int,
    device: str,
    backend_order: tuple[str, ...] = BACKENDS,
) -> list[dict]:
    """Measure both routers with identical model weights, examples, and encoded K/V."""
    cfg = _native_config(
        source,
        device,
        {
            "reference_encoding_strategy": "native_slice",
            "reference_position_mode": "global",
            "prompt_position_mode": "historical",
            "top_k_references": 8,
            "top_k_chunks_per_reference": 1,
            "collect_rank_diagnostics": False,
            "recursive_max_total_references": 512,
            "recursive_max_total_tokens": 65_536,
        },
    )
    model = convert_sa_model_to_pra(source, cfg).to(device).eval()
    rows = []
    for split_count in splits:
        datamodule = modules[split_count]
        datamodule.test_dataset = Subset(
            datamodule.dataset,
            range(min(eval_examples, len(datamodule.dataset))),
        )
        encoded_entry_cache: dict = {}
        evaluation_dataset = datamodule.test_dataset
        # Warm both implementations on the same row so one-time CUDA kernel setup
        # is excluded from the measured 32-example comparison.
        datamodule.test_dataset = Subset(datamodule.dataset, range(1))
        for backend in BACKENDS:
            model.cfg.routing_backend = backend
            evaluate_reference_ablation(
                model=model,
                loader=datamodule.test_loader(),
                tokenizer=tokenizer,
                device=device,
                condition="valid",
                collect_per_example=False,
                encoded_entry_cache=encoded_entry_cache,
            )
        datamodule.test_dataset = evaluation_dataset
        condition_rows = []
        for backend in backend_order:
            model.cfg.routing_backend = backend
            if device == "cuda":
                torch.cuda.synchronize()
            result = evaluate_reference_ablation(
                model=model,
                loader=datamodule.test_loader(),
                tokenizer=tokenizer,
                device=device,
                condition="valid",
                collect_per_example=True,
                encoded_entry_cache=encoded_entry_cache,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            condition_rows.append(
                _condition_row(
                    result,
                    backend,
                    split_count,
                    expected_gist_comparisons=(split_count - 1) * source.cfg.n_layers,
                )
            )
        by_backend = {row["routing_backend"]: row for row in condition_rows}
        legacy = by_backend["legacy"]
        tensorized = by_backend["tensorized"]
        if abs(legacy["loss"] - tensorized["loss"]) > 1e-6:
            raise AssertionError(
                f"Routing backend changed loss at split {split_count}: "
                f"{legacy['loss']} != {tensorized['loss']}"
            )
        for row in condition_rows:
            row["routing_speedup"] = (
                legacy["routing_latency_seconds"]
                / max(tensorized["routing_latency_seconds"], 1e-12)
            )
            row["loss_delta_vs_legacy"] = row["loss"] - legacy["loss"]
            rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["dataset"], row["model_tier"], row["split_count"], row["routing_backend"])
        groups.setdefault(key, []).append(row)
    aggregate = []
    metric_keys = (
        "loss",
        "token_accuracy",
        "duration_seconds",
        "routing_latency_seconds",
        "materialization_latency_seconds",
        "attention_latency_seconds",
        "gist_comparisons",
        "retrieved_physical_kv_tokens",
        "routing_speedup",
        "loss_delta_vs_legacy",
    )
    for (dataset, tier, split_count, backend), members in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "model_tier": tier,
            "split_count": split_count,
            "routing_backend": backend,
            "seeds": len(members),
        }
        for metric in metric_keys:
            values = [float(row[metric]) for row in members]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(item)
    return aggregate


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, aggregate: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in aggregate})
    tiers = [tier for tier in ("tiny", "small") if any(row["model_tier"] == tier for row in aggregate)]
    figure, axes = plt.subplots(2, len(datasets), figsize=(5.4 * len(datasets), 7.2), squeeze=False)
    for column, dataset in enumerate(datasets):
        for tier in tiers:
            for backend, style in (("legacy", "--"), ("tensorized", "-")):
                rows = sorted(
                    (
                        row
                        for row in aggregate
                        if row["dataset"] == dataset
                        and row["model_tier"] == tier
                        and row["routing_backend"] == backend
                    ),
                    key=lambda row: row["split_count"],
                )
                axes[0][column].plot(
                    [row["split_count"] for row in rows],
                    [1_000 * row["routing_latency_seconds_mean"] for row in rows],
                    marker="o",
                    linestyle=style,
                    label=f"{tier} {backend}",
                )
            speed_rows = sorted(
                (
                    row
                    for row in aggregate
                    if row["dataset"] == dataset
                    and row["model_tier"] == tier
                    and row["routing_backend"] == "tensorized"
                ),
                key=lambda row: row["split_count"],
            )
            axes[1][column].plot(
                [row["split_count"] for row in speed_rows],
                [row["routing_speedup_mean"] for row in speed_rows],
                marker="o",
                label=tier,
            )
        axes[0][column].set_title(dataset)
        axes[0][column].set_ylabel("Routing latency (ms/example)")
        axes[0][column].set_yscale("log")
        axes[0][column].grid(alpha=0.25)
        axes[0][column].legend(fontsize=8)
        axes[1][column].set_xlabel("Nominal split count")
        axes[1][column].set_ylabel("Legacy / tensorized speedup")
        axes[1][column].grid(alpha=0.25)
        axes[1][column].legend(fontsize=8)
    figure.suptitle("Exact tensorized PRA routing speed (five seeds, CUDA)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result_root = REPO / "out" / "pra_routing_speed"
    published = REPO / "docs" / "papers" / "shared"
    result_root.mkdir(parents=True, exist_ok=True)
    (published / "results").mkdir(parents=True, exist_ok=True)
    (published / "figures").mkdir(parents=True, exist_ok=True)
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
                    if (
                        payload.get("version") == VERSION
                        and payload.get("eval_examples") == args.eval_examples
                        and payload.get("splits") == args.splits
                    ):
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
                print(f"benchmark {dataset}/{tier}/seed-{seed}", flush=True)
                rows = benchmark_seed(
                    source=source,
                    tokenizer=tokenizer,
                    modules=modules,
                    splits=args.splits,
                    eval_examples=args.eval_examples,
                    device=device,
                    backend_order=(
                        BACKENDS
                        if args.seeds.index(seed) % 2 == 0
                        else tuple(reversed(BACKENDS))
                    ),
                )
                rows = [
                    {"dataset": dataset, "model_tier": tier, "seed": seed, **row}
                    for row in rows
                ]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "version": VERSION,
                            "device": device,
                            "eval_examples": args.eval_examples,
                            "splits": args.splits,
                            "results": rows,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                all_rows.extend(rows)
                del source
                if device == "cuda":
                    torch.cuda.empty_cache()

    aggregate = _aggregate(all_rows)
    payload = {
        "manifest": {
            "version": VERSION,
            "device": device,
            "datasets": args.datasets,
            "model_tiers": args.model_tiers,
            "seeds": args.seeds,
            "splits": args.splits,
            "eval_examples_per_seed": args.eval_examples,
            "top_k_references": 8,
            "routing_semantics": "exact cosine scoring and exact torch.topk selection",
            "timing_scope": "routing search including per-example packed-index construction",
        },
        "aggregate": aggregate,
        "raw": all_rows,
    }
    json_path = published / "results" / "pra_routing_speed.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(published / "results" / "pra_routing_speed.csv", aggregate)
    _plot(published / "figures" / "pra_routing_speed.pdf", aggregate)
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SETTINGS), default=sorted(DATASET_SETTINGS))
    parser.add_argument("--model-tiers", nargs="+", choices=sorted(MODEL_TIERS), default=["tiny", "small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--splits", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
