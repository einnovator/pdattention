"""Benchmark model-bounded PRA encoding, materialization, and stream rollover."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pra_torch.model import convert_sa_model_to_pra  # noqa: E402
from run_native_kv_benchmark import (  # noqa: E402
    DATASET_DEFAULTS,
    SEEDS,
    _native_config,
    _prepare_synthetic,
    _set_seed,
    train_full_context_sa,
)
from run_pra_long_prompt_head import (  # noqa: E402
    CONDITIONS,
    _evaluate_case,
    _json_safe,
)


VERSION = "pra_model_bounded_context_v2"
MODEL_MAX = 32
DIRECT_TOKENS = 8
RESERVE_TOKENS = 4
ENCODING_TOKENS = 16
ROUTING_TOKENS = 4
DEFAULT_MEMORY_BUDGET = 20


def _mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else math.nan


def _aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        record = {name: value for name, value in zip(keys, key)}
        record["seeds"] = len({row["seed"] for row in members})
        record["examples"] = len(members)
        for metric in metrics:
            values = [row.get(metric, math.nan) for row in members]
            record[f"{metric}_mean"] = _mean(values)
            finite = [float(value) for value in values if math.isfinite(float(value))]
            record[f"{metric}_stddev"] = statistics.stdev(finite) if len(finite) > 1 else 0.0
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_artifact(name: str, manifest: dict, raw: list[dict], aggregate: list[dict]) -> None:
    result_dir = REPO / "docs" / "papers" / "shared" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "aggregate": aggregate, "raw": raw}
    (result_dir / f"{name}.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(result_dir / f"{name}.csv", aggregate)


def _base_manifest(device: str, args: argparse.Namespace) -> dict:
    return {
        "version": VERSION,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset": "synthetic_native_kv_fixed_target_v5",
        "seeds": args.seeds,
        "examples_per_seed": args.examples,
        "model_max_context_tokens": MODEL_MAX,
        "direct_context_tokens": DIRECT_TOKENS,
        "context_safety_reserve_tokens": RESERVE_TOKENS,
        "encoding_chunk_tokens": ENCODING_TOKENS,
        "encoding_overlap_tokens": 4,
        "routing_chunk_tokens": ROUTING_TOKENS,
        "default_materialization_budget_tokens": DEFAULT_MEMORY_BUDGET,
        "routing_top_k_chunks": 8,
    }


def _model(source, device: str, *, memory_budget: int = DEFAULT_MEMORY_BUDGET):
    cfg = _native_config(
        source,
        device,
        {
            "model_max_context_tokens": MODEL_MAX,
            "max_prompt_direct_tokens": DIRECT_TOKENS,
            "context_safety_reserve_tokens": RESERVE_TOKENS,
            "max_materialized_memory_tokens": memory_budget,
            "prompt_overflow_mode": "implicit_reference",
            "prompt_position_mode": "historical",
            "reference_position_mode": "global",
            "encoding_chunking": {
                "mode": "fixed",
                "chunk_tokens": ENCODING_TOKENS,
                "overlap_tokens": 4,
            },
            "routing_chunking": {"mode": "fixed", "chunk_tokens": ROUTING_TOKENS},
            "encoding_context_mode": "overlap",
            # Preserve these legacy aliases for older experiment helpers.
            "chunking_mode": "fixed",
            "fixed_chunk_tokens": ROUTING_TOKENS,
            "fixed_chunk_overlap_tokens": 0,
            "max_prompt_gists": None,
            "top_k_references": 1,
            "top_k_chunks_per_reference": 8,
            "routing_backend": "tensorized",
            "collect_detailed_timing": True,
        },
    )
    return convert_sa_model_to_pra(source, cfg).to(device).eval()


def _normalized_case_rows(rows: list[dict], *, seed: int, sample_id: str, position: str) -> list[dict]:
    defaults = {
        "encoding_calls": 0.0,
        "max_encoding_input_tokens": 0.0,
        "memory_budget_tokens": 0.0,
        "memory_tokens_materialized": 0.0,
        "chunks_budget_rejected": 0.0,
        "materialization_budget_utilization": 0.0,
        "memory_attention_ms": 0.0,
        "transfer_ms": 0.0,
        "transfer_bytes": 0.0,
    }
    return [
        {
            **defaults,
            **row,
            "seed": seed,
            "example_id": sample_id,
            "target_position": position,
            "model_max_context_tokens": MODEL_MAX,
            "logical_to_native_context_ratio": row["total_prompt_tokens"] / MODEL_MAX,
        }
        for row in rows
    ]


def _run_seed(seed: int, source, model, tokenizer, evaluation, args, device: str) -> dict:
    head_rows = []
    budget_rows = []
    for example_index in range(min(args.examples, len(evaluation))):
        sample = evaluation[example_index]
        for length in args.lengths:
            for position in args.positions:
                rows = _evaluate_case(
                    source,
                    model,
                    tokenizer,
                    sample,
                    total_tokens=length,
                    direct_budget=DIRECT_TOKENS,
                    position=position,
                    device=device,
                )
                head_rows.extend(
                    _normalized_case_rows(
                        rows, seed=seed, sample_id=str(sample.id), position=position
                    )
                )
        for budget in args.budgets:
            model.cfg.max_materialized_memory_tokens = budget
            for position in args.positions:
                rows = _evaluate_case(
                    source,
                    model,
                    tokenizer,
                    sample,
                    total_tokens=max(args.lengths),
                    direct_budget=DIRECT_TOKENS,
                    position=position,
                    device=device,
                    pra_conditions=("head_routed",),
                )
                routed = next(row for row in rows if row["condition"] == "head_routed")
                budget_rows.extend(
                    _normalized_case_rows(
                        [{**routed, "materialization_budget": budget}],
                        seed=seed,
                        sample_id=str(sample.id),
                        position=position,
                    )
                )
    model.cfg.max_materialized_memory_tokens = DEFAULT_MEMORY_BUDGET

    stream_rows = []
    initial = torch.tensor(
        [list(tokenizer.encode(evaluation[0].question))[-DIRECT_TOKENS:]],
        dtype=torch.long,
        device=device,
    )
    for generated_tokens in args.generated_lengths:
        model.clear_pra_cache()
        with torch.no_grad():
            output = model.generate(
                initial,
                max_new_tokens=generated_tokens,
                tokenizer=tokenizer,
                do_sample=False,
            )
        stream_rows.append(
            {
                "seed": seed,
                "generated_tokens": generated_tokens,
                "output_tokens": int(output.shape[1]),
                **model.last_generation_stats,
                "routing_ms_total": 1_000.0
                * model.last_generation_stats["routing_duration_seconds_total"],
                "native_limit_violations": int(
                    model.last_generation_stats["max_native_operation_tokens"] > MODEL_MAX
                ),
            }
        )
    return {"head": head_rows, "budget": budget_rows, "stream": stream_rows}


def _plot_head(path: Path, aggregate: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    colors = {
        "dense_full": "#287a5b",
        "direct_truncation": "#777777",
        "head_routed": "#2867a5",
        "head_oracle": "#c9563d",
        "head_shuffled": "#98518e",
        "head_independent": "#aa7a24",
    }
    for condition in CONDITIONS:
        rows = sorted(
            (row for row in aggregate if row["condition"] == condition),
            key=lambda row: row["total_prompt_tokens"],
        )
        x = [row["total_prompt_tokens"] for row in rows]
        color = colors[condition]
        axes[0, 0].plot(x, [row["loss_mean"] for row in rows], marker="o", color=color, label=condition)
        axes[0, 1].plot(x, [row["accuracy_mean"] for row in rows], marker="o", color=color)
        axes[1, 0].plot(x, [row["target_chunk_recall_mean"] for row in rows], marker="o", color=color)
        axes[1, 1].plot(x, [row["memory_tokens_materialized_mean"] for row in rows], marker="o", color=color)
    labels = ("Answer-token loss", "Answer-token accuracy", "Target-chunk recall", "Materialized K/V tokens")
    for axis, label in zip(axes.flat, labels):
        axis.set_xlabel("Logical prompt tokens")
        axis.set_ylabel(label)
        axis.axvline(MODEL_MAX, color="#222222", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("Implicit prompt head beyond the 32-token native context (five seeds)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_context(path: Path, aggregate: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
    x = [row["logical_context_tokens"] for row in aggregate]
    axes[0].plot(x, [row["encoding_calls_mean"] for row in aggregate], marker="o", color="#2867a5")
    axes[0].set_ylabel("Bounded encoding calls")
    axes[1].plot(x, [row["logical_to_native_context_ratio_mean"] for row in aggregate], marker="o", color="#287a5b", label="logical/native")
    axes[1].plot(x, [row["max_encoding_input_tokens_mean"] for row in aggregate], marker="s", color="#c9563d", label="max encoding input")
    axes[1].axhline(MODEL_MAX, color="#222222", linestyle=":", label="native limit")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Logical context tokens")
        axis.grid(alpha=0.25)
    figure.suptitle("Logical context scales while native operations remain bounded")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_budget(path: Path, aggregate: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.3, 7.0))
    x = [row["materialization_budget"] for row in aggregate]
    axes[0, 0].plot(x, [row["loss_mean"] for row in aggregate], marker="o", color="#2867a5")
    axes[0, 1].plot(x, [row["target_chunk_recall_mean"] for row in aggregate], marker="o", color="#287a5b")
    axes[1, 0].plot(x, [row["memory_tokens_materialized_mean"] for row in aggregate], marker="o", color="#c9563d")
    axes[1, 1].plot(x, [row["request_forward_ms_mean"] for row in aggregate], marker="o", color="#98518e")
    labels = ("Answer-token loss", "Target-chunk recall", "Materialized K/V tokens", "Forward latency (ms)")
    for axis, label in zip(axes.flat, labels):
        axis.set_xlabel("Materialization budget (tokens)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Whole-chunk materialization budget sweep at 192 logical tokens")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_stream(path: Path, aggregate: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
    x = [row["generated_tokens"] for row in aggregate]
    axes[0].plot(x, [row["head_tokens_mean"] for row in aggregate], marker="o", label="PRA head", color="#2867a5")
    axes[0].plot(x, [row["direct_tokens_mean"] for row in aggregate], marker="s", label="direct tail", color="#c9563d")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Tokens after generation")
    axes[1].plot(x, [row["rollover_events_mean"] for row in aggregate], marker="o", color="#287a5b", label="rollovers")
    axes[1].plot(x, [row["max_native_operation_tokens_mean"] for row in aggregate], marker="s", color="#98518e", label="max native operation")
    axes[1].axhline(MODEL_MAX, color="#222222", linestyle=":", label="native limit")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Generated tokens")
        axis.grid(alpha=0.25)
    figure.suptitle("Streaming rollover beyond the native context horizon (five seeds)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    settings = dict(DATASET_DEFAULTS["synthetic"])
    tokenizer, training_module, modules = _prepare_synthetic(settings)
    evaluation = modules[64].dataset
    run_dir = REPO / "out" / "pra_bounded_context"
    run_dir.mkdir(parents=True, exist_ok=True)
    collected = {"head": [], "budget": [], "stream": []}
    for seed in args.seeds:
        cache_path = run_dir / f"seed-{seed}.json"
        if cache_path.exists() and not args.force:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("version") == VERSION and cached.get("args") == vars(args):
                for key in collected:
                    collected[key].extend(cached[key])
                print(f"reuse {cache_path.relative_to(REPO)}", flush=True)
                continue
        _set_seed(seed)
        source, _ = train_full_context_sa(
            seed=seed,
            tokenizer=tokenizer,
            datamodule=training_module,
            settings=settings,
            run_dir=REPO / "out" / "native_kv_benchmarks" / "synthetic" / f"seed-{seed}",
            device=device,
            force=False,
        )
        model = _model(source, device)
        rows = _run_seed(seed, source, model, tokenizer, evaluation, args, device)
        cache_path.write_text(
            json.dumps({"version": VERSION, "args": vars(args), **_json_safe(rows)}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        for key in collected:
            collected[key].extend(rows[key])
        print(f"completed bounded-context seed {seed}", flush=True)
        del model, source
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    head_aggregate = _aggregate(
        collected["head"],
        ("condition", "total_prompt_tokens"),
        (
            "loss", "accuracy", "target_chunk_recall", "retrieved_kv_tokens",
            "memory_tokens_materialized", "encoding_calls", "max_encoding_input_tokens",
            "request_forward_ms", "memory_attention_ms", "transfer_bytes",
        ),
    )
    context_raw = [
        {
            "seed": row["seed"],
            "logical_context_tokens": row["implicit_head_tokens"],
            "logical_to_native_context_ratio": row["implicit_head_tokens"] / MODEL_MAX,
            "encoding_calls": row["encoding_calls"],
            "max_encoding_input_tokens": row["max_encoding_input_tokens"],
            "native_limit_violation": int(row["max_encoding_input_tokens"] > MODEL_MAX),
        }
        for row in collected["head"]
        if row["condition"] == "head_routed"
    ]
    context_aggregate = _aggregate(
        context_raw,
        ("logical_context_tokens",),
        ("logical_to_native_context_ratio", "encoding_calls", "max_encoding_input_tokens", "native_limit_violation"),
    )
    budget_aggregate = _aggregate(
        collected["budget"],
        ("materialization_budget",),
        (
            "loss", "accuracy", "target_chunk_recall", "memory_tokens_materialized",
            "chunks_budget_rejected", "materialization_budget_utilization",
            "request_forward_ms", "memory_attention_ms", "transfer_bytes", "peak_cuda_allocated",
        ),
    )
    stream_aggregate = _aggregate(
        collected["stream"],
        ("generated_tokens",),
        (
            "direct_tokens", "head_tokens", "rollover_events", "tokens_migrated",
            "max_direct_tokens_observed", "max_native_operation_tokens",
            "routing_steps", "routing_ms_total", "materialized_memory_tokens_total",
            "max_materialized_memory_tokens_observed", "native_limit_violations",
        ),
    )
    manifest = _base_manifest(device, args)
    _save_artifact("pra_head_beyond_native_context", manifest, collected["head"], head_aggregate)
    _save_artifact("pra_context_budget", manifest, context_raw, context_aggregate)
    _save_artifact("pra_materialization_budget", manifest, collected["budget"], budget_aggregate)
    _save_artifact("pra_streaming_rollover", manifest, collected["stream"], stream_aggregate)

    figure_dir = REPO / "docs" / "papers" / "shared" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_head(figure_dir / "pra_head_beyond_native_context.pdf", head_aggregate)
    _plot_context(figure_dir / "pra_context_budget.pdf", context_aggregate)
    _plot_budget(figure_dir / "pra_materialization_budget.pdf", budget_aggregate)
    _plot_stream(figure_dir / "pra_streaming_rollover.pdf", stream_aggregate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 64, 128, 192])
    parser.add_argument("--positions", nargs="+", choices=["early", "middle", "late", "boundary"], default=["early", "middle", "late", "boundary"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 12, 16, 20])
    parser.add_argument("--generated-lengths", nargs="+", type=int, default=[16, 32, 48])
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
