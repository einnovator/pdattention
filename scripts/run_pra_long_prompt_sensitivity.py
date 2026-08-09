"""Small top-k, chunk-size, and overlap sensitivity sweep for implicit #__head."""

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
from run_pra_long_prompt_head import _evaluate_case  # noqa: E402


VERSION = "long_prompt_head_sensitivity_v1"


def _settings() -> list[tuple[int, int, int, str]]:
    values = {(top_k, 8, 0, "top_k") for top_k in (2, 4, 8, 16)}
    values.update((8, chunk_size, 0, "chunk_size") for chunk_size in (4, 8, 16))
    values.update((8, 8, overlap, "overlap") for overlap in (0, 2))
    return sorted(values)


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        if row["condition"] != "head_routed":
            continue
        key = (row["sweep"], row["top_k"], row["chunk_size"], row["overlap"])
        groups.setdefault(key, []).append(row)
    output = []
    for (sweep, top_k, chunk_size, overlap), members in sorted(groups.items()):
        output.append(
            {
                "sweep": sweep,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "seeds": len({row["seed"] for row in members}),
                "examples": len(members),
                **{
                    f"{metric}_mean": statistics.fmean(float(row[metric]) for row in members)
                    for metric in (
                        "loss",
                        "accuracy",
                        "target_chunk_recall",
                        "active_kv_fraction",
                        "retrieved_kv_tokens",
                        "routing_ms",
                    )
                },
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, aggregate: list[dict]) -> None:
    topk = sorted((row for row in aggregate if row["sweep"] == "top_k"), key=lambda row: row["top_k"])
    chunks = sorted((row for row in aggregate if row["sweep"] == "chunk_size"), key=lambda row: row["chunk_size"])
    overlap = sorted((row for row in aggregate if row["sweep"] == "overlap"), key=lambda row: row["overlap"])
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for axis, rows, key, label in (
        (axes[0], topk, "top_k", "Top-k chunks"),
        (axes[1], chunks, "chunk_size", "Chunk size"),
        (axes[2], overlap, "overlap", "Overlap tokens"),
    ):
        x = [row[key] for row in rows]
        axis.plot(x, [row["target_chunk_recall_mean"] for row in rows], marker="o", label="target recall")
        axis.plot(x, [row["accuracy_mean"] for row in rows], marker="s", label="accuracy")
        axis.plot(x, [row["active_kv_fraction_mean"] for row in rows], marker="^", label="active K/V")
        axis.set_xlabel(label)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Implicit #__head sensitivity at 192 tokens (five seeds)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_settings = dict(DATASET_DEFAULTS["synthetic"])
    tokenizer, training_module, modules = _prepare_synthetic(base_settings)
    evaluation = modules[64].dataset
    published = REPO / "docs" / "papers" / "shared"
    result_root = REPO / "out" / "pra_long_prompt_sensitivity"
    result_root.mkdir(parents=True, exist_ok=True)
    (published / "results").mkdir(parents=True, exist_ok=True)
    (published / "figures").mkdir(parents=True, exist_ok=True)
    sweep_settings = _settings()
    all_rows = []
    for seed in args.seeds:
        output = result_root / f"seed-{seed}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload.get("version") == VERSION and payload.get("examples") == args.examples:
                all_rows.extend(payload["results"])
                print(f"reuse {output.relative_to(REPO)}", flush=True)
                continue
        _set_seed(seed)
        source, _ = train_full_context_sa(
            seed=seed,
            tokenizer=tokenizer,
            datamodule=training_module,
            settings=base_settings,
            run_dir=REPO / "out" / "native_kv_benchmarks" / "synthetic" / f"seed-{seed}",
            device=device,
            force=False,
        )
        seed_rows = []
        for top_k, chunk_size, overlap, sweep in sweep_settings:
            cfg = _native_config(
                source,
                device,
                {
                    "max_prompt_direct_tokens": 24,
                    "prompt_overflow_mode": "implicit_reference",
                    "prompt_position_mode": "historical",
                    "chunking_mode": "fixed",
                    "fixed_chunk_tokens": chunk_size,
                    "fixed_chunk_overlap_tokens": overlap,
                    "top_k_references": 1,
                    "top_k_chunks_per_reference": top_k,
                    "collect_detailed_timing": True,
                    "routing_backend": "tensorized",
                },
            )
            model = convert_sa_model_to_pra(source, cfg).to(device).eval()
            for example_index in range(min(args.examples, len(evaluation))):
                sample = evaluation[example_index]
                for position in ("early", "middle", "late", "boundary"):
                    rows = _evaluate_case(
                        source,
                        model,
                        tokenizer,
                        sample,
                        total_tokens=192,
                        direct_budget=24,
                        position=position,
                        device=device,
                        pra_conditions=("head_routed",),
                    )
                    seed_rows.extend(
                        {
                            "seed": seed,
                            "example_id": str(sample.id),
                            "sweep": sweep,
                            "top_k": top_k,
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            **row,
                        }
                        for row in rows
                    )
            del model
        output.write_text(
            json.dumps({"version": VERSION, "examples": args.examples, "results": seed_rows}, indent=2),
            encoding="utf-8",
        )
        all_rows.extend(seed_rows)
        del source
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"done seed-{seed}", flush=True)

    aggregate = _aggregate(all_rows)
    payload = {
        "manifest": {
            "version": VERSION,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "device": device,
            "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "seeds": args.seeds,
            "examples_per_seed": args.examples,
            "prompt_tokens": 192,
            "direct_budget": 24,
            "positions": ["early", "middle", "late", "boundary"],
            "design": "one-factor-at-a-time top-k, chunk-size, and overlap sweep",
        },
        "aggregate": aggregate,
        "raw": all_rows,
    }
    path = published / "results" / "pra_long_prompt_head_sensitivity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(published / "results" / "pra_long_prompt_head_sensitivity.csv", aggregate)
    _plot(published / "figures" / "pra_long_prompt_head_sensitivity.pdf", aggregate)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
