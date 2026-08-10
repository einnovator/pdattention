"""Run the first-priority matched RoPE/native-KV experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_native_kv_benchmark as native  # noqa: E402

from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    SPLIT_COUNTS,
    TIERS,
    environment_metadata,
    refresh_manifest,
    set_seed,
    write_csv,
    write_json,
)
from experiments.paper1_5_rope.eval_translation import (  # noqa: E402
    run_pre_post,
    run_translation,
)
from experiments.paper1_5_rope.reporting import (  # noqa: E402
    aggregate_fragmentation,
    plot_fragmentation,
    plot_translation,
)


def _settings(tier: str, position_mode: str, smoke: bool) -> dict:
    values = dict(native.DATASET_DEFAULTS["synthetic"])
    values.update(TIERS[tier])
    values.update(
        {
            "position_encoding": position_mode,
            "rope_theta": 10_000.0,
            "max_examples": 2 if smoke else 12,
            "train_examples": 64 if smoke else 2_048,
            "batch_size": 2 if smoke else (32 if tier == "tiny" else 16),
            "validation_examples": 2 if smoke else 32,
            "steps": 2 if smoke else TIERS[tier]["steps"],
            "pra_overrides": {
                # Individual benchmark references each begin at local position zero.
                "reference_position_mode": "local",
                "max_gists_per_reference": 128,
                "max_materialized_memory_tokens": 160,
                "context_safety_reserve_tokens": 4,
            },
        }
    )
    return values


def _run_training_matrix(args, metadata: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    training_rows = []
    for tier in args.tiers:
        for mode in args.position_modes:
            settings = _settings(tier, mode, args.smoke)
            tokenizer, training_module, all_modules = native._prepare_synthetic(settings)
            modules = {
                split: module
                for split, module in all_modules.items()
                if split in (SPLIT_COUNTS if not args.smoke else (2, 5))
            }
            native._assert_fixed_target_invariants(modules)
            for seed in args.seeds:
                set_seed(seed)
                run_dir = (
                    REPO / "out" / "paper1_5_rope" / tier / mode / f"seed-{seed}"
                )
                source, training = native.train_full_context_sa(
                    seed=seed,
                    tokenizer=tokenizer,
                    datamodule=training_module,
                    settings=settings,
                    run_dir=run_dir,
                    device=args.device,
                    force=args.force,
                )
                result_rows, _raw = native.evaluate_seed(
                    source=source,
                    tokenizer=tokenizer,
                    modules=modules,
                    settings=settings,
                    device=args.device,
                )
                parameter_count = sum(parameter.numel() for parameter in source.parameters())
                for row in result_rows:
                    rows.append(
                        {
                            "seed": seed,
                            "model_tier": tier,
                            "position_mode": mode,
                            "model_parameters": parameter_count,
                            "native_context": settings["max_seq_len"],
                            "logical_context": settings["max_seq_len"],
                            "logical_native_ratio": 1.0,
                            "maximum_native_operation": settings["max_seq_len"],
                            "encoding_chunk_size": settings["max_seq_len"],
                            "encoding_overlap": 0.0,
                            "routing_chunk_size": None,
                            "materialization_budget": 160,
                            "k_storage_mode": "post_position",
                            "position_policy": "per_reference_local",
                            **row,
                        }
                    )
                training_rows.append(
                    {
                        "seed": seed,
                        "model_tier": tier,
                        "position_mode": mode,
                        "parameters": parameter_count,
                        "steps": training["final_step"],
                        "training_seconds": training["train_seconds"],
                        "checkpoint": training["checkpoint"],
                        "history": training["history"],
                        "settings": settings,
                    }
                )
                del source
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    return rows, training_rows


def run(args) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = environment_metadata()
    translation = run_translation()
    pre_post = run_pre_post()
    fragmentation, training = _run_training_matrix(args, metadata)
    aggregate = aggregate_fragmentation(fragmentation)

    write_json(RESULTS / "rope_translation.json", {"metadata": metadata, "rows": translation})
    write_csv(RESULTS / "rope_translation.csv", translation)
    write_json(RESULTS / "rope_pre_post_k.json", {"metadata": metadata, "rows": pre_post})
    write_csv(RESULTS / "rope_pre_post_k.csv", pre_post)
    write_json(
        RESULTS / "rope_fragmentation.json",
        {
            "metadata": metadata,
            "seeds": args.seeds,
            "tiers": args.tiers,
            "position_modes": args.position_modes,
            "training": training,
            "rows": fragmentation,
            "aggregate": aggregate,
        },
    )
    write_csv(RESULTS / "rope_fragmentation.csv", fragmentation)
    write_csv(RESULTS / "rope_fragmentation_aggregate.csv", aggregate)
    plot_translation(translation, RESULTS / "rope_translation.png")
    plot_fragmentation(aggregate, RESULTS / "rope_fragmentation.png")
    return refresh_manifest(metadata=metadata, smoke=args.smoke)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument(
        "--position-modes",
        nargs="+",
        choices=("absolute", "rope"),
        default=["absolute", "rope"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    path = run(parse_args())
    print(json.dumps({"manifest": str(path)}, indent=2))
