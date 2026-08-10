"""Train or reuse matched synthetic checkpoints needed by the validation matrix."""

from __future__ import annotations

import argparse
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
    TIERS,
    environment_metadata,
    set_seed,
    write_csv,
    write_json,
)
from experiments.paper1_5_rope.run_core_experiments import _settings  # noqa: E402


VALIDATION = RESULTS / "validation"


def run(args) -> Path:
    metadata = environment_metadata()
    rows = []
    for tier in args.tiers:
        for mode in args.position_modes:
            settings = _settings(tier, mode, args.smoke)
            tokenizer, training_module, _ = native._prepare_synthetic(settings)
            for seed in args.seeds:
                set_seed(seed)
                run_dir = REPO / "out" / "paper1_5_rope" / tier / mode / f"seed-{seed}"
                checkpoint = run_dir / "checkpoint.pt"
                reused = checkpoint.exists() and not args.force
                source, training = native.train_full_context_sa(
                    seed=seed,
                    tokenizer=tokenizer,
                    datamodule=training_module,
                    settings=settings,
                    run_dir=run_dir,
                    device=args.device,
                    force=args.force,
                )
                rows.append(
                    {
                        "git_sha": metadata["git_sha"],
                        "seed": seed,
                        "model_tier": tier,
                        "position_mode": mode,
                        "parameter_count": sum(p.numel() for p in source.parameters()),
                        "training_dataset": "synthetic_native_kv_full_context",
                        "training_examples": settings["train_examples"],
                        "training_tokens": int(settings["steps"])
                        * int(settings["batch_size"])
                        * int(settings["max_seq_len"]),
                        "native_training_context": settings["max_seq_len"],
                        "optimizer": "AdamW",
                        "scheduler": "constant",
                        "batch_size": settings["batch_size"],
                        "steps": training["final_step"],
                        "final_train_loss": training["history"][-1]["train_loss"]
                        if training["history"]
                        else None,
                        "validation_loss": training["history"][-1]["validation_loss"]
                        if training["history"]
                        else None,
                        "checkpoint": training["checkpoint"],
                        "checkpoint_reused": reused,
                    }
                )
                del source
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    VALIDATION.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "smoke": args.smoke,
        "matched_dimensions": [
            "architecture",
            "training corpus",
            "training examples",
            "optimizer",
            "batching",
            "steps",
            "native training context",
            "seeds",
        ],
        "intentional_difference": "position_mode",
        "rows": rows,
    }
    path = VALIDATION / "synthetic_checkpoint_matrix.json"
    write_json(path, payload)
    write_csv(VALIDATION / "synthetic_checkpoint_matrix.csv", rows)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument(
        "--position-modes",
        nargs="+",
        choices=("absolute", "sinusoidal", "rope"),
        default=["absolute", "sinusoidal", "rope"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
