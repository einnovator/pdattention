"""Run resumable Paper 3.2 cross-document composition seed campaigns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _complete(path: Path) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(manifest.get("completed_unix") and manifest.get("rows"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--precision-mode", default="INT4")
    parser.add_argument("--source-checkpoint", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--source-weight-dtype", default="bfloat16")
    parser.add_argument("--seeds", type=_seeds, default=(11, 23, 37, 71, 101))
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--composition-residual-scale", type=float, default=1.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    campaign: dict[str, object] = {
        "schema_version": "paper3.2-crossdoc-campaign-v1",
        "model": args.model,
        "revision": args.revision,
        "precision_mode": args.precision_mode,
        "seeds": list(args.seeds),
        "max_examples": args.max_examples,
        "token_budget": args.token_budget,
        "max_resources": args.max_resources,
        "composition_residual_scale": args.composition_residual_scale,
        "started_unix": time.time(),
        "runs": [],
    }
    campaign_path = args.output / "campaign_manifest.json"
    runs = campaign["runs"]
    assert isinstance(runs, list)
    for seed in args.seeds:
        run_dir = args.output / f"seed{seed}"
        manifest_path = run_dir / "manifest.json"
        if _complete(manifest_path):
            runs.append(
                {"seed": seed, "status": "REUSED", "manifest": str(manifest_path)}
            )
            continue
        command = [
            sys.executable,
            "-m",
            "experiments.paper3_2_rag.run_prerope_causal_decomposition",
            "--dataset",
            "multihop_rag",
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--precision-mode",
            args.precision_mode,
            "--source-checkpoint",
            args.source_checkpoint,
            "--source-weight-dtype",
            args.source_weight_dtype,
            "--seed",
            str(seed),
            "--max-examples",
            str(args.max_examples),
            "--token-budget",
            str(args.token_budget),
            "--max-resources",
            str(args.max_resources),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--mask-policies",
            "PREVIOUS_DOC_ONLY",
            "--boundary-windows",
            "8",
            "--composition-modes",
            "append,boundary8,boundary32",
            "--composition-residual-scale",
            str(args.composition_residual_scale),
            "--output",
            str(run_dir),
        ]
        started = time.time()
        result = subprocess.run(command, check=False)
        runs.append(
            {
                "seed": seed,
                "status": "COMPLETE" if result.returncode == 0 else "FAILED",
                "returncode": result.returncode,
                "elapsed_seconds": time.time() - started,
                "manifest": str(manifest_path),
            }
        )
        campaign_path.write_text(
            json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    campaign["completed_unix"] = time.time()
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
