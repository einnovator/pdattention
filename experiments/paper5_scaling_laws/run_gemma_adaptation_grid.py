"""Run the five-seed Gemma consumer/interface adaptation ladder for Paper 5.

Each model/seed cell executes in a fresh Python process so MPS allocations and
injected attention modules cannot leak into the next cell. Existing summaries
make the grid restartable after interruption. The resulting grid measures
consumer plasticity at two parameter scales; it is not a logical-memory sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper4_training.aggregate_pretrained_tier1 import aggregate


MODELS = (
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
)
SEEDS = (11, 23, 37, 53, 71)
REGIMES = ("frozen_pra", "consumer_lora", "interface_lora")


def model_slug(model: str) -> str:
    """Return a stable filesystem-safe model identity."""

    return model.replace("/", "--")


def cell_command(args: argparse.Namespace, model: str, seed: int, output: Path) -> list[str]:
    """Build one isolated, fully explicit adaptation command."""

    command = [
        sys.executable,
        "-u",
        "experiments/paper4_training/run_pretrained_tier1.py",
        "--model",
        model,
        "--device",
        args.device,
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-dir",
        str(output),
        "--split-seed",
        str(seed),
        "--train-examples",
        str(args.train_examples),
        "--validation-examples",
        str(args.validation_examples),
        "--steps",
        str(args.steps),
        "--ordinary-every",
        str(args.ordinary_every),
        "--context-limit",
        str(args.context_limit),
        "--reference-limit",
        str(args.reference_limit),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--learning-rate",
        str(args.learning_rate),
    ]
    for regime in args.regime:
        command.extend(("--regime", regime))
    return command


def run_grid(args: argparse.Namespace) -> dict:
    """Run missing cells and emit one strict five-seed aggregate per model."""

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    source = str(Path("src").resolve())
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    aggregates = []
    for model in args.model:
        model_root = root / model_slug(model)
        summaries = []
        for seed in SEEDS:
            cell = model_root / f"seed_{seed}"
            summary = cell / "summary.json"
            if not summary.is_file() or args.force:
                cell.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    cell_command(args, model, seed, cell),
                    check=True,
                    cwd=ROOT,
                    env=env,
                )
            summaries.append(summary)
        result = aggregate(summaries)
        result["paper5_scope"] = (
            "two-scale pretrained consumer adaptation; no logical-memory scaling claim"
        )
        aggregate_path = model_root / "aggregate.json"
        aggregate_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        aggregates.append(result)
    payload = {
        "schema_version": "paper5-gemma-adaptation-grid-v1",
        "models": list(args.model),
        "seeds": list(SEEDS),
        "regimes": list(args.regime),
        "aggregates": aggregates,
    }
    (root / "grid_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append")
    parser.add_argument("--regime", action="append", choices=REGIMES)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/wikitext2_references_v2")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/papers/shared/results/paper5_gemma_adaptation_grid"),
    )
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--validation-examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ordinary-every", type=int, default=4)
    parser.add_argument("--context-limit", type=int, default=192)
    parser.add_argument("--reference-limit", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.model = args.model or list(MODELS)
    args.regime = args.regime or list(REGIMES)
    return args


if __name__ == "__main__":
    summary = run_grid(parse_args())
    print(json.dumps({"models": summary["models"], "seeds": summary["seeds"]}, indent=2))
