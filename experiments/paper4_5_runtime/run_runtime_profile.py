"""Run the portable Paper 4.5 CPU and CUDA mechanism profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pra_hf.runtime_benchmark import run_runtime_microbenchmark, write_runtime_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/paper4_5_runtime"),
    )
    parser.add_argument("--candidate-tokens", type=int, default=8192)
    parser.add_argument("--selected-tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cpu-repeats", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu = run_runtime_microbenchmark(
        device="cpu",
        candidate_tokens=args.candidate_tokens,
        selected_tokens=args.selected_tokens,
        batches=(1, 4),
        warmups=min(args.warmups, 5),
        repeats=args.cpu_repeats,
    )
    write_runtime_benchmark(cpu, args.output / "cpu")
    if torch.cuda.is_available():
        cuda = run_runtime_microbenchmark(
            device="cuda",
            candidate_tokens=args.candidate_tokens,
            selected_tokens=args.selected_tokens,
            batches=(1, 4),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        write_runtime_benchmark(cuda, args.output / "cuda")


if __name__ == "__main__":
    main()
