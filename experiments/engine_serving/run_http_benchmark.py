"""Run the fixed PRA/prefix serving matrix against an engine HTTP endpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_benchmark():
    """Load the stdlib-only harness without importing the Torch-heavy SDK API."""

    source = Path(__file__).resolve().parents[2] / "src" / "pra_hf" / "serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("pra_serving_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving benchmark from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_serving_benchmark


def main() -> None:
    run_serving_benchmark = _load_benchmark()
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("vllm", "sglang", "mlx"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    payload = run_serving_benchmark(
        args.base_url,
        model=args.model,
        engine=args.engine,
        repeats=args.repeats,
        timeout_seconds=args.timeout_seconds,
        use_cache_salt=args.engine == "vllm",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "engine": args.engine,
        "samples": len(payload["samples"]),
    }))


if __name__ == "__main__":
    main()
