"""Run the frozen local Qwen3-Coder SWE-bench calibration profile."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..context_treatment import ContextTreatment
from .swebench_verified import PINNED_DATASET_REVISION, run


MODEL = "qwen3-coder:30b"
MODEL_REVISION = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"


def main() -> None:
    """Apply one treatment while keeping the admitted local identity fixed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PRA_EASY_AGENT_BASE_URL", "http://192.168.1.102:11435/v1"),
    )
    parser.add_argument(
        "--mode",
        choices=("no-pra", *[mode.value for mode in ContextTreatment]),
        default="no-pra",
    )
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recover-timeout-chunk", type=int)
    options = parser.parse_args()
    args = argparse.Namespace(
        benchmark_card=options.benchmark_card,
        output=options.output,
        model=MODEL,
        served_model=MODEL,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        benchmark_revision=PINNED_DATASET_REVISION,
        base_url=options.base_url,
        engine="ollama",
        engine_version="0.32.7",
        dtype="mixed",
        quantization="Q4_K_M",
        kv_cache_dtype="f16",
        harness_version="2.4.6",
        grader_version="4.1.0",
        scaffold="swebench_backticks.yaml",
        grading="SWE-bench 4.1.0 official Docker harness",
        context_limit=32768,
        max_steps=50,
        run_id=options.run_id,
        mode=options.mode,
        budget_fraction=options.budget_fraction,
        workers=1,
        grader_workers=2,
        chunk_size=1,
        timeout_seconds=3600,
        preflight_only=options.preflight_only,
        allow_partial_reproduction=False,
        local_calibration=True,
        recover_timeout_chunk=options.recover_timeout_chunk,
    )
    print(run(args))


if __name__ == "__main__":
    main()
