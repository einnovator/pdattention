"""Run the frozen local Qwen3-Coder SWE-bench calibration profile."""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

from ..context_treatment import CONSUMPTION_POLICIES, ContextTreatment
from .swebench_verified import PINNED_DATASET_REVISION, run


MODEL = "qwen3-coder:30b"
MODEL_REVISION = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"


def official_image_platform(machine: str | None = None) -> str | None:
    """Request the official x86 evaluator images explicitly on ARM hosts."""

    architecture = (machine or platform.machine()).lower()
    return "linux/amd64" if architecture in {"arm64", "aarch64"} else None


def main() -> None:
    """Apply one treatment while keeping the admitted local identity fixed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-card", type=Path, required=True)
    parser.add_argument(
        "--task-index",
        type=int,
        help="Run one 1-based task from the locked benchmark card without resampling.",
    )
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
    parser.add_argument("--recent-completed-turns", type=int, default=2)
    parser.add_argument("--recent-records-per-turn", type=int, default=2)
    parser.add_argument("--recent-source-turns", type=int, default=1)
    parser.add_argument("--recent-progress-turns", type=int, default=1)
    parser.add_argument("--recent-mutation-turns", type=int, default=1)
    parser.add_argument("--recent-verification-turns", type=int, default=1)
    parser.add_argument("--large-record-chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--max-records-per-turn-before-chunking", type=int, default=8,
    )
    parser.add_argument(
        "--preserve-action-observation-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--causal-bundle-round-up",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--consumption-policy",
        choices=CONSUMPTION_POLICIES,
        default="standard",
        help="Hold PRA selection fixed while varying how the agent consumes retained state.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Per-turn safety ceiling forwarded identically in every Easy-50 arm.",
    )
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--selection-replay", type=Path)
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        help="Require and record an endpoint with active sequential-prefix caching.",
    )
    parser.add_argument("--engine", default="ollama")
    parser.add_argument("--engine-version", default="0.32.7")
    parser.add_argument(
        "--require-endpoint-preflight",
        action="store_true",
        help="Validate a direct engine control endpoint before running the cohort.",
    )
    parser.add_argument(
        "--skip-image-prepull",
        action="store_true",
        help="Skip explicit task-image acquisition before mini-swe-agent starts.",
    )
    parser.add_argument(
        "--image-pull-timeout-seconds", type=int, default=3600,
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--endpoint-preflight-receipt", type=Path,
        help=(
            "Reuse a previously completed generation-probe receipt after a clean "
            "engine restart; health, capabilities, and model identity are rechecked."
        ),
    )
    parser.add_argument("--recover-timeout-chunk", type=int)
    options = parser.parse_args()
    args = argparse.Namespace(
        benchmark_card=options.benchmark_card,
        task_index=options.task_index,
        output=options.output,
        model=MODEL,
        served_model=MODEL,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        benchmark_revision=PINNED_DATASET_REVISION,
        base_url=options.base_url,
        engine=options.engine,
        engine_version=options.engine_version,
        dtype="mixed",
        quantization="Q4_K_M",
        kv_cache_dtype="f16",
        harness_version="2.4.6",
        grader_version="4.1.0",
        scaffold="swebench_backticks.yaml",
        grading="SWE-bench 4.1.0 official Docker harness",
        context_limit=32768,
        max_steps=50,
        max_completion_tokens=options.max_completion_tokens,
        run_id=options.run_id,
        mode=options.mode,
        budget_fraction=options.budget_fraction,
        recent_completed_turns=options.recent_completed_turns,
        recent_records_per_turn=options.recent_records_per_turn,
        recent_source_turns=options.recent_source_turns,
        recent_progress_turns=options.recent_progress_turns,
        recent_mutation_turns=options.recent_mutation_turns,
        recent_verification_turns=options.recent_verification_turns,
        large_record_chunk_tokens=options.large_record_chunk_tokens,
        max_records_per_turn_before_chunking=(
            options.max_records_per_turn_before_chunking
        ),
        preserve_action_observation_pairs=options.preserve_action_observation_pairs,
        causal_bundle_round_up=options.causal_bundle_round_up,
        consumption_policy=options.consumption_policy,
        workers=1,
        grader_workers=2,
        chunk_size=1,
        timeout_seconds=3600,
        preflight_only=options.preflight_only,
        endpoint_preflight_receipt=options.endpoint_preflight_receipt,
        allow_partial_reproduction=False,
        local_calibration=True,
        recover_timeout_chunk=options.recover_timeout_chunk,
        selection_record=options.selection_record,
        selection_replay=options.selection_replay,
        prefix_caching=options.prefix_caching,
        require_endpoint_preflight=options.require_endpoint_preflight,
        prepull_images=not options.skip_image_prepull,
        docker_platform=official_image_platform(),
        image_pull_timeout_seconds=options.image_pull_timeout_seconds,
    )
    print(run(args))


if __name__ == "__main__":
    main()
