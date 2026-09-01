"""Run one process-isolated OpenVINO GenAI cache-lifecycle cell.

A reference-count failure can poison the pipeline and emit additional teardown
diagnostics.  Each version/cache/scenario cell therefore runs in a fresh
process; orchestration should invoke this script once per cell and retain both
the JSON artifact and process stderr.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper6_3_openvino.run_genai_e0 import (
    _load_harness,
    _runtime_failure,
    _sample,
)


def request_plan(
    scenario: str,
    short_messages: Sequence[Mapping[str, str]],
    long_messages: Sequence[Mapping[str, str]],
    repeats: int,
) -> list[tuple[str, Sequence[Mapping[str, str]]]]:
    """Build the exact ordered workload used to expose lifecycle transitions."""

    if repeats < 2:
        raise ValueError("repeats must be at least two")
    if scenario == "short":
        return [("short", short_messages)] * repeats
    if scenario == "long":
        return [("long", long_messages)]
    if scenario == "long_short":
        return [("long", long_messages)] + [("short", short_messages)] * repeats
    if scenario == "long_long":
        return [("long", long_messages)] * repeats
    raise ValueError(f"Unknown lifecycle scenario: {scenario}")


def run(args: argparse.Namespace) -> Mapping[str, object]:
    import openvino_genai as genai

    harness = _load_harness()
    messages = harness.benchmark_messages()
    plan = request_plan(
        args.scenario,
        messages["pra_only"],
        messages["full_context"],
        args.repeats,
    )

    scheduler = genai.SchedulerConfig()
    scheduler.enable_prefix_caching = args.cache_mode == "enabled"
    scheduler.use_cache_eviction = args.cache_mode == "enabled"
    scheduler.dynamic_split_fuse = True
    scheduler.max_num_seqs = args.max_num_seqs
    scheduler.max_num_batched_tokens = args.max_num_batched_tokens
    scheduler.num_kv_blocks = args.num_kv_blocks

    compiled_at = time.perf_counter()
    pipe = genai.LLMPipeline(args.model, args.device, scheduler_config=scheduler)
    compile_ms = (time.perf_counter() - compiled_at) * 1000.0

    rows: list[Mapping[str, object]] = []
    terminal_status: str | None = None
    for step, (context_kind, step_messages) in enumerate(plan):
        started = time.perf_counter()
        try:
            measured = _sample(pipe, genai, list(step_messages), args.max_tokens)
            rows.append(
                {
                    "step": step,
                    "context_kind": context_kind,
                    "transition": (
                        context_kind
                        if step == 0
                        else f"{plan[step - 1][0]}_to_{context_kind}"
                    ),
                    **measured,
                }
            )
        except RuntimeError as error:
            terminal_status = _runtime_failure(error) or "UNCLASSIFIED_RUNTIME_FAILURE"
            rows.append(
                {
                    "step": step,
                    "context_kind": context_kind,
                    "transition": (
                        context_kind
                        if step == 0
                        else f"{plan[step - 1][0]}_to_{context_kind}"
                    ),
                    "measurement_status": terminal_status,
                    "wall_latency_ms": (time.perf_counter() - started) * 1000.0,
                    "error": str(error),
                }
            )
            break

    del pipe
    gc.collect()
    measured = [row for row in rows if row["measurement_status"] == "MEASURED"]
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_3_openvino_cache_lifecycle_cell_v1",
        "evidence_tier": "LIVE_ENGINE_LIFECYCLE_REGRESSION",
        "measurement_status": terminal_status or "MEASURED",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("openvino", "openvino-genai")
        },
        "model": str(args.model),
        "device": args.device,
        "scenario": args.scenario,
        "cache_mode": args.cache_mode,
        "scheduler": {
            "prefix_caching": scheduler.enable_prefix_caching,
            "cache_eviction": scheduler.use_cache_eviction,
            "dynamic_split_fuse": scheduler.dynamic_split_fuse,
            "num_kv_blocks": scheduler.num_kv_blocks,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        },
        "compile_ms": compile_ms,
        "requested_steps": len(plan),
        "completed_steps": len(measured),
        "all_expected_answers": bool(measured)
        and all(bool(row["expected_answer_present"]) for row in measured),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=("short", "long", "long_short", "long_long"), required=True
    )
    parser.add_argument("--cache-mode", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--num-kv-blocks", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["measurement_status"],
                "completed_steps": payload["completed_steps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
