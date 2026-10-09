"""Run one immutable frozen request sequence through a tensor engine.

This is deliberately a request-sequence qualification driver, not an
autonomous-agent evaluator.  Every request is executed independently through
the already-qualified engine lifecycle harness, and the resulting artifacts
are reduced only after all requests pass their per-request gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.paper4_5_agent.summarize_frozen_request_sequence import summarize


ENGINE_MODULES = {
    "hf": "experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle",
    "mlx": "experiments.paper4_5_agent.run_mlx_live_agent_kv_lifecycle",
    "sglang": "experiments.paper4_5_agent.run_sglang_live_agent_kv_lifecycle",
    "vllm-metal": "experiments.paper4_5_agent.run_vllm_metal_live_kv_lifecycle",
}


def _request_indices(request_replay: Path) -> tuple[int, ...]:
    rows = [
        json.loads(line)
        for line in request_replay.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Request replay is empty.")
    indices = tuple(int(row.get("request_index", offset)) for offset, row in enumerate(rows, 1))
    expected = tuple(range(1, len(rows) + 1))
    if indices != expected:
        raise ValueError(
            f"Request replay must contain contiguous identities {expected}; observed {indices}."
        )
    return indices


def _selected_request_indices(
    available: tuple[int, ...], value: str | None
) -> tuple[int, ...]:
    if value is None:
        return available
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Request-index selection contains an empty item.")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError("Request-index range is reversed.")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    result = tuple(index for index in available if index in selected)
    if not result or set(result) != selected:
        raise ValueError("Requested index is outside the frozen replay.")
    return result


def build_command(args: argparse.Namespace, request_index: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        ENGINE_MODULES[args.engine],
        "--request-replay",
        str(args.request_replay),
        "--selection-fixture",
        str(args.selection_fixture),
        "--request-index",
        str(request_index),
        "--output",
        str(output),
        "--model",
        args.model,
        "--continuation-tokens",
        str(args.continuation_tokens),
    ]
    if args.full_retention:
        command.append("--frozen-full-retention")
    if args.engine == "hf":
        command.extend(
            [
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--prefill-step-size",
                str(args.prefill_step_size),
                "--max-abs-logit-delta",
                str(args.max_abs_logit_delta),
                "--revision",
                args.revision,
            ]
        )
        if args.local_files_only:
            command.append("--local-files-only")
    elif args.engine in ("mlx", "sglang"):
        command.extend(
            [
                "--prefill-step-size",
                str(args.prefill_step_size),
                "--max-abs-logit-delta",
                str(args.max_abs_logit_delta),
                "--source-prefill-step-size",
                str(args.source_prefill_step_size),
                "--materialization-policy",
                "disjoint_segmented",
                "--hardware-label",
                args.hardware_label,
            ]
        )
    if args.engine == "mlx":
        command.append("--fused-disjoint-attention")
    elif args.engine == "sglang":
        command.extend(["--provenance", str(args.provenance), "--revision", args.revision])
    elif args.engine == "vllm-metal":
        command.extend(
            [
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--engine-source-revision",
                args.engine_source_revision,
                "--hardware-label",
                args.hardware_label,
            ]
        )
        if args.artifact_overlay:
            command.append("--artifact-overlay")
    return command


def run_sequence(args: argparse.Namespace) -> dict[str, Any]:
    for source in (args.request_replay, args.selection_fixture):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.engine == "sglang" and (args.provenance is None or not args.provenance.is_file()):
        raise FileNotFoundError(args.provenance)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output directory already exists: {args.output_dir}"
        )

    available_indices = _request_indices(args.request_replay)
    indices = _selected_request_indices(
        available_indices, getattr(args, "request_indices", None)
    )
    shared_hf_loader = None
    if args.engine == "hf" and getattr(args, "reuse_hf_model", False):
        from experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle import (
            _require_live_kv_api,
            load_hf_components,
        )

        _require_live_kv_api()
        shared_hf_loader = load_hf_components
    args.output_dir.mkdir(parents=True)
    artifacts: list[Path] = []
    shared_hf = None
    if shared_hf_loader is not None:
        shared_hf = shared_hf_loader(args)
    for request_index in indices:
        output = args.output_dir / f"request_{request_index:02d}.json"
        if shared_hf is None:
            command = build_command(args, request_index, output)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{args.engine} request {request_index} failed with exit code "
                    f"{completed.returncode}; partial evidence remains immutable in "
                    f"{args.output_dir}."
                )
        else:
            from experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle import (
                run as run_hf_lifecycle,
            )

            lifecycle_args = argparse.Namespace(**vars(args))
            lifecycle_args.trajectory = None
            lifecycle_args.request_index = request_index
            lifecycle_args.output = output
            lifecycle_args.frozen_full_retention = args.full_retention
            lifecycle_args.turn = 4
            lifecycle_args.retention_fraction = 0.9
            lifecycle_args.wire_tail_tokens = 32
            payload = run_hf_lifecycle(
                lifecycle_args,
                tokenizer=shared_hf[0],
                model=shared_hf[1],
            )
            if not payload.get("engine_lifecycle_qualified", False):
                raise RuntimeError(
                    f"HF request {request_index} did not qualify; partial evidence "
                    f"remains immutable in {args.output_dir}."
                )
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not payload.get("engine_lifecycle_qualified", False):
            raise RuntimeError(
                f"{args.engine} request {request_index} did not qualify; "
                f"partial evidence remains immutable in {args.output_dir}."
            )
        artifacts.append(output)

    result = summarize(
        artifacts,
        indices,
        expect_full_retention=args.full_retention,
        selection_fixture=args.selection_fixture,
        request_replay=args.request_replay,
    )
    result["available_requests"] = list(available_indices)
    result["requested_requests"] = list(indices)
    result["shared_model_process"] = bool(shared_hf)
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["sequence_qualified"]:
        raise RuntimeError(f"Frozen sequence failed qualification; see {summary}.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=tuple(ENGINE_MODULES), required=True)
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--request-indices",
        help="Optional immutable subset such as 4-8 or 1,3,5-7.",
    )
    parser.add_argument("--full-retention", action="store_true")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--continuation-tokens", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=1)
    parser.add_argument("--source-prefill-step-size", type=int, default=512)
    parser.add_argument("--max-abs-logit-delta", type=float, default=0.01)
    parser.add_argument("--hardware-label", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--engine-source-revision", default="unknown")
    parser.add_argument("--artifact-overlay", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--reuse-hf-model",
        action="store_true",
        help=(
            "Load the pinned HF model once while creating a fresh PRA runtime "
            "and lifecycle state for every requested index."
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_sequence(parse_args())
    print(
        json.dumps(
            {
                "engine": result["engine"],
                "completed_requests": result["completed_requests"],
                "weighted_realized_retention_fraction": result[
                    "weighted_realized_retention_fraction"
                ],
                "historical_kv_omission_fraction": result[
                    "historical_kv_omission_fraction"
                ],
                "visible_context_omission_fraction": result[
                    "visible_context_omission_fraction"
                ],
                "sequence_qualified": result["sequence_qualified"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
