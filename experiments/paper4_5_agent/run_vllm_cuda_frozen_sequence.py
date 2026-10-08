"""Run one immutable frozen request sequence through vLLM/CUDA page aliasing.

This driver qualifies the same hash-bound logical request ledger used by the
other engine sequence runners.  It intentionally makes the narrower vLLM
claim established by the lifecycle harness: repeat-exact attachment of the
selected original pages with no history re-encoding or K/V copying.  It is not
an independent dense-logit reference and it is not an autonomous task run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.paper4_5_agent.run_frozen_request_sequence import _request_indices


MODULE = "experiments.paper4_5_agent.run_vllm_cuda_frozen_receipt_lifecycle"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(args: argparse.Namespace, request_index: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--request-replay",
        str(args.request_replay),
        "--selection-fixture",
        str(args.selection_fixture),
        "--request-index",
        str(request_index),
        "--output",
        str(output),
        "--storage",
        str(args.storage_root / f"request_{request_index:02d}"),
        "--model",
        args.model,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-transfer-buffer-bytes",
        str(args.kv_transfer_buffer_bytes),
        "--continuation-tokens",
        str(args.continuation_tokens),
    ]
    if args.full_retention:
        command.append("--frozen-full-retention")
    return command


def summarize(paths: list[Path], expected: tuple[int, ...], full: bool) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed = tuple(int(payload["request_index"]) for payload in payloads)
    if observed != expected:
        raise ValueError(f"Expected request identities {expected}, observed {observed}.")
    digests = [str(payload["request_input_sha256"]) for payload in payloads]
    if len(set(digests)) != len(digests):
        raise ValueError("Frozen request digests are not unique.")
    engines = {str(payload["engine"]) for payload in payloads}
    models = {str(payload["model"]) for payload in payloads}
    policies = {str(payload["source_policy"]) for payload in payloads}
    if len(engines) != 1 or len(models) != 1 or len(policies) != 1:
        raise ValueError("Artifacts mix engine, model, or source-policy identities.")

    copy_fields = (
        "selected_history_kv_copy_bytes",
        "receipt_capture_d2h_bytes",
        "receipt_host_pack_copy_bytes",
        "receipt_h2d_bytes",
        "receipt_intermediate_cow_bytes",
        "receipt_final_cow_bytes",
        "terminal_page_copy_bytes_per_request",
    )
    rows: list[dict[str, Any]] = []
    for path, payload in zip(paths, payloads):
        checks = dict(payload.get("checks") or {})
        logical_source = int(payload["logical_source_tokens"])
        resident_source = int(payload["resident_source_tokens"])
        selected_logical = int(payload["selected_logical_kv_tokens"])
        selected_pages = int(payload["selected_original_page_tokens"])
        visible = int(payload["total_visible_tokens"])
        full_visible = int(payload["full_visible_tokens"])
        copies = {name: int(payload.get(name, -1)) for name in copy_fields}
        qualified = (
            bool(payload.get("qualified"))
            and not payload.get("qualification_blockers")
            and checks.get("all_receipts_materialized") is True
            and checks.get("all_original_pages_zero_copy") is True
            and checks.get("selected_history_not_reencoded") is True
            and checks.get("repeat_token_exact") is True
            and checks.get("final_alias_installed_twice") is True
            and checks.get("final_alias_released_twice") is True
            and checks.get("lifecycle_empty") is True
            and int(payload.get("selected_history_reencoded_tokens", -1)) == 0
            and int(payload.get("materialized_history_tokens", -1)) == 0
            and all(value == 0 for value in copies.values())
            and (
                not full
                or (
                    selected_logical == logical_source
                    and visible == full_visible
                )
            )
        )
        rows.append(
            {
                "request_index": int(payload["request_index"]),
                "request_input_sha256": str(payload["request_input_sha256"]),
                "logical_source_tokens": logical_source,
                "resident_source_tokens": resident_source,
                "selected_logical_kv_tokens": selected_logical,
                "selected_original_page_tokens": selected_pages,
                "selected_history_page_overhead_tokens": int(
                    payload.get("selected_history_page_overhead_tokens", 0)
                ),
                "wire_suffix_tokens": int(payload["wire_suffix_tokens"]),
                "total_visible_tokens": visible,
                "full_visible_tokens": full_visible,
                "selected_history_reencoded_tokens": int(
                    payload["selected_history_reencoded_tokens"]
                ),
                "materialized_history_tokens": int(
                    payload["materialized_history_tokens"]
                ),
                "copy_bytes": copies,
                "request_qualified": qualified,
                "artifact": str(path),
                "artifact_sha256": _sha256(path),
            }
        )

    logical_source = sum(row["logical_source_tokens"] for row in rows)
    resident_source = sum(row["resident_source_tokens"] for row in rows)
    selected_logical = sum(row["selected_logical_kv_tokens"] for row in rows)
    selected_pages = sum(row["selected_original_page_tokens"] for row in rows)
    visible = sum(row["total_visible_tokens"] for row in rows)
    full_visible = sum(row["full_visible_tokens"] for row in rows)
    return {
        "schema_version": "paper4.5.vllm-cuda-frozen-request-sequence.v1",
        "claim_boundary": (
            "repeat_exact_selected_page_alias_request_sequence_"
            "not_independent_dense_logit_reference_or_autonomous_task"
        ),
        "engine": next(iter(engines)),
        "model": next(iter(models)),
        "source_policy": next(iter(policies)),
        "expect_full_retention": full,
        "expected_requests": list(expected),
        "completed_requests": len(rows),
        "qualified_requests": sum(int(row["request_qualified"]) for row in rows),
        "sequence_qualified": all(row["request_qualified"] for row in rows),
        "cumulative_logical_source_tokens": logical_source,
        "cumulative_selected_logical_kv_tokens": selected_logical,
        "logical_history_omission_fraction": 1.0
        - selected_logical / max(logical_source, 1),
        "cumulative_resident_source_tokens": resident_source,
        "cumulative_selected_original_page_tokens": selected_pages,
        "resident_original_kv_omission_fraction": 1.0
        - selected_pages / max(resident_source, 1),
        "visible_context_omission_fraction": 1.0
        - visible / max(full_visible, 1),
        "selected_history_reencoded_tokens": sum(
            row["selected_history_reencoded_tokens"] for row in rows
        ),
        "materialized_history_tokens": sum(
            row["materialized_history_tokens"] for row in rows
        ),
        "selected_history_kv_copy_bytes": sum(
            row["copy_bytes"]["selected_history_kv_copy_bytes"] for row in rows
        ),
        "page_rounding_overhead_tokens": sum(
            row["selected_history_page_overhead_tokens"] for row in rows
        ),
        "rows": rows,
    }


def run_sequence(args: argparse.Namespace) -> dict[str, Any]:
    for source in (args.request_replay, args.selection_fixture):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.output_dir.exists() or args.storage_root.exists():
        raise FileExistsError("Output and storage roots must be new immutable paths.")

    expected = _request_indices(args.request_replay)
    args.output_dir.mkdir(parents=True)
    args.storage_root.mkdir(parents=True)
    paths: list[Path] = []
    for request_index in expected:
        output = args.output_dir / f"request_{request_index:02d}.json"
        completed = subprocess.run(build_command(args, request_index, output), check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"vLLM/CUDA request {request_index} failed with exit code "
                f"{completed.returncode}; partial evidence remains immutable in "
                f"{args.output_dir}."
            )
        paths.append(output)

    result = summarize(paths, expected, args.full_retention)
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["sequence_qualified"]:
        raise RuntimeError(f"vLLM/CUDA sequence failed qualification; see {summary}.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--full-retention", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--kv-transfer-buffer-bytes", type=int, default=200_000_000)
    parser.add_argument("--continuation-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    result = run_sequence(parse_args())
    print(
        json.dumps(
            {
                "completed_requests": result["completed_requests"],
                "qualified_requests": result["qualified_requests"],
                "resident_original_kv_omission_fraction": result[
                    "resident_original_kv_omission_fraction"
                ],
                "sequence_qualified": result["sequence_qualified"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
