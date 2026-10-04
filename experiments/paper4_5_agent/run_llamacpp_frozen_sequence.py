"""Run one immutable llama.cpp Full or frozen-subset request sequence.

Unlike the tensor-engine sequence reducer, llama.cpp currently establishes
repeat-exact selected-page attachment rather than an independent dense-logit
reference.  The summary names that narrower claim boundary explicitly.
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


MODULE = "experiments.paper4_5_agent.run_llamacpp_frozen_receipt_lifecycle"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(args: argparse.Namespace, request_index: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--base-url",
        args.base_url,
        "--request-replay",
        str(args.request_replay),
        "--selection-fixture",
        str(args.selection_fixture),
        "--request-index",
        str(request_index),
        "--output",
        str(output),
        "--tokenizer",
        args.tokenizer,
        "--model-label",
        args.model_label,
        "--model-fingerprint",
        args.model_fingerprint,
        "--tokenizer-fingerprint",
        args.tokenizer_fingerprint,
        "--engine-revision",
        args.engine_revision,
        "--experiment-revision",
        args.experiment_revision,
        "--source-slot",
        str(args.source_slot),
        "--request-slot",
        str(args.request_slot),
        "--continuation-tokens",
        str(args.continuation_tokens),
        "--seed",
        str(args.seed),
        "--require-strict-resident-subset",
    ]
    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", args.tokenizer_revision])
    if args.local_files_only:
        command.append("--local-files-only")
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
    if len(engines) != 1 or len(models) != 1:
        raise ValueError("Artifacts mix engine/model identities.")

    rows: list[dict[str, Any]] = []
    for path, payload in zip(paths, payloads):
        checks = dict(payload.get("checks") or {})
        source_tokens = int(payload["logical_source_tokens"])
        selected_tokens = int(payload["selected_logical_kv_tokens"])
        full_visible = int(payload["full_visible_tokens"])
        visible = int(payload["total_visible_tokens"])
        row_qualified = (
            bool(payload.get("qualified"))
            and not payload.get("qualification_blockers")
            and checks.get("repeat_token_exact") is True
            and checks.get("selected_history_not_reencoded") is True
            and checks.get("selected_original_kv_not_copied") is True
            and checks.get("source_positions_preserved") is True
            and checks.get("lifecycle_empty") is True
            and int(payload.get("selected_history_reencoded_tokens", -1)) == 0
            and int(payload.get("selected_history_kv_copy_bytes", -1)) == 0
            and (not full or (selected_tokens == source_tokens and visible == full_visible))
        )
        rows.append(
            {
                "request_index": int(payload["request_index"]),
                "request_input_sha256": str(payload["request_input_sha256"]),
                "source_tokens": source_tokens,
                "selected_kv_tokens": selected_tokens,
                "wire_suffix_tokens": int(payload["wire_suffix_tokens"]),
                "full_visible_tokens": full_visible,
                "total_visible_tokens": visible,
                "realized_retention_fraction": visible / max(full_visible, 1),
                "repeat_token_exact": bool(checks.get("repeat_token_exact")),
                "selected_history_reencoded_tokens": int(
                    payload["selected_history_reencoded_tokens"]
                ),
                "selected_history_kv_copy_bytes": int(
                    payload["selected_history_kv_copy_bytes"]
                ),
                "request_qualified": row_qualified,
                "artifact": str(path),
                "artifact_sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "paper4.5.llamacpp-frozen-request-sequence.v1",
        "claim_boundary": (
            "repeat_exact_selected_page_attachment_request_sequence_"
            "not_independent_dense_logit_reference_or_autonomous_task"
        ),
        "engine": next(iter(engines)),
        "model": next(iter(models)),
        "expect_full_retention": full,
        "expected_requests": list(expected),
        "completed_requests": len(rows),
        "qualified_requests": sum(int(row["request_qualified"]) for row in rows),
        "sequence_qualified": all(row["request_qualified"] for row in rows),
        "weighted_realized_retention_fraction": sum(
            row["total_visible_tokens"] for row in rows
        )
        / max(sum(row["full_visible_tokens"] for row in rows), 1),
        "selected_history_reencoded_tokens": sum(
            row["selected_history_reencoded_tokens"] for row in rows
        ),
        "selected_history_kv_copy_bytes": sum(
            row["selected_history_kv_copy_bytes"] for row in rows
        ),
        "rows": rows,
    }


def run_sequence(args: argparse.Namespace) -> dict[str, Any]:
    for source in (args.request_replay, args.selection_fixture):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output directory already exists: {args.output_dir}"
        )
    expected = _request_indices(args.request_replay)
    args.output_dir.mkdir(parents=True)
    paths: list[Path] = []
    for request_index in expected:
        output = args.output_dir / f"request_{request_index:02d}.json"
        completed = subprocess.run(
            build_command(args, request_index, output), check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp request {request_index} failed with exit code "
                f"{completed.returncode}; partial evidence remains in {args.output_dir}."
            )
        paths.append(output)
    result = summarize(paths, expected, args.full_retention)
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["sequence_qualified"]:
        raise RuntimeError(f"llama.cpp sequence failed qualification; see {summary}.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-retention", action="store_true")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--tokenizer-fingerprint", required=True)
    parser.add_argument("--engine-revision", required=True)
    parser.add_argument("--experiment-revision", required=True)
    parser.add_argument("--source-slot", type=int, default=0)
    parser.add_argument("--request-slot", type=int, default=1)
    parser.add_argument("--continuation-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_sequence(parse_args())
    print(
        json.dumps(
            {
                "completed_requests": result["completed_requests"],
                "qualified_requests": result["qualified_requests"],
                "weighted_realized_retention_fraction": result[
                    "weighted_realized_retention_fraction"
                ],
                "sequence_qualified": result["sequence_qualified"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
