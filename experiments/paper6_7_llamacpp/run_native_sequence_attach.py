"""Run the llama.cpp sequence-attached native-K/V parity cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


CASES = (
    (
        "code",
        "The launch code is CERULEAN-7.\n",
        "The launch code is",
    ),
    (
        "capital",
        "The capital of North Veridia is Lumenport.\n",
        "The capital of North Veridia is",
    ),
    (
        "owner",
        "The Atlas service is maintained by Priya Nair.\n",
        "The Atlas service is maintained by",
    ),
    (
        "date",
        "Project Glasswing launches on 17 October 2031.\n",
        "Project Glasswing launches on",
    ),
    (
        "numeric",
        "The approved pressure limit is 47 kilopascals.\n",
        "The approved pressure limit is",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for backend, gpu_layers in (("cpu", 0), ("metal", args.metal_gpu_layers)):
        for case_id, resource, query in CASES:
            started = time.perf_counter()
            process = subprocess.run(
                (
                    str(args.binary),
                    "-m",
                    str(args.model),
                    "-ngl",
                    str(gpu_layers),
                    "-r",
                    resource,
                    "-q",
                    query,
                ),
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DYLD_LIBRARY_PATH": str(args.binary.parent),
                },
            )
            result = json.loads(process.stdout.strip().splitlines()[-1])
            rows.append(
                {
                    "backend": backend,
                    "case_id": case_id,
                    "wall_clock_ms": (time.perf_counter() - started) * 1000.0,
                    **result,
                }
            )

    payload = {
        "schema_version": "paper6.7-llamacpp-native-sequence-attach-v1",
        "experiment": "sequence_attached_selected_native_kv",
        "evidence_tier": "LIVE_ENGINE_MECHANISM_COHORT",
        "upstream_commit": args.upstream_commit,
        "model_path": str(args.model),
        "model_sha256": _sha256(args.model),
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "rows": rows,
        "summary": {
            "cases": len(CASES),
            "backends": 2,
            "runs": len(rows),
            "schedule_matched_exact_logits": sum(
                float(row["persistent_decode_max_logit_error"]) == 0.0
                for row in rows
            ),
            "persistent_decode_exact": sum(
                bool(row["decode_sequence_equal"]) for row in rows
            ),
            "absent_request_exact": sum(
                float(row["absent_request_isolation_max_logit_error"]) == 0.0
                for row in rows
            ),
            "absent_request_bounded_1e_2": sum(
                float(row["absent_request_isolation_max_logit_error"]) <= 1e-2
                for row in rows
            ),
            "absent_request_top_token_equal": sum(
                bool(row["absent_request_top_token_equal"]) for row in rows
            ),
            "warm_resource_reuse_exact": sum(
                float(row["warm_resource_reuse_max_logit_error"]) == 0.0
                for row in rows
            ),
            "full_top_token_equal": sum(
                bool(row["full_top_token_equal"]) for row in rows
            ),
            "physical_kv_copy": any(bool(row["physical_kv_copy"]) for row in rows),
        },
        "interpretation": (
            "llama_memory_seq_cp attaches existing resource K/V cells to the "
            "request sequence in unified-cache mode without a physical K/V copy. "
            "Schedule-matched E0/E2 isolates this mechanism; FULL-vs-E2 also "
            "includes ordinary split-prefill numerical effects."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metal-gpu-layers", type=int, default=99)
    parser.add_argument(
        "--upstream-commit",
        default="458681e1d5d4a29a1463c4732e03226cf384b997",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2))
