"""Run the source-matched mini-swe-agent fixed-50 no-PRA baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..benchmark import load_benchmark_card


EXPECTED_PACKAGES = {
    "mini-swe-agent": "2.4.0",
    "swebench": "4.1.0",
    "vllm": "0.22.1",
}


def package_versions() -> dict[str, str | None]:
    """Return installed versions without converting missing packages to zero-like values."""

    versions: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def preflight(args: argparse.Namespace, card: dict[str, Any]) -> dict[str, Any]:
    """Capture identity and host compatibility before expensive inference starts."""

    versions = package_versions()
    differences = [
        f"{name}={versions[name]!r}; source requires {expected!r}"
        for name, expected in EXPECTED_PACKAGES.items()
        if versions[name] != expected
    ]
    if args.model_revision == "NOT_REPORTED_BY_SOURCE":
        differences.append("source study did not publish an immutable model revision")
    if args.tokenizer_revision == "NOT_REPORTED_BY_SOURCE":
        differences.append("source study did not publish an immutable tokenizer revision")
    gpu = _nvidia_gpu()
    if not _is_h100_80gb(gpu):
        differences.append(f"hardware={gpu or 'no NVIDIA GPU detected'}; source used one H100 80GB")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_source_revision": card["source_revision"],
        "benchmark_ids_sha256": card["canonical_ids_sha256"],
        "instance_count": len(card["instance_ids"]),
        "model": args.model,
        "served_model": args.served_model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "engine": "vllm",
        "engine_versions": versions,
        "dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "context_limit": 16384,
        "max_steps": 40,
        "temperature": 0,
        "harness_config": "swebench_backticks.yaml",
        "model_class": "litellm_textbased",
        "official_grader": "SWE-bench Docker harness",
        "python": sys.version,
        "os": platform.platform(),
        "gpu": gpu,
        "configuration_differences": differences,
        "exact_environment": not differences,
    }


def run(args: argparse.Namespace) -> Path:
    """Execute all fixed IDs in resumable chunks and normalize official grading."""

    if args.mode != "no-pra":
        raise ValueError("reproduction runner supports only the no-PRA baseline")
    card = load_benchmark_card(args.benchmark_card)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = preflight(args, card)
    _write_json(output / "run_manifest.json", receipt)
    if args.preflight_only:
        return output / "run_manifest.json"
    if receipt["configuration_differences"] and not args.allow_partial_reproduction:
        raise RuntimeError(
            "source-matched preflight failed; use --allow-partial-reproduction only for a "
            "diagnostic that must remain locked from PRA"
        )

    submitted: set[str] = set()
    resolved: set[str] = set()
    errors: set[str] = set()
    instance_ids = card["instance_ids"]
    for chunk_index in range(0, len(instance_ids), args.chunk_size):
        chunk_ids = instance_ids[chunk_index:chunk_index + args.chunk_size]
        chunk_number = chunk_index // args.chunk_size
        chunk_dir = output / f"chunk_{chunk_number:02d}"
        report_receipt = chunk_dir / "official_chunk_result.json"
        if report_receipt.is_file():
            chunk_result = json.loads(report_receipt.read_text(encoding="utf-8"))
        else:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            pattern = "(" + "|".join(re.escape(item) for item in chunk_ids) + ")"
            agent_command = [
                sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
                "--subset", "verified", "--split", "test", "--filter", pattern,
                "-m", f"openai/{args.served_model}", "--model-class", "litellm_textbased",
                "-c", "swebench_backticks.yaml",
                "-c", f"model.model_kwargs.api_base={args.base_url.rstrip('/')}",
                "-c", "model.model_kwargs.temperature=0",
                "-c", "agent.step_limit=40", "-w", str(args.workers), "-o", str(chunk_dir),
            ]
            _run(agent_command, output / f"chunk_{chunk_number:02d}.agent.log", args.timeout_seconds)
            predictions = chunk_dir / "preds.json"
            if not predictions.is_file():
                raise RuntimeError(f"mini-swe-agent did not produce {predictions}")
            grade_command = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "-d", "princeton-nlp/SWE-bench_Verified", "-s", "test",
                "-p", str(predictions), "--instance_ids", *chunk_ids,
                "--run_id", f"{args.run_id}_c{chunk_number}",
                "--max_workers", str(args.grader_workers), "--cache_level", "base",
                "--clean", "True", "--report_dir", str(chunk_dir),
            ]
            _run(grade_command, output / f"chunk_{chunk_number:02d}.grader.log", args.timeout_seconds)
            raw_report = _find_report(chunk_dir, f"{args.run_id}_c{chunk_number}")
            chunk_result = _normalize_report(raw_report, chunk_ids)
            _write_json(report_receipt, chunk_result)
        submitted.update(chunk_result["submitted_ids"])
        resolved.update(chunk_result["resolved_ids"])
        errors.update(chunk_result["error_ids"])

    expected = set(instance_ids)
    if submitted != expected:
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        raise RuntimeError(f"official cohort mismatch; missing={missing}, extra={extra}")
    ordered_resolved = [item for item in instance_ids if item in resolved]
    result = {
        "official_grader": True,
        "score": len(ordered_resolved) / len(instance_ids),
        "resolved": len(ordered_resolved),
        "total": len(instance_ids),
        "task_ids": instance_ids,
        "configuration_differences": receipt["configuration_differences"],
        "grader_artifact": str(output / "official_aggregate.json"),
    }
    _write_json(output / "official_aggregate.json", {
        "submitted_ids": instance_ids,
        "resolved_ids": ordered_resolved,
        "error_ids": [item for item in instance_ids if item in errors],
    })
    _write_json(output / "official_result.json", result)
    _write_null_safe_task_rows(output / "results.jsonl", args, instance_ids, resolved, errors)
    return output / "official_result.json"


def _nvidia_gpu() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _is_h100_80gb(gpu: str | None) -> bool:
    """Accept nvidia-smi's MiB form while rejecting smaller H100 variants."""

    if not gpu or "H100" not in gpu.upper():
        return False
    memory_values = [int(value) for value in re.findall(r"(\d+)\s*MiB", gpu, re.IGNORECASE)]
    return bool(memory_values) and max(memory_values) >= 79_000


def _run(command: list[str], log: Path, timeout_seconds: int) -> None:
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "dummy")
    environment.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment,
        timeout=timeout_seconds, check=False,
    )
    log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}; see {log}")


def _find_report(directory: Path, run_id: str) -> Path:
    matches = sorted(directory.glob(f"*.{run_id}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one official report for {run_id}, found {len(matches)}")
    return matches[0]


def _normalize_report(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    submitted = set(payload.get("submitted_ids") or ())
    if submitted != set(expected_ids):
        raise RuntimeError(f"grader report {path} does not match its frozen chunk")
    return {
        "submitted_ids": expected_ids,
        "resolved_ids": [item for item in expected_ids if item in set(payload.get("resolved_ids") or ())],
        "error_ids": [item for item in expected_ids if item in set(payload.get("error_ids") or ())],
    }


def _write_null_safe_task_rows(
    destination: Path, args: argparse.Namespace, instance_ids: list[str],
    resolved: set[str], errors: set[str],
) -> None:
    fields = {
        "logical_input_tokens": None, "physical_input_tokens": None,
        "output_tokens": None, "materialized_tokens": None, "selected_tokens": None,
        "token_saving_fraction": None, "request_count": None, "step_count": None,
        "wall_time_s": None, "ttft_s": None, "prefill_time_s": None,
        "decode_time_s": None, "tool_time_s": None, "pra_route_time_s": None,
        "pra_materialize_time_s": None, "gateway_overhead_s": None,
        "peak_memory_bytes": None, "kv_bytes": None, "pra_memory_bytes": None,
    }
    rows = []
    for instance_id in instance_ids:
        rows.append({
            "run_id": args.run_id, "benchmark": "SWE-bench Verified",
            "instance_id": instance_id, "harness": "mini-swe-agent",
            "harness_version": "2.4.0", "model": args.model,
            "model_revision": args.model_revision, "engine": "vllm",
            "engine_version": "0.22.1", "quantization": None,
            "dtype": "bfloat16", "mode": "native", "pra_config_id": None,
            "context_budget": 16384, "seed": 0, "resolved": instance_id in resolved,
            "benchmark_score": 1.0 if instance_id in resolved else 0.0,
            **fields, "termination_reason": None,
            "error_type": "official_grader_error" if instance_id in errors else None,
            "empty_patch": None, "invalid_patch": None,
            "trajectory_path": None, "patch_path": None,
        })
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-revision", default="NOT_REPORTED_BY_SOURCE")
    parser.add_argument("--tokenizer-revision", default="NOT_REPORTED_BY_SOURCE")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("no-pra",), default="no-pra")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--grader-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-partial-reproduction", action="store_true")
    result = run(parser.parse_args())
    print(result)


if __name__ == "__main__":
    main()
