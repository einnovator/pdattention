"""Run the source-matched mini-swe-agent fixed-50 no-PRA baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..benchmark import load_benchmark_card
from ..context_treatment import ContextTreatment, TreatmentProxy, session_id_for_messages


EXPECTED_PACKAGES = {
    "mini-swe-agent": "2.4.0",
    "swebench": "4.1.0",
    "vllm": "0.22.1",
}
PINNED_DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"


def package_versions() -> dict[str, str | None]:
    """Return installed versions without converting missing packages to zero-like values."""

    versions: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def package_record_hashes() -> dict[str, str | None]:
    """Fingerprint installed distribution manifests when the source gives only versions."""

    values: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            record = importlib.metadata.distribution(package).read_text("RECORD")
        except importlib.metadata.PackageNotFoundError:
            record = None
        values[package] = hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None
    return values


def preflight(args: argparse.Namespace, card: dict[str, Any]) -> dict[str, Any]:
    """Capture identity and host compatibility before expensive inference starts."""

    versions = package_versions()
    expected_packages = {
        "mini-swe-agent": args.harness_version,
        "swebench": args.grader_version,
    }
    if not args.local_calibration and args.engine == "vllm":
        expected_packages["vllm"] = args.engine_version
    differences = [
        f"{name}={versions[name]!r}; execution requires {expected!r}"
        for name, expected in expected_packages.items()
        if versions[name] != expected
    ]
    if args.model_revision == "NOT_REPORTED_BY_SOURCE":
        differences.append("execution model revision is not pinned")
    if args.tokenizer_revision == "NOT_REPORTED_BY_SOURCE":
        differences.append("execution tokenizer revision is not pinned")
    gpu = _nvidia_gpu()
    if not args.local_calibration and not _is_h100_80gb(gpu):
        differences.append(f"hardware={gpu or 'no NVIDIA GPU detected'}; source used one H100 80GB")
    current_dataset_revision = _dataset_revision()
    if current_dataset_revision != args.benchmark_revision:
        differences.append(
            f"SWE-bench dataset revision={current_dataset_revision!r}; "
            f"execution requires {args.benchmark_revision!r}"
        )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_source_revision": card["source_revision"],
        "benchmark_execution_revision": args.benchmark_revision,
        "benchmark_ids_sha256": card["canonical_ids_sha256"],
        "instance_count": len(card["instance_ids"]),
        "model": args.model,
        "served_model": args.served_model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "engine_versions": versions,
        "package_record_sha256": package_record_hashes(),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "context_limit": args.context_limit,
        "max_steps": args.max_steps,
        "temperature": 0,
        "campaign_mode": args.mode,
        "context_budget_fraction": args.budget_fraction,
        "harness": "mini-swe-agent",
        "harness_version": args.harness_version,
        "harness_config": args.scaffold,
        "model_class": "litellm_textbased",
        "official_grader": args.grading,
        "grader_version": args.grader_version,
        "python": sys.version,
        "os": platform.platform(),
        "gpu": gpu,
        "configuration_differences": differences,
        "source_provenance_limitations": [] if args.local_calibration else [
            "source study did not publish an immutable model revision",
            "source study did not publish an immutable tokenizer revision",
            "source study did not publish package or grader-image hashes",
            "source study did not publish the SWE-bench dataset revision",
        ],
        "exact_environment": not differences,
    }


def run(args: argparse.Namespace) -> Path:
    """Execute all fixed IDs in resumable chunks and normalize official grading."""

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

    proxy = None
    agent_base_url = args.base_url.rstrip("/")
    if args.mode != "no-pra":
        proxy = TreatmentProxy(
            agent_base_url,
            mode=ContextTreatment(args.mode),
            budget_fraction=args.budget_fraction,
            trace_path=output / "request_telemetry.jsonl",
        )
        agent_base_url = proxy.start()
    try:
        return _execute_chunks(args, card, output, agent_base_url)
    finally:
        if proxy is not None:
            proxy.close()


def _execute_chunks(
    args: argparse.Namespace, card: dict[str, Any], output: Path, agent_base_url: str,
) -> Path:
    """Run resumable agent/grader chunks against the direct endpoint or treatment proxy."""

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
                "-c", args.scaffold,
                "-c", f"model.model_kwargs.api_base={agent_base_url}",
                "-c", "model.model_kwargs.temperature=0",
                "-c", f"agent.step_limit={args.max_steps}", "-w", str(args.workers), "-o", str(chunk_dir),
            ]
            dataset_environment = {"HF_DATASETS_CACHE": str(output / "hf_datasets_cache")}
            _run(
                agent_command, output / f"chunk_{chunk_number:02d}.agent.log",
                args.timeout_seconds, extra_environment=dataset_environment,
            )
            predictions = chunk_dir / "preds.json"
            if not predictions.is_file():
                raise RuntimeError(f"mini-swe-agent did not produce {predictions}")
            grade_command = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "-d", card["dataset"], "-s", card["split"],
                "-p", str(predictions), "--instance_ids", *chunk_ids,
                "--run_id", f"{args.run_id}_c{chunk_number}",
                "--max_workers", str(args.grader_workers), "--cache_level", "base",
                "--clean", "True", "--report_dir", str(chunk_dir),
            ]
            grader_wall_time_s = _run(
                grade_command, output / f"chunk_{chunk_number:02d}.grader.log",
                args.timeout_seconds, extra_environment=dataset_environment,
                cwd=chunk_dir,
            )
            raw_report = _find_report(chunk_dir, f"{args.run_id}_c{chunk_number}")
            chunk_result = _normalize_report(raw_report, chunk_ids)
            chunk_result["grader_wall_time_s"] = grader_wall_time_s
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
        "execution_identity": {
            "cohort_sha256": card["canonical_ids_sha256"],
            "benchmark_revision": args.benchmark_revision,
            "harness": "mini-swe-agent",
            "harness_version": args.harness_version,
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "engine": args.engine,
            "engine_version": args.engine_version,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "scaffold": args.scaffold,
            "context_limit": args.context_limit,
            "max_steps": args.max_steps,
            "temperature": 0.0,
            "function_calling": False,
            "prefix_caching": False,
            "grading": args.grading,
        },
    }
    _write_json(output / "official_aggregate.json", {
        "submitted_ids": instance_ids,
        "resolved_ids": ordered_resolved,
        "error_ids": [item for item in instance_ids if item in errors],
    })
    _write_json(output / "official_result.json", result)
    _write_task_rows(output / "results.jsonl", output, args, instance_ids, resolved, errors)
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


def _dataset_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info("princeton-nlp/SWE-bench_Verified").sha
    except Exception:  # The receipt records an unavailable identity as a blocking difference.
        return None


def _is_h100_80gb(gpu: str | None) -> bool:
    """Accept nvidia-smi's MiB form while rejecting smaller H100 variants."""

    if not gpu or "H100" not in gpu.upper():
        return False
    memory_values = [int(value) for value in re.findall(r"(\d+)\s*MiB", gpu, re.IGNORECASE)]
    return bool(memory_values) and max(memory_values) >= 79_000


def _run(
    command: list[str], log: Path, timeout_seconds: int,
    *, extra_environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> float:
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "dummy")
    environment.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.update(extra_environment or {})
    started = time.perf_counter()
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment,
        cwd=cwd, timeout=timeout_seconds, check=False,
    )
    elapsed = time.perf_counter() - started
    log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}; see {log}")
    return elapsed


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


def _write_task_rows(
    destination: Path, output: Path, args: argparse.Namespace, instance_ids: list[str],
    resolved: set[str], errors: set[str],
) -> None:
    """Join agent-visible trajectories and proxy traces with official outcomes."""

    traces_by_session = _load_treatment_traces(output / "request_telemetry.jsonl")
    unavailable = {
        "wall_time_s": None, "ttft_s": None, "prefill_time_s": None,
        "decode_time_s": None, "tool_time_s": None, "pra_route_time_s": None,
        "pra_materialize_time_s": None, "gateway_overhead_s": None,
        "peak_memory_bytes": None, "kv_bytes": None, "pra_memory_bytes": None,
    }
    rows = []
    for task_index, instance_id in enumerate(instance_ids):
        trajectory_path = _find_trajectory(output, instance_id)
        trajectory = _trajectory_metrics(trajectory_path) if trajectory_path else {}
        chunk_number = task_index // args.chunk_size
        chunk_receipt = _read_json(output / f"chunk_{chunk_number:02d}" / "official_chunk_result.json")
        grader_error_type = _grader_error_type(
            output / f"chunk_{chunk_number:02d}.grader.log", instance_id
        ) if instance_id in errors else None
        task_traces = traces_by_session.get(trajectory.get("session_id"), [])
        trace = _aggregate_traces(task_traces)
        prompt_tokens = trajectory.get("cumulative_prompt_tokens")
        logical_tokens = prompt_tokens if args.mode == "no-pra" else None
        physical_tokens = prompt_tokens
        patch = str(trajectory.get("patch") or "")
        rows.append({
            "run_id": args.run_id, "benchmark": "SWE-bench Verified",
            "instance_id": instance_id, "harness": "mini-swe-agent",
            "harness_version": args.harness_version, "model": args.model,
            "model_revision": args.model_revision, "engine": args.engine,
            "engine_version": args.engine_version, "quantization": args.quantization,
            "dtype": args.dtype, "mode": args.mode,
            "pra_config_id": "swebench-balanced-v1" if args.mode == "gateway-pra" else None,
            "context_budget": args.context_limit, "seed": 0, "resolved": instance_id in resolved,
            "benchmark_score": 1.0 if instance_id in resolved else 0.0,
            "logical_input_tokens": logical_tokens,
            "physical_input_tokens": physical_tokens,
            "cumulative_prompt_tokens": prompt_tokens,
            "unique_context_tokens_estimate": trajectory.get("max_prompt_tokens"),
            "repeated_context_tokens_estimate": trajectory.get("repeated_context_tokens_estimate"),
            "repeated_context_fraction_estimate": trajectory.get("repeated_context_fraction_estimate"),
            "context_estimate_semantics": "max_prompt_under_accumulating_minisweagent_trajectory",
            "max_prompt_tokens": trajectory.get("max_prompt_tokens"),
            "logical_input_tokens_estimate": trace.get("logical_input_tokens_estimate"),
            "physical_input_tokens_estimate": trace.get("physical_input_tokens_estimate"),
            "selected_tokens_estimate": trace.get("selected_tokens_estimate"),
            "tokens_avoided_estimate": trace.get("tokens_avoided_estimate"),
            "token_saving_fraction_estimate": trace.get("token_saving_fraction_estimate"),
            "token_estimator": trace.get("token_estimator"),
            "output_tokens": trajectory.get("output_tokens"),
            "materialized_tokens": None,
            "selected_tokens": None,
            "token_saving_fraction": 0.0 if args.mode == "no-pra" and prompt_tokens is not None else None,
            "request_count": trajectory.get("model_call_count"),
            "model_call_count": trajectory.get("model_call_count"),
            "step_count": trajectory.get("model_call_count"),
            "trajectory_length": trajectory.get("model_call_count"),
            "tool_call_count": trajectory.get("tool_call_count"),
            "commands_executed": trajectory.get("commands_executed"),
            "files_inspected": trajectory.get("files_inspected"),
            "unique_files_inspected": trajectory.get("unique_files_inspected"),
            "files_modified": trajectory.get("files_modified"),
            "modified_file_paths": trajectory.get("modified_file_paths"),
            **unavailable,
            "wall_time_s": trajectory.get("wall_time_s"),
            "tool_time_s": trajectory.get("tool_time_s"),
            "grader_time_s": chunk_receipt.get("grader_wall_time_s"),
            "pra_route_time_s": trace.get("route_time_s"),
            "termination_reason": trajectory.get("termination_reason"),
            "error_type": grader_error_type,
            "grader_outcome": (
                "error" if instance_id in errors
                else "resolved" if instance_id in resolved
                else "unresolved"
            ),
            "empty_patch": not bool(patch.strip()) if trajectory_path else None,
            "invalid_patch": grader_error_type == "patch_apply_failed",
            "patch_bytes": len(patch.encode("utf-8")) if trajectory_path else None,
            "patch_lines": len(patch.splitlines()) if trajectory_path else None,
            "trajectory_path": trajectory_path.as_posix() if trajectory_path else None,
            "patch_path": trajectory_path.as_posix() if trajectory_path else None,
        })
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _find_trajectory(output: Path, instance_id: str) -> Path | None:
    matches = sorted(output.glob(f"chunk_*/*/{instance_id}.traj.json"))
    if not matches:
        matches = sorted(output.rglob(f"{instance_id}.traj.json"))
    return matches[0] if matches else None


def _trajectory_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages") or []
    calls = [
        row for row in messages
        if row.get("role") == "assistant" and (row.get("extra") or {}).get("response")
    ]
    prompt_per_call = [
        int((((row.get("extra") or {}).get("response") or {}).get("usage") or {}).get("prompt_tokens") or 0)
        for row in calls
    ]
    output_tokens = sum(
        int((((row.get("extra") or {}).get("response") or {}).get("usage") or {}).get("completion_tokens") or 0)
        for row in calls
    )
    cumulative = sum(prompt_per_call)
    unique = max(prompt_per_call, default=0)
    repeated = max(0, cumulative - unique)
    timestamps = [
        float((row.get("extra") or {}).get("timestamp"))
        for row in messages if (row.get("extra") or {}).get("timestamp") is not None
    ]
    tool_time = 0.0
    previous_assistant_time: float | None = None
    for row in messages:
        stamp = (row.get("extra") or {}).get("timestamp")
        if row.get("role") == "assistant" and stamp is not None:
            previous_assistant_time = float(stamp)
        elif row.get("role") == "user" and stamp is not None and previous_assistant_time is not None:
            tool_time += max(0.0, float(stamp) - previous_assistant_time)
            previous_assistant_time = None
    task_messages = [row for row in messages if row.get("role") != "exit"]
    commands = [
        str(action.get("command") or "")
        for row in calls for action in ((row.get("extra") or {}).get("actions") or ())
    ]
    mentioned_paths = sorted({
        match.group(0).lstrip("./")
        for command in commands
        for match in re.finditer(r"(?:\.?\.?/)?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]+", command)
        if match.group(0) != "patch.txt"
    })
    patch = str((payload.get("info") or {}).get("submission") or "")
    modified_paths = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)))
    return {
        "session_id": session_id_for_messages(task_messages) if task_messages else None,
        "cumulative_prompt_tokens": cumulative,
        "max_prompt_tokens": unique,
        "repeated_context_tokens_estimate": repeated,
        "repeated_context_fraction_estimate": repeated / cumulative if cumulative else 0.0,
        "output_tokens": output_tokens,
        "model_call_count": len(calls),
        "tool_call_count": sum(len(((row.get("extra") or {}).get("actions") or ())) for row in calls),
        "commands_executed": len(commands),
        "files_inspected": len(mentioned_paths),
        "unique_files_inspected": mentioned_paths,
        "files_modified": len(modified_paths),
        "modified_file_paths": modified_paths,
        "wall_time_s": max(timestamps) - min(timestamps) if len(timestamps) > 1 else None,
        "tool_time_s": tool_time if timestamps else None,
        "termination_reason": (payload.get("info") or {}).get("exit_status"),
        "patch": patch,
    }


def _load_treatment_traces(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not path.is_file():
        return grouped
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row.get("session_id")), []).append(row)
    return grouped


def _aggregate_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    logical = sum(int(row.get("logical_input_tokens_estimate") or 0) for row in rows)
    physical = sum(int(row.get("physical_input_tokens_estimate") or 0) for row in rows)
    return {
        "logical_input_tokens_estimate": logical,
        "physical_input_tokens_estimate": physical,
        "selected_tokens_estimate": sum(int(row.get("selected_tokens_estimate") or 0) for row in rows),
        "tokens_avoided_estimate": max(0, logical - physical),
        "token_saving_fraction_estimate": max(0, logical - physical) / logical if logical else 0.0,
        "route_time_s": sum(float(row.get("route_time_s") or 0) for row in rows),
        "token_estimator": rows[0].get("token_estimator"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _grader_error_type(log: Path, instance_id: str) -> str:
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if instance_id in text and "Patch Apply Failed" in text:
        return "patch_apply_failed"
    return "official_grader_error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-revision", default="NOT_REPORTED_BY_SOURCE")
    parser.add_argument("--tokenizer-revision", default="NOT_REPORTED_BY_SOURCE")
    parser.add_argument("--benchmark-revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--engine-version", default=EXPECTED_PACKAGES["vllm"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--harness-version", default=EXPECTED_PACKAGES["mini-swe-agent"])
    parser.add_argument("--grader-version", default=EXPECTED_PACKAGES["swebench"])
    parser.add_argument("--scaffold", default="swebench_backticks.yaml")
    parser.add_argument("--grading", default="SWE-bench 4.1.0 official Docker harness")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode", choices=("no-pra", *[mode.value for mode in ContextTreatment]),
        default="no-pra",
    )
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--grader-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-partial-reproduction", action="store_true")
    parser.add_argument(
        "--local-calibration",
        action="store_true",
        help="Validate a pinned local configuration without imposing the H100 source host.",
    )
    result = run(parser.parse_args())
    print(result)


if __name__ == "__main__":
    main()
