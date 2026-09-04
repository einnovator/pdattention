"""Run an ordinary R2E-Gym agent and grade its patches with SWE-bench."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def trajectories_to_predictions(source: Path, destination: Path, model: str) -> list[dict[str, Any]]:
    """Convert R2E-Gym trajectories without inspecting hidden benchmark fields."""

    predictions: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trajectory = json.loads(line)
        dataset_row = trajectory.get("ds") or {}
        instance_id = dataset_row.get("instance_id")
        if not instance_id:
            raise ValueError("R2E-Gym trajectory does not contain ds.instance_id")
        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": trajectory.get("output_patch") or "",
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8"
    )
    return predictions


def normalize_official_report(
    report: Path,
    destination: Path,
    *,
    configuration_differences: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reduce the official SWE-bench report to the campaign receipt contract."""

    payload = json.loads(report.read_text(encoding="utf-8"))
    submitted = tuple(payload.get("submitted_ids") or ())
    resolved = len(payload.get("resolved_ids") or ())
    total = len(submitted)
    if total == 0:
        raise ValueError("official SWE-bench report contains no submitted instances")
    receipt = {
        "official_grader": True,
        "score": resolved / total,
        "resolved": resolved,
        "total": total,
        "task_ids": list(submitted),
        "configuration_differences": list(configuration_differences),
        "grader_artifact": str(report),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def write_task_results(
    trajectories: Path,
    aggregate_report: Path,
    destination: Path,
    *,
    model: str,
    model_revision: str,
    engine: str,
    engine_version: str,
    quantization: str | None,
    harness_version: str,
) -> list[dict[str, Any]]:
    """Join visible trajectory telemetry with official per-instance outcomes."""

    report = json.loads(aggregate_report.read_text(encoding="utf-8"))
    resolved_ids = set(report.get("resolved_ids") or ())
    error_ids = set(report.get("error_ids") or ())
    rows: list[dict[str, Any]] = []
    for line in trajectories.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trajectory = json.loads(line)
        instance_id = (trajectory.get("ds") or {}).get("instance_id")
        steps = trajectory.get("trajectory_steps") or []
        prompt_tokens = sum(int(step.get("token_usage_prompt") or 0) for step in steps)
        output_tokens = sum(int(step.get("token_usage_completion") or 0) for step in steps)
        inference_s = sum(float(step.get("llm_exec_time") or 0) for step in steps)
        tool_s = sum(float(step.get("env_exec_time") or 0) for step in steps)
        wall_s = float(steps[-1].get("total_time_traj") or 0) if steps else 0.0
        rows.append({
            "run_id": f"{instance_id}:no-pra",
            "benchmark": "SWE-bench Verified",
            "instance_id": instance_id,
            "harness": "R2E-Gym",
            "harness_version": harness_version,
            "model": model,
            "model_revision": model_revision,
            "engine": engine,
            "engine_version": engine_version,
            "quantization": quantization,
            "mode": "native",
            "pra_config_id": None,
            "context_budget": int(trajectory.get("max_token_limit") or 0),
            "seed": 42,
            "resolved": instance_id in resolved_ids,
            "benchmark_score": 1.0 if instance_id in resolved_ids else 0.0,
            "logical_input_tokens": prompt_tokens,
            "physical_input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "materialized_tokens": 0,
            "token_saving_fraction": 0.0,
            "request_count": len(steps),
            "wall_time_s": wall_s,
            "prefill_time_s": None,
            "decode_time_s": inference_s,
            "tool_time_s": tool_s,
            "pra_route_time_s": 0.0,
            "pra_materialize_time_s": 0.0,
            "gateway_overhead_s": 0.0,
            "peak_memory_bytes": None,
            "kv_bytes": None,
            "pra_memory_bytes": 0,
            "cost_usd": 0.0,
            "termination_reason": trajectory.get("exit_reason"),
            "error_type": "official_grader_error" if instance_id in error_ids else None,
            "trajectory_path": trajectories.as_posix(),
            "patch_path": trajectories.as_posix(),
        })
    destination.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    return rows


def locate_official_report(output: Path, root: Path, run_id: str) -> Path:
    """Find and retain the aggregate report across SWE-bench versions."""

    reports = sorted(output.glob(f"*.{run_id}.json"))
    if not reports:
        reports = sorted(root.glob(f"*.{run_id}.json"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one official report, found {len(reports)}")
    raw_report = output / reports[0].name
    if reports[0] != raw_report:
        raw_report.write_bytes(reports[0].read_bytes())
    return raw_report


def execute(args: argparse.Namespace) -> Path:
    """Run no-PRA inference, conversion, and the official container grader."""

    if args.mode != "no-pra":
        raise ValueError("this runner intentionally supports only the no-PRA baseline")
    root = args.r2egym_root.resolve()
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"R2E-Gym environment is missing: {python}")
    output = args.output.resolve()
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = args.api_key
    environment["LLM_BASE_URL"] = args.base_url.rstrip("/")
    source_root = str(root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    _write_run_manifest(args, root, output)

    inference = [
        str(python), "-m", "r2egym.agenthub.run.edit", "runagent_multiple",
        "--dataset", args.dataset, "--split", args.split,
        "--start_idx", str(args.start_idx), "--k", str(args.count),
        "--traj_dir", str(trajectory_dir), "--exp_name", args.run_id,
        "--llm_name", f"openai/{args.served_model}",
        "--scaffold", args.scaffold, "--backend", "docker",
        "--use_fn_calling", "False", "--temperature", str(args.temperature),
        "--max_steps", str(args.max_steps),
        "--max_steps_absolute", str(args.max_steps_absolute),
        "--max_workers", str(args.workers),
        "--max_reward_calc_time", str(args.grader_timeout),
        "--max_tokens", str(args.context_limit), "--use_existing", "True",
    ]
    _run(inference, cwd=root, environment=environment, log=output / "inference.log")

    trajectories = trajectory_dir / f"{args.run_id}.jsonl"
    predictions = output / "predictions.jsonl"
    rows = trajectories_to_predictions(trajectories, predictions, args.model)
    if len(rows) != args.count:
        raise RuntimeError(f"expected {args.count} trajectories, found {len(rows)}")

    grade = [
        str(python), "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--split", "test", "--predictions_path", str(predictions),
        "--max_workers", str(args.workers), "--run_id", args.run_id,
        "--timeout", str(args.grader_timeout), "--report_dir", str(output),
        "--cache_level", "env", "--clean", "false",
    ]
    _run(grade, cwd=root, environment=environment, log=output / "official_grader.log")
    # SWE-bench 3.0.2 may ignore report_dir for the aggregate report and write
    # it in the process working directory. Accept either documented location,
    # then copy the raw grader receipt into the campaign artifact directory.
    raw_report = locate_official_report(output, root, args.run_id)
    normalize_official_report(
        raw_report, output / "official_result.json",
        configuration_differences=tuple(args.configuration_difference),
    )
    write_task_results(
        trajectories, raw_report, output / "results.jsonl",
        model=args.model, model_revision=args.model_revision,
        engine=args.engine, engine_version=args.engine_version,
        quantization=args.quantization, harness_version=args.harness_version,
    )
    return output / "official_result.json"


def _write_run_manifest(args: argparse.Namespace, root: Path, output: Path) -> None:
    """Capture exact baseline identity without serializing endpoint credentials."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip())
    runner_bytes = Path(__file__).read_bytes()
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "native",
        "pra_enabled": False,
        "gateway_enabled": False,
        "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
        "r2egym_commit": commit,
        "r2egym_dirty": dirty,
        "swebench_version": importlib.metadata.version("swebench"),
        "dataset": args.dataset,
        "split": args.split,
        "start_idx": args.start_idx,
        "count": args.count,
        "shuffle_seed": 42,
        "model": args.model,
        "model_revision": args.model_revision,
        "served_model": args.served_model,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "quantization": args.quantization,
        "harness_version": args.harness_version,
        "scaffold": args.scaffold,
        "context_limit": args.context_limit,
        "max_steps": args.max_steps,
        "max_steps_absolute": args.max_steps_absolute,
        "temperature": args.temperature,
        "workers": args.workers,
        "grader": "swebench.harness.run_evaluation",
        "grader_timeout": args.grader_timeout,
        "configuration_differences": args.configuration_difference,
        "hardware": {
            "node": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _run(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}; see {log}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2egym-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="NOT_RECORDED")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--engine-version", default="NOT_RECORDED")
    parser.add_argument("--quantization")
    parser.add_argument("--harness-version", default="NOT_RECORDED")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="R2E-Gym/SWE-Bench-Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scaffold", default="r2egym")
    parser.add_argument("--context-limit", type=int, default=65536)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-steps-absolute", type=int, default=100)
    parser.add_argument("--grader-timeout", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mode", choices=("no-pra",), default="no-pra")
    parser.add_argument("--configuration-difference", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(execute(args))


if __name__ == "__main__":
    main()
